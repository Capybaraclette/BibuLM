//! Headless node composition: database, hardware monitor, supervisor, downloads, Link and gateway.

use bibulm_core::{AppSettings, Database, default_data_dir};
use bibulm_hardware::HardwareMonitor;
use bibulm_link::LinkManager;
use bibulm_models::{DownloadManager, JobKind, JobStatus};
use bibulm_runtime::{HardwareMatchContext, ModelSupervisor};
use bibulm_server::{ApiGateway, GatewayContext, LinkHardwareSnapshot};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::RwLock;

pub struct HeadlessNode {
    pub data_dir: PathBuf,
    pub db: Database,
    pub settings: Arc<RwLock<AppSettings>>,
    pub hardware: Arc<Mutex<HardwareMonitor>>,
    pub supervisor: Arc<ModelSupervisor>,
    pub downloads: Arc<DownloadManager>,
    pub link: Arc<LinkManager>,
    pub hw_snapshot: Arc<RwLock<LinkHardwareSnapshot>>,
    pub gateway: Arc<ApiGateway>,
}

impl HeadlessNode {
    /// Opens (creating if needed) the data directory and wires every subsystem together.
    pub fn open() -> anyhow::Result<Self> {
        let data_dir = default_data_dir();
        std::fs::create_dir_all(&data_dir)?;
        let db_path = data_dir.join("bibulm.db");

        let db = Database::open(&db_path)?;
        let settings_val = db.load_settings().unwrap_or_default();
        let _ = std::fs::create_dir_all(settings_val.primary_models_dir());
        let _ = std::fs::create_dir_all(&settings_val.runtimes_dir);
        let _ = std::fs::create_dir_all(&settings_val.logs_dir);

        let link_cfg = settings_val.link.clone();
        let disabled = settings_val.disabled_gpu_ids.clone();
        let settings = Arc::new(RwLock::new(settings_val));

        let mut monitor = HardwareMonitor::new();
        let snapshot = LinkHardwareSnapshot::from_monitor(&mut monitor, &disabled);
        let hw_snapshot = Arc::new(RwLock::new(snapshot));
        let hardware = Arc::new(Mutex::new(monitor));

        let supervisor = Arc::new(ModelSupervisor::new());
        let downloads = Arc::new(DownloadManager::new(db.clone(), 2));
        let link = Arc::new(LinkManager::new_with_data_dir(&link_cfg, &data_dir));
        let gw_ctx = GatewayContext::new(db.clone(), supervisor.clone(), settings.clone())
            .with_link(link.clone())
            .with_hw_snapshot(hw_snapshot.clone())
            .with_downloads(downloads.clone())
            .with_headless(true);
        let gateway = Arc::new(ApiGateway::new(gw_ctx));

        Ok(Self {
            data_dir,
            db,
            settings,
            hardware,
            supervisor,
            downloads,
            link,
            hw_snapshot,
            gateway,
        })
    }

    /// Hardware capabilities used to pick compatible llama.cpp builds.
    pub async fn hardware_match_context(&self) -> HardwareMatchContext {
        self.hw_snapshot.read().await.to_runtime_match_context()
    }

    /// Total VRAM / RAM in bytes used for model fit verdicts.
    pub fn total_memory_bytes(&self, disabled: &[u32]) -> (u64, u64) {
        let Ok(hw) = self.hardware.lock() else {
            return (0, 0);
        };
        let prof = hw.profile();
        let vram: u64 = prof
            .gpus
            .iter()
            .filter(|g| !disabled.contains(&g.device_id))
            .map(|g| g.vram_total_bytes)
            .sum();
        (vram, prof.ram_total_bytes)
    }

    /// Starts the telemetry refresh loop and the completed-download watcher.
    pub fn spawn_background_tasks(&self, announce: bool) {
        let hardware = self.hardware.clone();
        let supervisor = self.supervisor.clone();
        let settings = self.settings.clone();
        let snapshot = self.hw_snapshot.clone();
        let link = self.link.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(2));
            let mut last_epoch = supervisor.memory_epoch();
            loop {
                interval.tick().await;
                let disabled = settings.read().await.disabled_gpu_ids.clone();
                let pids = supervisor.loaded_pids().await;
                let epoch = supervisor.memory_epoch();
                let epoch_changed = epoch != last_epoch;
                last_epoch = epoch;
                let sampled = match hardware.lock() {
                    Ok(mut hw) => {
                        let snap = if epoch_changed {
                            hw.reset_memory_average(&disabled)
                        } else {
                            hw.sample(&disabled)
                        };
                        let pid_mem = hw.sample_process_memory(&pids, &disabled);
                        let budget = hw.memory_budget(&disabled);
                        Some((snap, pid_mem, budget))
                    }
                    Err(_) => None,
                };
                let Some((s, pid_mem, budget)) = sampled else {
                    continue;
                };
                const MIB: f64 = 1024.0 * 1024.0;
                let shared_mib = s.shared_gpu_mem_used_bytes as f64 / MIB;
                let _ = supervisor
                    .sync_live_telemetry(s.wddm_spillover_warning, shared_mib, &pid_mem)
                    .await;
                snapshot.write().await.apply_sample(&s, &budget);
                if epoch_changed {
                    link.notify_peers_state_changed();
                }
            }
        });

        let mut rx = self.downloads.subscribe();
        let gateway = self.gateway.clone();
        let link = self.link.clone();
        tokio::spawn(async move {
            while let Ok(job) = rx.recv().await {
                if job.status == JobStatus::Completed && job.kind == JobKind::Model {
                    gateway.invalidate_models_cache().await;
                    link.notify_peers_state_changed();
                    if announce {
                        println!("[download] completed: {}", job.title);
                    }
                } else if job.status == JobStatus::Failed && announce {
                    println!(
                        "[download] failed: {} ({})",
                        job.title,
                        job.error.as_deref().unwrap_or("unknown error")
                    );
                }
            }
        });
    }
}

/// Resolves when the process receives Ctrl+C (or SIGTERM on Unix).
pub async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        match signal(SignalKind::terminate()) {
            Ok(mut term) => {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {}
                    _ = term.recv() => {}
                }
            }
            Err(_) => {
                let _ = tokio::signal::ctrl_c().await;
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}
