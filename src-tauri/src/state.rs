use bibulm_core::{AppSettings, Database, default_data_dir, legacy_data_dir};
use bibulm_hardware::HardwareMonitor;
use bibulm_link::LinkManager;
use bibulm_models::DownloadManager;
use bibulm_runtime::ModelSupervisor;
use bibulm_server::{ApiGateway, GatewayContext, LinkHardwareSnapshot};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use tokio::sync::RwLock;

pub struct AppState {
    pub db: Database,
    pub settings: Arc<RwLock<AppSettings>>,
    pub hardware: Arc<Mutex<HardwareMonitor>>,
    pub supervisor: Arc<ModelSupervisor>,
    pub downloads: Arc<DownloadManager>,
    pub link: Arc<LinkManager>,
    pub link_hw_snapshot: Arc<RwLock<LinkHardwareSnapshot>>,
    pub gateway: Arc<ApiGateway>,
    pub allow_exit: Arc<AtomicBool>,
}

impl AppState {
    pub fn initialize() -> anyhow::Result<Self> {
        let data_dir = default_data_dir();
        std::fs::create_dir_all(&data_dir)?;
        let db_path = data_dir.join("bibulm.db");

        // One-time non-destructive migration from pre-rebrand local database if present
        if !db_path.exists()
            && let Some(old_dir) = legacy_data_dir()
        {
            let old_db = old_dir.join("ruststudio.db");
            if old_db.is_file() {
                if let Err(e) = std::fs::copy(&old_db, &db_path) {
                    tracing::warn!("Could not copy legacy database: {e}");
                } else {
                    tracing::info!("Migrated legacy database into {}", db_path.display());
                }
            }
        }

        let db = Database::open(&db_path)?;

        let settings_val = db.load_settings().unwrap_or_default();
        // Ensure configured directories exist
        let _ = std::fs::create_dir_all(settings_val.primary_models_dir());
        let _ = std::fs::create_dir_all(&settings_val.runtimes_dir);
        let _ = std::fs::create_dir_all(&settings_val.logs_dir);

        let initial_link_cfg = settings_val.link.clone();
        let settings = Arc::new(RwLock::new(settings_val));
        let mut hw_monitor = HardwareMonitor::new();
        let gpu_summary = hw_monitor.profile().format_summary();
        let cpu_logical_cores = hw_monitor.profile().cpu.logical_cores as u32;
        let initial_sample = hw_monitor.sample(&[]);
        let initial_budget = hw_monitor.memory_budget(&[]);
        let (has_nvidia, nvidia_cuda_major, nvidia_cuda_minor, has_amd_gpu, has_intel_gpu) = {
            let prof = hw_monitor.profile();
            let nv = prof.gpus.iter().find(|g| g.vendor == "NVIDIA");
            (
                nv.is_some(),
                nv.and_then(|g| g.cuda_version_major),
                nv.and_then(|g| g.cuda_version_minor),
                prof.gpus.iter().any(|g| g.vendor == "AMD"),
                prof.gpus.iter().any(|g| g.vendor == "Intel"),
            )
        };
        let has_any_gpu = !hw_monitor.profile().gpus.is_empty();
        let has_avx512 = hw_monitor.profile().cpu.has_avx512;
        let has_avx2 = hw_monitor.profile().cpu.has_avx2;
        let link_hw_snapshot = Arc::new(RwLock::new(LinkHardwareSnapshot {
            gpu_summary,
            cpu_logical_cores,
            vram_used_mib: initial_sample.vram_used_bytes as f64 / (1024.0 * 1024.0),
            vram_used_avg_mib: initial_budget.vram_used_avg_mib,
            vram_total_mib: initial_sample.vram_total_bytes as f64 / (1024.0 * 1024.0),
            ram_used_mib: initial_sample.ram_used_bytes as f64 / (1024.0 * 1024.0),
            ram_used_avg_mib: initial_budget.ram_used_avg_mib,
            ram_total_mib: initial_sample.ram_total_bytes as f64 / (1024.0 * 1024.0),
            vram_sample_count: initial_budget.sample_count,
            has_nvidia,
            nvidia_cuda_major,
            nvidia_cuda_minor,
            has_amd_gpu,
            has_intel_gpu,
            has_any_gpu,
            has_avx512,
            has_avx2,
        }));

        let hardware = Arc::new(Mutex::new(hw_monitor));
        let supervisor = Arc::new(ModelSupervisor::new());
        let downloads = Arc::new(DownloadManager::new(db.clone(), 2));
        let link = Arc::new(LinkManager::new(&initial_link_cfg));
        let gw_ctx = GatewayContext::new(db.clone(), supervisor.clone(), settings.clone())
            .with_link(link.clone())
            .with_hw_snapshot(link_hw_snapshot.clone())
            .with_downloads(downloads.clone());
        let gateway = Arc::new(ApiGateway::new(gw_ctx));

        Ok(Self {
            db,
            settings,
            hardware,
            supervisor,
            downloads,
            link,
            link_hw_snapshot,
            gateway,
            allow_exit: Arc::new(AtomicBool::new(false)),
        })
    }
}
