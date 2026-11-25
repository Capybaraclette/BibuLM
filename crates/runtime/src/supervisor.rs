use crate::cli_builder::build_llama_server_args;
use crate::placement_parser::parse_placement_line;
use chrono::{DateTime, Utc};
use bibulm_core::{
    LoadConfig, MemoryComponentBreakdown, MemoryPlacementReport, PlacementStatus,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::net::TcpListener;
use std::path::Path;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;
use thiserror::Error;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{Mutex, RwLock, broadcast};
use tracing::{info, warn};

const LOG_RING_CAPACITY: usize = 2000;

#[derive(Debug, Error)]
pub enum SupervisorError {
    #[error("Failed to spawn llama-server at {path}: {source}")]
    SpawnFailed {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("Runtime binary not found: {0}")]
    RuntimeNotFound(String),
    #[error("No ephemeral TCP port available")]
    NoFreePort,
    #[error("llama-server exited early (code {code:?}) before becoming healthy:\n{tail}")]
    ExitedEarly { code: Option<i32>, tail: String },
    #[error("Timed out waiting for llama-server to become ready:\n{tail}")]
    StartupTimeout { tail: String },
    #[error("Model not loaded: {0}")]
    NotLoaded(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerLogEntry {
    pub timestamp: String,
    /// `"INFO"`, `"WARN"`, `"ERROR"`, `"LLAMA"`
    pub level: String,
    pub source: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadedModelInfo {
    pub model_id: String,
    pub display_name: String,
    pub alias: Option<String>,
    pub runtime_id: String,
    pub runtime_tag: String,
    pub port: u16,
    pub pid: Option<u32>,
    pub is_jit: bool,
    pub loaded_at: String,
    pub last_used_at: String,
    pub idle_seconds: u64,
    pub active_requests: u32,
    pub placement: MemoryPlacementReport,
    pub placement_status: PlacementStatus,
    pub breakdown: MemoryComponentBreakdown,
    pub load_config: LoadConfig,
    pub cli_args: Vec<String>,
}

struct ManagedInstance {
    model_id: String,
    display_name: String,
    alias: Option<String>,
    runtime_id: String,
    runtime_tag: String,
    port: u16,
    pid: Option<u32>,
    is_jit: bool,
    loaded_at: DateTime<Utc>,
    last_used_at: DateTime<Utc>,
    active_requests: u32,
    placement: Arc<RwLock<MemoryPlacementReport>>,
    load_config: LoadConfig,
    cli_args: Vec<String>,
    child: Child,
}

/// Manages `llama-server` child processes behind a Windows Job Object guard,
/// streams stdout/stderr logs, parses memory placement in real time, and handles health/lifecycle.
pub struct ModelSupervisor {
    instances: RwLock<HashMap<String, ManagedInstance>>,
    load_lock: Mutex<()>,
    placement_dirty: Arc<AtomicBool>,
    /// Monotonically incremented whenever a model is loaded, unloaded, or reaped so the
    /// hardware monitor automatically resets its rolling VRAM/RAM baseline average.
    memory_epoch: AtomicU64,
    log_tx: broadcast::Sender<ServerLogEntry>,
    log_ring: Arc<Mutex<VecDeque<ServerLogEntry>>>,
    #[cfg(target_os = "windows")]
    job_object: Option<WindowsJobGuard>,
}

impl ModelSupervisor {
    pub fn new() -> Self {
        let (log_tx, _) = broadcast::channel(512);
        Self {
            instances: RwLock::new(HashMap::new()),
            load_lock: Mutex::new(()),
            placement_dirty: Arc::new(AtomicBool::new(false)),
            memory_epoch: AtomicU64::new(0),
            log_tx,
            log_ring: Arc::new(Mutex::new(VecDeque::with_capacity(LOG_RING_CAPACITY))),
            #[cfg(target_os = "windows")]
            job_object: WindowsJobGuard::create().ok(),
        }
    }

    pub fn memory_epoch(&self) -> u64 {
        self.memory_epoch.load(Ordering::Acquire)
    }

    fn bump_memory_epoch(&self) {
        self.memory_epoch.fetch_add(1, Ordering::Release);
    }

    pub fn subscribe_logs(&self) -> broadcast::Receiver<ServerLogEntry> {
        self.log_tx.subscribe()
    }

    pub async fn recent_logs(&self) -> Vec<ServerLogEntry> {
        self.log_ring.lock().await.iter().cloned().collect()
    }

    pub async fn clear_logs(&self) {
        self.log_ring.lock().await.clear();
    }

    pub async fn push_log(&self, level: &str, source: &str, message: impl Into<String>) {
        let entry = ServerLogEntry {
            timestamp: Utc::now().format("%Y-%m-%d %H:%M:%S").to_string(),
            level: level.to_string(),
            source: source.to_string(),
            message: message.into(),
        };
        {
            let mut ring = self.log_ring.lock().await;
            if ring.len() >= LOG_RING_CAPACITY {
                ring.pop_front();
            }
            ring.push_back(entry.clone());
        }
        let _ = self.log_tx.send(entry);
    }

    /// Spawns a new `llama-server` child process on an ephemeral loopback port and waits until `/health` returns 200 OK.
    #[allow(clippy::too_many_arguments)]
    pub async fn load_model(
        &self,
        model_id: String,
        display_name: String,
        runtime_id: String,
        runtime_tag: String,
        server_exe: &Path,
        config: LoadConfig,
        estimated_breakdown: Option<MemoryComponentBreakdown>,
        is_jit: bool,
        evict_existing_before_load: bool,
    ) -> Result<LoadedModelInfo, SupervisorError> {
        let _guard = self.load_lock.lock().await;

        if is_jit {
            let mut map = self.instances.write().await;
            if let Some(existing) = map.get_mut(&model_id) {
                if matches!(existing.child.try_wait(), Ok(None)) {
                    existing.last_used_at = Utc::now();
                    return Ok(snapshot_instance(existing).await);
                }
                map.remove(&model_id);
            }
        }

        if !server_exe.is_file() {
            return Err(SupervisorError::RuntimeNotFound(
                server_exe.to_string_lossy().into_owned(),
            ));
        }

        // Unload existing instance with same model_id (or all instances if single-slot policy requested).
        if evict_existing_before_load {
            self.unload_all_inner().await;
        } else {
            self.unload_inner(&model_id).await;
        }

        let port = allocate_ephemeral_port().ok_or(SupervisorError::NoFreePort)?;
        let args = build_llama_server_args(&config, "127.0.0.1", port);

        self.push_log(
            "INFO",
            "SUPERVISOR",
            format!(
                "Loading '{}' via {} ({}) on 127.0.0.1:{}",
                display_name,
                runtime_tag,
                server_exe.display(),
                port
            ),
        )
        .await;

        let mut cmd = Command::new(server_exe);
        cmd.args(&args)
            // Ensure llama.cpp emits load_tensors / memory breakdown lines without ANSI color escapes
            // while remaining safely ignored by older/custom forks that don't read these env vars.
            .env("LLAMA_LOG_VERBOSITY", "4")
            .env("LLAMA_LOG_COLORS", "0")
            .env("NO_COLOR", "1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        if let Some(dir) = server_exe.parent() {
            cmd.current_dir(dir);
        }

        #[cfg(target_os = "windows")]
        {
            const CREATE_NO_WINDOW: u32 = 0x08000000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }

        #[cfg(target_os = "linux")]
        unsafe {
            cmd.pre_exec(|| {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }

        let mut child = cmd.spawn().map_err(|source| SupervisorError::SpawnFailed {
            path: server_exe.to_string_lossy().into_owned(),
            source,
        })?;

        let pid = child.id();

        #[cfg(target_os = "windows")]
        if let (Some(job), Some(p)) = (&self.job_object, pid)
            && let Err(e) = job.assign_pid(p)
        {
            warn!("Could not assign llama-server PID {p} to Windows Job Object: {e}");
        }

        let mut initial_placement = MemoryPlacementReport {
            estimated_breakdown,
            ..Default::default()
        };
        // Set cpu_moe_layers hint from config if applicable
        if let bibulm_core::MoeOffloadMode::FirstNLayersExpertsCpu(n) = config.moe_offload {
            initial_placement.cpu_moe_layers = Some(n);
        }
        let placement = Arc::new(RwLock::new(initial_placement));

        let startup_tail = Arc::new(Mutex::new(VecDeque::<String>::with_capacity(40)));

        // Spawn stdout/stderr pump tasks
        if let Some(stdout) = child.stdout.take() {
            spawn_stream_pump(
                stdout,
                model_id.clone(),
                placement.clone(),
                self.placement_dirty.clone(),
                startup_tail.clone(),
                self.log_tx.clone(),
                self.log_ring.clone(),
            );
        }
        if let Some(stderr) = child.stderr.take() {
            spawn_stream_pump(
                stderr,
                model_id.clone(),
                placement.clone(),
                self.placement_dirty.clone(),
                startup_tail.clone(),
                self.log_tx.clone(),
                self.log_ring.clone(),
            );
        }

        // Poll `/health` until 200 OK or process exits or 180s timeout.
        let health_url = format!("http://127.0.0.1:{port}/health");
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .expect("reqwest client");

        let deadline = tokio::time::Instant::now() + Duration::from_secs(180);
        loop {
            if let Ok(Some(status)) = child.try_wait() {
                let tail = startup_tail
                    .lock()
                    .await
                    .iter()
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("\n");
                self.push_log(
                    "ERROR",
                    "SUPERVISOR",
                    format!("llama-server exited prematurely ({status}): {tail}"),
                )
                .await;
                return Err(SupervisorError::ExitedEarly {
                    code: status.code(),
                    tail,
                });
            }

            if let Ok(resp) = client.get(&health_url).send().await
                && resp.status().is_success()
            {
                // Brief yield so buffered startup log lines drain through stream pumps before snapshot.
                tokio::time::sleep(Duration::from_millis(120)).await;
                break;
            }

            if tokio::time::Instant::now() >= deadline {
                let _ = child.kill().await;
                let tail = startup_tail
                    .lock()
                    .await
                    .iter()
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("\n");
                return Err(SupervisorError::StartupTimeout { tail });
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }

        let now = Utc::now();
        let alias = config.model_alias.clone();
        let inst = ManagedInstance {
            model_id: model_id.clone(),
            display_name: display_name.clone(),
            alias,
            runtime_id,
            runtime_tag,
            port,
            pid,
            is_jit,
            loaded_at: now,
            last_used_at: now,
            active_requests: 0,
            placement: placement.clone(),
            load_config: config,
            cli_args: args,
            child,
        };

        let info_snap = snapshot_instance(&inst).await;
        self.instances.write().await.insert(model_id.clone(), inst);
        self.bump_memory_epoch();
        self.push_log(
            "INFO",
            "SUPERVISOR",
            format!(
                "Model '{}' ready on 127.0.0.1:{} — placement: {:?} (VRAM {:.0} MiB / RAM {:.0} MiB)",
                display_name,
                port,
                info_snap.placement_status,
                info_snap.placement.total_gpu_mib(),
                info_snap.placement.total_cpu_mib()
            ),
        )
        .await;
        info!("Loaded model {model_id} on port {port}");
        Ok(info_snap)
    }

    pub async fn unload_model(&self, model_id: &str) -> bool {
        let _guard = self.load_lock.lock().await;
        self.unload_inner(model_id).await
    }

    async fn unload_inner(&self, model_id: &str) -> bool {
        let removed = {
            let mut map = self.instances.write().await;
            if let Some(inst) = map.remove(model_id) {
                Some(inst)
            } else {
                let qlow = model_id.to_ascii_lowercase();
                let matched_key = map
                    .iter()
                    .find(|(_, inst)| {
                        inst.model_id.to_ascii_lowercase() == qlow
                            || inst.display_name.to_ascii_lowercase() == qlow
                            || inst
                                .alias
                                .as_deref()
                                .is_some_and(|a| a.to_ascii_lowercase() == qlow)
                    })
                    .map(|(k, _)| k.clone());
                matched_key.and_then(|k| map.remove(&k))
            }
        };
        if let Some(mut inst) = removed {
            let _ = inst.child.kill().await;
            let _ = inst.child.wait().await;
            self.bump_memory_epoch();
            self.push_log(
                "INFO",
                "SUPERVISOR",
                format!("Unloaded model '{}' (port {})", inst.display_name, inst.port),
            )
            .await;
            true
        } else {
            false
        }
    }

    pub async fn unload_all(&self) {
        let _guard = self.load_lock.lock().await;
        self.unload_all_inner().await;
    }

    async fn unload_all_inner(&self) {
        let mut map = self.instances.write().await;
        let drained: Vec<ManagedInstance> = map.drain().map(|(_, v)| v).collect();
        drop(map);
        let had_any = !drained.is_empty();
        for mut inst in drained {
            let _ = inst.child.kill().await;
            let _ = inst.child.wait().await;
            self.push_log(
                "INFO",
                "SUPERVISOR",
                format!("Unloaded model '{}' (port {})", inst.display_name, inst.port),
            )
            .await;
        }
        if had_any {
            self.bump_memory_epoch();
        }
    }

    pub async fn list_loaded(&self) -> Vec<LoadedModelInfo> {
        let mut map = self.instances.write().await;
        // Reap any crashed child processes automatically
        let mut crashed = Vec::new();
        for (id, inst) in map.iter_mut() {
            if let Ok(Some(_)) = inst.child.try_wait() {
                crashed.push(id.clone());
            }
        }
        for id in &crashed {
            map.remove(id);
        }
        if !crashed.is_empty() {
            self.bump_memory_epoch();
        }
        let mut out = Vec::with_capacity(map.len());
        for inst in map.values() {
            out.push(snapshot_instance(inst).await);
        }
        out.sort_by(|a, b| b.loaded_at.cmp(&a.loaded_at));
        out
    }

    /// Resolves a loaded instance by exact `model_id`, `alias`, or `display_name` (case-insensitive).
    pub async fn resolve_loaded(&self, query: &str) -> Option<LoadedModelInfo> {
        let map = self.instances.read().await;
        if let Some(inst) = map.get(query) {
            return Some(snapshot_instance(inst).await);
        }
        let qlow = query.to_ascii_lowercase();
        for inst in map.values() {
            if inst.model_id.to_ascii_lowercase() == qlow
                || inst.display_name.to_ascii_lowercase() == qlow
                || inst
                    .alias
                    .as_deref()
                    .is_some_and(|a| a.to_ascii_lowercase() == qlow)
            {
                return Some(snapshot_instance(inst).await);
            }
        }
        None
    }

    /// Returns the first loaded model if any (used when a client omits or sends a generic model ID).
    pub async fn first_loaded(&self) -> Option<LoadedModelInfo> {
        let map = self.instances.read().await;
        if let Some(inst) = map.values().next() {
            Some(snapshot_instance(inst).await)
        } else {
            None
        }
    }

    pub async fn begin_request(&self, model_id: &str) {
        let mut map = self.instances.write().await;
        if let Some(inst) = map.get_mut(model_id) {
            inst.active_requests = inst.active_requests.saturating_add(1);
            inst.last_used_at = Utc::now();
        }
    }

    pub async fn end_request(&self, model_id: &str) {
        let mut map = self.instances.write().await;
        if let Some(inst) = map.get_mut(model_id) {
            inst.active_requests = inst.active_requests.saturating_sub(1);
            inst.last_used_at = Utc::now();
        }
    }

    /// Returns OS PIDs of all currently managed `llama-server` instances.
    pub async fn loaded_pids(&self) -> Vec<u32> {
        let map = self.instances.read().await;
        map.values().filter_map(|i| i.pid).collect()
    }

    /// Updates WDDM shared-memory spillover status and live per-PID VRAM/RAM telemetry on all
    /// loaded models. Returns `true` if any loaded model's placement or memory figures changed
    /// (including asynchronous log-parsed updates from `spawn_stream_pump`).
    pub async fn sync_live_telemetry(
        &self,
        spillover: bool,
        shared_mib: f64,
        pid_memory: &HashMap<u32, (Option<f64>, Option<f64>)>,
    ) -> bool {
        let mut changed = self.placement_dirty.swap(false, Ordering::AcqRel);
        let map = self.instances.read().await;
        for inst in map.values() {
            let mut p = inst.placement.write().await;
            if p.wddm_shared_spillover_detected != spillover {
                p.wddm_shared_spillover_detected = spillover;
                changed = true;
            }
            p.shared_gpu_mem_mib = Some(shared_mib);

            if let Some(pid) = inst.pid
                && let Some(&(vram_opt, ram_opt)) = pid_memory.get(&pid)
            {
                if let Some(vram) = vram_opt
                    && p.process_vram_mib.is_none_or(|prev| (prev - vram).abs() >= 4.0)
                {
                    p.process_vram_mib = Some(vram);
                    changed = true;
                }
                if let Some(ram) = ram_opt
                    && p.process_ram_mib.is_none_or(|prev| (prev - ram).abs() >= 4.0)
                {
                    p.process_ram_mib = Some(ram);
                    changed = true;
                }
            }
        }
        changed
    }

    /// Evicts JIT-loaded (or idle) models that have exceeded `max_idle_ttl_minutes` with 0 active requests.
    pub async fn evict_idle_models(&self, max_idle_ttl_minutes: u32) -> Vec<String> {
        let ttl_secs = (max_idle_ttl_minutes as i64).max(1) * 60;
        let now = Utc::now();
        let to_evict: Vec<String> = {
            let map = self.instances.read().await;
            map.iter()
                .filter(|(_, inst)| {
                    inst.is_jit
                        && inst.active_requests == 0
                        && (now - inst.last_used_at).num_seconds() >= ttl_secs
                })
                .map(|(id, _)| id.clone())
                .collect()
        };
        let mut evicted = Vec::new();
        for id in to_evict {
            if self.unload_model(&id).await {
                self.push_log(
                    "INFO",
                    "TTL",
                    format!("Auto-unloaded idle JIT model '{id}' after {max_idle_ttl_minutes}m"),
                )
                .await;
                evicted.push(id);
            }
        }
        evicted
    }
}

impl Default for ModelSupervisor {
    fn default() -> Self {
        Self::new()
    }
}

async fn snapshot_instance(inst: &ManagedInstance) -> LoadedModelInfo {
    let placement = inst.placement.read().await.clone();
    let status = placement.status();
    let breakdown = placement.resolved_breakdown();
    let idle_seconds = (Utc::now() - inst.last_used_at).num_seconds().max(0) as u64;
    LoadedModelInfo {
        model_id: inst.model_id.clone(),
        display_name: inst.display_name.clone(),
        alias: inst.alias.clone(),
        runtime_id: inst.runtime_id.clone(),
        runtime_tag: inst.runtime_tag.clone(),
        port: inst.port,
        pid: inst.pid,
        is_jit: inst.is_jit,
        loaded_at: inst.loaded_at.to_rfc3339(),
        last_used_at: inst.last_used_at.to_rfc3339(),
        idle_seconds,
        active_requests: inst.active_requests,
        placement,
        placement_status: status,
        breakdown,
        load_config: inst.load_config.clone(),
        cli_args: inst.cli_args.clone(),
    }
}

fn spawn_stream_pump<R: tokio::io::AsyncRead + Unpin + Send + 'static>(
    stream: R,
    source_label: String,
    placement: Arc<RwLock<MemoryPlacementReport>>,
    placement_dirty: Arc<AtomicBool>,
    tail: Arc<Mutex<VecDeque<String>>>,
    log_tx: broadcast::Sender<ServerLogEntry>,
    log_ring: Arc<Mutex<VecDeque<ServerLogEntry>>>,
) {
    tokio::spawn(async move {
        let mut lines = BufReader::new(stream).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            {
                let mut report = placement.write().await;
                if parse_placement_line(&mut report, trimmed) {
                    placement_dirty.store(true, Ordering::Release);
                }
            }
            {
                let mut t = tail.lock().await;
                if t.len() >= 40 {
                    t.pop_front();
                }
                t.push_back(trimmed.to_string());
            }
            let level = if trimmed.contains("error") || trimmed.contains("ERROR") || trimmed.contains("CUDA error") {
                "ERROR"
            } else if trimmed.contains("warn") || trimmed.contains("WARN") {
                "WARN"
            } else {
                "LLAMA"
            };
            let entry = ServerLogEntry {
                timestamp: Utc::now().format("%Y-%m-%d %H:%M:%S").to_string(),
                level: level.to_string(),
                source: source_label.clone(),
                message: trimmed.to_string(),
            };
            {
                let mut ring = log_ring.lock().await;
                if ring.len() >= LOG_RING_CAPACITY {
                    ring.pop_front();
                }
                ring.push_back(entry.clone());
            }
            let _ = log_tx.send(entry);
        }
    });
}

fn allocate_ephemeral_port() -> Option<u16> {
    let listener = TcpListener::bind("127.0.0.1:0").ok()?;
    Some(listener.local_addr().ok()?.port())
}

// --- Windows Job Object Guard ---

#[cfg(target_os = "windows")]
struct WindowsJobGuard {
    handle: isize,
}

#[cfg(target_os = "windows")]
unsafe impl Send for WindowsJobGuard {}
#[cfg(target_os = "windows")]
unsafe impl Sync for WindowsJobGuard {}

#[cfg(target_os = "windows")]
impl WindowsJobGuard {
    fn create() -> std::io::Result<Self> {
        use std::ffi::c_void;
        use std::mem::size_of;
        use windows::Win32::System::JobObjects::{
            CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
            SetInformationJobObject,
        };

        unsafe {
            let job = CreateJobObjectW(None, None)
                .map_err(|e| std::io::Error::other(e.to_string()))?;
            let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                &info as *const _ as *const c_void,
                size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            )
            .map_err(|e| std::io::Error::other(e.to_string()))?;
            Ok(Self {
                handle: job.0 as isize,
            })
        }
    }

    fn assign_pid(&self, pid: u32) -> std::io::Result<()> {
        use std::ffi::c_void;
        use windows::Win32::Foundation::{CloseHandle, HANDLE};
        use windows::Win32::System::JobObjects::AssignProcessToJobObject;
        use windows::Win32::System::Threading::{
            OpenProcess, PROCESS_SET_QUOTA, PROCESS_TERMINATE,
        };

        unsafe {
            let proc_handle = OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, false, pid)
                .map_err(|e| std::io::Error::other(e.to_string()))?;
            let job_handle = HANDLE(self.handle as *mut c_void);
            let res = AssignProcessToJobObject(job_handle, proc_handle)
                .map_err(|e| std::io::Error::other(e.to_string()));
            let _ = CloseHandle(proc_handle);
            res
        }
    }
}

#[cfg(target_os = "windows")]
impl Drop for WindowsJobGuard {
    fn drop(&mut self) {
        use std::ffi::c_void;
        use windows::Win32::Foundation::{CloseHandle, HANDLE};
        unsafe {
            let _ = CloseHandle(HANDLE(self.handle as *mut c_void));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn spawn_dummy_child() -> Child {
        #[cfg(target_os = "windows")]
        let mut cmd = {
            let mut c = Command::new("cmd");
            c.args(["/C", "ping", "-n", "10", "127.0.0.1"]);
            c
        };
        #[cfg(not(target_os = "windows"))]
        let mut cmd = {
            let mut c = Command::new("sleep");
            c.arg("10");
            c
        };
        cmd.stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .expect("spawn dummy child")
    }

    #[tokio::test]
    async fn concurrent_jit_load_reuses_existing_instance() {
        let sup = Arc::new(ModelSupervisor::new());
        let model_id = "qwen-concurrent-test".to_string();

        // Hold load_lock while two concurrent JIT load_model calls queue behind it,
        // simulating two parallel requests racing after a cold start / TTL eviction.
        let guard = sup.load_lock.lock().await;

        let sup1 = sup.clone();
        let id1 = model_id.clone();
        let t1 = tokio::spawn(async move {
            sup1.load_model(
                id1,
                "Qwen Test".into(),
                "rt-1".into(),
                "b5200".into(),
                &PathBuf::from("/nonexistent/llama-server"),
                LoadConfig::new(PathBuf::from("model.gguf")),
                None,
                true,
                true,
            )
            .await
        });

        let sup2 = sup.clone();
        let id2 = model_id.clone();
        let t2 = tokio::spawn(async move {
            sup2.load_model(
                id2,
                "Qwen Test".into(),
                "rt-1".into(),
                "b5200".into(),
                &PathBuf::from("/nonexistent/llama-server"),
                LoadConfig::new(PathBuf::from("model.gguf")),
                None,
                true,
                true,
            )
            .await
        });

        // Simulate the first loader finishing and registering the live child before releasing the lock.
        let child = spawn_dummy_child();
        let expected_pid = child.id();
        let now = Utc::now();
        sup.instances.write().await.insert(
            model_id.clone(),
            ManagedInstance {
                model_id: model_id.clone(),
                display_name: "Qwen Test".into(),
                alias: None,
                runtime_id: "rt-1".into(),
                runtime_tag: "b5200".into(),
                port: 43210,
                pid: expected_pid,
                is_jit: true,
                loaded_at: now,
                last_used_at: now,
                active_requests: 0,
                placement: Arc::new(RwLock::new(MemoryPlacementReport::default())),
                load_config: LoadConfig::new(PathBuf::from("model.gguf")),
                cli_args: vec![],
                child,
            },
        );
        drop(guard);

        let (r1, r2) = tokio::join!(t1, t2);
        let info1 = r1.unwrap().expect("first concurrent JIT call should reuse instance");
        let info2 = r2.unwrap().expect("second concurrent JIT call should reuse instance");
        assert_eq!(info1.port, 43210);
        assert_eq!(info2.port, 43210);
        assert_eq!(info1.pid, expected_pid);
        assert_eq!(info2.pid, expected_pid);

        sup.unload_all().await;
    }
}
