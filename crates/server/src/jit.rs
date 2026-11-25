use axum::http::StatusCode;
use bibulm_core::{AppSettings, Database, LiveMemoryContext, LoadConfig, canonical_file_within};
use bibulm_hardware::{HardwareMonitor, LiveMemoryBudget, TelemetrySnapshot};
use bibulm_link::{LinkManager, RemotePeer};
use bibulm_models::{LocalModelEntry, scan_local_models_with_external};
use bibulm_runtime::{LoadedModelInfo, ModelSupervisor};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

type ModelsCacheEntry = (Instant, Vec<PathBuf>, bool, Vec<LocalModelEntry>);

/// Lightweight hardware summary snapshot updated by the app telemetry loop
/// so `/link/v1/info` and `/link/v1/models/inspect` respond instantaneously without blocking on GPU probes.
#[derive(Debug, Clone, Default)]
pub struct LinkHardwareSnapshot {
    pub gpu_summary: String,
    pub cpu_logical_cores: u32,
    pub vram_used_mib: f64,
    pub vram_used_avg_mib: f64,
    pub vram_total_mib: f64,
    pub ram_used_mib: f64,
    pub ram_used_avg_mib: f64,
    pub ram_total_mib: f64,
    pub vram_sample_count: u32,
    pub has_nvidia: bool,
    pub nvidia_cuda_major: Option<u32>,
    pub nvidia_cuda_minor: Option<u32>,
    pub has_amd_gpu: bool,
    pub has_intel_gpu: bool,
    pub has_any_gpu: bool,
    pub has_avx512: bool,
    pub has_avx2: bool,
}

impl LinkHardwareSnapshot {
    /// Static hardware capabilities plus the current memory sample.
    pub fn from_monitor(hw: &mut HardwareMonitor, disabled_gpu_ids: &[u32]) -> Self {
        let prof = hw.profile();
        let nv = prof.gpus.iter().find(|g| g.vendor == "NVIDIA");
        let mut snap = Self {
            gpu_summary: prof.format_summary(),
            cpu_logical_cores: prof.cpu.logical_cores as u32,
            has_nvidia: nv.is_some(),
            nvidia_cuda_major: nv.and_then(|g| g.cuda_version_major),
            nvidia_cuda_minor: nv.and_then(|g| g.cuda_version_minor),
            has_amd_gpu: prof.gpus.iter().any(|g| g.vendor == "AMD"),
            has_intel_gpu: prof.gpus.iter().any(|g| g.vendor == "Intel"),
            has_any_gpu: !prof.gpus.is_empty(),
            has_avx512: prof.cpu.has_avx512,
            has_avx2: prof.cpu.has_avx2,
            ..Default::default()
        };
        let sample = hw.sample(disabled_gpu_ids);
        let budget = hw.memory_budget(disabled_gpu_ids);
        snap.apply_sample(&sample, &budget);
        snap
    }

    /// Refreshes the live memory fields from a telemetry sample.
    pub fn apply_sample(&mut self, s: &TelemetrySnapshot, budget: &LiveMemoryBudget) {
        const MIB: f64 = 1024.0 * 1024.0;
        self.vram_used_mib = s.vram_used_bytes as f64 / MIB;
        self.vram_used_avg_mib = budget.vram_used_avg_mib;
        self.vram_total_mib = s.vram_total_bytes as f64 / MIB;
        self.ram_used_mib = s.ram_used_bytes as f64 / MIB;
        self.ram_used_avg_mib = budget.ram_used_avg_mib;
        self.ram_total_mib = s.ram_total_bytes as f64 / MIB;
        self.vram_sample_count = budget.sample_count;
    }

    /// Hardware capabilities used to pick compatible `llama.cpp` runtime builds.
    pub fn to_runtime_match_context(&self) -> bibulm_runtime::HardwareMatchContext {
        bibulm_runtime::HardwareMatchContext {
            has_nvidia: self.has_nvidia,
            nvidia_cuda_major: self.nvidia_cuda_major,
            nvidia_cuda_minor: self.nvidia_cuda_minor,
            has_amd_gpu: self.has_amd_gpu,
            has_intel_gpu: self.has_intel_gpu,
            has_any_gpu: self.has_any_gpu,
            has_avx512: self.has_avx512,
            has_avx2: self.has_avx2,
        }
    }
}

/// Target resolved for an incoming OpenAI API inference request.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum ResolvedTarget {
    Local(LoadedModelInfo),
    Remote {
        peer: RemotePeer,
        remote_model_id: String,
    },
}

/// Shared state backing JIT model resolution, Link mesh routing & lifecycle inside the OpenAI gateway.
#[derive(Clone)]
pub struct GatewayContext {
    pub db: Database,
    pub supervisor: Arc<ModelSupervisor>,
    pub settings: Arc<RwLock<AppSettings>>,
    pub http_client: reqwest::Client,
    pub link: Option<Arc<LinkManager>>,
    pub hw_snapshot: Arc<RwLock<LinkHardwareSnapshot>>,
    pub models_cache: Arc<RwLock<Option<ModelsCacheEntry>>>,
    pub downloads: Option<Arc<bibulm_models::DownloadManager>>,
    pub is_headless: bool,
}

impl GatewayContext {
    pub fn new(
        db: Database,
        supervisor: Arc<ModelSupervisor>,
        settings: Arc<RwLock<AppSettings>>,
    ) -> Self {
        let http_client = reqwest::Client::builder()
            .pool_idle_timeout(Duration::from_secs(90))
            .build()
            .expect("gateway reqwest client");
        Self {
            db,
            supervisor,
            settings,
            http_client,
            link: None,
            hw_snapshot: Arc::new(RwLock::new(LinkHardwareSnapshot::default())),
            models_cache: Arc::new(RwLock::new(None)),
            downloads: None,
            is_headless: false,
        }
    }

    pub fn with_downloads(mut self, downloads: Arc<bibulm_models::DownloadManager>) -> Self {
        self.downloads = Some(downloads);
        self
    }

    pub fn with_headless(mut self, is_headless: bool) -> Self {
        self.is_headless = is_headless;
        self
    }

    pub fn with_link(mut self, link: Arc<LinkManager>) -> Self {
        self.link = Some(link);
        self
    }

    pub fn with_hw_snapshot(mut self, hw_snapshot: Arc<RwLock<LinkHardwareSnapshot>>) -> Self {
        self.hw_snapshot = hw_snapshot;
        self
    }

    pub async fn invalidate_models_cache(&self) {
        *self.models_cache.write().await = None;
    }

    /// Returns local models (configured dirs + optional auto-discovered external apps) using a 20s TTL cache.
    pub async fn cached_local_models(&self) -> Vec<LocalModelEntry> {
        let (dirs, auto_discover) = {
            let s = self.settings.read().await;
            (s.models_dirs.clone(), s.auto_discover_external_models)
        };
        {
            let guard = self.models_cache.read().await;
            if let Some((ts, cached_dirs, cached_auto, entries)) = guard.as_ref()
                && ts.elapsed() < Duration::from_secs(20)
                && *cached_dirs == dirs
                && *cached_auto == auto_discover
            {
                return entries.clone();
            }
        }
        let dirs_clone = dirs.clone();
        let entries = tokio::task::spawn_blocking(move || {
            scan_local_models_with_external(&dirs_clone, auto_discover)
        })
        .await
        .unwrap_or_default();
        *self.models_cache.write().await =
            Some((Instant::now(), dirs, auto_discover, entries.clone()));
        entries
    }

    /// Finds a local model matching `query` (by id, display_name, or file stem), falling back to a fresh scan if needed.
    pub async fn find_local_model(&self, query: &str) -> Option<LocalModelEntry> {
        let loaded_id_hint = self
            .supervisor
            .resolve_loaded(query)
            .await
            .map(|l| l.model_id);
        let qlow = query.trim().to_ascii_lowercase();
        let match_fn = |locals: &[LocalModelEntry]| -> Option<LocalModelEntry> {
            locals
                .iter()
                .find(|m| {
                    m.id.to_ascii_lowercase() == qlow
                        || m.display_name.to_ascii_lowercase() == qlow
                        || loaded_id_hint
                            .as_deref()
                            .is_some_and(|lid| m.id.eq_ignore_ascii_case(lid))
                        || m.path
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .is_some_and(|s| s.to_ascii_lowercase() == qlow)
                })
                .cloned()
        };

        let cached = self.cached_local_models().await;
        if let Some(found) = match_fn(&cached) {
            return Some(found);
        }

        // Cache miss: rescan in case a model was just added on disk, but at most every few
        // seconds — unknown names (peer models, generic ids) would otherwise rescan per request.
        let recently_scanned = self
            .models_cache
            .read()
            .await
            .as_ref()
            .is_some_and(|(ts, ..)| ts.elapsed() < Duration::from_secs(5));
        if recently_scanned {
            return None;
        }
        self.invalidate_models_cache().await;
        let fresh = self.cached_local_models().await;
        match_fn(&fresh)
    }

    /// Computes live VRAM/RAM headroom for remote inspection / estimation, reclaiming memory
    /// currently held by `replacing_model_id` if that model is already loaded.
    pub async fn compute_link_live_memory_context(
        &self,
        replacing_model_id: Option<&str>,
    ) -> LiveMemoryContext {
        let hw = self.hw_snapshot.read().await.clone();
        let (reclaim_vram, reclaim_ram) = if let Some(mid) = replacing_model_id {
            if let Some(loaded) = self.supervisor.resolve_loaded(mid).await {
                let v = loaded
                    .placement
                    .process_vram_mib
                    .filter(|&x| x > 1.0)
                    .unwrap_or_else(|| loaded.placement.total_gpu_mib())
                    .max(0.0);
                let r = loaded
                    .placement
                    .process_ram_mib
                    .filter(|&x| x > 1.0)
                    .unwrap_or_else(|| loaded.placement.total_cpu_mib())
                    .max(0.0);
                (v, r)
            } else {
                (0.0, 0.0)
            }
        } else {
            (0.0, 0.0)
        };

        let base_vram_used = if hw.vram_used_avg_mib > 0.0 {
            hw.vram_used_avg_mib
        } else {
            hw.vram_used_mib
        };
        let base_ram_used = if hw.ram_used_avg_mib > 0.0 {
            hw.ram_used_avg_mib
        } else {
            hw.ram_used_mib
        };

        let effective_vram_used = (base_vram_used - reclaim_vram).max(0.0);
        let effective_vram_avail = (hw.vram_total_mib - effective_vram_used)
            .clamp(0.0, hw.vram_total_mib.max(0.0));
        let effective_ram_used = (base_ram_used - reclaim_ram).max(0.0);
        let effective_ram_avail = (hw.ram_total_mib - effective_ram_used)
            .clamp(0.0, hw.ram_total_mib.max(0.0));

        LiveMemoryContext {
            hardware_vram_total_mib: hw.vram_total_mib,
            current_vram_used_mib: effective_vram_used,
            available_vram_mib: effective_vram_avail,
            hardware_ram_total_mib: hw.ram_total_mib,
            current_ram_used_mib: effective_ram_used,
            available_ram_mib: effective_ram_avail,
            vram_sample_count: hw.vram_sample_count,
        }
    }

    /// Resolves either a local model (loaded or JIT) or a remote Tailscale Link peer model.
    /// When `is_link_inbound` is true, never forwards to another remote peer (prevents loops)
    /// and enforces `settings.link.allow_remote_load` before JIT-loading.
    pub async fn resolve_target(
        &self,
        requested_model: Option<&str>,
        is_link_inbound: bool,
    ) -> Result<ResolvedTarget, (StatusCode, String)> {
        let raw_trimmed = requested_model.map(str::trim).filter(|s| !s.is_empty());
        let cfg_snap = self.settings.read().await.clone();

        let aliased_target: Option<String> = if !is_link_inbound
            && let Some(req) = raw_trimmed
        {
            cfg_snap
                .server
                .model_aliases
                .iter()
                .find(|r| r.matches(req))
                .map(|r| r.target.trim().to_string())
                .filter(|t| !t.is_empty())
        } else {
            None
        };

        if let (Some(orig), Some(mapped)) = (raw_trimmed, aliased_target.as_deref()) {
            self.supervisor
                .push_log(
                    "INFO",
                    "GATEWAY",
                    format!("Model alias matched: '{orig}' -> '{mapped}'"),
                )
                .await;
        }

        let req_trimmed = aliased_target.as_deref().or(raw_trimmed);

        // 0. If this is a local client request with an explicit `<peer>/<model>` prefix, route directly.
        if !is_link_inbound
            && cfg_snap.link.expose_remote_in_openai_api
            && let (Some(query), Some(link)) = (req_trimmed, self.link.as_ref())
            && query.contains('/')
            && let Some((peer, remote_model_id)) = link.resolve_remote_model(query).await
        {
            return Ok(ResolvedTarget::Remote {
                peer,
                remote_model_id,
            });
        }

        // 1. Try local loaded model (or local JIT).
        let local_res = self
            .resolve_or_jit_load_local(req_trimmed, false, is_link_inbound)
            .await;

        match local_res {
            Ok(loaded) => Ok(ResolvedTarget::Local(loaded)),
            Err(local_err) => {
                // 2. If local didn't match and this is NOT an inbound mesh request, check Link peers.
                if !is_link_inbound
                    && cfg_snap.link.expose_remote_in_openai_api
                    && let (Some(query), Some(link)) = (req_trimmed, self.link.as_ref())
                    && let Some((peer, remote_model_id)) = link.resolve_remote_model(query).await
                {
                    return Ok(ResolvedTarget::Remote {
                        peer,
                        remote_model_id,
                    });
                }

                // 3. Optional OpenAI/Anthropic placeholder fallback when explicitly enabled.
                if !is_link_inbound
                    && cfg_snap.server.fallback_to_first_loaded
                    && let Some(first) = self.supervisor.first_loaded().await
                {
                    return Ok(ResolvedTarget::Local(first));
                }

                Err(local_err)
            }
        }
    }

    /// Resolves a currently loaded local model, or triggers Just-in-Time loading from the local GGUF library.
    pub async fn resolve_or_jit_load_local(
        &self,
        requested_model: Option<&str>,
        force_allow_jit: bool,
        is_link_inbound: bool,
    ) -> Result<LoadedModelInfo, (StatusCode, String)> {
        let req_trimmed = requested_model.map(str::trim).filter(|s| !s.is_empty());

        // 1. Check already loaded models
        if let Some(name) = req_trimmed {
            if let Some(loaded) = self.supervisor.resolve_loaded(name).await {
                return Ok(loaded);
            }
        } else if let Some(first) = self.supervisor.first_loaded().await {
            return Ok(first);
        }

        // 2. Check JIT permission
        let cfg_snap = self.settings.read().await.clone();
        if is_link_inbound && !cfg_snap.link.allow_remote_load {
            return Err((
                StatusCode::FORBIDDEN,
                format!(
                    "Model '{}' is not loaded and remote model loading is disabled on this host.",
                    req_trimmed.unwrap_or("<none>")
                ),
            ));
        }

        if !force_allow_jit && !cfg_snap.server.jit_loading {
            if let Some(first) = self.supervisor.first_loaded().await
                && req_trimmed.is_none()
            {
                return Ok(first);
            }
            return Err((
                StatusCode::NOT_FOUND,
                format!(
                    "Model '{}' is not loaded and Just-in-Time loading is disabled.",
                    req_trimmed.unwrap_or("<none>")
                ),
            ));
        }

        // 3. Locate matching local GGUF model
        let Some(query) = req_trimmed else {
            return Err((
                StatusCode::BAD_REQUEST,
                "No model loaded and no 'model' field specified in request.".to_string(),
            ));
        };

        let Some(local) = self.find_local_model(query).await else {
            return Err((
                StatusCode::NOT_FOUND,
                format!("No local GGUF model matching '{query}' found in library."),
            ));
        };

        // 4. Load saved per-model LoadConfig preset or build smart default
        let mut load_cfg = self
            .db
            .get_model_preset(&local.id)
            .ok()
            .flatten()
            .unwrap_or_else(|| {
                let mut c = LoadConfig::new(&local.path);
                c.model_alias = Some(local.id.clone());
                c.mmproj_path = local.mmproj_path.clone();
                c.offload_kv_cache = cfg_snap.default_offload_kv_cache;
                if let Some(max_ctx) = local.max_context_length {
                    c.ctx_size = c.ctx_size.min(max_ctx);
                }
                c
            });
        load_cfg.model_path = local.path.clone();
        if load_cfg.model_alias.is_none() {
            load_cfg.model_alias = Some(local.id.clone());
        }

        // 5. Resolve runtime binary (model-pinned > active default > latest installed)
        let runtimes = self
            .db
            .list_runtimes()
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        let preferred_rt_id = load_cfg
            .runtime_id
            .as_deref()
            .or(cfg_snap.active_runtime_id.as_deref());
        let runtime = preferred_rt_id
            .and_then(|id| runtimes.iter().find(|r| r.id == id))
            .or_else(|| runtimes.first())
            .cloned()
            .ok_or_else(|| {
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "No llama.cpp runtime installed. Install a release in the Runtime tab first."
                        .to_string(),
                )
            })?;

        self.supervisor
            .push_log(
                "INFO",
                if is_link_inbound { "LINK" } else { "JIT" },
                format!("Loading '{}' via {}", local.id, runtime.tag),
            )
            .await;

        let live_ctx = self
            .compute_link_live_memory_context(Some(&local.id))
            .await;
        let estimate = local.gguf.as_ref().map(|meta| {
            bibulm_core::estimate_memory_usage_with_context(
                meta,
                &load_cfg,
                live_ctx,
                cfg_snap.guardrail_level,
            )
        });
        if let Some(ref est) = estimate {
            if cfg_snap.guardrail_level == bibulm_core::GuardrailLevel::Strict
                && let Some(ref warn_msg) = est.guardrail_warning
            {
                return Err((
                    StatusCode::INSUFFICIENT_STORAGE,
                    format!("Strict memory guardrail blocked JIT load: {warn_msg}"),
                ));
            }
            if cfg_snap.limit_offload_to_dedicated_vram
                && matches!(
                    cfg_snap.guardrail_level,
                    bibulm_core::GuardrailLevel::Balanced | bibulm_core::GuardrailLevel::Strict
                )
                && live_ctx.available_vram_mib > 0.0
                && est.total_vram_mib > live_ctx.available_vram_mib * 1.02
            {
                return Err((
                    StatusCode::INSUFFICIENT_STORAGE,
                    format!(
                        "Dedicated VRAM guardrail blocked JIT load ({:.0} MiB required > {:.0} MiB available).",
                        est.total_vram_mib, live_ctx.available_vram_mib
                    ),
                ));
            }
        }
        let estimated_breakdown = estimate.map(|e| e.breakdown);

        let loaded = self
            .supervisor
            .load_model(
                local.id.clone(),
                local.display_name.clone(),
                runtime.id,
                runtime.tag,
                &PathBuf::from(&runtime.server_exe),
                load_cfg,
                estimated_breakdown,
                true,
                cfg_snap.server.only_keep_last_jit_model,
            )
            .await
            .map_err(|e| (StatusCode::BAD_GATEWAY, format!("JIT load failed: {e}")))?;

        Ok(loaded)
    }

    /// Loads (or reloads) a local model with an explicit `LoadConfig` supplied by a Link peer.
    /// Sanitizes local filesystem paths against discovered library models before spawning `llama-server`.
    pub async fn load_local_with_config(
        &self,
        model_id: &str,
        mut load_cfg: LoadConfig,
        remember_preset: bool,
    ) -> Result<LoadedModelInfo, (StatusCode, String)> {
        let Some(local) = self.find_local_model(model_id).await else {
            return Err((
                StatusCode::NOT_FOUND,
                format!("No local GGUF model matching '{model_id}' found in library."),
            ));
        };

        let all_locals = self.cached_local_models().await;
        let cfg_snap = self.settings.read().await.clone();

        // Force model_path to the verified local GGUF path on this host.
        load_cfg.model_path = local.path.clone();

        // Validate mmproj_path: allow paired sibling mmproj or another existing .gguf in allowed dirs.
        if let Some(ref mm) = load_cfg.mmproj_path {
            let is_paired = local.mmproj_path.as_ref().is_some_and(|p| p == mm);
            let is_known_mmproj = all_locals
                .iter()
                .any(|m| m.mmproj_path.as_ref().is_some_and(|p| p == mm));
            let in_allowed_root =
                canonical_file_within(mm, &cfg_snap.models_dirs, &["gguf"]).is_some();
            if !is_paired && !is_known_mmproj && !in_allowed_root {
                load_cfg.mmproj_path = local.mmproj_path.clone();
            }
        }

        // Validate draft_model_path against scanned local models.
        if let Some(ref draft) = load_cfg.draft_model_path {
            let known_draft = all_locals.iter().any(|m| &m.path == draft);
            if !known_draft {
                load_cfg.draft_model_path = None;
            }
        }

        // Validate optional chat_template_file: must be a template file inside allowed directories.
        if let Some(ref tpl) = load_cfg.chat_template_file
            && !is_allowed_template_file(tpl, &cfg_snap.models_dirs)
        {
            self.supervisor
                .push_log(
                    "WARN",
                    "LINK",
                    format!(
                        "Stripped out-of-scope remote chat_template_file '{}'",
                        tpl.display()
                    ),
                )
                .await;
            load_cfg.chat_template_file = None;
        }

        // Security: never allow arbitrary raw CLI args or unsafe flags from remote Link peers.
        if !load_cfg.extra_args.is_empty() {
            self.supervisor
                .push_log(
                    "WARN",
                    "LINK",
                    "Stripped raw extra_args from remote Link load request.",
                )
                .await;
            load_cfg.extra_args.clear();
        }
        let mut stripped_flags = Vec::new();
        load_cfg.cli_flags.retain(|flag, _| {
            if is_safe_remote_cli_flag(flag) {
                true
            } else {
                stripped_flags.push(flag.clone());
                false
            }
        });
        if !stripped_flags.is_empty() {
            self.supervisor
                .push_log(
                    "WARN",
                    "LINK",
                    format!(
                        "Stripped disallowed remote cli_flags: {}",
                        stripped_flags.join(", ")
                    ),
                )
                .await;
        }

        if load_cfg.model_alias.is_none() {
            load_cfg.model_alias = Some(local.id.clone());
        }

        if remember_preset {
            let _ = self.db.save_model_preset(&local.id, &load_cfg);
        }

        let runtimes = self
            .db
            .list_runtimes()
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        let preferred_rt_id = load_cfg
            .runtime_id
            .as_deref()
            .or(cfg_snap.active_runtime_id.as_deref());
        let runtime = preferred_rt_id
            .and_then(|id| runtimes.iter().find(|r| r.id == id))
            .or_else(|| runtimes.first())
            .cloned()
            .ok_or_else(|| {
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "No llama.cpp runtime installed. Install a release in the Runtime tab first."
                        .to_string(),
                )
            })?;

        self.supervisor
            .push_log(
                "INFO",
                "LINK",
                format!(
                    "Remote-configured load of '{}' via {} (ctx={}, ngl={:?})",
                    local.id, runtime.tag, load_cfg.ctx_size, load_cfg.gpu_layers
                ),
            )
            .await;

        let live_ctx = self
            .compute_link_live_memory_context(Some(&local.id))
            .await;
        let estimate = local.gguf.as_ref().map(|meta| {
            bibulm_core::estimate_memory_usage_with_context(
                meta,
                &load_cfg,
                live_ctx,
                cfg_snap.guardrail_level,
            )
        });
        if let Some(ref est) = estimate {
            if cfg_snap.guardrail_level == bibulm_core::GuardrailLevel::Strict
                && let Some(ref warn_msg) = est.guardrail_warning
            {
                return Err((
                    StatusCode::INSUFFICIENT_STORAGE,
                    format!("Strict memory guardrail blocked JIT load: {warn_msg}"),
                ));
            }
            if cfg_snap.limit_offload_to_dedicated_vram
                && matches!(
                    cfg_snap.guardrail_level,
                    bibulm_core::GuardrailLevel::Balanced | bibulm_core::GuardrailLevel::Strict
                )
                && live_ctx.available_vram_mib > 0.0
                && est.total_vram_mib > live_ctx.available_vram_mib * 1.02
            {
                return Err((
                    StatusCode::INSUFFICIENT_STORAGE,
                    format!(
                        "Dedicated VRAM guardrail blocked JIT load ({:.0} MiB required > {:.0} MiB available).",
                        est.total_vram_mib, live_ctx.available_vram_mib
                    ),
                ));
            }
        }
        let estimated_breakdown = estimate.map(|e| e.breakdown);

        let loaded = self
            .supervisor
            .load_model(
                local.id.clone(),
                local.display_name.clone(),
                runtime.id,
                runtime.tag,
                &PathBuf::from(&runtime.server_exe),
                load_cfg,
                estimated_breakdown,
                false,
                false,
            )
            .await
            .map_err(|e| (StatusCode::BAD_GATEWAY, format!("Remote load failed: {e}")))?;

        Ok(loaded)
    }
}

fn is_allowed_template_file(path: &PathBuf, models_dirs: &[PathBuf]) -> bool {
    let templates_dir = bibulm_core::default_data_dir().join("templates");
    let roots = models_dirs.iter().chain(std::iter::once(&templates_dir));
    canonical_file_within(path, roots, &["jinja", "j2", "txt", "tmpl"]).is_some()
}

fn is_safe_remote_cli_flag(flag: &str) -> bool {
    matches!(
        flag.trim(),
        "--temp"
            | "--top-k"
            | "--top-p"
            | "--min-p"
            | "--typical"
            | "--repeat-penalty"
            | "--repeat-last-n"
            | "--presence-penalty"
            | "--frequency-penalty"
            | "--dry-multiplier"
            | "--dry-base"
            | "--dry-allowed-length"
            | "--dry-penalty-last-n"
            | "--xtc-probability"
            | "--xtc-threshold"
            | "--mirostat"
            | "--mirostat-lr"
            | "--mirostat-ent"
            | "--dynatemp-range"
            | "--dynatemp-exp"
            | "--ctx-size"
            | "--batch-size"
            | "--ubatch-size"
            | "--parallel"
            | "--flash-attn"
            | "--cache-type-k"
            | "--cache-type-v"
            | "--kv-unified"
            | "--cache-ram"
            | "--reasoning-format"
            | "--reasoning-budget"
            | "--reasoning-budget-message"
            | "--context-shift"
            | "--no-context-shift"
            | "--swa-full"
            | "--no-warmup"
            | "--seed"
            | "--rope-scaling"
            | "--rope-freq-base"
            | "--rope-freq-scale"
            | "--yarn-ext-factor"
            | "--yarn-attn-factor"
            | "--yarn-beta-fast"
            | "--yarn-beta-slow"
            | "--spec-draft-n-max"
            | "--spec-draft-n-min"
            | "--spec-draft-p-min"
            | "--split-mode"
            | "--n-gpu-layers"
            | "--cpu-moe"
            | "--n-cpu-moe"
            | "--fit"
            | "--fit-ctx"
            | "--fit-target"
            | "--no-mmap"
            | "--mlock"
            | "--direct-io"
            | "--load-mode"
            | "--cache-reuse"
            | "--n-predict"
            | "--keep"
    )
}
