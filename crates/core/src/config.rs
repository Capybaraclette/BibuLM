use crate::{SamplingParams, ServerConfig};
use directories::{BaseDirs, ProjectDirs, UserDirs};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

fn default_true() -> bool {
    true
}

/// Kind of external application model storage directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExternalSourceKind {
    GgufDir,
    Ollama,
}

/// Discovered external AI application model directory on the local machine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalModelSource {
    pub app_name: String,
    pub path: PathBuf,
    pub kind: ExternalSourceKind,
}

/// Model loading memory safety guardrail level.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GuardrailLevel {
    /// No precautions against system overload.
    Off,
    /// Mild precautions (warn above 95% total memory).
    #[default]
    Relaxed,
    /// Moderate precautions (warn above 88% VRAM or 85% RAM).
    Balanced,
    /// Strict precautions (block/warn above 80% VRAM or 75% RAM).
    Strict,
}

impl GuardrailLevel {
    /// Maximum recommended fraction of VRAM and System RAM before warning.
    pub fn thresholds(self) -> Option<(f64, f64)> {
        match self {
            Self::Off => None,
            Self::Relaxed => Some((0.97, 0.92)),
            Self::Balanced => Some((0.90, 0.85)),
            Self::Strict => Some((0.82, 0.75)),
        }
    }
}

/// Access policy governing which Tailscale peers may reach this Bibulm node.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkAccessPolicy {
    /// Only devices signed into your exact same Tailscale account.
    #[default]
    SelfOnly,
    /// Any device in your Tailnet allowed by your Tailscale ACLs.
    SameTailnet,
}

impl LinkAccessPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SelfOnly => "self_only",
            Self::SameTailnet => "same_tailnet",
        }
    }
}

/// Configuration for Bibulm Link (BYO Tailscale userspace mesh).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct LinkConfig {
    pub enabled: bool,
    pub auto_connect: bool,
    pub device_name: Option<String>,
    pub expose_local_models: bool,
    pub allow_remote_load: bool,
    pub allow_remote_manage: bool,
    pub access_policy: LinkAccessPolicy,
    pub control_url: Option<String>,
    pub expose_remote_in_openai_api: bool,
    pub mesh_port: u16,
}

impl Default for LinkConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            auto_connect: true,
            device_name: None,
            expose_local_models: true,
            allow_remote_load: true,
            allow_remote_manage: false,
            access_policy: LinkAccessPolicy::SelfOnly,
            control_url: None,
            expose_remote_in_openai_api: true,
            mesh_port: 11435,
        }
    }
}

/// Top-level persisted application configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AppSettings {
    pub models_dirs: Vec<PathBuf>,
    #[serde(default = "default_true")]
    pub auto_discover_external_models: bool,
    pub runtimes_dir: PathBuf,
    pub logs_dir: PathBuf,
    pub hf_token: Option<String>,
    pub github_token: Option<String>,
    pub active_runtime_id: Option<String>,
    pub auto_start_server: bool,
    pub minimize_to_tray: bool,
    pub start_minimized: bool,
    pub auto_check_runtime_updates: bool,
    pub auto_check_app_updates: bool,
    pub log_retention_days: u32,
    pub guardrail_level: GuardrailLevel,
    pub limit_offload_to_dedicated_vram: bool,
    pub default_offload_kv_cache: bool,
    pub disabled_gpu_ids: Vec<u32>,
    pub server: ServerConfig,
    pub link: LinkConfig,
    pub default_sampling: SamplingParams,
}

impl Default for AppSettings {
    fn default() -> Self {
        data_dir_settings_default(&default_data_dir())
    }
}

/// Constructs default `AppSettings` rooted at the provided `base` data directory.
pub fn data_dir_settings_default(base: &std::path::Path) -> AppSettings {
    let models_dirs = vec![base.join("models")];
    AppSettings {
        models_dirs,
        auto_discover_external_models: true,
        runtimes_dir: base.join("runtimes"),
        logs_dir: base.join("logs"),
        hf_token: None,
        github_token: None,
        active_runtime_id: None,
        auto_start_server: true,
        minimize_to_tray: true,
        start_minimized: false,
        auto_check_runtime_updates: true,
        auto_check_app_updates: true,
        log_retention_days: 14,
        guardrail_level: GuardrailLevel::Relaxed,
        limit_offload_to_dedicated_vram: true,
        default_offload_kv_cache: true,
        disabled_gpu_ids: Vec::new(),
        server: ServerConfig::default(),
        link: LinkConfig::default(),
        default_sampling: SamplingParams::default(),
    }
}

impl AppSettings {
    /// Primary directory where newly downloaded models are stored.
    pub fn primary_models_dir(&self) -> PathBuf {
        self.models_dirs
            .first()
            .cloned()
            .unwrap_or_else(|| default_data_dir().join("models"))
    }
}

/// Resolves the platform-appropriate persistent data directory for Bibulm.
/// Respects `BIBULM_DATA_DIR` when set and non-empty.
/// Canonical form of `path` when it resolves (after `..` and symlinks) inside one of `roots`.
pub fn canonical_within<'a>(
    path: &Path,
    roots: impl IntoIterator<Item = &'a PathBuf>,
) -> Option<PathBuf> {
    let canon = std::fs::canonicalize(path).ok()?;
    roots
        .into_iter()
        .any(|root| std::fs::canonicalize(root).is_ok_and(|r| canon.starts_with(r)))
        .then_some(canon)
}

pub fn default_data_dir() -> PathBuf {
    if let Ok(override_dir) = std::env::var("BIBULM_DATA_DIR") {
        let trimmed = override_dir.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    if let Some(proj) = ProjectDirs::from("com", "bibulm", "Bibulm") {
        return proj.data_local_dir().to_path_buf();
    }
    if let Some(user) = UserDirs::new() {
        return user.home_dir().join(".bibulm");
    }
    PathBuf::from("./data")
}

/// Resolves the pre-rebrand persistent data directory, if present, for one-time DB migration.
pub fn legacy_data_dir() -> Option<PathBuf> {
    if let Some(proj) = ProjectDirs::from("com", "ruststudio", "rustStudio") {
        let dir = proj.data_local_dir().to_path_buf();
        if dir.is_dir() {
            return Some(dir);
        }
    }
    if let Some(user) = UserDirs::new() {
        let dir = user.home_dir().join(".ruststudio");
        if dir.is_dir() {
            return Some(dir);
        }
    }
    None
}

/// Discovers model directories managed by other local AI applications (Ollama, LM Studio,
/// Jan, GPT4All, llama.cpp cache, HuggingFace Hub cache, AnythingLLM, Msty, Backyard AI, etc.).
pub fn discover_external_model_sources() -> Vec<ExternalModelSource> {
    let mut candidates: Vec<(&'static str, PathBuf, ExternalSourceKind)> = Vec::new();
    let user_dirs = UserDirs::new();
    let base_dirs = BaseDirs::new();
    let home = user_dirs
        .as_ref()
        .map(|u| u.home_dir().to_path_buf())
        .or_else(|| base_dirs.as_ref().map(|b| b.home_dir().to_path_buf()));

    // Legacy rustStudio data/models
    if let Some(old_dir) = legacy_data_dir() {
        candidates.push(("rustStudio", old_dir.join("models"), ExternalSourceKind::GgufDir));
    }

    // LM Studio (custom downloadsFolder in ~/.lmstudio/settings.json + default locations)
    if let Some(h) = &home {
        let lms_settings = h.join(".lmstudio").join("settings.json");
        if let Ok(raw) = std::fs::read_to_string(&lms_settings)
            && let Ok(val) = serde_json::from_str::<serde_json::Value>(&raw)
            && let Some(folder) = val.get("downloadsFolder").and_then(|v| v.as_str())
        {
            let trimmed = folder.trim();
            if !trimmed.is_empty() {
                candidates.push(("LM Studio", PathBuf::from(trimmed), ExternalSourceKind::GgufDir));
            }
        }
        candidates.push((
            "LM Studio",
            h.join(".lmstudio").join("models"),
            ExternalSourceKind::GgufDir,
        ));
        candidates.push((
            "LM Studio",
            h.join(".cache").join("lm-studio").join("models"),
            ExternalSourceKind::GgufDir,
        ));
    }

    // Ollama (OLLAMA_MODELS env + standard paths)
    if let Ok(ollama_env) = std::env::var("OLLAMA_MODELS") {
        let trimmed = ollama_env.trim();
        if !trimmed.is_empty() {
            candidates.push(("Ollama", PathBuf::from(trimmed), ExternalSourceKind::Ollama));
        }
    }
    if let Some(h) = &home {
        candidates.push((
            "Ollama",
            h.join(".ollama").join("models"),
            ExternalSourceKind::Ollama,
        ));
    }
    if let Some(b) = &base_dirs {
        candidates.push((
            "Ollama",
            b.data_local_dir().join("Ollama").join("models"),
            ExternalSourceKind::Ollama,
        ));
    }
    #[cfg(target_os = "linux")]
    candidates.push((
        "Ollama",
        PathBuf::from("/usr/share/ollama/.ollama/models"),
        ExternalSourceKind::Ollama,
    ));

    // Jan
    if let Some(h) = &home {
        candidates.push(("Jan", h.join("jan").join("models"), ExternalSourceKind::GgufDir));
    }
    if let Some(b) = &base_dirs {
        candidates.push((
            "Jan",
            b.data_dir().join("Jan").join("data").join("models"),
            ExternalSourceKind::GgufDir,
        ));
        candidates.push((
            "Jan",
            b.data_local_dir().join("Jan").join("data").join("models"),
            ExternalSourceKind::GgufDir,
        ));
    }

    // GPT4All
    if let Some(b) = &base_dirs {
        candidates.push((
            "GPT4All",
            b.data_local_dir().join("nomic.ai").join("GPT4All"),
            ExternalSourceKind::GgufDir,
        ));
        candidates.push((
            "GPT4All",
            b.data_dir().join("nomic.ai").join("GPT4All"),
            ExternalSourceKind::GgufDir,
        ));
    }

    // llama.cpp cache
    if let Ok(llama_cache) = std::env::var("LLAMA_CACHE") {
        let trimmed = llama_cache.trim();
        if !trimmed.is_empty() {
            candidates.push(("llama.cpp", PathBuf::from(trimmed), ExternalSourceKind::GgufDir));
        }
    }
    if let Some(b) = &base_dirs {
        candidates.push((
            "llama.cpp",
            b.data_local_dir().join("llama.cpp"),
            ExternalSourceKind::GgufDir,
        ));
        candidates.push((
            "llama.cpp",
            b.cache_dir().join("llama.cpp"),
            ExternalSourceKind::GgufDir,
        ));
    }

    // HuggingFace Hub cache
    if let Ok(hf_hub_cache) = std::env::var("HF_HUB_CACHE") {
        let trimmed = hf_hub_cache.trim();
        if !trimmed.is_empty() {
            candidates.push(("HuggingFace", PathBuf::from(trimmed), ExternalSourceKind::GgufDir));
        }
    }
    if let Ok(hf_home) = std::env::var("HF_HOME") {
        let trimmed = hf_home.trim();
        if !trimmed.is_empty() {
            candidates.push((
                "HuggingFace",
                PathBuf::from(trimmed).join("hub"),
                ExternalSourceKind::GgufDir,
            ));
        }
    }
    if let Some(h) = &home {
        candidates.push((
            "HuggingFace",
            h.join(".cache").join("huggingface").join("hub"),
            ExternalSourceKind::GgufDir,
        ));
    }

    // AnythingLLM
    if let Some(b) = &base_dirs {
        candidates.push((
            "AnythingLLM",
            b.data_dir()
                .join("anythingllm-desktop")
                .join("storage")
                .join("models")
                .join("downloaded"),
            ExternalSourceKind::GgufDir,
        ));
        candidates.push((
            "AnythingLLM",
            b.config_dir()
                .join("anythingllm-desktop")
                .join("storage")
                .join("models")
                .join("downloaded"),
            ExternalSourceKind::GgufDir,
        ));
    }

    // Msty
    if let Some(b) = &base_dirs {
        candidates.push((
            "Msty",
            b.data_dir().join("Msty").join("models"),
            ExternalSourceKind::GgufDir,
        ));
        candidates.push((
            "Msty",
            b.config_dir().join("Msty").join("models"),
            ExternalSourceKind::GgufDir,
        ));
    }

    // Backyard AI (formerly Faraday)
    if let Some(b) = &base_dirs {
        candidates.push((
            "Backyard AI",
            b.data_dir().join("Backyard AI").join("models"),
            ExternalSourceKind::GgufDir,
        ));
        candidates.push((
            "Backyard AI",
            b.data_dir().join("faraday").join("models"),
            ExternalSourceKind::GgufDir,
        ));
    }

    // LocalAI
    if let Some(b) = &base_dirs {
        candidates.push((
            "LocalAI",
            b.data_local_dir().join("local-ai").join("models"),
            ExternalSourceKind::GgufDir,
        ));
    }

    // Nexa & Docker Models
    if let Some(h) = &home {
        candidates.push((
            "Nexa",
            h.join(".cache").join("nexa").join("hub"),
            ExternalSourceKind::GgufDir,
        ));
        candidates.push((
            "Docker Models",
            h.join(".docker").join("models"),
            ExternalSourceKind::GgufDir,
        ));
    }

    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for (app_name, path, kind) in candidates {
        if !path.is_dir() {
            continue;
        }
        let canonical = std::fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
        if seen.insert(canonical) {
            out.push(ExternalModelSource {
                app_name: app_name.to_string(),
                path,
                kind,
            });
        }
    }
    out
}
