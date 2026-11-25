use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkConnectionState {
    #[default]
    Disconnected,
    Starting,
    NeedsLogin,
    Connecting,
    Connected,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinkStatus {
    pub state: LinkConnectionState,
    pub auth_url: Option<String>,
    pub hostname: String,
    pub node_id: Option<String>,
    pub tailscale_ip: Option<String>,
    pub magic_dns: Option<String>,
    pub tailnet_name: Option<String>,
    pub account_login: Option<String>,
    pub account_name: Option<String>,
    pub access_policy: String,
    pub expose_local_models: bool,
    pub sidecar_available: bool,
    pub sidecar_path: Option<String>,
    pub error: Option<String>,
}

impl LinkStatus {
    pub fn disconnected(
        hostname: String,
        access_policy: String,
        expose_local_models: bool,
        sidecar_path: Option<String>,
    ) -> Self {
        Self {
            state: LinkConnectionState::Disconnected,
            auth_url: None,
            hostname,
            node_id: None,
            tailscale_ip: None,
            magic_dns: None,
            tailnet_name: None,
            account_login: None,
            account_name: None,
            access_policy,
            expose_local_models,
            sidecar_available: sidecar_path.is_some(),
            sidecar_path,
            error: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RemoteModelSummary {
    pub id: String,
    pub display_name: String,
    pub loaded: bool,
    pub quantization: Option<String>,
    pub architecture: Option<String>,
    #[serde(default)]
    pub size_bytes: u64,
    pub max_context_length: Option<u32>,
    #[serde(default)]
    pub supports_vision: bool,
    #[serde(default)]
    pub supports_reasoning: bool,
    pub placement_status: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PeerInfoPayload {
    pub device_name: String,
    pub app_version: String,
    pub expose_local_models: bool,
    pub allow_remote_load: bool,
    #[serde(default)]
    pub allow_remote_manage: bool,
    #[serde(default)]
    pub is_headless: bool,
    #[serde(default)]
    pub platform: String,
    #[serde(default)]
    pub runtimes_count: usize,
    #[serde(default)]
    pub active_runtime_tag: Option<String>,
    #[serde(default)]
    pub active_downloads_count: usize,
    pub gpu_summary: String,
    pub vram_used_mib: f64,
    pub vram_total_mib: f64,
    pub ram_used_mib: f64,
    pub ram_total_mib: f64,
    pub loaded_models: Vec<RemoteModelSummary>,
    pub available_models: Vec<RemoteModelSummary>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RemotePeer {
    pub node_id: String,
    pub slug: String,
    pub hostname: String,
    pub device_name: String,
    pub tailscale_ip: String,
    pub magic_dns: String,
    pub os: String,
    pub online: bool,
    pub connection_type: String,
    pub relay: Option<String>,
    pub latency_ms: Option<u64>,
    pub app_version: String,
    pub expose_local_models: bool,
    pub allow_remote_load: bool,
    #[serde(default)]
    pub allow_remote_manage: bool,
    #[serde(default)]
    pub is_headless: bool,
    #[serde(default)]
    pub platform: String,
    #[serde(default)]
    pub runtimes_count: usize,
    #[serde(default)]
    pub active_runtime_tag: Option<String>,
    #[serde(default)]
    pub active_downloads_count: usize,
    pub gpu_summary: String,
    pub vram_used_mib: f64,
    pub vram_total_mib: f64,
    pub ram_used_mib: f64,
    pub ram_total_mib: f64,
    #[serde(default)]
    pub loaded_models: Vec<RemoteModelSummary>,
    #[serde(default)]
    pub available_models: Vec<RemoteModelSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteLoadRequest {
    pub model_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config: Option<bibulm_core::LoadConfig>,
    #[serde(default)]
    pub remember_preset: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteLoadResponse {
    pub ok: bool,
    pub model_id: String,
    pub message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteModelInspectRequest {
    pub model_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteRuntimeSummary {
    pub id: String,
    pub tag: String,
    pub backend: String,
    pub display_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteDraftCandidate {
    pub id: String,
    pub display_name: String,
    pub path: String,
    pub quantization: Option<String>,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteModelInspectResponse {
    pub model: bibulm_models::LocalModelEntry,
    pub saved_preset: Option<bibulm_core::LoadConfig>,
    pub active_load_config: Option<bibulm_core::LoadConfig>,
    pub runtimes: Vec<RemoteRuntimeSummary>,
    pub active_runtime_id: Option<String>,
    pub default_offload_kv_cache: bool,
    pub guardrail_level: bibulm_core::GuardrailLevel,
    pub cpu_logical_cores: u32,
    pub live_memory: bibulm_core::LiveMemoryContext,
    pub draft_models: Vec<RemoteDraftCandidate>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteRuntimesResponse {
    pub installed: Vec<bibulm_core::InstalledRuntimeRecord>,
    pub active_runtime_id: Option<String>,
    pub platform: String,
    pub gpu_summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteInstallRuntimeRequest {
    pub variant: bibulm_runtime::RuntimeBackendVariant,
    #[serde(default)]
    pub release_notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteInstallRuntimeResponse {
    pub ok: bool,
    pub job_id: String,
    pub variant_id: String,
    pub message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteSetActiveRuntimeRequest {
    pub runtime_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteUninstallRuntimeRequest {
    pub runtime_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteModelDownloadRequest {
    pub title: String,
    pub repo_id: String,
    pub files: Vec<bibulm_models::HfDownloadFile>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteDownloadActionRequest {
    pub id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteDeleteModelRequest {
    pub model_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteActionResponse {
    pub ok: bool,
    pub message: Option<String>,
}
