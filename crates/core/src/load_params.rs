use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// Typed value for a generic `llama-server` CLI flag entry in [`LoadConfig::cli_flags`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum CliFlagValue {
    /// Presence-only flag (emits `[flag]` when `true`, nothing when `false`).
    /// For paired `--foo` / `--no-foo` flags, the map key itself is the chosen flag.
    Flag(bool),
    /// Single-argument flag: emits `[flag, value]`.
    Value(String),
    /// Single flag followed by multiple positional arguments: emits `[flag, v1, v2, ...]`
    /// (e.g. `--control-vector-layer-range 10 20`).
    MultiArg(Vec<String>),
    /// Repeatable flag: emits `[flag, v1, flag, v2, ...]`
    /// (e.g. `--logit-bias 15043+1 --logit-bias 1234-1`).
    Repeatable(Vec<String>),
}

/// Controls how Mixture-of-Experts (MoE) layers are distributed between GPU VRAM and System RAM.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", content = "value", rename_all = "snake_case")]
pub enum MoeOffloadMode {
    /// Keep all layers (including MoE routed experts) on GPU VRAM subject to `gpu_layers`.
    #[default]
    FullGpu,
    /// `--cpu-moe`: Keep attention, shared experts, embeddings, and KV cache on VRAM;
    /// keep all routed MoE expert weights in System RAM (CPU).
    AllExpertsCpu,
    /// `--n-cpu-moe N`: Keep the routed MoE expert weights of the first `N` layers in System RAM,
    /// letting remaining MoE layers fill available VRAM.
    FirstNLayersExpertsCpu(u32),
    /// `--override-tensor <pattern>=<buffer>`: Explicit tensor placement rules.
    CustomOverrides(Vec<TensorOverride>),
}

/// Single `--override-tensor` mapping (`<regex_pattern>=<buffer_name>`, e.g. `CPU`, `CUDA0`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorOverride {
    pub pattern: String,
    pub buffer: String,
}

/// Thinking / reasoning effort levels supported by modern reasoning models via llama.cpp.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    Default,
    Off,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

impl ReasoningEffort {
    pub fn as_cli_str(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Off => "none",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
            Self::Max => "max",
        }
    }
}

/// How `llama-server` separates `<think>` blocks in OpenAI API responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningFormat {
    Auto,
    Deepseek,
    DeepseekLegacy,
    None,
}

impl ReasoningFormat {
    pub fn as_cli_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Deepseek => "deepseek",
            Self::DeepseekLegacy => "deepseek-legacy",
            Self::None => "none",
        }
    }
}

/// Reasoning / thinking controls passed to `llama-server`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReasoningConfig {
    pub effort: ReasoningEffort,
    pub format: ReasoningFormat,
    /// `--reasoning-budget`: `-1` unlimited, `0` disabled, `> 0` token cap.
    pub budget: Option<i32>,
    /// `--reasoning-budget-message`: optional graceful transition prompt on budget cutoff.
    pub budget_message: Option<String>,
    /// Optional explicit boolean toggle (`{"enable_thinking": true|false}`) for Qwen3-style templates.
    pub enable_thinking: Option<bool>,
}

impl Default for ReasoningConfig {
    fn default() -> Self {
        Self {
            effort: ReasoningEffort::Default,
            format: ReasoningFormat::Auto,
            budget: None,
            budget_message: None,
            enable_thinking: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FlashAttentionMode {
    Auto,
    On,
    Off,
}

impl FlashAttentionMode {
    pub fn as_cli_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::On => "on",
            Self::Off => "off",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvQuantType {
    F32,
    F16,
    Bf16,
    F8E4m3,
    Q8_0,
    Q5_1,
    Q5_0,
    Q4_1,
    Q4_0,
    Iq4Nl,
}

impl KvQuantType {
    pub fn as_cli_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::F8E4m3 => "f8_e4m3",
            Self::Q8_0 => "q8_0",
            Self::Q5_1 => "q5_1",
            Self::Q5_0 => "q5_0",
            Self::Q4_1 => "q4_1",
            Self::Q4_0 => "q4_0",
            Self::Iq4Nl => "iq4_nl",
        }
    }

    /// Average bytes per element for KV cache memory estimation.
    pub fn bytes_per_element(self) -> f64 {
        match self {
            Self::F32 => 4.0,
            Self::F16 | Self::Bf16 => 2.0,
            Self::F8E4m3 => 1.0,
            Self::Q8_0 => 1.0625, // 34 bytes / 32 elems
            Self::Q5_1 => 0.75,   // 24 bytes / 32 elems
            Self::Q5_0 => 0.6875, // 22 bytes / 32 elems
            Self::Q4_1 => 0.625,  // 20 bytes / 32 elems
            Self::Q4_0 | Self::Iq4Nl => 0.5625, // 18 bytes / 32 elems
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RopeScalingType {
    None,
    Linear,
    Yarn,
}

impl RopeScalingType {
    pub fn as_cli_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Linear => "linear",
            Self::Yarn => "yarn",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SplitMode {
    None,
    Layer,
    Row,
    Tensor,
}

impl SplitMode {
    pub fn as_cli_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Layer => "layer",
            Self::Row => "row",
            Self::Tensor => "tensor",
        }
    }
}

/// Extended sampling & generation defaults applied to benchmark runs and server requests
/// when omitted by the OpenAI client.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SamplingParams {
    pub system_prompt: String,
    pub temperature: Option<f32>,
    pub top_k: Option<i32>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub typical_p: Option<f32>,
    pub repeat_penalty: Option<f32>,
    pub repeat_last_n: Option<i32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub dry_multiplier: Option<f32>,
    pub dry_base: Option<f32>,
    pub dry_allowed_length: Option<i32>,
    pub xtc_probability: Option<f32>,
    pub xtc_threshold: Option<f32>,
    pub dynatemp_range: Option<f32>,
    pub mirostat: Option<u8>,
    pub mirostat_tau: Option<f32>,
    pub mirostat_eta: Option<f32>,
    pub max_tokens: Option<u32>,
    pub stop_strings: Vec<String>,
    pub structured_output_enabled: bool,
    pub json_schema: Option<String>,
    pub grammar: Option<String>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            system_prompt: String::new(),
            temperature: Some(0.7),
            top_k: Some(40),
            top_p: Some(0.95),
            min_p: Some(0.05),
            typical_p: None,
            repeat_penalty: Some(1.05),
            repeat_last_n: Some(64),
            presence_penalty: None,
            frequency_penalty: None,
            dry_multiplier: None,
            dry_base: None,
            dry_allowed_length: None,
            xtc_probability: None,
            xtc_threshold: None,
            dynatemp_range: None,
            mirostat: None,
            mirostat_tau: None,
            mirostat_eta: None,
            max_tokens: Some(1024),
            stop_strings: Vec::new(),
            structured_output_enabled: false,
            json_schema: None,
            grammar: None,
        }
    }
}

/// Complete model load specification translated into `llama-server` CLI arguments.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LoadConfig {
    pub model_path: PathBuf,
    pub model_alias: Option<String>,
    pub runtime_id: Option<String>,
    /// When `false`, vision (`--mmproj`) is omitted even if a paired `mmproj_path` exists.
    pub enable_vision: bool,
    pub mmproj_path: Option<PathBuf>,
    pub offload_mmproj: bool,

    // Context & batching
    pub ctx_size: u32,
    pub batch_size: u32,
    pub ubatch_size: u32,
    pub parallel_slots: u32,
    pub unified_kv_cache: bool,
    pub cache_ram_mib: Option<i32>,
    pub context_shift: Option<bool>,

    // GPU / CPU offload & MoE placement
    /// `None` lets llama.cpp decide (`-ngl auto` or all layers via `999`).
    pub gpu_layers: Option<u32>,
    pub moe_offload: MoeOffloadMode,
    pub auto_fit: Option<bool>,
    pub fit_ctx: Option<u32>,
    pub fit_target_mib: Option<u32>,
    pub threads: Option<u32>,
    pub threads_batch: Option<u32>,
    pub devices: Vec<String>,
    pub split_mode: Option<SplitMode>,
    pub main_gpu: Option<u32>,
    pub tensor_split: Option<String>,

    // KV Cache & Attention
    pub offload_kv_cache: bool,
    pub flash_attention: FlashAttentionMode,
    pub cache_type_k: Option<KvQuantType>,
    pub cache_type_v: Option<KvQuantType>,

    // Memory / IO
    pub use_mmap: bool,
    pub use_mlock: bool,
    pub direct_io: bool,

    // RoPE
    pub rope_scaling: Option<RopeScalingType>,
    pub rope_freq_base: Option<f32>,
    pub rope_freq_scale: Option<f32>,
    pub yarn_ext_factor: Option<f32>,
    pub yarn_attn_factor: Option<f32>,
    pub yarn_beta_fast: Option<f32>,
    pub yarn_beta_slow: Option<f32>,

    // Reasoning & templates
    pub jinja: bool,
    pub chat_template_override: Option<String>,
    /// Direct custom chat template file (`--chat-template-file <path>`).
    pub chat_template_file: Option<PathBuf>,
    pub reasoning: ReasoningConfig,

    // Multi-Token Prediction (MTP) & Speculative Decoding
    /// Enables native built-in MTP heads (`--spec-type draft-mtp`).
    pub mtp_enabled: bool,
    /// Optional spec-type string for MTP (`"draft-mtp"` default, or `"mtp"` for legacy builds).
    pub mtp_spec_type: Option<String>,
    pub draft_model_path: Option<PathBuf>,
    pub draft_gpu_layers: Option<u32>,
    pub draft_max: Option<u32>,
    pub draft_min: Option<u32>,
    pub draft_p_min: Option<f32>,

    // Misc
    pub seed: Option<u32>,
    pub embeddings: bool,
    /// Structured overrides for all additional `llama-server` CLI flags (keyed by canonical `--flag`).
    #[serde(default)]
    pub cli_flags: BTreeMap<String, CliFlagValue>,
    pub extra_args: Vec<String>,
    /// Optional raw `llama-server` CLI arguments replacing auto-generated flags (except managed `--host` / `--port`).
    #[serde(default)]
    pub custom_cli_args: Option<String>,

    // Optional per-model sampling defaults
    pub sampling_defaults: Option<SamplingParams>,
}

impl Default for LoadConfig {
    fn default() -> Self {
        Self::new(PathBuf::new())
    }
}

impl LoadConfig {
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            model_alias: None,
            runtime_id: None,
            enable_vision: true,
            mmproj_path: None,
            offload_mmproj: true,
            ctx_size: 8192,
            batch_size: 2048,
            ubatch_size: 512,
            parallel_slots: 1,
            unified_kv_cache: true,
            cache_ram_mib: None,
            context_shift: None,
            gpu_layers: Some(999),
            moe_offload: MoeOffloadMode::FullGpu,
            auto_fit: None,
            fit_ctx: None,
            fit_target_mib: None,
            threads: None,
            threads_batch: None,
            devices: Vec::new(),
            split_mode: None,
            main_gpu: None,
            tensor_split: None,
            offload_kv_cache: true,
            flash_attention: FlashAttentionMode::Auto,
            cache_type_k: Some(KvQuantType::Q8_0),
            cache_type_v: Some(KvQuantType::Q8_0),
            use_mmap: true,
            use_mlock: false,
            direct_io: false,
            rope_scaling: None,
            rope_freq_base: None,
            rope_freq_scale: None,
            yarn_ext_factor: None,
            yarn_attn_factor: None,
            yarn_beta_fast: None,
            yarn_beta_slow: None,
            jinja: true,
            chat_template_override: None,
            chat_template_file: None,
            reasoning: ReasoningConfig::default(),
            mtp_enabled: false,
            mtp_spec_type: None,
            draft_model_path: None,
            draft_gpu_layers: None,
            draft_max: None,
            draft_min: None,
            draft_p_min: None,
            seed: None,
            embeddings: false,
            cli_flags: BTreeMap::new(),
            extra_args: Vec::new(),
            custom_cli_args: None,
            sampling_defaults: None,
        }
    }

    /// Returns `true` when native built-in MTP heads should be used (without an external `-md` draft file).
    pub fn uses_native_mtp(&self) -> bool {
        self.mtp_enabled && spec_type_is_native_mtp(self.mtp_spec_type.as_deref())
    }

    /// Returns `true` when an external draft GGUF model (`-md`) should be passed to `llama-server`.
    pub fn uses_external_draft(&self) -> bool {
        let has_draft_path = self
            .draft_model_path
            .as_deref()
            .is_some_and(|p| !p.as_os_str().is_empty());
        has_draft_path
            && (!self.mtp_enabled
                || self
                    .mtp_spec_type
                    .as_deref()
                    .is_some_and(spec_type_uses_external_draft))
    }
}

/// Returns `true` if a `--spec-type` value (including comma-separated stacked types like
/// `draft-dspark,ngram-mod`) relies on an external draft GGUF model (`-md`).
pub fn spec_type_uses_external_draft(spec_type: &str) -> bool {
    spec_type.split(',').any(|tok| {
        let t = tok.trim().to_ascii_lowercase();
        t.starts_with("draft-") && t != "draft-mtp"
    })
}

/// Returns `true` if an optional `--spec-type` value selects native built-in MTP heads (`draft-mtp` / `mtp`).
pub fn spec_type_is_native_mtp(spec_type: Option<&str>) -> bool {
    match spec_type.map(str::trim).filter(|s| !s.is_empty()) {
        None => true,
        Some(raw) => raw.split(',').any(|tok| {
            let t = tok.trim().to_ascii_lowercase();
            t == "draft-mtp" || t == "mtp"
        }),
    }
}

/// Applies configured [`SamplingParams`] onto an OpenAI-compatible JSON request object.
/// When `overwrite` is `false`, existing client-supplied keys are preserved.
pub fn apply_sampling_to_openai_body(
    obj: &mut serde_json::Map<String, serde_json::Value>,
    s: &SamplingParams,
    overwrite: bool,
) {
    use serde_json::json;

    let set_val = |map: &mut serde_json::Map<String, serde_json::Value>,
                   key: &str,
                   val: serde_json::Value| {
        if overwrite || !map.contains_key(key) {
            map.insert(key.to_string(), val);
        }
    };

    if let Some(v) = s.temperature {
        set_val(obj, "temperature", json!(v));
    }
    if let Some(v) = s.top_p {
        set_val(obj, "top_p", json!(v));
    }
    if let Some(v) = s.top_k {
        set_val(obj, "top_k", json!(v));
    }
    if let Some(v) = s.min_p {
        set_val(obj, "min_p", json!(v));
    }
    if let Some(v) = s.typical_p {
        set_val(obj, "typical_p", json!(v));
    }
    if let Some(v) = s.repeat_penalty {
        set_val(obj, "repeat_penalty", json!(v));
    }
    if let Some(v) = s.repeat_last_n {
        set_val(obj, "repeat_last_n", json!(v));
    }
    if let Some(v) = s.presence_penalty {
        set_val(obj, "presence_penalty", json!(v));
    }
    if let Some(v) = s.frequency_penalty {
        set_val(obj, "frequency_penalty", json!(v));
    }
    if let Some(v) = s.dry_multiplier {
        set_val(obj, "dry_multiplier", json!(v));
    }
    if let Some(v) = s.dry_base {
        set_val(obj, "dry_base", json!(v));
    }
    if let Some(v) = s.dry_allowed_length {
        set_val(obj, "dry_allowed_length", json!(v));
    }
    if let Some(v) = s.xtc_probability {
        set_val(obj, "xtc_probability", json!(v));
    }
    if let Some(v) = s.xtc_threshold {
        set_val(obj, "xtc_threshold", json!(v));
    }
    if let Some(v) = s.dynatemp_range {
        set_val(obj, "dynatemp_range", json!(v));
    }
    if let Some(v) = s.mirostat {
        set_val(obj, "mirostat", json!(v));
    }
    if let Some(v) = s.mirostat_tau {
        set_val(obj, "mirostat_tau", json!(v));
    }
    if let Some(v) = s.mirostat_eta {
        set_val(obj, "mirostat_eta", json!(v));
    }
    if !overwrite
        && let Some(v) = s.max_tokens
        && !obj.contains_key("max_tokens")
        && !obj.contains_key("max_completion_tokens")
    {
        obj.insert("max_tokens".to_string(), json!(v));
    }
    if !s.stop_strings.is_empty() {
        set_val(obj, "stop", json!(s.stop_strings));
    }
    if s.structured_output_enabled {
        if let Some(schema_str) = s.json_schema.as_deref().map(str::trim).filter(|x| !x.is_empty())
            && let Ok(schema_val) = serde_json::from_str::<serde_json::Value>(schema_str)
        {
            set_val(
                obj,
                "response_format",
                json!({
                    "type": "json_schema",
                    "json_schema": {
                        "name": "structured_output",
                        "strict": true,
                        "schema": schema_val
                    }
                }),
            );
        } else if let Some(grammar) = s.grammar.as_deref().map(str::trim).filter(|x| !x.is_empty())
        {
            set_val(obj, "grammar", json!(grammar));
        }
    }
}

fn default_true_alias() -> bool {
    true
}

/// Maps a requested client model name or glob pattern (e.g. `claude-*`, `gpt-4o`)
/// to a local model ID or remote Link `<peer>/<model>` target.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelAliasRule {
    pub pattern: String,
    pub target: String,
    #[serde(default = "default_true_alias")]
    pub enabled: bool,
}

impl ModelAliasRule {
    pub fn matches(&self, requested: &str) -> bool {
        if !self.enabled {
            return false;
        }
        let pat = self.pattern.trim().to_ascii_lowercase();
        let target_ok = !self.target.trim().is_empty();
        if pat.is_empty() || !target_ok {
            return false;
        }
        let req = requested.trim().to_ascii_lowercase();
        wildcard_match(&pat, &req)
    }
}

fn wildcard_match(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    let (mut pi, mut ti) = (0usize, 0usize);
    let (mut star_p, mut star_t) = (None, 0usize);
    while ti < t.len() {
        if pi < p.len() && p[pi] == t[ti] {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star_p = Some(pi);
            star_t = ti;
            pi += 1;
        } else if let Some(sp) = star_p {
            pi = sp + 1;
            star_t += 1;
            ti = star_t;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

/// OpenAI/Anthropic-compatible HTTP gateway configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    pub port: u16,
    pub serve_on_lan: bool,
    pub require_auth: bool,
    pub enable_cors: bool,
    pub jit_loading: bool,
    pub auto_unload_idle: bool,
    pub max_idle_ttl_minutes: u32,
    pub only_keep_last_jit_model: bool,
    #[serde(default)]
    pub model_aliases: Vec<ModelAliasRule>,
    #[serde(default)]
    pub fallback_to_first_loaded: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            port: 1234,
            serve_on_lan: false,
            require_auth: false,
            enable_cors: false,
            jit_loading: true,
            auto_unload_idle: true,
            max_idle_ttl_minutes: 60,
            only_keep_last_jit_model: true,
            model_aliases: Vec::new(),
            fallback_to_first_loaded: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_alias_matches_exact_and_wildcards() {
        let r1 = ModelAliasRule {
            pattern: "claude-*".into(),
            target: "qwen3-coder-30b".into(),
            enabled: true,
        };
        assert!(r1.matches("claude-sonnet-4-5-20250929"));
        assert!(r1.matches("CLAUDE-3-7-SONNET"));
        assert!(!r1.matches("gpt-4o"));

        let r2 = ModelAliasRule {
            pattern: "gpt-4o".into(),
            target: "llama-3.3".into(),
            enabled: true,
        };
        assert!(r2.matches("gpt-4o"));
        assert!(!r2.matches("gpt-4o-mini"));

        let disabled = ModelAliasRule {
            pattern: "*".into(),
            target: "fallback".into(),
            enabled: false,
        };
        assert!(!disabled.matches("anything"));
    }

    #[test]
    fn load_config_defaults_to_q8_kv_cache_and_no_custom_cli() {
        let cfg = LoadConfig::default();
        assert_eq!(cfg.cache_type_k, Some(KvQuantType::Q8_0));
        assert_eq!(cfg.cache_type_v, Some(KvQuantType::Q8_0));
        assert_eq!(cfg.custom_cli_args, None);
    }

    #[test]
    fn spec_type_classification_handles_native_mtp_and_external_draft() {
        assert!(spec_type_uses_external_draft("draft-dspark"));
        assert!(spec_type_uses_external_draft("draft-dflash"));
        assert!(spec_type_uses_external_draft("draft-dspark,ngram-mod"));
        assert!(!spec_type_uses_external_draft("draft-mtp"));
        assert!(!spec_type_uses_external_draft("mtp"));
        assert!(!spec_type_uses_external_draft("ngram-mod"));

        assert!(spec_type_is_native_mtp(None));
        assert!(spec_type_is_native_mtp(Some("draft-mtp")));
        assert!(spec_type_is_native_mtp(Some("mtp")));
        assert!(!spec_type_is_native_mtp(Some("draft-dspark")));

        let mut cfg = LoadConfig::new("main.gguf");
        cfg.mtp_enabled = true;
        cfg.mtp_spec_type = Some("draft-dspark".into());
        assert!(!cfg.uses_native_mtp());
        assert!(!cfg.uses_external_draft());

        cfg.draft_model_path = Some("dspark.gguf".into());
        assert!(cfg.uses_external_draft());

        cfg.mtp_spec_type = Some("draft-mtp".into());
        assert!(cfg.uses_native_mtp());
        assert!(!cfg.uses_external_draft());
    }
}
