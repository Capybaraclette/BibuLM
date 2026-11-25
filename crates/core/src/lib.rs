//! Core domain types, native GGUF header parser, VRAM/RAM/MoE estimator,
//! load configurations, placement reports, and SQLite persistence shared across all Bibulm crates.

pub mod config;
pub mod db;
pub mod estimator;
pub mod gguf;
pub mod load_params;
pub mod placement;
pub mod secrets;

pub use config::{
    AppSettings, ExternalModelSource, ExternalSourceKind, GuardrailLevel, LinkAccessPolicy,
    LinkConfig, data_dir_settings_default, default_data_dir, discover_external_model_sources,
    legacy_data_dir,
};
pub use db::{
    ApiTokenRecord, BenchmarkRunRecord, Database, DbError, DownloadRecord, InstalledRuntimeRecord,
    NamedPresetRecord,
};
pub use estimator::{
    FitVerdict, LiveMemoryContext, MemoryEstimate, auto_fit_moe, auto_fit_moe_with_sizes,
    estimate_compute_buffer_mib, estimate_hf_file_fit, estimate_kv_cache_bytes,
    estimate_memory_usage, estimate_memory_usage_with_context, estimate_memory_usage_with_sizes,
    estimate_vision_mib, estimate_vision_mib_from_bytes,
};
pub use gguf::{
    DraftCompatibilityLevel, DraftCompatibilityReport, GgmlType, GgufError, GgufMetadata,
    TensorMemorySummary, check_draft_compatibility, discover_split_shards,
    infer_quant_from_filename, is_mtp_tensor_name, is_routed_moe_expert_tensor,
    is_secondary_split_shard, parse_gguf_file, parse_split_shard_pattern,
};
pub use load_params::{
    CliFlagValue, FlashAttentionMode, KvQuantType, LoadConfig, ModelAliasRule, MoeOffloadMode,
    ReasoningConfig, ReasoningEffort, ReasoningFormat, RopeScalingType, SamplingParams,
    ServerConfig, SplitMode, TensorOverride, apply_sampling_to_openai_body,
};
pub use placement::{
    BufferAllocation, MemoryComponentBreakdown, MemoryPlacementReport, PlacementStatus,
};
