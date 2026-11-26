//! `llama.cpp` runtime manager, GitHub release fetcher, binary installer,
//! CLI argument builder, `llama-server` process supervisor, and startup log memory-placement parser.

pub mod cli_builder;
pub mod github;
pub mod help_parser;
pub mod installer;
pub mod placement_parser;
pub mod supervisor;
pub mod telemetry;

pub use cli_builder::{
    build_editable_llama_args, build_llama_server_args, format_cli_args, format_cli_preview,
    split_extra_args,
};
pub use github::{
    GitHubError, HardwareMatchContext, LlamaRelease, ReleaseAssetInfo, RuntimeBackendVariant,
    TargetPlatform, categorize_platform_assets, categorize_win_x64_assets,
    fetch_llama_cpp_releases,
};
pub use help_parser::{DiscoveredCliFlag, discover_runtime_cli_flags, parse_llama_help_output};
pub use installer::{
    InstallProgress, InstallerError, install_runtime_variant, prune_old_runtimes,
    register_custom_runtime, resolve_bench_exe, runtime_family, uninstall_runtime,
};
pub use placement_parser::{parse_placement_line, parse_placement_logs};
pub use supervisor::{
    LoadedModelInfo, ModelSupervisor, ServerLogEntry, SupervisorError, configure_runtime_command,
};
pub use telemetry::{
    LlamaServerMetrics, LlamaSlotInfo, ParsedPrometheusMetrics, fetch_llama_server_metrics,
    parse_prometheus_metrics,
};
