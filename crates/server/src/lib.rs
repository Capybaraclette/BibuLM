//! OpenAI-compatible HTTP gateway (`/v1/models`, `/v1/chat/completions`,
//! `/v1/completions`, `/v1/embeddings`, `/v1/responses`) with API key auth, CORS,
//! Just-in-Time model loading, idle TTL auto-unloading, and SSE streaming pass-through.

pub mod anthropic;
pub mod jit;
pub mod proxy;
pub mod router;

pub use jit::{GatewayContext, LinkHardwareSnapshot, ResolvedTarget};
pub use router::{
    ApiGateway, GatewayStatus, build_router, is_safe_single_filename, is_valid_hf_download_url,
    is_valid_llama_release_url, sanitize_repo_subpath,
};
