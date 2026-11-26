//! HuggingFace Hub GGUF search, quantization/fit inspection,
//! resumable multi-part downloader, and local model library scanner.

pub mod downloader;
pub mod hf_client;
pub mod library;

pub use downloader::{DownloadError, DownloadJobState, DownloadManager, JobKind, JobStatus};
pub use hf_client::{
    HfDownloadFile, HfError, HfModelDetail, HfModelSummary, HfQuantOption, HfSortOrder,
    ModelCapabilities, get_hf_model_detail, search_hf_gguf_models,
};
pub use library::{
    LocalModelEntry, delete_local_model, scan_local_models, scan_local_models_with_external,
};
