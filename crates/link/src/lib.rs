//! Bibulm Link: non-invasive BYO Tailscale userspace mesh powered by the
//! `bibulm-linkd` (`tsnet`) sidecar. Handles authentication, peer discovery,
//! remote model resolution, and unbuffered streaming proxy calls.

pub mod manager;
pub mod types;

pub use manager::{LINK_SECRET_HEADER, LinkManager};
pub use types::{
    LinkConnectionState, LinkStatus, PeerInfoPayload, RemoteActionResponse,
    RemoteDeleteModelRequest, RemoteDownloadActionRequest, RemoteDraftCandidate,
    RemoteInstallRuntimeRequest, RemoteInstallRuntimeResponse, RemoteLoadRequest,
    RemoteLoadResponse, RemoteModelDownloadRequest, RemoteModelInspectRequest,
    RemoteModelInspectResponse, RemoteModelSummary, RemotePeer, RemoteRuntimeSummary,
    RemoteRuntimesResponse, RemoteSetActiveRuntimeRequest, RemoteUninstallRuntimeRequest,
};
