use crate::types::{
    LinkConnectionState, LinkStatus, RemoteActionResponse, RemoteDeleteModelRequest,
    RemoteDownloadActionRequest, RemoteInstallRuntimeRequest, RemoteInstallRuntimeResponse,
    RemoteLoadRequest, RemoteLoadResponse, RemoteModelDownloadRequest, RemoteModelInspectRequest,
    RemoteModelInspectResponse, RemotePeer, RemoteRuntimesResponse, RemoteSetActiveRuntimeRequest,
    RemoteUninstallRuntimeRequest,
};
use bibulm_core::{LinkConfig, LoadConfig, default_data_dir};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{Mutex, RwLock, broadcast};
use tracing::{info, warn};
use uuid::Uuid;

/// Shared secret header authenticating requests between BibuLM and `bibulm-linkd`.
pub const LINK_SECRET_HEADER: &str = "X-Link-Secret";

#[derive(Debug, Deserialize)]
struct SidecarHandshake {
    event: String,
    bridge_port: u16,
    hostname: String,
}

#[derive(Debug, Deserialize)]
struct SidecarNodeStatus {
    state: String,
    #[serde(default)]
    auth_url: Option<String>,
    hostname: String,
    #[serde(default)]
    node_id: Option<String>,
    #[serde(default)]
    tailscale_ip: Option<String>,
    #[serde(default)]
    magic_dns: Option<String>,
    #[serde(default)]
    tailnet_name: Option<String>,
    #[serde(default)]
    account_login: Option<String>,
    #[serde(default)]
    account_name: Option<String>,
    access_policy: String,
    expose_local_models: bool,
    #[serde(default)]
    error: Option<String>,
}

struct SidecarHandle {
    child: Child,
    _stdin: ChildStdin,
    bridge_port: u16,
    poller_abort: tokio::task::AbortHandle,
    #[cfg(target_os = "windows")]
    _job_guard: Option<WindowsJobGuard>,
}

/// Supervises the `bibulm-linkd` (`tsnet`) sidecar, manages Tailscale login/logout,
/// caches discovered BibuLM peers, and proxies requests over the encrypted mesh.
pub struct LinkManager {
    secret: String,
    state_dir: PathBuf,
    http_client: reqwest::Client,
    running: Mutex<Option<SidecarHandle>>,
    active_bridge_port: Arc<RwLock<Option<u16>>>,
    status: Arc<RwLock<LinkStatus>>,
    peers: Arc<RwLock<Vec<RemotePeer>>>,
    status_tx: broadcast::Sender<LinkStatus>,
    peers_tx: broadcast::Sender<Vec<RemotePeer>>,
}

impl LinkManager {
    pub fn new(initial_cfg: &LinkConfig) -> Self {
        Self::new_with_data_dir(initial_cfg, &default_data_dir())
    }

    pub fn new_with_data_dir(initial_cfg: &LinkConfig, data_dir: &Path) -> Self {
        let secret = format!("{}{}", Uuid::new_v4().simple(), Uuid::new_v4().simple());
        let state_dir = data_dir.join("link").join("tsnet-state");
        let http_client = reqwest::Client::builder()
            .pool_idle_timeout(Duration::from_secs(60))
            .build()
            .expect("link reqwest client");
        let (status_tx, _) = broadcast::channel(64);
        let (peers_tx, _) = broadcast::channel(64);

        let sidecar = locate_sidecar_binary();
        let hostname = effective_device_name(initial_cfg);
        let initial_status = LinkStatus::disconnected(
            hostname,
            initial_cfg.access_policy.as_str().to_string(),
            initial_cfg.expose_local_models,
            sidecar.map(|p| p.to_string_lossy().into_owned()),
        );

        Self {
            secret,
            state_dir,
            http_client,
            running: Mutex::new(None),
            active_bridge_port: Arc::new(RwLock::new(None)),
            status: Arc::new(RwLock::new(initial_status)),
            peers: Arc::new(RwLock::new(Vec::new())),
            status_tx,
            peers_tx,
        }
    }

    /// Constant-time check used by `bibulm-server` to verify requests forwarded by the sidecar.
    pub fn verify_secret(&self, candidate: &str) -> bool {
        bibulm_core::secrets::constant_time_eq(self.secret.as_bytes(), candidate.as_bytes())
    }

    pub fn subscribe_status(&self) -> broadcast::Receiver<LinkStatus> {
        self.status_tx.subscribe()
    }

    pub fn subscribe_peers(&self) -> broadcast::Receiver<Vec<RemotePeer>> {
        self.peers_tx.subscribe()
    }

    pub async fn status(&self) -> LinkStatus {
        let mut st = self.status.read().await.clone();
        if !st.sidecar_available
            && let Some(p) = locate_sidecar_binary()
        {
            st.sidecar_available = true;
            st.sidecar_path = Some(p.to_string_lossy().into_owned());
        }
        st
    }

    pub async fn peers(&self) -> Vec<RemotePeer> {
        self.peers.read().await.clone()
    }

    /// Starts the `bibulm-linkd` sidecar and joins (or prompts login for) the user's Tailnet.
    pub async fn connect(&self, cfg: &LinkConfig, gateway_port: u16) -> Result<LinkStatus, String> {
        self.connect_with_auth_key(cfg, gateway_port, None).await
    }

    /// Starts the `bibulm-linkd` sidecar with an optional headless Tailscale auth key (`TS_AUTHKEY`).
    pub async fn connect_with_auth_key(
        &self,
        cfg: &LinkConfig,
        gateway_port: u16,
        auth_key: Option<&str>,
    ) -> Result<LinkStatus, String> {
        // Stop any existing sidecar cleanly first.
        self.disconnect(cfg).await;

        let bin_path = locate_sidecar_binary().ok_or_else(|| {
            "bibulm-linkd sidecar binary not found. Build it with `scripts/build-linkd.ps1` (or reinstall BibuLM).".to_string()
        })?;

        std::fs::create_dir_all(&self.state_dir)
            .map_err(|e| format!("Failed to create Link state directory: {e}"))?;

        let hostname = effective_device_name(cfg);
        let mut cmd = Command::new(&bin_path);
        cmd.arg("--state-dir")
            .arg(&self.state_dir)
            .arg("--hostname")
            .arg(&hostname)
            .arg("--gateway-port")
            .arg(gateway_port.to_string())
            .arg("--mesh-port")
            .arg(cfg.mesh_port.to_string())
            .arg("--access-policy")
            .arg(cfg.access_policy.as_str())
            .arg(format!("--expose-models={}", cfg.expose_local_models))
            .env("BIBULM_LINK_SECRET", &self.secret)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        if let Some(key) = auth_key.map(str::trim).filter(|s| !s.is_empty()) {
            cmd.env("TS_AUTHKEY", key);
        }

        if let Some(url) = cfg
            .control_url
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            cmd.arg("--control-url").arg(url);
        }

        #[cfg(target_os = "windows")]
        {
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }

        let mut child = cmd
            .spawn()
            .map_err(|e| format!("Failed to launch bibulm-linkd ({}): {e}", bin_path.display()))?;

        #[cfg(target_os = "windows")]
        let job_guard = {
            let guard = WindowsJobGuard::create().ok();
            if let (Some(g), Some(pid)) = (&guard, child.id()) {
                let _ = g.assign_pid(pid);
            }
            guard
        };

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "Failed to attach stdin to bibulm-linkd".to_string())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "Failed to attach stdout to bibulm-linkd".to_string())?;
        let stderr = child.stderr.take();

        // Drain stderr to tracing logs so pipe never blocks.
        if let Some(err_stream) = stderr {
            tokio::spawn(async move {
                let mut lines = BufReader::new(err_stream).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    let trimmed = line.trim();
                    if !trimmed.is_empty() {
                        info!(target: "bibulm_linkd", "{trimmed}");
                    }
                }
            });
        }

        // Wait for the JSON readiness handshake line on stdout.
        let mut reader = BufReader::new(stdout).lines();
        let handshake = tokio::time::timeout(Duration::from_secs(12), async {
            while let Ok(Some(line)) = reader.next_line().await {
                if let Ok(hs) = serde_json::from_str::<SidecarHandshake>(&line)
                    && hs.event == "ready"
                {
                    return Ok(hs);
                }
            }
            Err("bibulm-linkd exited before sending ready handshake".to_string())
        })
        .await
        .map_err(|_| "Timed out waiting for bibulm-linkd startup".to_string())??;

        let bridge_port = handshake.bridge_port;
        let sidecar_path_str = Some(bin_path.to_string_lossy().into_owned());

        // Set initial "starting" status.
        {
            let mut st = self.status.write().await;
            *st = LinkStatus {
                state: LinkConnectionState::Starting,
                auth_url: None,
                hostname: handshake.hostname.clone(),
                node_id: None,
                tailscale_ip: None,
                magic_dns: None,
                tailnet_name: None,
                account_login: None,
                account_name: None,
                access_policy: cfg.access_policy.as_str().to_string(),
                expose_local_models: cfg.expose_local_models,
                sidecar_available: true,
                sidecar_path: sidecar_path_str.clone(),
                error: None,
            };
            let _ = self.status_tx.send(st.clone());
        }

        // Spawn background poller synchronizing sidecar status & peer list into Rust state.
        let poll_client = self.http_client.clone();
        let poll_secret = self.secret.clone();
        let poll_status = self.status.clone();
        let poll_peers = self.peers.clone();
        let poll_status_tx = self.status_tx.clone();
        let poll_peers_tx = self.peers_tx.clone();
        let poll_sidecar_path = sidecar_path_str.clone();

        let poller = tokio::spawn(async move {
            let mut consecutive_errors = 0u32;
            loop {
                let status_res =
                    fetch_bridge_status(&poll_client, bridge_port, &poll_secret, &poll_sidecar_path)
                        .await;
                let mut is_connected = false;
                match status_res {
                    Ok(next_status) => {
                        consecutive_errors = 0;
                        is_connected = next_status.state == LinkConnectionState::Connected;
                        let changed = {
                            let mut cur = poll_status.write().await;
                            if *cur != next_status {
                                *cur = next_status.clone();
                                true
                            } else {
                                false
                            }
                        };
                        if changed {
                            let _ = poll_status_tx.send(next_status);
                        }
                    }
                    Err(e) => {
                        consecutive_errors += 1;
                        if consecutive_errors >= 5 {
                            warn!("bibulm-linkd bridge unreachable: {e}");
                            let mut cur = poll_status.write().await;
                            cur.state = LinkConnectionState::Error;
                            cur.error = Some("bibulm-linkd process stopped unexpectedly.".to_string());
                            let _ = poll_status_tx.send(cur.clone());
                            break;
                        }
                    }
                }

                if is_connected {
                    // The sidecar re-probes peers on its own 3 s ticker; just read its cached list.
                    let next_peers_res =
                        fetch_bridge_peers(&poll_client, bridge_port, &poll_secret).await;
                    if let Ok(next_peers) = next_peers_res {
                        let changed = {
                            let mut cur = poll_peers.write().await;
                            if *cur != next_peers {
                                *cur = next_peers.clone();
                                true
                            } else {
                                false
                            }
                        };
                        if changed {
                            let _ = poll_peers_tx.send(next_peers);
                        }
                    }
                    tokio::time::sleep(Duration::from_secs(3)).await;
                } else {
                    tokio::time::sleep(Duration::from_millis(900)).await;
                }
            }
        });

        *self.running.lock().await = Some(SidecarHandle {
            child,
            _stdin: stdin,
            bridge_port,
            poller_abort: poller.abort_handle(),
            #[cfg(target_os = "windows")]
            _job_guard: job_guard,
        });
        *self.active_bridge_port.write().await = Some(bridge_port);

        // Give tsnet a brief window (up to 3.5s) to emit `needs_login` (with auth_url) or `connected`.
        for _ in 0..14 {
            tokio::time::sleep(Duration::from_millis(250)).await;
            let cur = self.status.read().await.clone();
            if cur.state == LinkConnectionState::Connected
                || (cur.state == LinkConnectionState::NeedsLogin && cur.auth_url.is_some())
                || cur.state == LinkConnectionState::Error
            {
                return Ok(cur);
            }
        }

        Ok(self.status.read().await.clone())
    }

    /// Hot-updates gateway port / access policy / host sharing without restarting `tsnet`.
    pub async fn sync_config(&self, cfg: &LinkConfig, gateway_port: u16) {
        let port_opt = self.running.lock().await.as_ref().map(|h| h.bridge_port);
        if let Some(bridge_port) = port_opt {
            let url = format!("http://127.0.0.1:{bridge_port}/bridge/config");
            let _ = self
                .http_client
                .post(url)
                .header(LINK_SECRET_HEADER, &self.secret)
                .json(&json!({
                    "gateway_port": gateway_port,
                    "access_policy": cfg.access_policy.as_str(),
                    "expose_local_models": cfg.expose_local_models,
                }))
                .send()
                .await;
        } else {
            let mut st = self.status.write().await;
            st.hostname = effective_device_name(cfg);
            st.access_policy = cfg.access_policy.as_str().to_string();
            st.expose_local_models = cfg.expose_local_models;
            let _ = self.status_tx.send(st.clone());
        }
    }

    /// Disconnects the Tailscale node cleanly without revoking the device key.
    pub async fn disconnect(&self, cfg: &LinkConfig) -> LinkStatus {
        *self.active_bridge_port.write().await = None;
        let handle = self.running.lock().await.take();
        if let Some(mut h) = handle {
            h.poller_abort.abort();
            let url = format!("http://127.0.0.1:{}/bridge/shutdown", h.bridge_port);
            let _ = self
                .http_client
                .post(url)
                .header(LINK_SECRET_HEADER, &self.secret)
                .timeout(Duration::from_secs(2))
                .send()
                .await;
            let _ = tokio::time::timeout(Duration::from_secs(2), h.child.wait()).await;
            let _ = h.child.kill().await;
        }

        let sidecar = locate_sidecar_binary().map(|p| p.to_string_lossy().into_owned());
        let next = LinkStatus::disconnected(
            effective_device_name(cfg),
            cfg.access_policy.as_str().to_string(),
            cfg.expose_local_models,
            sidecar,
        );
        *self.status.write().await = next.clone();
        *self.peers.write().await = Vec::new();
        let _ = self.status_tx.send(next.clone());
        let _ = self.peers_tx.send(Vec::new());
        next
    }

    /// Logs out from Tailscale (removing this device from the user's tailnet) and clears local state.
    pub async fn logout(&self, cfg: &LinkConfig) -> Result<LinkStatus, String> {
        let bridge_port_opt = self.running.lock().await.as_ref().map(|h| h.bridge_port);
        if let Some(bridge_port) = bridge_port_opt {
            let url = format!("http://127.0.0.1:{bridge_port}/bridge/logout");
            let resp = self
                .http_client
                .post(url)
                .header(LINK_SECRET_HEADER, &self.secret)
                .timeout(Duration::from_secs(6))
                .send()
                .await;
            if let Ok(r) = resp
                && !r.status().is_success()
            {
                let txt = r.text().await.unwrap_or_default();
                warn!("Tailscale logout returned non-200: {txt}");
            }
        }
        let st = self.disconnect(cfg).await;
        if self.state_dir.exists() {
            let _ = std::fs::remove_dir_all(&self.state_dir);
        }
        Ok(st)
    }

    /// Forces an immediate discovery scan across all online Tailnet peers.
    pub async fn refresh_peers(&self) -> Result<Vec<RemotePeer>, String> {
        let bridge_port = self
            .bridge_port()
            .await
            .ok_or_else(|| "BibuLM Link is not connected.".to_string())?;

        let url = format!("http://127.0.0.1:{bridge_port}/bridge/peers/refresh");
        let resp = self
            .http_client
            .post(url)
            .header(LINK_SECRET_HEADER, &self.secret)
            .timeout(Duration::from_secs(10))
            .send()
            .await
            .map_err(|e| format!("Failed to refresh Link peers: {e}"))?;

        if !resp.status().is_success() {
            return Err(format!("Peer refresh returned HTTP {}", resp.status()));
        }
        let list: Vec<RemotePeer> = resp
            .json()
            .await
            .map_err(|e| format!("Failed to parse peer list: {e}"))?;

        *self.peers.write().await = list.clone();
        let _ = self.peers_tx.send(list.clone());
        Ok(list)
    }

    /// Returns `(bridge_proxy_url, secret)` for calling a remote peer endpoint through `tsnet.Dial`.
    pub async fn proxy_endpoint(
        &self,
        peer_ip: &str,
        remote_path: &str,
    ) -> Result<(String, String), String> {
        let bridge_port = self
            .bridge_port()
            .await
            .ok_or_else(|| "BibuLM Link is not connected.".to_string())?;
        let clean_path = remote_path.trim_start_matches('/');
        let url = format!(
            "http://127.0.0.1:{bridge_port}/bridge/proxy/{peer_ip}/{clean_path}"
        );
        Ok((url, self.secret.clone()))
    }

    /// Sends a JSON POST request to a remote peer through the `tsnet` bridge, returning the
    /// streaming `reqwest::Response` directly (suitable for SSE pass-through).
    pub async fn proxy_post_json(
        &self,
        peer_ip: &str,
        remote_path: &str,
        payload: &serde_json::Value,
    ) -> Result<reqwest::Response, String> {
        let (url, secret) = self.proxy_endpoint(peer_ip, remote_path).await?;
        self.http_client
            .post(url)
            .header(LINK_SECRET_HEADER, secret)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .json(payload)
            .send()
            .await
            .map_err(|e| format!("Link proxy error: {e}"))
    }

    /// Sends a GET request to a remote peer through the `tsnet` bridge.
    pub async fn proxy_get(
        &self,
        peer_ip: &str,
        remote_path: &str,
    ) -> Result<reqwest::Response, String> {
        let (url, secret) = self.proxy_endpoint(peer_ip, remote_path).await?;
        self.http_client
            .get(url)
            .header(LINK_SECRET_HEADER, secret)
            .send()
            .await
            .map_err(|e| format!("Link proxy error: {e}"))
    }

    /// Fires a non-blocking notification to all online Tailnet peers so they immediately
    /// refresh their cached model state (e.g., right after loading or unloading a local model).
    pub fn notify_peers_state_changed(&self) {
        let client = self.http_client.clone();
        let secret = self.secret.clone();
        let peers_arc = self.peers.clone();
        let port_arc = self.active_bridge_port.clone();
        tokio::spawn(async move {
            let Some(bridge_port) = *port_arc.read().await else {
                return;
            };
            let peer_ips: Vec<String> = {
                let peers = peers_arc.read().await;
                peers
                    .iter()
                    .filter(|p| p.online)
                    .map(|p| p.tailscale_ip.clone())
                    .collect()
            };
            for peer_ip in peer_ips {
                let c = client.clone();
                let s = secret.clone();
                tokio::spawn(async move {
                    let url = format!(
                        "http://127.0.0.1:{bridge_port}/bridge/proxy/{peer_ip}/link/v1/notify"
                    );
                    let _ = c
                        .post(url)
                        .header(LINK_SECRET_HEADER, s)
                        .timeout(Duration::from_secs(2))
                        .json(&json!({}))
                        .send()
                        .await;
                });
            }
        });
    }

    /// Inspects a remote model's GGUF metadata, saved preset, active config, runtimes, and live memory headroom.
    pub async fn remote_inspect_model(
        &self,
        peer_id_or_slug: &str,
        model_id: &str,
    ) -> Result<RemoteModelInspectResponse, String> {
        let peer = self
            .require_peer(peer_id_or_slug, PeerPermission::Load, "remote model configuration")
            .await?;
        let req = RemoteModelInspectRequest { model_id: model_id.to_string() };
        self.remote_post(&peer, "/link/v1/models/inspect", &req, "inspect").await
    }

    /// Asks a remote peer to load one of its local models (using explicit config, saved preset, or auto-fit).
    pub async fn remote_load_model(
        &self,
        peer_id_or_slug: &str,
        model_id: &str,
        config: Option<LoadConfig>,
        remember_preset: bool,
    ) -> Result<RemoteLoadResponse, String> {
        let peer = self
            .require_peer(peer_id_or_slug, PeerPermission::Load, "remote model loading")
            .await?;
        let req = RemoteLoadRequest { model_id: model_id.to_string(), config, remember_preset };
        let parsed = self.remote_post(&peer, "/link/v1/models/load", &req, "load").await?;
        let _ = self.refresh_peers().await;
        Ok(parsed)
    }

    /// Asks a remote peer to unload a loaded model.
    pub async fn remote_unload_model(
        &self,
        peer_id_or_slug: &str,
        model_id: &str,
    ) -> Result<RemoteLoadResponse, String> {
        let peer = self
            .require_peer(peer_id_or_slug, PeerPermission::Load, "remote model unloading")
            .await?;
        let req = RemoteLoadRequest {
            model_id: model_id.to_string(),
            config: None,
            remember_preset: false,
        };
        let parsed = self.remote_post(&peer, "/link/v1/models/unload", &req, "unload").await?;
        let _ = self.refresh_peers().await;
        Ok(parsed)
    }

    pub async fn remote_list_runtimes(
        &self,
        peer_id_or_slug: &str,
    ) -> Result<RemoteRuntimesResponse, String> {
        let peer = self.require_manageable_peer(peer_id_or_slug).await?;
        self.remote_get(&peer, "/link/v1/runtimes", "runtimes").await
    }

    pub async fn remote_fetch_runtime_releases(
        &self,
        peer_id_or_slug: &str,
    ) -> Result<Vec<bibulm_runtime::LlamaRelease>, String> {
        let peer = self.require_manageable_peer(peer_id_or_slug).await?;
        self.remote_get(&peer, "/link/v1/runtimes/releases", "runtime releases").await
    }

    pub async fn remote_install_runtime(
        &self,
        peer_id_or_slug: &str,
        variant: bibulm_runtime::RuntimeBackendVariant,
        release_notes: Option<String>,
    ) -> Result<RemoteInstallRuntimeResponse, String> {
        let peer = self.require_manageable_peer(peer_id_or_slug).await?;
        let req = RemoteInstallRuntimeRequest { variant, release_notes };
        self.remote_post(&peer, "/link/v1/runtimes/install", &req, "runtime install").await
    }

    pub async fn remote_set_active_runtime(
        &self,
        peer_id_or_slug: &str,
        runtime_id: &str,
    ) -> Result<RemoteActionResponse, String> {
        let peer = self.require_manageable_peer(peer_id_or_slug).await?;
        let req = RemoteSetActiveRuntimeRequest { runtime_id: runtime_id.to_string() };
        let parsed = self
            .remote_post(&peer, "/link/v1/runtimes/active", &req, "set active runtime")
            .await?;
        let _ = self.refresh_peers().await;
        Ok(parsed)
    }

    pub async fn remote_uninstall_runtime(
        &self,
        peer_id_or_slug: &str,
        runtime_id: &str,
    ) -> Result<RemoteActionResponse, String> {
        let peer = self.require_manageable_peer(peer_id_or_slug).await?;
        let req = RemoteUninstallRuntimeRequest { runtime_id: runtime_id.to_string() };
        let parsed = self
            .remote_post(&peer, "/link/v1/runtimes/uninstall", &req, "uninstall runtime")
            .await?;
        let _ = self.refresh_peers().await;
        Ok(parsed)
    }

    pub async fn remote_enqueue_model_download(
        &self,
        peer_id_or_slug: &str,
        title: String,
        repo_id: String,
        files: Vec<bibulm_models::HfDownloadFile>,
    ) -> Result<bibulm_models::DownloadJobState, String> {
        let peer = self.require_manageable_peer(peer_id_or_slug).await?;
        let req = RemoteModelDownloadRequest { title, repo_id, files };
        self.remote_post(&peer, "/link/v1/models/download", &req, "model download").await
    }

    pub async fn remote_list_downloads(
        &self,
        peer_id_or_slug: &str,
    ) -> Result<Vec<bibulm_models::DownloadJobState>, String> {
        let peer = self.require_manageable_peer(peer_id_or_slug).await?;
        self.remote_get(&peer, "/link/v1/downloads", "list downloads").await
    }

    pub async fn remote_pause_download(
        &self,
        peer_id_or_slug: &str,
        id: &str,
    ) -> Result<bool, String> {
        self.remote_download_action(peer_id_or_slug, id, "pause").await
    }

    pub async fn remote_resume_download(
        &self,
        peer_id_or_slug: &str,
        id: &str,
    ) -> Result<bool, String> {
        self.remote_download_action(peer_id_or_slug, id, "resume").await
    }

    pub async fn remote_cancel_download(
        &self,
        peer_id_or_slug: &str,
        id: &str,
    ) -> Result<bool, String> {
        self.remote_download_action(peer_id_or_slug, id, "cancel").await
    }

    async fn remote_download_action(
        &self,
        peer_id_or_slug: &str,
        id: &str,
        action: &str,
    ) -> Result<bool, String> {
        let peer = self.require_manageable_peer(peer_id_or_slug).await?;
        let req = RemoteDownloadActionRequest { id: id.to_string() };
        let path = format!("/link/v1/downloads/{action}");
        let parsed: RemoteActionResponse = self
            .remote_post(&peer, &path, &req, &format!("{action} download"))
            .await?;
        Ok(parsed.ok)
    }

    pub async fn remote_clear_completed_downloads(
        &self,
        peer_id_or_slug: &str,
    ) -> Result<(), String> {
        let peer = self.require_manageable_peer(peer_id_or_slug).await?;
        let _: RemoteActionResponse = self
            .remote_post(&peer, "/link/v1/downloads/clear", &json!({}), "clear downloads")
            .await?;
        Ok(())
    }

    pub async fn remote_delete_model(
        &self,
        peer_id_or_slug: &str,
        model_id: &str,
    ) -> Result<RemoteActionResponse, String> {
        let peer = self.require_manageable_peer(peer_id_or_slug).await?;
        let req = RemoteDeleteModelRequest { model_id: model_id.to_string() };
        let parsed = self.remote_post(&peer, "/link/v1/models/delete", &req, "delete model").await?;
        let _ = self.refresh_peers().await;
        Ok(parsed)
    }

    pub async fn remote_get_logs(
        &self,
        peer_id_or_slug: &str,
    ) -> Result<Vec<bibulm_runtime::ServerLogEntry>, String> {
        let peer = self.require_manageable_peer(peer_id_or_slug).await?;
        self.remote_get(&peer, "/link/v1/logs", "logs").await
    }

    async fn require_manageable_peer(&self, peer_id_or_slug: &str) -> Result<RemotePeer, String> {
        let what = "remote runtime/model management";
        self.require_peer(peer_id_or_slug, PeerPermission::Manage, what).await
    }

    /// Finds an online peer and checks it grants `perm`; `what` names the action in the error.
    async fn require_peer(
        &self,
        peer_id_or_slug: &str,
        perm: PeerPermission,
        what: &str,
    ) -> Result<RemotePeer, String> {
        let peer = self
            .find_peer(peer_id_or_slug)
            .await
            .ok_or_else(|| format!("Remote peer '{peer_id_or_slug}' not found or offline."))?;
        let allowed = match perm {
            PeerPermission::Load => peer.allow_remote_load,
            PeerPermission::Manage => peer.allow_remote_manage,
        };
        if !allowed {
            return Err(format!("Remote peer '{}' does not allow {what}.", peer.device_name));
        }
        Ok(peer)
    }

    async fn remote_get<T: DeserializeOwned>(
        &self,
        peer: &RemotePeer,
        path: &str,
        label: &str,
    ) -> Result<T, String> {
        let resp = self.proxy_get(&peer.tailscale_ip, path).await?;
        decode_remote(resp, label).await
    }

    async fn remote_post<T: DeserializeOwned>(
        &self,
        peer: &RemotePeer,
        path: &str,
        body: &impl Serialize,
        label: &str,
    ) -> Result<T, String> {
        let payload = serde_json::to_value(body).map_err(|e| e.to_string())?;
        let resp = self.proxy_post_json(&peer.tailscale_ip, path, &payload).await?;
        decode_remote(resp, label).await
    }

    pub async fn find_peer(&self, peer_id_or_slug: &str) -> Option<RemotePeer> {
        let q = peer_id_or_slug.trim().to_ascii_lowercase();
        let peers = self.peers.read().await;
        peers.iter().find(|p| {
            p.node_id.eq_ignore_ascii_case(&q)
                || p.slug.eq_ignore_ascii_case(&q)
                || p.device_name.eq_ignore_ascii_case(&q)
                || p.tailscale_ip == q
        }).cloned()
    }

    /// Resolves a model query (either explicit `<peer_slug>/<model_id>` or unqualified `<model_id>`)
    /// against currently connected, exposing remote peers.
    /// Returns `(peer, remote_model_id)`.
    pub async fn resolve_remote_model(&self, query: &str) -> Option<(RemotePeer, String)> {
        let peers = self.peers.read().await;
        resolve_remote_in_slice(&peers, query)
    }

    async fn bridge_port(&self) -> Option<u16> {
        self.running.lock().await.as_ref().map(|h| h.bridge_port)
    }
}

/// Pure peer resolution helper (extracted for easy unit testing).
pub fn resolve_remote_in_slice(peers: &[RemotePeer], query: &str) -> Option<(RemotePeer, String)> {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return None;
    }

    // 1. Explicit peer routing: `<peer_slug>/<model_id>`
    if let Some((prefix, rest)) = trimmed.split_once('/') {
        let p_low = prefix.trim().to_ascii_lowercase();
        let m_rest = rest.trim();
        if !m_rest.is_empty()
            && let Some(peer) = peers.iter().find(|p| {
                p.online
                    && p.expose_local_models
                    && (p.slug.eq_ignore_ascii_case(&p_low)
                        || p.device_name.eq_ignore_ascii_case(&p_low))
            })
        {
            return Some((peer.clone(), m_rest.to_string()));
        }
    }

    // Sort candidate peers: direct connections first, then lowest latency.
    let mut candidates: Vec<&RemotePeer> = peers
        .iter()
        .filter(|p| p.online && p.expose_local_models)
        .collect();
    candidates.sort_by_key(|p| {
        let is_direct = if p.connection_type == "direct" { 0u8 } else { 1u8 };
        let rtt = p.latency_ms.unwrap_or(u64::MAX);
        (is_direct, rtt)
    });

    let q_low = trimmed.to_ascii_lowercase();

    // 2. Unqualified match against ALREADY LOADED models on any peer
    for peer in &candidates {
        if let Some(m) = peer.loaded_models.iter().find(|m| {
            m.id.eq_ignore_ascii_case(&q_low) || m.display_name.eq_ignore_ascii_case(&q_low)
        }) {
            return Some(((*peer).clone(), m.id.clone()));
        }
    }

    // 3. Unqualified match against AVAILABLE (JIT-loadable) models on peers allowing remote load
    for peer in &candidates {
        if !peer.allow_remote_load {
            continue;
        }
        if let Some(m) = peer.available_models.iter().find(|m| {
            m.id.eq_ignore_ascii_case(&q_low) || m.display_name.eq_ignore_ascii_case(&q_low)
        }) {
            return Some(((*peer).clone(), m.id.clone()));
        }
    }

    None
}

async fn fetch_bridge_status(
    client: &reqwest::Client,
    bridge_port: u16,
    secret: &str,
    sidecar_path: &Option<String>,
) -> Result<LinkStatus, String> {
    let url = format!("http://127.0.0.1:{bridge_port}/bridge/status");
    let resp = client
        .get(url)
        .header(LINK_SECRET_HEADER, secret)
        .timeout(Duration::from_secs(3))
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("HTTP {}", resp.status()));
    }
    let raw: SidecarNodeStatus = resp.json().await.map_err(|e| e.to_string())?;
    let state = match raw.state.as_str() {
        "starting" => LinkConnectionState::Starting,
        "needs_login" => LinkConnectionState::NeedsLogin,
        "connecting" => LinkConnectionState::Connecting,
        "connected" => LinkConnectionState::Connected,
        "error" => LinkConnectionState::Error,
        _ => LinkConnectionState::Disconnected,
    };
    Ok(LinkStatus {
        state,
        auth_url: raw.auth_url.filter(|s| !s.is_empty()),
        hostname: raw.hostname,
        node_id: raw.node_id.filter(|s| !s.is_empty()),
        tailscale_ip: raw.tailscale_ip.filter(|s| !s.is_empty()),
        magic_dns: raw.magic_dns.filter(|s| !s.is_empty()),
        tailnet_name: raw.tailnet_name.filter(|s| !s.is_empty()),
        account_login: raw.account_login.filter(|s| !s.is_empty()),
        account_name: raw.account_name.filter(|s| !s.is_empty()),
        access_policy: raw.access_policy,
        expose_local_models: raw.expose_local_models,
        sidecar_available: true,
        sidecar_path: sidecar_path.clone(),
        error: raw.error.filter(|s| !s.is_empty()),
    })
}

async fn fetch_bridge_peers(
    client: &reqwest::Client,
    bridge_port: u16,
    secret: &str,
) -> Result<Vec<RemotePeer>, String> {
    let url = format!("http://127.0.0.1:{bridge_port}/bridge/peers");
    let resp = client
        .get(url)
        .header(LINK_SECRET_HEADER, secret)
        .timeout(Duration::from_secs(4))
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("HTTP {}", resp.status()));
    }
    resp.json::<Vec<RemotePeer>>().await.map_err(|e| e.to_string())
}

#[derive(Clone, Copy)]
enum PeerPermission {
    Load,
    Manage,
}

/// Turns a peer reply into `T`, or an error naming the failed action and the peer's reason.
async fn decode_remote<T: DeserializeOwned>(
    resp: reqwest::Response,
    label: &str,
) -> Result<T, String> {
    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("Remote {label} failed ({status}): {text}"));
    }
    resp.json()
        .await
        .map_err(|e| format!("Invalid remote {label} response: {e}"))
}

pub fn effective_device_name(cfg: &LinkConfig) -> String {
    if let Some(custom) = cfg
        .device_name
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        return custom.to_string();
    }
    std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "bibulm-node".to_string())
}

pub fn locate_sidecar_binary() -> Option<PathBuf> {
    if let Ok(explicit) = std::env::var("BIBULM_LINKD_PATH") {
        let p = PathBuf::from(explicit);
        if p.is_file() {
            return Some(p);
        }
    }

    let ext = if cfg!(target_os = "windows") { ".exe" } else { "" };
    let plain_name = format!("bibulm-linkd{ext}");
    let triple_name = format!("bibulm-linkd-{}{ext}", current_target_triple());

    let mut search_dirs: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        search_dirs.push(dir.to_path_buf());
        search_dirs.push(dir.join("binaries"));
        if let Some(parent) = dir.parent() {
            search_dirs.push(parent.join("binaries"));
        }
    }
    search_dirs.push(default_data_dir().join("link").join("bin"));
    if let Ok(cwd) = std::env::current_dir() {
        search_dirs.push(cwd.join("src-tauri").join("binaries"));
        search_dirs.push(cwd.join("binaries"));
    }

    for dir in search_dirs {
        for name in [&plain_name, &triple_name] {
            let candidate = dir.join(name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }

    // Fallback: search PATH
    if let Ok(path_env) = std::env::var("PATH") {
        for entry in std::env::split_paths(&path_env) {
            let candidate = Path::new(&entry).join(&plain_name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }

    None
}

fn current_target_triple() -> &'static str {
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    {
        "x86_64-pc-windows-msvc"
    }
    #[cfg(all(target_os = "windows", target_arch = "aarch64"))]
    {
        "aarch64-pc-windows-msvc"
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        "aarch64-apple-darwin"
    }
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    {
        "x86_64-apple-darwin"
    }
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        "x86_64-unknown-linux-gnu"
    }
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        "aarch64-unknown-linux-gnu"
    }
}

// --- Windows Job Object Guard (ensures sidecar dies if BibuLM crashes) ---

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
    use crate::types::RemoteModelSummary;

    fn sample_peer(
        slug: &str,
        conn_type: &str,
        latency: u64,
        expose: bool,
        allow_load: bool,
        loaded: Vec<&str>,
        avail: Vec<&str>,
    ) -> RemotePeer {
        RemotePeer {
            node_id: format!("node-{slug}"),
            slug: slug.to_string(),
            hostname: format!("bibulm-{slug}"),
            device_name: slug.to_string(),
            tailscale_ip: "100.64.0.10".to_string(),
            magic_dns: format!("bibulm-{slug}.ts.net"),
            os: "windows".to_string(),
            online: true,
            connection_type: conn_type.to_string(),
            relay: None,
            latency_ms: Some(latency),
            app_version: "0.2.0".to_string(),
            expose_local_models: expose,
            allow_remote_load: allow_load,
            allow_remote_manage: allow_load,
            is_headless: false,
            platform: "windows-x86_64".to_string(),
            runtimes_count: 1,
            active_runtime_tag: Some("b5200".to_string()),
            active_downloads_count: 0,
            gpu_summary: "RTX 4090 24 GB".to_string(),
            vram_used_mib: 4000.0,
            vram_total_mib: 24576.0,
            ram_used_mib: 12000.0,
            ram_total_mib: 65536.0,
            loaded_models: loaded
                .into_iter()
                .map(|id| RemoteModelSummary {
                    id: id.to_string(),
                    display_name: id.to_string(),
                    loaded: true,
                    quantization: Some("Q4_K_M".to_string()),
                    architecture: Some("qwen3".to_string()),
                    size_bytes: 0,
                    max_context_length: Some(32768),
                    supports_vision: false,
                    supports_reasoning: true,
                    placement_status: Some("gpu_only".to_string()),
                })
                .collect(),
            available_models: avail
                .into_iter()
                .map(|id| RemoteModelSummary {
                    id: id.to_string(),
                    display_name: id.to_string(),
                    loaded: false,
                    quantization: Some("Q4_K_M".to_string()),
                    architecture: Some("qwen3".to_string()),
                    size_bytes: 0,
                    max_context_length: Some(32768),
                    supports_vision: false,
                    supports_reasoning: false,
                    placement_status: None,
                })
                .collect(),
        }
    }

    #[test]
    fn explicit_peer_prefix_routes_directly() {
        let peers = vec![
            sample_peer("pc-salon", "direct", 5, true, true, vec!["qwen3-32b"], vec![]),
            sample_peer("server-lab", "derp", 45, true, true, vec!["qwen3-32b"], vec![]),
        ];
        let (peer, model) = resolve_remote_in_slice(&peers, "server-lab/qwen3-32b").unwrap();
        assert_eq!(peer.slug, "server-lab");
        assert_eq!(model, "qwen3-32b");
    }

    #[test]
    fn unqualified_prefers_loaded_direct_and_respects_permissions() {
        let peers = vec![
            sample_peer("relay-box", "derp", 40, true, true, vec!["llama-8b"], vec!["mistral-24b"]),
            sample_peer("pc-salon", "direct", 4, true, false, vec!["llama-8b"], vec!["deepseek-70b"]),
            sample_peer("private-lap", "direct", 2, false, true, vec!["secret-model"], vec![]),
        ];

        // Prefers direct over DERP for loaded model
        let (peer, model) = resolve_remote_in_slice(&peers, "llama-8b").unwrap();
        assert_eq!(peer.slug, "pc-salon");
        assert_eq!(model, "llama-8b");

        // Respects allow_remote_load = false for unloaded models
        assert!(resolve_remote_in_slice(&peers, "deepseek-70b").is_none());

        // Allows JIT remote load when allow_remote_load = true
        let (peer2, model2) = resolve_remote_in_slice(&peers, "mistral-24b").unwrap();
        assert_eq!(peer2.slug, "relay-box");
        assert_eq!(model2, "mistral-24b");

        // Never routes to a peer with expose_local_models = false
        assert!(resolve_remote_in_slice(&peers, "secret-model").is_none());
    }

    #[test]
    fn constant_time_secret_check() {
        let mgr = LinkManager::new(&LinkConfig::default());
        assert!(mgr.verify_secret(&mgr.secret));
        assert!(!mgr.verify_secret("wrong-secret"));
    }
}
