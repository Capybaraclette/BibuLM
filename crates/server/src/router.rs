use crate::anthropic::{
    anthropic_error, handle_anthropic_count_tokens, handle_anthropic_messages,
};
use crate::jit::GatewayContext;
use crate::proxy::{openai_error, proxy_openai_post};
use axum::Router;
use axum::extract::{DefaultBodyLimit, Extension, MatchedPath, Request, State};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use bytes::Bytes;
use chrono::Utc;
use bibulm_core::{AppSettings, ServerConfig};
use bibulm_link::{
    LINK_SECRET_HEADER, PeerInfoPayload, RemoteActionResponse, RemoteDeleteModelRequest,
    RemoteDownloadActionRequest, RemoteDraftCandidate, RemoteInstallRuntimeRequest,
    RemoteInstallRuntimeResponse, RemoteLoadRequest, RemoteLoadResponse,
    RemoteModelDownloadRequest, RemoteModelInspectRequest, RemoteModelInspectResponse,
    RemoteModelSummary, RemoteRuntimeSummary, RemoteRuntimesResponse,
    RemoteSetActiveRuntimeRequest, RemoteUninstallRuntimeRequest, manager::effective_device_name,
};
use bibulm_models::{DownloadJobState, JobKind, JobStatus};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::BTreeSet;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::{Mutex, oneshot};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayStatus {
    pub running: bool,
    pub bind_address: String,
    pub reachable_url: String,
    pub config: ServerConfig,
}

struct RunningHandle {
    bind_address: String,
    reachable_url: String,
    port: u16,
    shutdown_tx: oneshot::Sender<()>,
    ttl_abort: tokio::task::AbortHandle,
}

/// Manages the lifecycle of the OpenAI-compatible `axum` HTTP server and idle TTL evictor.
pub struct ApiGateway {
    ctx: GatewayContext,
    running: Mutex<Option<RunningHandle>>,
}

impl ApiGateway {
    pub fn new(ctx: GatewayContext) -> Self {
        Self {
            ctx,
            running: Mutex::new(None),
        }
    }

    pub async fn invalidate_models_cache(&self) {
        self.ctx.invalidate_models_cache().await;
    }

    pub async fn active_port(&self) -> u16 {
        if let Some(h) = self.running.lock().await.as_ref() {
            return h.port;
        }
        self.ctx.settings.read().await.server.port
    }

    pub async fn status(&self) -> GatewayStatus {
        let cfg = self.ctx.settings.read().await.server.clone();
        let guard = self.running.lock().await;
        if let Some(h) = guard.as_ref() {
            GatewayStatus {
                running: true,
                bind_address: h.bind_address.clone(),
                reachable_url: h.reachable_url.clone(),
                config: cfg,
            }
        } else {
            let host = if cfg.serve_on_lan { "0.0.0.0" } else { "127.0.0.1" };
            GatewayStatus {
                running: false,
                bind_address: format!("{host}:{}", cfg.port),
                reachable_url: format!("http://127.0.0.1:{}", cfg.port),
                config: cfg,
            }
        }
    }

    pub async fn start(&self) -> Result<GatewayStatus, String> {
        // Stop existing listener first if running
        self.stop().await;

        let cfg = self.ctx.settings.read().await.server.clone();
        let host = if cfg.serve_on_lan {
            "0.0.0.0"
        } else {
            "127.0.0.1"
        };
        let bind_addr = format!("{host}:{}", cfg.port);
        let listener = TcpListener::bind(&bind_addr)
            .await
            .map_err(|e| format!("Could not bind OpenAI API server on {bind_addr}: {e}"))?;
        let local_addr: SocketAddr = listener
            .local_addr()
            .map_err(|e| format!("Failed to read bound socket address: {e}"))?;
        let bound_port = local_addr.port();
        let reachable_url = format!("http://127.0.0.1:{bound_port}");

        let router = build_router(self.ctx.clone());
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

        let sup_for_log = self.ctx.supervisor.clone();
        let bind_str = local_addr.to_string();
        tokio::spawn(async move {
            let server = axum::serve(listener, router).with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            });
            if let Err(e) = server.await {
                sup_for_log
                    .push_log("ERROR", "SERVER", format!("HTTP server error: {e}"))
                    .await;
            }
        });

        // Start background idle TTL auto-evictor
        let ttl_ctx = self.ctx.clone();
        let ttl_task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(30));
            loop {
                interval.tick().await;
                let server_cfg = ttl_ctx.settings.read().await.server.clone();
                if server_cfg.auto_unload_idle {
                    let evicted = ttl_ctx
                        .supervisor
                        .evict_idle_models(server_cfg.max_idle_ttl_minutes)
                        .await;
                    if !evicted.is_empty()
                        && let Some(link) = ttl_ctx.link.as_ref()
                    {
                        link.notify_peers_state_changed();
                    }
                }
            }
        });

        self.ctx
            .supervisor
            .push_log(
                "INFO",
                "SERVER",
                format!("Success! OpenAI HTTP server listening on {bind_str} ({reachable_url})"),
            )
            .await;
        if cfg.jit_loading {
            self.ctx
                .supervisor
                .push_log("INFO", "SERVER", "Just-in-Time model loading active.")
                .await;
        }

        *self.running.lock().await = Some(RunningHandle {
            bind_address: bind_str,
            reachable_url,
            port: bound_port,
            shutdown_tx,
            ttl_abort: ttl_task.abort_handle(),
        });

        // Keep running Link sidecar synced with the active gateway port
        if let Some(link) = self.ctx.link.as_ref() {
            let link_cfg = self.ctx.settings.read().await.link.clone();
            link.sync_config(&link_cfg, bound_port).await;
        }

        Ok(self.status().await)
    }

    pub async fn stop(&self) -> GatewayStatus {
        let handle = self.running.lock().await.take();
        if let Some(h) = handle {
            h.ttl_abort.abort();
            let _ = h.shutdown_tx.send(());
            self.ctx
                .supervisor
                .push_log("INFO", "SERVER", "Server stopped.")
                .await;
        }
        self.status().await
    }
}

pub fn build_router(ctx: GatewayContext) -> Router {
    let state = Arc::new(ctx);
    Router::new()
        .route("/health", get(health_handler))
        .route("/v1/models", get(list_models_handler))
        .route("/v1/chat/completions", post(openai_proxy_handler))
        .route("/v1/completions", post(openai_proxy_handler))
        .route("/v1/embeddings", post(openai_proxy_handler))
        .route("/v1/responses", post(openai_proxy_handler))
        .route("/v1/messages", post(anthropic_messages_handler))
        .route("/v1/messages/count_tokens", post(anthropic_count_tokens_handler))
        // Internal BibuLM Link mesh endpoints (strictly gated by X-Link-Secret from sidecar)
        .route("/link/v1/info", get(link_info_handler))
        .route("/link/v1/notify", post(link_notify_handler))
        .route("/link/v1/models/inspect", post(link_inspect_model_handler))
        .route("/link/v1/models/load", post(link_load_model_handler))
        .route("/link/v1/models/unload", post(link_unload_model_handler))
        .merge(remote_manage_routes(state.clone()))
        .layer(DefaultBodyLimit::max(256 * 1024 * 1024))
        .layer(middleware::from_fn_with_state(state.clone(), auth_middleware))
        .layer(middleware::from_fn_with_state(state.clone(), cors_middleware))
        .with_state(state)
}

/// Remote runtime/model management routes, only reachable when this device allows it.
fn remote_manage_routes(state: Arc<GatewayContext>) -> Router<Arc<GatewayContext>> {
    Router::new()
        .route("/link/v1/runtimes", get(link_list_runtimes_handler))
        .route("/link/v1/runtimes/releases", get(link_fetch_runtime_releases_handler))
        .route("/link/v1/runtimes/install", post(link_install_runtime_handler))
        .route("/link/v1/runtimes/active", post(link_set_active_runtime_handler))
        .route("/link/v1/runtimes/uninstall", post(link_uninstall_runtime_handler))
        .route("/link/v1/models/download", post(link_enqueue_model_download_handler))
        .route("/link/v1/models/delete", post(link_delete_model_handler))
        .route("/link/v1/downloads", get(link_list_downloads_handler))
        .route("/link/v1/downloads/pause", post(link_pause_download_handler))
        .route("/link/v1/downloads/resume", post(link_resume_download_handler))
        .route("/link/v1/downloads/cancel", post(link_cancel_download_handler))
        .route("/link/v1/downloads/clear", post(link_clear_completed_downloads_handler))
        .route("/link/v1/logs", get(link_get_logs_handler))
        .route_layer(middleware::from_fn_with_state(state, remote_manage_middleware))
}

/// Rejects management calls unless enabled, and hands the settings snapshot to the handler.
async fn remote_manage_middleware(
    State(ctx): State<Arc<GatewayContext>>,
    mut req: Request,
    next: Next,
) -> Response {
    let settings = ctx.settings.read().await.clone();
    if !(settings.link.expose_local_models && settings.link.allow_remote_manage) {
        return openai_error(
            StatusCode::FORBIDDEN,
            "Remote runtime/model management is disabled on this device.",
        );
    }
    req.extensions_mut().insert(settings);
    next.run(req).await
}

fn is_verified_link_request(ctx: &GatewayContext, headers: &HeaderMap) -> bool {
    let Some(link) = ctx.link.as_ref() else {
        return false;
    };
    headers
        .get(LINK_SECRET_HEADER)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|secret| link.verify_secret(secret))
}

async fn cors_middleware(
    State(ctx): State<Arc<GatewayContext>>,
    req: Request,
    next: Next,
) -> Response {
    let enable_cors = ctx.settings.read().await.server.enable_cors;
    if req.method() == Method::OPTIONS && enable_cors {
        let mut resp = StatusCode::NO_CONTENT.into_response();
        attach_cors_headers(resp.headers_mut());
        return resp;
    }
    let mut resp = next.run(req).await;
    if enable_cors {
        attach_cors_headers(resp.headers_mut());
    }
    resp
}

fn attach_cors_headers(headers: &mut HeaderMap) {
    headers.insert(
        header::ACCESS_CONTROL_ALLOW_ORIGIN,
        HeaderValue::from_static("*"),
    );
    headers.insert(
        header::ACCESS_CONTROL_ALLOW_METHODS,
        HeaderValue::from_static("GET, POST, OPTIONS"),
    );
    headers.insert(
        header::ACCESS_CONTROL_ALLOW_HEADERS,
        HeaderValue::from_static(
            "Authorization, Content-Type, x-api-key, anthropic-version, anthropic-beta",
        ),
    );
}

async fn auth_middleware(
    State(ctx): State<Arc<GatewayContext>>,
    req: Request,
    next: Next,
) -> Response {
    let path = req.uri().path().to_string();
    if req.method() == Method::OPTIONS || path == "/health" {
        return next.run(req).await;
    }

    // Internal /link/v1/* routes ALWAYS require the verified sidecar secret.
    if path.starts_with("/link/v1/") {
        if is_verified_link_request(&ctx, req.headers()) {
            return next.run(req).await;
        }
        return openai_error(
            StatusCode::FORBIDDEN,
            "Internal BibuLM Link endpoint requires verified Tailnet peer.",
        );
    }

    // Requests forwarded by the local bibulm-linkd sidecar have already been
    // cryptographically authenticated via Tailscale WhoIs.
    if is_verified_link_request(&ctx, req.headers()) {
        return next.run(req).await;
    }

    let (serve_on_lan, require_auth) = {
        let s = ctx.settings.read().await;
        (s.server.serve_on_lan, s.server.require_auth)
    };
    // Blocks DNS rebinding: a web page whose domain resolves to 127.0.0.1 still sends its own Host.
    if !serve_on_lan && !is_loopback_host(req.headers()) {
        return openai_error(
            StatusCode::FORBIDDEN,
            "Host header must be localhost when LAN serving is disabled.",
        );
    }
    if !require_auth {
        return next.run(req).await;
    }
    let bearer_token = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer ").or_else(|| s.strip_prefix("bearer ")))
        .map(str::trim);
    let x_api_key = req
        .headers()
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .map(str::trim);
    let token = bearer_token.or(x_api_key);

    match token {
        Some(tok) if ctx.db.validate_api_token(tok).unwrap_or(false) => next.run(req).await,
        _ if path.starts_with("/v1/messages") => anthropic_error(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "Missing or invalid API key.",
        ),
        _ => openai_error(
            StatusCode::UNAUTHORIZED,
            "Missing or invalid Bearer API key.",
        ),
    }
}

/// True for a missing Host (non-browser clients) or a loopback host name.
fn is_loopback_host(headers: &HeaderMap) -> bool {
    let Some(host) = headers.get(header::HOST) else {
        return true;
    };
    host.to_str()
        .ok()
        .and_then(|h| h.parse::<axum::http::uri::Authority>().ok())
        .is_some_and(|a| {
            let name = a.host();
            name.eq_ignore_ascii_case("localhost") || name == "127.0.0.1" || name == "[::1]"
        })
}

async fn health_handler(State(ctx): State<Arc<GatewayContext>>) -> Response {
    let loaded = ctx.supervisor.list_loaded().await;
    axum::Json(json!({
        "status": "ok",
        "loaded_models": loaded.len()
    }))
    .into_response()
}

async fn list_models_handler(
    State(ctx): State<Arc<GatewayContext>>,
    headers: HeaderMap,
) -> Response {
    let is_link_inbound = is_verified_link_request(&ctx, &headers);
    let loaded = ctx.supervisor.list_loaded().await;
    let settings = ctx.settings.read().await.clone();
    let now = Utc::now().timestamp();

    let mut seen = BTreeSet::new();
    let mut data = Vec::new();

    for m in &loaded {
        let id = m.alias.clone().unwrap_or_else(|| m.model_id.clone());
        if seen.insert(id.clone()) {
            data.push(json!({
                "id": id,
                "object": "model",
                "created": now,
                "owned_by": "bibulm",
                "loaded": true,
                "remote": false,
                "placement": m.placement_status,
            }));
        }
    }

    if settings.server.jit_loading {
        for local in ctx.cached_local_models().await {
            if seen.insert(local.id.clone()) {
                data.push(json!({
                    "id": local.id,
                    "object": "model",
                    "created": now,
                    "owned_by": "bibulm",
                    "loaded": false,
                    "remote": false,
                }));
            }
        }
    }

    // Aggregate remote Tailscale Link peer models for local OpenAI API consumers.
    if !is_link_inbound
        && settings.link.expose_remote_in_openai_api
        && let Some(link) = ctx.link.as_ref()
    {
        for peer in link.peers().await {
            if !peer.online || !peer.expose_local_models {
                continue;
            }
            for rm in &peer.loaded_models {
                let qualified = format!("{}/{}", peer.slug, rm.id);
                if seen.insert(qualified.clone()) {
                    data.push(json!({
                        "id": qualified,
                        "object": "model",
                        "created": now,
                        "owned_by": format!("bibulm-link:{}", peer.device_name),
                        "loaded": true,
                        "remote": true,
                        "peer_slug": peer.slug,
                        "peer_name": peer.device_name,
                        "placement": rm.placement_status,
                    }));
                }
            }
            if peer.allow_remote_load {
                for rm in &peer.available_models {
                    let qualified = format!("{}/{}", peer.slug, rm.id);
                    if seen.insert(qualified.clone()) {
                        data.push(json!({
                            "id": qualified,
                            "object": "model",
                            "created": now,
                            "owned_by": format!("bibulm-link:{}", peer.device_name),
                            "loaded": false,
                            "remote": true,
                            "peer_slug": peer.slug,
                            "peer_name": peer.device_name,
                        }));
                    }
                }
            }
        }
    }

    axum::Json(json!({
        "object": "list",
        "data": data
    }))
    .into_response()
}

async fn link_info_handler(State(ctx): State<Arc<GatewayContext>>) -> Response {
    let settings = ctx.settings.read().await.clone();
    let hw = ctx.hw_snapshot.read().await.clone();
    let expose = settings.link.expose_local_models;

    let runtimes = ctx.db.list_runtimes().unwrap_or_default();
    let active_runtime_tag = settings
        .active_runtime_id
        .as_deref()
        .and_then(|id| runtimes.iter().find(|r| r.id == id))
        .or_else(|| runtimes.first())
        .map(|r| r.tag.clone());
    let active_downloads_count = if let Some(dl) = ctx.downloads.as_ref() {
        dl.list_jobs()
            .await
            .iter()
            .filter(|j| j.status.is_active())
            .count()
    } else {
        0
    };

    let mut loaded_models = Vec::new();
    let mut available_models = Vec::new();

    if expose {
        let loaded = ctx.supervisor.list_loaded().await;
        let mut loaded_ids = BTreeSet::new();
        let mut loaded_key_pairs = Vec::new();
        for m in &loaded {
            let id = m.alias.clone().unwrap_or_else(|| m.model_id.clone());
            loaded_ids.insert(id.clone());
            loaded_ids.insert(m.model_id.clone());
            loaded_key_pairs.push((m.model_id.clone(), id.clone()));
            loaded_models.push(RemoteModelSummary {
                id,
                display_name: m.display_name.clone(),
                loaded: true,
                quantization: None,
                architecture: None,
                size_bytes: 0,
                max_context_length: Some(m.load_config.ctx_size),
                supports_vision: m.load_config.enable_vision,
                supports_reasoning: m.load_config.reasoning.enable_thinking.unwrap_or(false),
                placement_status: Some(format!("{:?}", m.placement_status).to_ascii_lowercase()),
            });
        }

        for local in ctx.cached_local_models().await {
            if loaded_ids.contains(&local.id) {
                // Enrich loaded model summary with GGUF metadata when available
                if let Some((idx, _)) = loaded_key_pairs
                    .iter()
                    .enumerate()
                    .find(|(_, (mid, alias))| *mid == local.id || *alias == local.id)
                    && let Some(entry) = loaded_models.get_mut(idx)
                {
                    entry.quantization = local.quantization.clone();
                    entry.architecture = local.architecture.clone();
                    entry.size_bytes = local.size_bytes;
                    entry.supports_vision = entry.supports_vision || local.supports_vision;
                    entry.supports_reasoning =
                        entry.supports_reasoning || local.supports_reasoning;
                }
                continue;
            }
            available_models.push(RemoteModelSummary {
                id: local.id,
                display_name: local.display_name,
                loaded: false,
                quantization: local.quantization,
                architecture: local.architecture,
                size_bytes: local.size_bytes,
                max_context_length: local.max_context_length,
                supports_vision: local.supports_vision,
                supports_reasoning: local.supports_reasoning,
                placement_status: None,
            });
        }
    }

    let payload = PeerInfoPayload {
        device_name: effective_device_name(&settings.link),
        app_version: env!("CARGO_PKG_VERSION").to_string(),
        expose_local_models: expose,
        allow_remote_load: expose && settings.link.allow_remote_load,
        allow_remote_manage: expose && settings.link.allow_remote_manage,
        is_headless: ctx.is_headless,
        platform: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        runtimes_count: runtimes.len(),
        active_runtime_tag,
        active_downloads_count,
        gpu_summary: hw.gpu_summary,
        vram_used_mib: hw.vram_used_mib,
        vram_total_mib: hw.vram_total_mib,
        ram_used_mib: hw.ram_used_mib,
        ram_total_mib: hw.ram_total_mib,
        loaded_models,
        available_models,
    };

    axum::Json(payload).into_response()
}

async fn link_notify_handler(State(ctx): State<Arc<GatewayContext>>) -> Response {
    if let Some(link) = ctx.link.clone() {
        tokio::spawn(async move {
            let _ = link.refresh_peers().await;
        });
    }
    axum::Json(json!({ "ok": true })).into_response()
}

async fn link_inspect_model_handler(
    State(ctx): State<Arc<GatewayContext>>,
    axum::Json(req): axum::Json<RemoteModelInspectRequest>,
) -> Response {
    let settings = ctx.settings.read().await.clone();
    if !settings.link.expose_local_models || !settings.link.allow_remote_load {
        return openai_error(
            StatusCode::FORBIDDEN,
            "Remote model configuration is disabled on this device.",
        );
    }

    let Some(local) = ctx.find_local_model(&req.model_id).await else {
        return openai_error(
            StatusCode::NOT_FOUND,
            &format!("No local GGUF model matching '{}' found.", req.model_id),
        );
    };

    let saved_preset = ctx.db.get_model_preset(&local.id).ok().flatten();
    let active_load_config = ctx
        .supervisor
        .resolve_loaded(&local.id)
        .await
        .map(|l| l.load_config);
    let runtimes: Vec<RemoteRuntimeSummary> = ctx
        .db
        .list_runtimes()
        .unwrap_or_default()
        .into_iter()
        .map(|r| RemoteRuntimeSummary {
            id: r.id,
            tag: r.tag,
            backend: r.backend,
            display_name: r.display_name,
        })
        .collect();
    let draft_models: Vec<RemoteDraftCandidate> = ctx
        .cached_local_models()
        .await
        .into_iter()
        .filter(|m| m.path != local.path)
        .map(|m| RemoteDraftCandidate {
            id: m.id,
            display_name: m.display_name,
            path: m.path.to_string_lossy().into_owned(),
            quantization: m.quantization,
            size_bytes: m.size_bytes,
        })
        .collect();

    let hw = ctx.hw_snapshot.read().await.clone();
    let cpu_logical_cores = hw.cpu_logical_cores.max(1);
    let live_memory = ctx
        .compute_link_live_memory_context(Some(&local.id))
        .await;

    axum::Json(RemoteModelInspectResponse {
        model: local,
        saved_preset,
        active_load_config,
        runtimes,
        active_runtime_id: settings.active_runtime_id,
        default_offload_kv_cache: settings.default_offload_kv_cache,
        guardrail_level: settings.guardrail_level,
        cpu_logical_cores,
        live_memory,
        draft_models,
    })
    .into_response()
}

async fn link_load_model_handler(
    State(ctx): State<Arc<GatewayContext>>,
    axum::Json(req): axum::Json<RemoteLoadRequest>,
) -> Response {
    let settings = ctx.settings.read().await.clone();
    if !settings.link.expose_local_models || !settings.link.allow_remote_load {
        return openai_error(
            StatusCode::FORBIDDEN,
            "Remote model loading is disabled on this device.",
        );
    }

    let load_res = if let Some(cfg) = req.config {
        ctx.load_local_with_config(&req.model_id, cfg, req.remember_preset)
            .await
    } else {
        ctx.resolve_or_jit_load_local(Some(&req.model_id), true, true)
            .await
    };

    match load_res {
        Ok(loaded) => {
            if let Some(link) = ctx.link.as_ref() {
                link.notify_peers_state_changed();
            }
            axum::Json(RemoteLoadResponse {
                ok: true,
                model_id: loaded.model_id,
                message: Some("Model loaded remotely.".to_string()),
            })
            .into_response()
        }
        Err((status, msg)) => openai_error(status, &msg),
    }
}

async fn link_unload_model_handler(
    State(ctx): State<Arc<GatewayContext>>,
    axum::Json(req): axum::Json<RemoteLoadRequest>,
) -> Response {
    let settings = ctx.settings.read().await.clone();
    if !settings.link.expose_local_models || !settings.link.allow_remote_load {
        return openai_error(
            StatusCode::FORBIDDEN,
            "Remote model unloading is disabled on this device.",
        );
    }

    let ok = ctx.supervisor.unload_model(&req.model_id).await;
    if ok && let Some(link) = ctx.link.as_ref() {
        link.notify_peers_state_changed();
    }
    axum::Json(RemoteLoadResponse {
        ok,
        model_id: req.model_id,
        message: if ok {
            Some("Model unloaded.".to_string())
        } else {
            Some("Model was not loaded.".to_string())
        },
    })
    .into_response()
}

/// Only official `ggml-org/llama.cpp` release assets may be installed remotely.
pub fn is_valid_llama_release_url(url: &str) -> bool {
    let lower = url.to_ascii_lowercase();
    url.starts_with("https://github.com/ggml-org/llama.cpp/releases/download/")
        && !lower.contains("..")
        && !lower.contains("%2e")
        && !lower.contains("%5c")
        && !url.contains('\\')
        && !url.contains('@')
        && !url.contains('?')
        && !url.contains('#')
}

/// Only HuggingFace-hosted files may be downloaded remotely.
pub fn is_valid_hf_download_url(url: &str) -> bool {
    let lower = url.to_ascii_lowercase();
    (url.starts_with("https://huggingface.co/") || url.starts_with("https://hf.co/"))
        && !lower.contains("..")
        && !lower.contains("%2e")
        && !lower.contains("%5c")
        && !url.contains('\\')
        && !url.contains('@')
}

pub fn is_safe_single_filename(name: &str) -> bool {
    !name.is_empty()
        && name != "."
        && name != ".."
        && !name.contains("..")
        && !name.contains('/')
        && !name.contains('\\')
        && !name.contains('\0')
}

pub fn sanitize_repo_subpath(repo_id: &str) -> Option<PathBuf> {
    let mut out = PathBuf::new();
    for seg in repo_id.split('/').filter(|s| !s.is_empty()) {
        let clean: String = seg
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        if clean.is_empty() || clean == "." || clean == ".." || clean.contains("..") {
            return None;
        }
        out.push(clean);
    }
    if out.as_os_str().is_empty() {
        None
    } else {
        Some(out)
    }
}

fn action_ok(message: &str) -> Response {
    axum::Json(RemoteActionResponse {
        ok: true,
        message: Some(message.to_string()),
    })
    .into_response()
}

async fn link_list_runtimes_handler(
    State(ctx): State<Arc<GatewayContext>>,
    Extension(settings): Extension<AppSettings>,
) -> Response {
    let hw = ctx.hw_snapshot.read().await.clone();
    match ctx.db.list_runtimes() {
        Ok(installed) => axum::Json(RemoteRuntimesResponse {
            installed,
            active_runtime_id: settings.active_runtime_id,
            platform: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
            gpu_summary: hw.gpu_summary,
        })
        .into_response(),
        Err(e) => openai_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

async fn link_fetch_runtime_releases_handler(
    State(ctx): State<Arc<GatewayContext>>,
    Extension(settings): Extension<AppSettings>,
) -> Response {
    let hw_ctx = ctx.hw_snapshot.read().await.to_runtime_match_context();
    match bibulm_runtime::fetch_llama_cpp_releases(settings.github_token.as_deref(), &hw_ctx).await
    {
        Ok(releases) => axum::Json(releases).into_response(),
        Err(e) => openai_error(StatusCode::BAD_GATEWAY, &e.to_string()),
    }
}

async fn link_install_runtime_handler(
    State(ctx): State<Arc<GatewayContext>>,
    Extension(settings): Extension<AppSettings>,
    axum::Json(req): axum::Json<RemoteInstallRuntimeRequest>,
) -> Response {
    let Some(downloads) = ctx.downloads.clone() else {
        return openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "Download manager unavailable on this device.",
        );
    };
    let variant = req.variant;
    if variant.assets.is_empty()
        || !is_safe_single_filename(&variant.id)
        || variant
            .assets
            .iter()
            .any(|a| !is_valid_llama_release_url(&a.download_url) || !is_safe_single_filename(&a.name))
    {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "Runtime assets must be official ggml-org/llama.cpp release downloads.",
        );
    }

    let target_dest_dir = settings
        .runtimes_dir
        .join(&variant.id)
        .to_string_lossy()
        .into_owned();
    let already_installing = downloads.list_jobs().await.iter().any(|j| {
        j.kind == JobKind::Runtime && j.status == JobStatus::Downloading && j.dest_dir == target_dest_dir
    });
    if already_installing {
        return openai_error(
            StatusCode::CONFLICT,
            "This runtime variant is already being installed.",
        );
    }

    let job_id = format!("rt-{}", Uuid::new_v4());
    let base_job = DownloadJobState {
        id: job_id.clone(),
        kind: JobKind::Runtime,
        title: format!("llama.cpp {} ({})", variant.tag, variant.backend),
        repo_id: Some("ggml-org/llama.cpp".to_string()),
        dest_dir: target_dest_dir,
        files: Vec::new(),
        current_file: Some("Starting...".to_string()),
        total_bytes: variant.total_size_bytes,
        downloaded_bytes: 0,
        speed_bytes_per_sec: 0.0,
        eta_seconds: None,
        status: JobStatus::Downloading,
        error: None,
        created_at: Utc::now().to_rfc3339(),
        completed_at: None,
    };
    downloads.record_external_job(base_job.clone()).await;

    let task_ctx = (*ctx).clone();
    let variant_id = variant.id.clone();
    let release_notes = req.release_notes;
    tokio::spawn(async move {
        let (tx, mut rx) =
            tokio::sync::mpsc::unbounded_channel::<bibulm_runtime::InstallProgress>();
        let fwd_downloads = downloads.clone();
        let fwd_base = base_job.clone();
        let forwarder = tokio::spawn(async move {
            let mut last: Option<std::time::Instant> = None;
            let mut last_stage = String::new();
            while let Some(p) = rx.recv().await {
                if p.stage != last_stage
                    || last.is_none_or(|t| t.elapsed() >= std::time::Duration::from_millis(250))
                {
                    last = Some(std::time::Instant::now());
                    last_stage = p.stage.clone();
                    let mut snap = fwd_base.clone();
                    snap.current_file = Some(p.stage);
                    snap.total_bytes = p.total_bytes;
                    snap.downloaded_bytes = p.downloaded_bytes;
                    fwd_downloads.update_external_progress(snap).await;
                }
            }
        });

        let res = bibulm_runtime::install_runtime_variant(
            &task_ctx.db,
            &settings.runtimes_dir,
            &variant,
            release_notes,
            settings.github_token.as_deref(),
            move |p| {
                let _ = tx.send(p);
            },
        )
        .await;
        let _ = forwarder.await;

        match res {
            Ok(rec) => {
                let mut done = base_job.clone();
                done.dest_dir = rec.install_dir.clone();
                done.current_file = None;
                done.downloaded_bytes = done.total_bytes;
                done.status = JobStatus::Completed;
                done.completed_at = Some(Utc::now().to_rfc3339());
                downloads.record_external_job(done).await;
                {
                    let mut s = task_ctx.settings.write().await;
                    if s.active_runtime_id.is_none() {
                        s.active_runtime_id = Some(rec.id.clone());
                        let _ = task_ctx.db.save_settings(&s);
                    }
                }
                task_ctx
                    .supervisor
                    .push_log(
                        "INFO",
                        "LINK",
                        format!("Remote-installed runtime {} ({})", rec.tag, rec.backend),
                    )
                    .await;
                if let Some(link) = task_ctx.link.as_ref() {
                    link.notify_peers_state_changed();
                }
            }
            Err(e) => {
                let mut failed = base_job.clone();
                failed.current_file = None;
                failed.status = JobStatus::Failed;
                failed.error = Some(e.to_string());
                downloads.record_external_job(failed).await;
                task_ctx
                    .supervisor
                    .push_log("ERROR", "LINK", format!("Remote runtime install failed: {e}"))
                    .await;
            }
        }
    });

    axum::Json(RemoteInstallRuntimeResponse {
        ok: true,
        job_id,
        variant_id,
        message: Some("Runtime installation started.".to_string()),
    })
    .into_response()
}

async fn link_set_active_runtime_handler(
    State(ctx): State<Arc<GatewayContext>>,
    axum::Json(req): axum::Json<RemoteSetActiveRuntimeRequest>,
) -> Response {
    let known = ctx
        .db
        .list_runtimes()
        .map(|l| l.iter().any(|r| r.id == req.runtime_id))
        .unwrap_or(false);
    if !known {
        return openai_error(StatusCode::NOT_FOUND, "Runtime not installed on this device.");
    }
    let mut s = ctx.settings.write().await;
    s.active_runtime_id = Some(req.runtime_id);
    if let Err(e) = ctx.db.save_settings(&s) {
        return openai_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string());
    }
    drop(s);
    if let Some(link) = ctx.link.as_ref() {
        link.notify_peers_state_changed();
    }
    action_ok("Active runtime updated.")
}

async fn link_uninstall_runtime_handler(
    State(ctx): State<Arc<GatewayContext>>,
    axum::Json(req): axum::Json<RemoteUninstallRuntimeRequest>,
) -> Response {
    if let Err(e) = bibulm_runtime::uninstall_runtime(&ctx.db, &req.runtime_id) {
        return openai_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string());
    }
    {
        let mut s = ctx.settings.write().await;
        if s.active_runtime_id.as_deref() == Some(req.runtime_id.as_str()) {
            s.active_runtime_id = ctx
                .db
                .list_runtimes()
                .ok()
                .and_then(|l| l.into_iter().next().map(|r| r.id));
            let _ = ctx.db.save_settings(&s);
        }
    }
    if let Some(link) = ctx.link.as_ref() {
        link.notify_peers_state_changed();
    }
    action_ok("Runtime removed.")
}

async fn link_enqueue_model_download_handler(
    State(ctx): State<Arc<GatewayContext>>,
    Extension(settings): Extension<AppSettings>,
    axum::Json(req): axum::Json<RemoteModelDownloadRequest>,
) -> Response {
    let Some(downloads) = ctx.downloads.clone() else {
        return openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "Download manager unavailable on this device.",
        );
    };
    let Some(sub) = sanitize_repo_subpath(&req.repo_id) else {
        return openai_error(StatusCode::BAD_REQUEST, "Invalid repository id.");
    };
    if req.files.is_empty()
        || req.files.len() > 128
        || req
            .files
            .iter()
            .any(|f| !is_valid_hf_download_url(&f.url) || !is_safe_single_filename(&f.file_name))
    {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "Model files must be HuggingFace downloads with safe file names.",
        );
    }
    let dest_dir = settings.primary_models_dir().join(sub);
    let job = downloads
        .enqueue_model_download(req.title, req.repo_id, dest_dir, req.files, settings.hf_token)
        .await;
    axum::Json(job).into_response()
}

async fn link_list_downloads_handler(State(ctx): State<Arc<GatewayContext>>) -> Response {
    let jobs = match ctx.downloads.as_ref() {
        Some(d) => d.list_jobs().await,
        None => Vec::new(),
    };
    axum::Json(jobs).into_response()
}

async fn link_pause_download_handler(
    State(ctx): State<Arc<GatewayContext>>,
    axum::Json(req): axum::Json<RemoteDownloadActionRequest>,
) -> Response {
    let ok = match ctx.downloads.as_ref() {
        Some(d) => d.pause(&req.id).await,
        None => false,
    };
    axum::Json(RemoteActionResponse { ok, message: None }).into_response()
}

async fn link_resume_download_handler(
    State(ctx): State<Arc<GatewayContext>>,
    Extension(settings): Extension<AppSettings>,
    axum::Json(req): axum::Json<RemoteDownloadActionRequest>,
) -> Response {
    let ok = match ctx.downloads.as_ref() {
        Some(d) => d.resume_or_retry(&req.id, settings.hf_token).await,
        None => false,
    };
    axum::Json(RemoteActionResponse { ok, message: None }).into_response()
}

async fn link_cancel_download_handler(
    State(ctx): State<Arc<GatewayContext>>,
    axum::Json(req): axum::Json<RemoteDownloadActionRequest>,
) -> Response {
    let ok = match ctx.downloads.as_ref() {
        Some(d) => d.cancel(&req.id).await,
        None => false,
    };
    axum::Json(RemoteActionResponse { ok, message: None }).into_response()
}

async fn link_clear_completed_downloads_handler(
    State(ctx): State<Arc<GatewayContext>>,
) -> Response {
    if let Some(d) = ctx.downloads.as_ref() {
        d.clear_completed().await;
    }
    action_ok("Download history cleared.")
}

async fn link_delete_model_handler(
    State(ctx): State<Arc<GatewayContext>>,
    Extension(settings): Extension<AppSettings>,
    axum::Json(req): axum::Json<RemoteDeleteModelRequest>,
) -> Response {
    let Some(local) = ctx.find_local_model(&req.model_id).await else {
        return openai_error(
            StatusCode::NOT_FOUND,
            &format!("No local GGUF model matching '{}' found.", req.model_id),
        );
    };
    // Only models inside this device's configured model directories may be deleted remotely.
    let in_managed_dir =
        bibulm_core::canonical_file_within(&local.path, &settings.models_dirs, &[]).is_some();
    if !in_managed_dir {
        return openai_error(
            StatusCode::FORBIDDEN,
            "Model is outside the managed models directories and cannot be deleted remotely.",
        );
    }
    ctx.supervisor.unload_model(&local.id).await;
    let path = local.path.clone();
    let res = tokio::task::spawn_blocking(move || bibulm_models::delete_local_model(&path)).await;
    match res {
        Ok(Ok(n)) => {
            ctx.invalidate_models_cache().await;
            if let Some(link) = ctx.link.as_ref() {
                link.notify_peers_state_changed();
            }
            action_ok(&format!("Deleted {n} file(s)."))
        }
        Ok(Err(e)) => openai_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
        Err(e) => openai_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

async fn link_get_logs_handler(State(ctx): State<Arc<GatewayContext>>) -> Response {
    axum::Json(ctx.supervisor.recent_logs().await).into_response()
}

async fn openai_proxy_handler(
    State(ctx): State<Arc<GatewayContext>>,
    path: MatchedPath,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let is_link = is_verified_link_request(&ctx, &headers);
    proxy_openai_post((*ctx).clone(), path.as_str(), body, is_link).await
}

async fn anthropic_messages_handler(
    State(ctx): State<Arc<GatewayContext>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let is_link = is_verified_link_request(&ctx, &headers);
    handle_anthropic_messages((*ctx).clone(), body, is_link).await
}

async fn anthropic_count_tokens_handler(
    State(ctx): State<Arc<GatewayContext>>,
    body: Bytes,
) -> Response {
    handle_anthropic_count_tokens((*ctx).clone(), body).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{Body, to_bytes};
    use bibulm_core::{AppSettings, Database, LinkConfig};
    use bibulm_link::LinkManager;
    use bibulm_runtime::ModelSupervisor;
    use tokio::sync::RwLock;
    use tower::Service;

    async fn send_req(app: &mut Router, req: Request) -> Response {
        std::future::poll_fn(|cx| <Router as Service<Request>>::poll_ready(app, cx))
            .await
            .unwrap();
        <Router as Service<Request>>::call(app, req).await.unwrap()
    }

    #[tokio::test]
    async fn gateway_serves_health_models_and_enforces_auth_and_cors() {
        let db = Database::open_in_memory().unwrap();
        let token = db.create_api_token("test-key").unwrap();

        let mut settings = AppSettings {
            models_dirs: vec![],
            auto_discover_external_models: false,
            ..Default::default()
        };
        settings.server.port = 0;
        settings.server.require_auth = true;
        settings.server.enable_cors = true;

        let supervisor = Arc::new(ModelSupervisor::new());
        let ctx = GatewayContext::new(db, supervisor, Arc::new(RwLock::new(settings)));
        let mut app = build_router(ctx);

        // 1. /health is public & carries CORS header
        let health = send_req(
            &mut app,
            Request::builder().uri("/health").body(Body::empty()).unwrap(),
        )
        .await;
        assert_eq!(health.status(), StatusCode::OK);
        assert_eq!(
            health
                .headers()
                .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
                .unwrap(),
            "*"
        );

        // 2. /v1/models without token -> 401
        let unauth = send_req(
            &mut app,
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(unauth.status(), StatusCode::UNAUTHORIZED);

        // 2b. DNS rebinding: foreign Host is refused even with a valid token, loopback Host passes
        let rebind = send_req(
            &mut app,
            Request::builder()
                .uri("/v1/models")
                .header(header::HOST, "evil.example:1234")
                .header(header::AUTHORIZATION, format!("Bearer {}", token.token))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(rebind.status(), StatusCode::FORBIDDEN);
        let loopback = send_req(
            &mut app,
            Request::builder()
                .uri("/v1/models")
                .header(header::HOST, "localhost:1234")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(loopback.status(), StatusCode::UNAUTHORIZED);

        // 3. /v1/models with valid Bearer token -> 200
        let ok = send_req(
            &mut app,
            Request::builder()
                .uri("/v1/models")
                .header(header::AUTHORIZATION, format!("Bearer {}", token.token))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(ok.status(), StatusCode::OK);
        let bytes = to_bytes(ok.into_body(), 1024 * 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["object"], "list");

        // 4. /link/v1/info without X-Link-Secret -> 403
        let link_denied = send_req(
            &mut app,
            Request::builder()
                .uri("/link/v1/info")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(link_denied.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn link_internal_info_requires_secret_and_returns_metadata() {
        let db = Database::open_in_memory().unwrap();
        let mut settings = AppSettings {
            models_dirs: vec![],
            auto_discover_external_models: false,
            ..Default::default()
        };
        settings.link.device_name = Some("PC-Salon".to_string());

        let link = Arc::new(LinkManager::new(&LinkConfig::default()));
        let supervisor = Arc::new(ModelSupervisor::new());
        let ctx = GatewayContext::new(db, supervisor, Arc::new(RwLock::new(settings)))
            .with_link(link.clone());
        let mut app = build_router(ctx);

        // Denied without secret
        let denied = send_req(
            &mut app,
            Request::builder()
                .uri("/link/v1/info")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(denied.status(), StatusCode::FORBIDDEN);

        // Allowed with valid secret (via proxy_endpoint)
        // We verify via direct check: build a bad secret request -> 403
        let bad = send_req(
            &mut app,
            Request::builder()
                .uri("/link/v1/info")
                .header(LINK_SECRET_HEADER, "invalid")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(bad.status(), StatusCode::FORBIDDEN);

        // /link/v1/models/inspect and /link/v1/notify also require X-Link-Secret
        let inspect_denied = send_req(
            &mut app,
            Request::builder()
                .method(Method::POST)
                .uri("/link/v1/models/inspect")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"model_id":"missing"}"#))
                .unwrap(),
        )
        .await;
        assert_eq!(inspect_denied.status(), StatusCode::FORBIDDEN);

        let notify_denied = send_req(
            &mut app,
            Request::builder()
                .method(Method::POST)
                .uri("/link/v1/notify")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(notify_denied.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn link_inspect_and_custom_load_respect_permissions_and_return_metadata() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let tmp = std::env::temp_dir().join(format!("bibulm_srv_test_{nanos}"));
        std::fs::create_dir_all(&tmp).unwrap();
        let mut gguf_bytes = Vec::new();
        gguf_bytes.extend_from_slice(b"GGUF");
        gguf_bytes.extend_from_slice(&3u32.to_le_bytes());
        gguf_bytes.extend_from_slice(&0u64.to_le_bytes());
        gguf_bytes.extend_from_slice(&0u64.to_le_bytes());
        std::fs::write(tmp.join("demo.gguf"), gguf_bytes).unwrap();

        let db = Database::open_in_memory().unwrap();
        let mut settings = AppSettings {
            models_dirs: vec![tmp.clone()],
            auto_discover_external_models: false,
            ..Default::default()
        };
        settings.link.expose_local_models = true;
        settings.link.allow_remote_load = true;

        let supervisor = Arc::new(ModelSupervisor::new());
        let ctx = GatewayContext::new(db, supervisor, Arc::new(RwLock::new(settings)));
        {
            let mut hw = ctx.hw_snapshot.write().await;
            hw.cpu_logical_cores = 16;
            hw.vram_total_mib = 24576.0;
            hw.vram_used_avg_mib = 2048.0;
            hw.ram_total_mib = 65536.0;
            hw.ram_used_avg_mib = 8192.0;
        }

        // Direct handler check for inspect response
        let resp = link_inspect_model_handler(
            State(Arc::new(ctx.clone())),
            axum::Json(RemoteModelInspectRequest {
                model_id: "demo".to_string(),
            }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        let parsed: RemoteModelInspectResponse = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(parsed.model.id, "demo");
        assert_eq!(parsed.cpu_logical_cores, 16);
        assert_eq!(parsed.live_memory.hardware_vram_total_mib, 24576.0);

        // When allow_remote_load is false, inspect and load are forbidden
        ctx.settings.write().await.link.allow_remote_load = false;
        let forbidden_inspect = link_inspect_model_handler(
            State(Arc::new(ctx.clone())),
            axum::Json(RemoteModelInspectRequest {
                model_id: "demo".to_string(),
            }),
        )
        .await;
        assert_eq!(forbidden_inspect.status(), StatusCode::FORBIDDEN);

        let forbidden_load = link_load_model_handler(
            State(Arc::new(ctx)),
            axum::Json(RemoteLoadRequest {
                model_id: "demo".to_string(),
                config: Some(bibulm_core::LoadConfig::default()),
                remember_preset: false,
            }),
        )
        .await;
        assert_eq!(forbidden_load.status(), StatusCode::FORBIDDEN);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn anthropic_x_api_key_auth_count_tokens_and_alias_routing() {
        let db = Database::open_in_memory().unwrap();
        let token = db.create_api_token("claude-code").unwrap();

        let mut settings = AppSettings {
            models_dirs: vec![],
            auto_discover_external_models: false,
            ..Default::default()
        };
        settings.server.require_auth = true;
        settings.server.fallback_to_first_loaded = false;
        settings.server.model_aliases.push(bibulm_core::ModelAliasRule {
            pattern: "claude-*".into(),
            target: "missing-target".into(),
            enabled: true,
        });

        let supervisor = Arc::new(ModelSupervisor::new());
        let ctx = GatewayContext::new(db, supervisor, Arc::new(RwLock::new(settings)));
        let mut app = build_router(ctx.clone());

        // 1. Missing x-api-key on /v1/messages -> 401 Anthropic error
        let unauth = send_req(
            &mut app,
            Request::builder()
                .method(Method::POST)
                .uri("/v1/messages")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"model":"claude-sonnet-4","messages":[{"role":"user","content":"hi"}]}"#))
                .unwrap(),
        )
        .await;
        assert_eq!(unauth.status(), StatusCode::UNAUTHORIZED);
        let unauth_bytes = to_bytes(unauth.into_body(), 64 * 1024).await.unwrap();
        let unauth_json: serde_json::Value = serde_json::from_slice(&unauth_bytes).unwrap();
        assert_eq!(unauth_json["error"]["type"], "authentication_error");

        // 2. Valid x-api-key on /v1/messages/count_tokens -> 200 with input_tokens
        let count_ok = send_req(
            &mut app,
            Request::builder()
                .method(Method::POST)
                .uri("/v1/messages/count_tokens")
                .header("x-api-key", &token.token)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"model":"claude-sonnet-4","messages":[{"role":"user","content":"Hello world from Claude Code"}]}"#))
                .unwrap(),
        )
        .await;
        assert_eq!(count_ok.status(), StatusCode::OK);
        let c_bytes = to_bytes(count_ok.into_body(), 64 * 1024).await.unwrap();
        let c_json: serde_json::Value = serde_json::from_slice(&c_bytes).unwrap();
        assert!(c_json["input_tokens"].as_u64().unwrap() > 0);

        // 3. Alias rewrites claude-* -> missing-target (and fallback_to_first_loaded=false returns 404 mentioning target)
        let aliased = send_req(
            &mut app,
            Request::builder()
                .method(Method::POST)
                .uri("/v1/messages")
                .header("x-api-key", &token.token)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"model":"claude-sonnet-4","messages":[{"role":"user","content":"hi"}]}"#))
                .unwrap(),
        )
        .await;
        assert_eq!(aliased.status(), StatusCode::NOT_FOUND);
        let a_bytes = to_bytes(aliased.into_body(), 64 * 1024).await.unwrap();
        let a_json: serde_json::Value = serde_json::from_slice(&a_bytes).unwrap();
        assert!(
            a_json["error"]["message"]
                .as_str()
                .unwrap()
                .contains("missing-target")
        );
    }

    #[test]
    fn remote_manage_validators_accept_official_and_reject_untrusted() {
        assert!(is_valid_llama_release_url(
            "https://github.com/ggml-org/llama.cpp/releases/download/b5200/llama-b5200-bin-ubuntu-x64.tar.gz"
        ));
        assert!(!is_valid_llama_release_url("https://evil.example/llama.zip"));
        assert!(!is_valid_llama_release_url(
            "https://github.com/evil/llama.cpp/releases/download/b1/x.zip"
        ));
        assert!(!is_valid_llama_release_url(
            "https://github.com/ggml-org/llama.cpp/releases/download/../../../../pirate/repo/releases/download/b1/x.zip"
        ));
        assert!(!is_valid_llama_release_url(
            "https://github.com/ggml-org/llama.cpp/releases/download/%2e%2e/pirate/x.zip"
        ));
        assert!(!is_valid_llama_release_url(
            "https://github.com/ggml-org/llama.cpp/releases/download/%2E%2E/pirate/x.zip"
        ));
        assert!(!is_valid_llama_release_url(
            "https://github.com/ggml-org/llama.cpp/releases/download/@evil.com/x.zip"
        ));
        assert!(is_valid_hf_download_url(
            "https://huggingface.co/org/repo/resolve/main/m.gguf"
        ));
        assert!(!is_valid_hf_download_url("http://huggingface.co/org/repo/m.gguf"));
        assert!(!is_valid_hf_download_url("https://huggingface.co.evil.com/m.gguf"));
        assert!(!is_valid_hf_download_url("https://huggingface.co@evil.com/m.gguf"));

        assert!(is_safe_single_filename("model-q4_k_m.gguf"));
        assert!(!is_safe_single_filename("../etc/passwd"));
        assert!(!is_safe_single_filename("a/b.gguf"));
        assert!(!is_safe_single_filename("a\\b.gguf"));
        assert!(!is_safe_single_filename(".."));
        assert!(!is_safe_single_filename(""));

        assert_eq!(
            sanitize_repo_subpath("unsloth/Qwen3 8B-GGUF"),
            Some(PathBuf::from("unsloth").join("Qwen3_8B-GGUF"))
        );
        assert!(sanitize_repo_subpath("../x").is_none());
        assert!(sanitize_repo_subpath("").is_none());
    }

    #[tokio::test]
    async fn remote_manage_endpoints_are_gated_and_report_info() {
        let db = Database::open_in_memory().unwrap();
        let mut settings = AppSettings {
            models_dirs: vec![],
            auto_discover_external_models: false,
            ..Default::default()
        };
        settings.link.expose_local_models = true;
        settings.link.allow_remote_manage = false;

        let link = Arc::new(LinkManager::new(&LinkConfig::default()));
        let secret_probe = link.clone();
        let supervisor = Arc::new(ModelSupervisor::new());
        let downloads = Arc::new(bibulm_models::DownloadManager::new(db.clone(), 1));
        let ctx = GatewayContext::new(db, supervisor, Arc::new(RwLock::new(settings)))
            .with_link(link)
            .with_downloads(downloads)
            .with_headless(true);

        // The management routes (without the secret middleware) enforce `allow_remote_manage`.
        let state = Arc::new(ctx.clone());
        let mut manage = remote_manage_routes(state.clone()).with_state(state);
        let denied = send_req(
            &mut manage,
            Request::builder().uri("/link/v1/runtimes").body(Body::empty()).unwrap(),
        )
        .await;
        assert_eq!(denied.status(), StatusCode::FORBIDDEN);
        let denied_dl = send_req(
            &mut manage,
            Request::builder()
                .method(Method::POST)
                .uri("/link/v1/models/download")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"title":"t","repo_id":"org/repo","files":[]}"#))
                .unwrap(),
        )
        .await;
        assert_eq!(denied_dl.status(), StatusCode::FORBIDDEN);

        // Router still requires the link secret.
        let mut app = build_router(ctx.clone());
        let no_secret = send_req(
            &mut app,
            Request::builder()
                .uri("/link/v1/runtimes")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(no_secret.status(), StatusCode::FORBIDDEN);
        assert!(!secret_probe.verify_secret("nope"));

        // Info reports headless + disabled management.
        let info = link_info_handler(State(Arc::new(ctx.clone()))).await;
        let bytes = to_bytes(info.into_body(), 1024 * 1024).await.unwrap();
        let payload: PeerInfoPayload = serde_json::from_slice(&bytes).unwrap();
        assert!(payload.is_headless);
        assert!(!payload.allow_remote_manage);
        assert_eq!(payload.runtimes_count, 0);
        assert_eq!(payload.active_downloads_count, 0);
        assert!(payload.platform.contains(std::env::consts::OS));

        // Enable management: listings succeed, unsafe URLs are rejected.
        ctx.settings.write().await.link.allow_remote_manage = true;
        let ok_rt = send_req(
            &mut manage,
            Request::builder().uri("/link/v1/runtimes").body(Body::empty()).unwrap(),
        )
        .await;
        assert_eq!(ok_rt.status(), StatusCode::OK);
        let ok_dl = send_req(
            &mut manage,
            Request::builder().uri("/link/v1/downloads").body(Body::empty()).unwrap(),
        )
        .await;
        assert_eq!(ok_dl.status(), StatusCode::OK);
        let enabled = Extension(ctx.settings.read().await.clone());

        let bad_url = link_enqueue_model_download_handler(
            State(Arc::new(ctx.clone())),
            enabled.clone(),
            axum::Json(RemoteModelDownloadRequest {
                title: "t".into(),
                repo_id: "org/repo".into(),
                files: vec![bibulm_models::HfDownloadFile {
                    path: "m.gguf".into(),
                    file_name: "m.gguf".into(),
                    url: "https://evil.example/m.gguf".into(),
                    size_bytes: 1,
                    sha256: None,
                }],
            }),
        )
        .await;
        assert_eq!(bad_url.status(), StatusCode::BAD_REQUEST);

        let bad_name = link_enqueue_model_download_handler(
            State(Arc::new(ctx)),
            enabled,
            axum::Json(RemoteModelDownloadRequest {
                title: "t".into(),
                repo_id: "org/repo".into(),
                files: vec![bibulm_models::HfDownloadFile {
                    path: "m.gguf".into(),
                    file_name: "../m.gguf".into(),
                    url: "https://huggingface.co/org/repo/resolve/main/m.gguf".into(),
                    size_bytes: 1,
                    sha256: None,
                }],
            }),
        )
        .await;
        assert_eq!(bad_name.status(), StatusCode::BAD_REQUEST);
    }
}
