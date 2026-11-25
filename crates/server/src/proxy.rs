use crate::jit::{GatewayContext, ResolvedTarget};
use axum::body::Body;
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use futures_util::StreamExt;
use bibulm_core::{SamplingParams, apply_sampling_to_openai_body};
use bibulm_runtime::ModelSupervisor;
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Instant;

pub(crate) struct ActiveRequestGuard {
    supervisor: Arc<ModelSupervisor>,
    model_id: String,
}

impl ActiveRequestGuard {
    pub(crate) fn new(supervisor: Arc<ModelSupervisor>, model_id: String) -> Self {
        Self {
            supervisor,
            model_id,
        }
    }
}

impl Drop for ActiveRequestGuard {
    fn drop(&mut self) {
        let sup = self.supervisor.clone();
        let id = self.model_id.clone();
        tokio::spawn(async move {
            sup.end_request(&id).await;
        });
    }
}

/// Proxies an OpenAI-compatible JSON POST request to either a local `llama-server`
/// or a remote BibuLM Link peer over Tailscale, supporting non-streaming JSON
/// and unbuffered SSE streaming pass-through.
pub async fn proxy_openai_post(
    ctx: GatewayContext,
    path: &str,
    body_bytes: Bytes,
    is_link_inbound: bool,
) -> Response {
    let started = Instant::now();
    let mut payload: Value = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(e) => {
            return openai_error(
                StatusCode::BAD_REQUEST,
                &format!("Invalid JSON request body: {e}"),
            );
        }
    };

    let requested_model = payload
        .get("model")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);

    let target = match ctx
        .resolve_target(requested_model.as_deref(), is_link_inbound)
        .await
    {
        Ok(t) => t,
        Err((status, msg)) => {
            ctx.supervisor
                .push_log("WARN", "GATEWAY", format!("POST {path} -> {status}: {msg}"))
                .await;
            return openai_error(status, &msg);
        }
    };

    let is_stream = payload
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    match target {
        ResolvedTarget::Local(loaded) => {
            // Inject configured default sampling params if omitted on chat/text/responses endpoints
            if path.ends_with("/completions") || path.ends_with("/responses") {
                let defaults = {
                    let s = ctx.settings.read().await;
                    loaded
                        .load_config
                        .sampling_defaults
                        .clone()
                        .unwrap_or_else(|| s.default_sampling.clone())
                };
                apply_sampling_defaults(&mut payload, &defaults, path.ends_with("/responses"));
            }

            ctx.supervisor.begin_request(&loaded.model_id).await;
            let guard = ActiveRequestGuard {
                supervisor: ctx.supervisor.clone(),
                model_id: loaded.model_id.clone(),
            };

            let log_src = if is_link_inbound { "LINK" } else { "GATEWAY" };
            ctx.supervisor
                .push_log(
                    "INFO",
                    log_src,
                    format!(
                        "POST {path} -> '{}' (:{}){}",
                        loaded.model_id,
                        loaded.port,
                        if is_stream { " [stream]" } else { "" }
                    ),
                )
                .await;

            let upstream_url = format!("http://127.0.0.1:{}{path}", loaded.port);
            let upstream_resp = match ctx
                .http_client
                .post(&upstream_url)
                .header(header::CONTENT_TYPE, "application/json")
                .json(&payload)
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    let msg = format!("Upstream llama-server error: {e}");
                    ctx.supervisor.push_log("ERROR", log_src, &msg).await;
                    return openai_error(StatusCode::BAD_GATEWAY, &msg);
                }
            };

            forward_upstream_response(
                upstream_resp,
                is_stream,
                Some(guard),
                ctx.supervisor.clone(),
                log_src,
                path,
                loaded.model_id,
                started,
            )
            .await
        }
        ResolvedTarget::Remote {
            peer,
            remote_model_id,
        } => {
            let Some(link) = ctx.link.as_ref() else {
                return openai_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "BibuLM Link is not initialized.",
                );
            };

            // Rewrite the model field to the bare model ID expected by the remote peer.
            if let Some(obj) = payload.as_object_mut() {
                obj.insert("model".into(), json!(remote_model_id));
            }

            let label = format!("{}/{}", peer.slug, remote_model_id);
            ctx.supervisor
                .push_log(
                    "INFO",
                    "LINK",
                    format!(
                        "POST {path} -> remote '{}' ({}){}",
                        label,
                        peer.tailscale_ip,
                        if is_stream { " [stream]" } else { "" }
                    ),
                )
                .await;

            let upstream_resp = match link
                .proxy_post_json(&peer.tailscale_ip, path, &payload)
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    let link_c = link.clone();
                    tokio::spawn(async move {
                        let _ = link_c.refresh_peers().await;
                    });
                    let msg = format!("Remote peer '{}' error: {e}", peer.device_name);
                    ctx.supervisor.push_log("ERROR", "LINK", &msg).await;
                    return openai_error(StatusCode::BAD_GATEWAY, &msg);
                }
            };

            if matches!(
                upstream_resp.status().as_u16(),
                403 | 404 | 502
            ) {
                let link_c = link.clone();
                tokio::spawn(async move {
                    let _ = link_c.refresh_peers().await;
                });
            }

            forward_upstream_response(
                upstream_resp,
                is_stream,
                None,
                ctx.supervisor.clone(),
                "LINK",
                path,
                label,
                started,
            )
            .await
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn forward_upstream_response(
    upstream_resp: reqwest::Response,
    is_stream: bool,
    guard: Option<ActiveRequestGuard>,
    supervisor: Arc<ModelSupervisor>,
    log_src: &'static str,
    path: &str,
    model_label: String,
    started: Instant,
) -> Response {
    let status = StatusCode::from_u16(upstream_resp.status().as_u16())
        .unwrap_or(StatusCode::BAD_GATEWAY);
    let content_type = upstream_resp
        .headers()
        .get(header::CONTENT_TYPE)
        .cloned()
        .unwrap_or_else(|| {
            HeaderValue::from_static(if is_stream {
                "text/event-stream; charset=utf-8"
            } else {
                "application/json"
            })
        });

    if is_stream {
        let path_owned = path.to_string();
        let stream = async_stream_body(
            upstream_resp,
            guard,
            supervisor,
            log_src,
            path_owned,
            model_label,
            started,
        );
        let mut headers = HeaderMap::new();
        headers.insert(header::CONTENT_TYPE, content_type);
        headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
        headers.insert(
            header::HeaderName::from_static("x-accel-buffering"),
            HeaderValue::from_static("no"),
        );
        (status, headers, Body::from_stream(stream)).into_response()
    } else {
        let resp_bytes = match upstream_resp.bytes().await {
            Ok(b) => b,
            Err(e) => {
                return openai_error(
                    StatusCode::BAD_GATEWAY,
                    &format!("Failed reading upstream response: {e}"),
                );
            }
        };
        drop(guard);
        supervisor
            .push_log(
                "INFO",
                log_src,
                format!(
                    "POST {path} '{}' completed ({}) in {} ms",
                    model_label,
                    status.as_u16(),
                    started.elapsed().as_millis()
                ),
            )
            .await;
        let mut headers = HeaderMap::new();
        headers.insert(header::CONTENT_TYPE, content_type);
        (status, headers, resp_bytes).into_response()
    }
}

fn async_stream_body(
    upstream_resp: reqwest::Response,
    guard: Option<ActiveRequestGuard>,
    supervisor: Arc<ModelSupervisor>,
    log_src: &'static str,
    path: String,
    model_id: String,
    started: Instant,
) -> impl futures_util::Stream<Item = Result<Bytes, std::io::Error>> + Send + 'static {
    let inner = upstream_resp.bytes_stream();
    futures_util::stream::unfold(
        (inner, guard, supervisor, log_src, path, model_id, started),
        |(mut stream, guard, supervisor, log_src, path, model_id, started)| async move {
            match stream.next().await {
                Some(Ok(chunk)) => Some((
                    Ok(chunk),
                    (stream, guard, supervisor, log_src, path, model_id, started),
                )),
                Some(Err(e)) => Some((
                    Err(std::io::Error::other(e.to_string())),
                    (stream, guard, supervisor, log_src, path, model_id, started),
                )),
                None => {
                    drop(guard);
                    supervisor
                        .push_log(
                            "INFO",
                            log_src,
                            format!(
                                "POST {path} '{model_id}' stream finished in {} ms",
                                started.elapsed().as_millis()
                            ),
                        )
                        .await;
                    None
                }
            }
        },
    )
}

fn apply_sampling_defaults(payload: &mut Value, defaults: &SamplingParams, is_responses_api: bool) {
    let Some(obj) = payload.as_object_mut() else {
        return;
    };
    apply_sampling_to_openai_body(obj, defaults, false);
    if is_responses_api
        && let Some(mt) = obj.remove("max_tokens")
        && !obj.contains_key("max_output_tokens")
    {
        obj.insert("max_output_tokens".to_string(), mt);
    }
}

pub fn openai_error(status: StatusCode, message: &str) -> Response {
    let body = json!({
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "code": status.as_u16()
        }
    });
    (status, axum::Json(body)).into_response()
}
