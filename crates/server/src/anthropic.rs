use crate::jit::{GatewayContext, ResolvedTarget};
use crate::proxy::ActiveRequestGuard;
use axum::body::Body;
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use futures_util::StreamExt;
use bibulm_core::apply_sampling_to_openai_body;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::time::Instant;
use uuid::Uuid;

pub fn anthropic_error(status: StatusCode, err_type: &str, message: &str) -> Response {
    let body = json!({
        "type": "error",
        "error": {
            "type": err_type,
            "message": message
        }
    });
    (status, axum::Json(body)).into_response()
}

/// Handles Anthropic-compatible `POST /v1/messages` (supports Claude Code, tool calling,
/// thinking blocks, base64 vision images, and SSE streaming).
pub async fn handle_anthropic_messages(
    ctx: GatewayContext,
    body_bytes: Bytes,
    is_link_inbound: bool,
) -> Response {
    let started = Instant::now();
    let req_json: Value = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(e) => {
            return anthropic_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                &format!("Invalid JSON request body: {e}"),
            );
        }
    };

    let requested_model = req_json
        .get("model")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToOwned::to_owned);
    let display_model = requested_model
        .clone()
        .unwrap_or_else(|| "local-model".to_string());
    let is_stream = req_json
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let target = match ctx
        .resolve_target(requested_model.as_deref(), is_link_inbound)
        .await
    {
        Ok(t) => t,
        Err((status, msg)) => {
            let err_type = if status == StatusCode::NOT_FOUND {
                "not_found_error"
            } else {
                "invalid_request_error"
            };
            ctx.supervisor
                .push_log(
                    "WARN",
                    "ANTHROPIC",
                    format!("POST /v1/messages -> {status}: {msg}"),
                )
                .await;
            return anthropic_error(status, err_type, &msg);
        }
    };

    match target {
        ResolvedTarget::Local(loaded) => {
            let mut openai_payload = match anthropic_to_openai_request(&req_json, &loaded.model_id)
            {
                Ok(v) => v,
                Err(msg) => {
                    return anthropic_error(StatusCode::BAD_REQUEST, "invalid_request_error", &msg);
                }
            };

            if let Some(obj) = openai_payload.as_object_mut() {
                let defaults = {
                    let s = ctx.settings.read().await;
                    loaded
                        .load_config
                        .sampling_defaults
                        .clone()
                        .unwrap_or_else(|| s.default_sampling.clone())
                };
                apply_sampling_to_openai_body(obj, &defaults, false);
            }

            ctx.supervisor.begin_request(&loaded.model_id).await;
            let guard = ActiveRequestGuard::new(ctx.supervisor.clone(), loaded.model_id.clone());

            let log_src = if is_link_inbound { "LINK" } else { "ANTHROPIC" };
            ctx.supervisor
                .push_log(
                    "INFO",
                    log_src,
                    format!(
                        "POST /v1/messages -> '{}' (:{}){}",
                        loaded.model_id,
                        loaded.port,
                        if is_stream { " [stream]" } else { "" }
                    ),
                )
                .await;

            let upstream_url = format!("http://127.0.0.1:{}/v1/chat/completions", loaded.port);
            let upstream_resp = match ctx
                .http_client
                .post(&upstream_url)
                .header(header::CONTENT_TYPE, "application/json")
                .json(&openai_payload)
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    return anthropic_error(
                        StatusCode::BAD_GATEWAY,
                        "api_error",
                        &format!("Upstream llama-server error: {e}"),
                    );
                }
            };

            if !upstream_resp.status().is_success() {
                let st = StatusCode::from_u16(upstream_resp.status().as_u16())
                    .unwrap_or(StatusCode::BAD_GATEWAY);
                let txt = upstream_resp.text().await.unwrap_or_default();
                return anthropic_error(st, "api_error", &txt);
            }

            if is_stream {
                let stream = openai_sse_to_anthropic_sse_stream(
                    upstream_resp,
                    display_model,
                    Some(guard),
                    ctx.supervisor.clone(),
                    log_src,
                    loaded.model_id,
                    started,
                );
                let mut headers = HeaderMap::new();
                headers.insert(
                    header::CONTENT_TYPE,
                    HeaderValue::from_static("text/event-stream; charset=utf-8"),
                );
                headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
                headers.insert(
                    header::HeaderName::from_static("x-accel-buffering"),
                    HeaderValue::from_static("no"),
                );
                (StatusCode::OK, headers, Body::from_stream(stream)).into_response()
            } else {
                let bytes = match upstream_resp.bytes().await {
                    Ok(b) => b,
                    Err(e) => {
                        return anthropic_error(
                            StatusCode::BAD_GATEWAY,
                            "api_error",
                            &format!("Failed reading upstream response: {e}"),
                        );
                    }
                };
                drop(guard);
                let openai_json: Value = match serde_json::from_slice(&bytes) {
                    Ok(v) => v,
                    Err(e) => {
                        return anthropic_error(
                            StatusCode::BAD_GATEWAY,
                            "api_error",
                            &format!("Invalid upstream JSON: {e}"),
                        );
                    }
                };
                ctx.supervisor
                    .push_log(
                        "INFO",
                        log_src,
                        format!(
                            "POST /v1/messages '{}' completed in {} ms",
                            loaded.model_id,
                            started.elapsed().as_millis()
                        ),
                    )
                    .await;
                axum::Json(openai_to_anthropic_response(&openai_json, &display_model))
                    .into_response()
            }
        }
        ResolvedTarget::Remote {
            peer,
            remote_model_id,
        } => {
            let Some(link) = ctx.link.as_ref() else {
                return anthropic_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "api_error",
                    "BibuLM Link is not initialized.",
                );
            };
            let mut forwarded = req_json.clone();
            if let Some(obj) = forwarded.as_object_mut() {
                obj.insert("model".into(), json!(remote_model_id));
            }
            let upstream_resp = match link
                .proxy_post_json(&peer.tailscale_ip, "/v1/messages", &forwarded)
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    return anthropic_error(
                        StatusCode::BAD_GATEWAY,
                        "api_error",
                        &format!("Remote peer '{}' error: {e}", peer.device_name),
                    );
                }
            };
            let status = StatusCode::from_u16(upstream_resp.status().as_u16())
                .unwrap_or(StatusCode::BAD_GATEWAY);
            if is_stream {
                let stream = upstream_resp
                    .bytes_stream()
                    .map(|r| r.map_err(|e| std::io::Error::other(e.to_string())));
                let mut headers = HeaderMap::new();
                headers.insert(
                    header::CONTENT_TYPE,
                    HeaderValue::from_static("text/event-stream; charset=utf-8"),
                );
                headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
                (status, headers, Body::from_stream(stream)).into_response()
            } else {
                let bytes = upstream_resp.bytes().await.unwrap_or_default();
                let mut headers = HeaderMap::new();
                headers.insert(
                    header::CONTENT_TYPE,
                    HeaderValue::from_static("application/json"),
                );
                (status, headers, bytes).into_response()
            }
        }
    }
}

/// Handles `POST /v1/messages/count_tokens` so Claude Code / Anthropic SDKs can estimate prompt tokens.
pub async fn handle_anthropic_count_tokens(
    ctx: GatewayContext,
    body_bytes: Bytes,
) -> Response {
    let req_json: Value = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(e) => {
            return anthropic_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                &format!("Invalid JSON request body: {e}"),
            );
        }
    };

    // If a local model is already loaded, query llama-server's exact /tokenize endpoint when possible.
    if let Some(model_name) = req_json.get("model").and_then(Value::as_str)
        && let Some(loaded) = ctx.supervisor.resolve_loaded(model_name).await
    {
        let flat = flatten_anthropic_text_for_count(&req_json);
        let url = format!("http://127.0.0.1:{}/tokenize", loaded.port);
        if let Ok(resp) = ctx
            .http_client
            .post(&url)
            .json(&json!({ "content": flat }))
            .send()
            .await
            && resp.status().is_success()
            && let Ok(val) = resp.json::<Value>().await
            && let Some(arr) = val.get("tokens").and_then(Value::as_array)
        {
            return axum::Json(json!({ "input_tokens": arr.len() })).into_response();
        }
    }

    let approx_chars = flatten_anthropic_text_for_count(&req_json).chars().count();
    let approx_tokens = approx_chars.div_ceil(4).max(1);
    axum::Json(json!({ "input_tokens": approx_tokens })).into_response()
}

fn flatten_anthropic_text_for_count(req: &Value) -> String {
    let mut out = String::new();
    if let Some(sys) = req.get("system") {
        extract_text_blocks(sys, &mut out);
    }
    if let Some(msgs) = req.get("messages").and_then(Value::as_array) {
        for m in msgs {
            if let Some(c) = m.get("content") {
                extract_text_blocks(c, &mut out);
            }
        }
    }
    if let Some(tools) = req.get("tools") {
        out.push_str(&tools.to_string());
    }
    out
}

fn extract_text_blocks(val: &Value, out: &mut String) {
    match val {
        Value::String(s) => {
            out.push_str(s);
            out.push('\n');
        }
        Value::Array(arr) => {
            for item in arr {
                if let Some(t) = item.get("text").and_then(Value::as_str) {
                    out.push_str(t);
                    out.push('\n');
                } else if let Some(c) = item.get("content") {
                    extract_text_blocks(c, out);
                }
            }
        }
        _ => {}
    }
}

/// Translates an Anthropic `/v1/messages` JSON payload into an OpenAI `/v1/chat/completions` payload.
pub fn anthropic_to_openai_request(req: &Value, target_model: &str) -> Result<Value, String> {
    let mut messages: Vec<Value> = Vec::new();

    if let Some(sys) = req.get("system") {
        let mut sys_text = String::new();
        extract_text_blocks(sys, &mut sys_text);
        let trimmed = sys_text.trim();
        if !trimmed.is_empty() {
            messages.push(json!({ "role": "system", "content": trimmed }));
        }
    }

    let raw_msgs = req
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| "Missing 'messages' array in Anthropic request.".to_string())?;

    for msg in raw_msgs {
        let role = msg
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("user");
        let content = msg.get("content").unwrap_or(&Value::Null);

        match content {
            Value::String(s) => {
                messages.push(json!({ "role": role, "content": s }));
            }
            Value::Array(blocks) => {
                let mut content_parts: Vec<Value> = Vec::new();
                let mut tool_calls: Vec<Value> = Vec::new();
                let mut tool_results: Vec<Value> = Vec::new();

                for block in blocks {
                    let btype = block.get("type").and_then(Value::as_str).unwrap_or("");
                    match btype {
                        "text" => {
                            if let Some(t) = block.get("text").and_then(Value::as_str) {
                                content_parts.push(json!({ "type": "text", "text": t }));
                            }
                        }
                        "image" => {
                            if let Some(src) = block.get("source")
                                && src.get("type").and_then(Value::as_str) == Some("base64")
                                && let Some(data) = src.get("data").and_then(Value::as_str)
                            {
                                let media_type = src
                                    .get("media_type")
                                    .and_then(Value::as_str)
                                    .unwrap_or("image/png");
                                content_parts.push(json!({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": format!("data:{media_type};base64,{data}")
                                    }
                                }));
                            }
                        }
                        "tool_use" => {
                            let id = block
                                .get("id")
                                .and_then(Value::as_str)
                                .unwrap_or("call_0");
                            let name = block
                                .get("name")
                                .and_then(Value::as_str)
                                .unwrap_or("");
                            let input_str = block
                                .get("input")
                                .map(|v| v.to_string())
                                .unwrap_or_else(|| "{}".to_string());
                            tool_calls.push(json!({
                                "id": id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": input_str
                                }
                            }));
                        }
                        "tool_result" => {
                            let tc_id = block
                                .get("tool_use_id")
                                .and_then(Value::as_str)
                                .unwrap_or("call_0");
                            let mut text_out = String::new();
                            if let Some(inner) = block.get("content") {
                                extract_text_blocks(inner, &mut text_out);
                                if text_out.is_empty() && !inner.is_null() {
                                    text_out = inner.to_string();
                                }
                            }
                            tool_results.push(json!({
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "content": text_out.trim_end()
                            }));
                        }
                        _ => {}
                    }
                }

                if role == "assistant" {
                    let mut assistant_msg = serde_json::Map::new();
                    assistant_msg.insert("role".into(), json!("assistant"));
                    let joined_text: String = content_parts
                        .iter()
                        .filter_map(|p| p.get("text").and_then(Value::as_str))
                        .collect::<Vec<_>>()
                        .join("\n");
                    if !joined_text.is_empty() {
                        assistant_msg.insert("content".into(), json!(joined_text));
                    } else if tool_calls.is_empty() {
                        assistant_msg.insert("content".into(), json!(""));
                    } else {
                        assistant_msg.insert("content".into(), Value::Null);
                    }
                    if !tool_calls.is_empty() {
                        assistant_msg.insert("tool_calls".into(), json!(tool_calls));
                    }
                    messages.push(Value::Object(assistant_msg));
                } else {
                    for tr in tool_results {
                        messages.push(tr);
                    }
                    if !content_parts.is_empty() {
                        let has_images = content_parts
                            .iter()
                            .any(|p| p.get("type").and_then(Value::as_str) == Some("image_url"));
                        if has_images {
                            messages.push(json!({ "role": role, "content": content_parts }));
                        } else {
                            let joined: String = content_parts
                                .iter()
                                .filter_map(|p| p.get("text").and_then(Value::as_str))
                                .collect::<Vec<_>>()
                                .join("\n");
                            messages.push(json!({ "role": role, "content": joined }));
                        }
                    }
                }
            }
            _ => {
                messages.push(json!({ "role": role, "content": "" }));
            }
        }
    }

    let is_stream = req.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let mut out = json!({
        "model": target_model,
        "messages": messages,
        "stream": is_stream,
    });
    if is_stream {
        out.as_object_mut()
            .unwrap()
            .insert("stream_options".into(), json!({ "include_usage": true }));
    }

    let obj = out.as_object_mut().unwrap();
    if let Some(mt) = req.get("max_tokens").and_then(Value::as_u64) {
        obj.insert("max_tokens".into(), json!(mt));
    }
    if let Some(t) = req.get("temperature") {
        obj.insert("temperature".into(), t.clone());
    }
    if let Some(tp) = req.get("top_p") {
        obj.insert("top_p".into(), tp.clone());
    }
    if let Some(tk) = req.get("top_k") {
        obj.insert("top_k".into(), tk.clone());
    }
    if let Some(stops) = req.get("stop_sequences") {
        obj.insert("stop".into(), stops.clone());
    }

    if let Some(tools) = req.get("tools").and_then(Value::as_array) {
        let converted: Vec<Value> = tools
            .iter()
            .filter_map(|t| {
                let name = t.get("name")?.as_str()?;
                let desc = t.get("description").and_then(Value::as_str).unwrap_or("");
                let schema = t.get("input_schema").cloned().unwrap_or_else(|| json!({}));
                Some(json!({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": desc,
                        "parameters": schema
                    }
                }))
            })
            .collect();
        if !converted.is_empty() {
            obj.insert("tools".into(), json!(converted));
        }
    }

    if let Some(tc) = req.get("tool_choice") {
        match tc.get("type").and_then(Value::as_str) {
            Some("auto") => {
                obj.insert("tool_choice".into(), json!("auto"));
            }
            Some("any") => {
                obj.insert("tool_choice".into(), json!("required"));
            }
            Some("none") => {
                obj.insert("tool_choice".into(), json!("none"));
            }
            Some("tool") => {
                if let Some(name) = tc.get("name").and_then(Value::as_str) {
                    obj.insert(
                        "tool_choice".into(),
                        json!({ "type": "function", "function": { "name": name } }),
                    );
                }
            }
            _ => {}
        }
    }

    if let Some(thinking) = req.get("thinking") {
        if thinking.get("type").and_then(Value::as_str) == Some("disabled") {
            obj.insert("reasoning_effort".into(), json!("none"));
        } else if let Some(budget) = thinking.get("budget_tokens").and_then(Value::as_u64) {
            let effort = if budget <= 1024 {
                "low"
            } else if budget <= 4096 {
                "medium"
            } else {
                "high"
            };
            obj.insert("reasoning_effort".into(), json!(effort));
        }
    }

    Ok(out)
}

/// Translates a non-streaming OpenAI chat completion JSON payload into an Anthropic Message response.
pub fn openai_to_anthropic_response(openai: &Value, requested_model: &str) -> Value {
    let msg_id = format!("msg_{}", Uuid::new_v4().to_string().replace('-', ""));
    let choice = openai
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|arr| arr.first());
    let msg = choice.and_then(|c| c.get("message"));

    let mut content: Vec<Value> = Vec::new();
    if let Some(reasoning) = msg
        .and_then(|m| m.get("reasoning_content").or_else(|| m.get("reasoning")))
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        content.push(json!({
            "type": "thinking",
            "thinking": reasoning
        }));
    }

    if let Some(text) = msg
        .and_then(|m| m.get("content"))
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        content.push(json!({
            "type": "text",
            "text": text
        }));
    }

    if let Some(tcs) = msg
        .and_then(|m| m.get("tool_calls"))
        .and_then(Value::as_array)
    {
        for (i, tc) in tcs.iter().enumerate() {
            let id = tc
                .get("id")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| format!("toolu_{i}"));
            let name = tc
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(Value::as_str)
                .unwrap_or("");
            let args_str = tc
                .get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(Value::as_str)
                .unwrap_or("{}");
            let input_val: Value = serde_json::from_str(args_str).unwrap_or_else(|_| json!({}));
            content.push(json!({
                "type": "tool_use",
                "id": id,
                "name": name,
                "input": input_val
            }));
        }
    }

    if content.is_empty() {
        content.push(json!({ "type": "text", "text": "" }));
    }

    let stop_reason = map_finish_reason(choice.and_then(|c| c.get("finish_reason")).and_then(Value::as_str));
    let input_tokens = openai
        .get("usage")
        .and_then(|u| u.get("prompt_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let output_tokens = openai
        .get("usage")
        .and_then(|u| u.get("completion_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);

    json!({
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": Value::Null,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
    })
}

fn map_finish_reason(fr: Option<&str>) -> &'static str {
    match fr {
        Some("length") => "max_tokens",
        Some("tool_calls") | Some("function_call") => "tool_use",
        _ => "end_turn",
    }
}

fn format_anthropic_sse_event(event_name: &str, payload: &Value) -> Bytes {
    Bytes::from(format!("event: {event_name}\ndata: {payload}\n\n"))
}

struct StreamTranslatorState {
    msg_id: String,
    model: String,
    started_message: bool,
    current_block_index: Option<usize>,
    current_block_kind: Option<&'static str>,
    next_block_index: usize,
    tool_block_indices: BTreeMap<usize, usize>,
    stop_reason: &'static str,
    input_tokens: u64,
    output_tokens: u64,
    finished: bool,
}

impl StreamTranslatorState {
    fn new(model: String) -> Self {
        Self {
            msg_id: format!("msg_{}", Uuid::new_v4().to_string().replace('-', "")),
            model,
            started_message: false,
            current_block_index: None,
            current_block_kind: None,
            next_block_index: 0,
            tool_block_indices: BTreeMap::new(),
            stop_reason: "end_turn",
            input_tokens: 0,
            output_tokens: 0,
            finished: false,
        }
    }

    fn ensure_started(&mut self, out: &mut Vec<Bytes>) {
        if !self.started_message {
            self.started_message = true;
            out.push(format_anthropic_sse_event(
                "message_start",
                &json!({
                    "type": "message_start",
                    "message": {
                        "id": self.msg_id,
                        "type": "message",
                        "role": "assistant",
                        "model": self.model,
                        "content": [],
                        "stop_reason": Value::Null,
                        "stop_sequence": Value::Null,
                        "usage": { "input_tokens": self.input_tokens, "output_tokens": 0 }
                    }
                }),
            ));
        }
    }

    fn close_current_block(&mut self, out: &mut Vec<Bytes>) {
        if let Some(idx) = self.current_block_index.take() {
            self.current_block_kind = None;
            out.push(format_anthropic_sse_event(
                "content_block_stop",
                &json!({ "type": "content_block_stop", "index": idx }),
            ));
        }
    }

    fn process_openai_sse_line(&mut self, line: &str) -> Vec<Bytes> {
        let mut out = Vec::new();
        let Some(data) = line.strip_prefix("data:") else {
            return out;
        };
        let data = data.trim();
        if data.is_empty() {
            return out;
        }
        if data == "[DONE]" {
            self.finalize(&mut out);
            return out;
        }
        let Ok(v) = serde_json::from_str::<Value>(data) else {
            return out;
        };

        if let Some(usage) = v.get("usage") {
            if let Some(pt) = usage.get("prompt_tokens").and_then(Value::as_u64) {
                self.input_tokens = pt;
            }
            if let Some(ct) = usage.get("completion_tokens").and_then(Value::as_u64) {
                self.output_tokens = ct;
            }
        }
        if let Some(timings) = v.get("timings") {
            if let Some(pn) = timings.get("prompt_n").and_then(Value::as_u64) {
                self.input_tokens = pn;
            }
            if let Some(gn) = timings.get("predicted_n").and_then(Value::as_u64) {
                self.output_tokens = gn;
            }
        }

        if let Some(choice) = v
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|arr| arr.first())
        {
            if let Some(fr) = choice.get("finish_reason").and_then(Value::as_str) {
                self.stop_reason = map_finish_reason(Some(fr));
            }
            if let Some(delta) = choice.get("delta") {
                if let Some(rc) = delta
                    .get("reasoning_content")
                    .or_else(|| delta.get("reasoning"))
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                {
                    self.ensure_started(&mut out);
                    if self.current_block_kind != Some("thinking") {
                        self.close_current_block(&mut out);
                        let idx = self.next_block_index;
                        self.next_block_index += 1;
                        self.current_block_index = Some(idx);
                        self.current_block_kind = Some("thinking");
                        out.push(format_anthropic_sse_event(
                            "content_block_start",
                            &json!({
                                "type": "content_block_start",
                                "index": idx,
                                "content_block": { "type": "thinking", "thinking": "" }
                            }),
                        ));
                    }
                    let idx = self.current_block_index.unwrap();
                    out.push(format_anthropic_sse_event(
                        "content_block_delta",
                        &json!({
                            "type": "content_block_delta",
                            "index": idx,
                            "delta": { "type": "thinking_delta", "thinking": rc }
                        }),
                    ));
                }

                if let Some(cc) = delta
                    .get("content")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                {
                    self.ensure_started(&mut out);
                    if self.current_block_kind != Some("text") {
                        self.close_current_block(&mut out);
                        let idx = self.next_block_index;
                        self.next_block_index += 1;
                        self.current_block_index = Some(idx);
                        self.current_block_kind = Some("text");
                        out.push(format_anthropic_sse_event(
                            "content_block_start",
                            &json!({
                                "type": "content_block_start",
                                "index": idx,
                                "content_block": { "type": "text", "text": "" }
                            }),
                        ));
                    }
                    let idx = self.current_block_index.unwrap();
                    out.push(format_anthropic_sse_event(
                        "content_block_delta",
                        &json!({
                            "type": "content_block_delta",
                            "index": idx,
                            "delta": { "type": "text_delta", "text": cc }
                        }),
                    ));
                }

                if let Some(tcs) = delta.get("tool_calls").and_then(Value::as_array) {
                    self.ensure_started(&mut out);
                    self.stop_reason = "tool_use";
                    for tc in tcs {
                        let tc_idx = tc.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                        let block_idx = if let Some(&existing) = self.tool_block_indices.get(&tc_idx)
                        {
                            existing
                        } else {
                            self.close_current_block(&mut out);
                            let idx = self.next_block_index;
                            self.next_block_index += 1;
                            self.current_block_index = Some(idx);
                            self.current_block_kind = Some("tool_use");
                            self.tool_block_indices.insert(tc_idx, idx);

                            let id = tc
                                .get("id")
                                .and_then(Value::as_str)
                                .map(ToOwned::to_owned)
                                .unwrap_or_else(|| format!("toolu_{tc_idx}"));
                            let name = tc
                                .get("function")
                                .and_then(|f| f.get("name"))
                                .and_then(Value::as_str)
                                .unwrap_or("");
                            out.push(format_anthropic_sse_event(
                                "content_block_start",
                                &json!({
                                    "type": "content_block_start",
                                    "index": idx,
                                    "content_block": {
                                        "type": "tool_use",
                                        "id": id,
                                        "name": name,
                                        "input": {}
                                    }
                                }),
                            ));
                            idx
                        };
                        if let Some(args_part) = tc
                            .get("function")
                            .and_then(|f| f.get("arguments"))
                            .and_then(Value::as_str)
                            .filter(|s| !s.is_empty())
                        {
                            out.push(format_anthropic_sse_event(
                                "content_block_delta",
                                &json!({
                                    "type": "content_block_delta",
                                    "index": block_idx,
                                    "delta": {
                                        "type": "input_json_delta",
                                        "partial_json": args_part
                                    }
                                }),
                            ));
                        }
                    }
                }
            }
        }
        out
    }

    fn finalize(&mut self, out: &mut Vec<Bytes>) {
        if self.finished {
            return;
        }
        self.finished = true;
        self.ensure_started(out);
        self.close_current_block(out);
        out.push(format_anthropic_sse_event(
            "message_delta",
            &json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": self.stop_reason,
                    "stop_sequence": Value::Null
                },
                "usage": { "output_tokens": self.output_tokens }
            }),
        ));
        out.push(format_anthropic_sse_event(
            "message_stop",
            &json!({ "type": "message_stop" }),
        ));
    }
}

fn openai_sse_to_anthropic_sse_stream(
    upstream_resp: reqwest::Response,
    display_model: String,
    guard: Option<ActiveRequestGuard>,
    supervisor: std::sync::Arc<bibulm_runtime::ModelSupervisor>,
    log_src: &'static str,
    model_id: String,
    started: Instant,
) -> impl futures_util::Stream<Item = Result<Bytes, std::io::Error>> + Send + 'static {
    let inner = upstream_resp.bytes_stream();
    let state = StreamTranslatorState::new(display_model);
    let pending: std::collections::VecDeque<Bytes> = std::collections::VecDeque::new();
    let byte_buf: Vec<u8> = Vec::new();

    futures_util::stream::unfold(
        (
            inner,
            byte_buf,
            pending,
            state,
            guard,
            supervisor,
            log_src,
            model_id,
            started,
        ),
        |(
            mut stream,
            mut byte_buf,
            mut pending,
            mut state,
            guard,
            supervisor,
            log_src,
            model_id,
            started,
        )| async move {
            loop {
                if let Some(item) = pending.pop_front() {
                    return Some((
                        Ok(item),
                        (
                            stream, byte_buf, pending, state, guard, supervisor, log_src, model_id,
                            started,
                        ),
                    ));
                }
                if state.finished {
                    drop(guard);
                    supervisor
                        .push_log(
                            "INFO",
                            log_src,
                            format!(
                                "POST /v1/messages '{model_id}' stream finished in {} ms",
                                started.elapsed().as_millis()
                            ),
                        )
                        .await;
                    return None;
                }
                match stream.next().await {
                    Some(Ok(chunk)) => {
                        byte_buf.extend_from_slice(&chunk);
                        while let Some(nl_pos) = byte_buf.iter().position(|&b| b == b'\n') {
                            let line_bytes: Vec<u8> = byte_buf.drain(..=nl_pos).collect();
                            let line = String::from_utf8_lossy(&line_bytes).trim().to_string();
                            for ev in state.process_openai_sse_line(&line) {
                                pending.push_back(ev);
                            }
                        }
                    }
                    Some(Err(e)) => {
                        return Some((
                            Err(std::io::Error::other(e.to_string())),
                            (
                                stream, byte_buf, pending, state, guard, supervisor, log_src,
                                model_id, started,
                            ),
                        ));
                    }
                    None => {
                        let mut tail = Vec::new();
                        state.finalize(&mut tail);
                        for ev in tail {
                            pending.push_back(ev);
                        }
                    }
                }
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translates_anthropic_request_with_system_images_tools_and_thinking() {
        let req = json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 2048,
            "system": [{ "type": "text", "text": "Be concise." }],
            "thinking": { "type": "enabled", "budget_tokens": 2048 },
            "tools": [{
                "name": "read_file",
                "description": "Reads a file",
                "input_schema": { "type": "object", "properties": { "path": { "type": "string" } } }
            }],
            "tool_choice": { "type": "any" },
            "messages": [
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": "Look at this:" },
                        { "type": "image", "source": { "type": "base64", "media_type": "image/png", "data": "AAAA" } }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        { "type": "tool_use", "id": "toolu_1", "name": "read_file", "input": { "path": "a.rs" } }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        { "type": "tool_result", "tool_use_id": "toolu_1", "content": "fn main() {}" }
                    ]
                }
            ]
        });

        let openai = anthropic_to_openai_request(&req, "qwen3-coder").unwrap();
        assert_eq!(openai["model"], "qwen3-coder");
        assert_eq!(openai["max_tokens"], 2048);
        assert_eq!(openai["reasoning_effort"], "medium");
        assert_eq!(openai["tool_choice"], "required");
        assert_eq!(openai["tools"][0]["function"]["name"], "read_file");
        let msgs = openai["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[1]["content"][1]["type"], "image_url");
        assert_eq!(msgs[2]["tool_calls"][0]["id"], "toolu_1");
        assert_eq!(msgs[3]["role"], "tool");
        assert_eq!(msgs[3]["tool_call_id"], "toolu_1");
    }

    #[test]
    fn translates_non_streaming_and_streaming_openai_to_anthropic() {
        let openai_resp = json!({
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "reasoning_content": "Let me check.",
                    "content": "Calling tool.",
                    "tool_calls": [{
                        "id": "call_1",
                        "function": { "name": "get_weather", "arguments": "{\"city\":\"Paris\"}" }
                    }]
                }
            }],
            "usage": { "prompt_tokens": 42, "completion_tokens": 17 }
        });
        let anth = openai_to_anthropic_response(&openai_resp, "claude-test");
        assert_eq!(anth["type"], "message");
        assert_eq!(anth["stop_reason"], "tool_use");
        assert_eq!(anth["content"][0]["type"], "thinking");
        assert_eq!(anth["content"][1]["type"], "text");
        assert_eq!(anth["content"][2]["type"], "tool_use");
        assert_eq!(anth["content"][2]["input"]["city"], "Paris");
        assert_eq!(anth["usage"]["input_tokens"], 42);

        let mut st = StreamTranslatorState::new("claude-test".into());
        let mut all_events = String::new();
        for line in [
            r#"data: {"choices":[{"delta":{"reasoning_content":"Think..."}}]}"#,
            r#"data: {"choices":[{"delta":{"content":"Hello"}}]}"#,
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_9","function":{"name":"ls","arguments":"{\"p\":"}}]}}]}"#,
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"/\"}"}}]},"finish_reason":"tool_calls"}]}"#,
            r#"data: {"usage":{"prompt_tokens":10,"completion_tokens":5}}"#,
            r#"data: [DONE]"#,
        ] {
            for b in st.process_openai_sse_line(line) {
                all_events.push_str(&String::from_utf8_lossy(&b));
            }
        }
        assert!(all_events.contains("event: message_start"));
        assert!(all_events.contains("thinking_delta"));
        assert!(all_events.contains("text_delta"));
        assert!(all_events.contains("input_json_delta"));
        assert!(all_events.contains("event: content_block_stop"));
        assert!(all_events.contains("event: message_delta"));
        assert!(all_events.contains("event: message_stop"));
        assert!(all_events.contains("\"stop_reason\":\"tool_use\""));
    }
}
