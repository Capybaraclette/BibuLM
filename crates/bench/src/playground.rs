use crate::quick_test::{
    BenchError, StreamUsage, drain_complete_sse_lines, post_chat_stream, sse_data_json,
};
use futures_util::StreamExt;
use bibulm_core::{SamplingParams, apply_sampling_to_openai_body};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::time::Instant;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaygroundMessage {
    pub role: String,
    pub content: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaygroundTurnRequest {
    pub messages: Vec<PlaygroundMessage>,
    pub max_tokens: u32,
    pub reasoning_effort: Option<String>,
    pub sampling: Option<SamplingParams>,
    pub tools_json: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaygroundStreamChunk {
    pub turn_id: String,
    pub reasoning_delta: String,
    pub content_delta: String,
    pub elapsed_ms: f64,
    pub approx_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaygroundTurnResult {
    pub turn_id: String,
    pub reasoning_text: String,
    pub output_text: String,
    pub tool_calls: Vec<Value>,
    pub finish_reason: Option<String>,
    pub prompt_tokens: u32,
    pub cached_tokens: u32,
    pub completion_tokens: u32,
    pub ttft_ms: f64,
    pub prompt_eval_tok_s: f64,
    pub generation_tok_s: f64,
    pub total_duration_ms: f64,
}

#[derive(Default)]
struct PartialToolCall {
    id: String,
    kind: String,
    name: String,
    arguments: String,
}

/// Runs a multi-turn, multimodal, tool-calling, structured-output playground request
/// against `/v1/chat/completions` and returns detailed per-turn timings & KV cache reuse stats.
pub async fn run_playground_turn<F>(
    endpoint_url: &str,
    extra_headers: &[(&str, &str)],
    request_model_id: String,
    req: PlaygroundTurnRequest,
    mut on_chunk: F,
) -> Result<PlaygroundTurnResult, BenchError>
where
    F: FnMut(PlaygroundStreamChunk) + Send,
{
    let turn_id = Uuid::new_v4().to_string();
    let mut messages_json: Vec<Value> = Vec::new();
    if let Some(sp) = req
        .sampling
        .as_ref()
        .map(|s| s.system_prompt.trim())
        .filter(|s| !s.is_empty())
    {
        let has_sys = req.messages.first().is_some_and(|m| m.role == "system");
        if !has_sys {
            messages_json.push(json!({ "role": "system", "content": sp }));
        }
    }
    for m in &req.messages {
        let mut obj = serde_json::Map::new();
        obj.insert("role".into(), json!(m.role));
        obj.insert("content".into(), m.content.clone());
        if let Some(ref tc_id) = m.tool_call_id {
            obj.insert("tool_call_id".into(), json!(tc_id));
        }
        if let Some(ref tcs) = m.tool_calls {
            obj.insert("tool_calls".into(), json!(tcs));
        }
        messages_json.push(Value::Object(obj));
    }

    let mut body = json!({
        "model": request_model_id,
        "messages": messages_json,
        "stream": true,
        "stream_options": { "include_usage": true },
        "max_tokens": req.max_tokens.max(16),
        "timings_per_token": true,
    });

    if let Some(obj) = body.as_object_mut() {
        if let Some(effort) = req
            .reasoning_effort
            .as_deref()
            .filter(|s| !s.is_empty() && *s != "default")
        {
            obj.insert("reasoning_effort".into(), json!(effort));
        }
        if let Some(s) = &req.sampling {
            apply_sampling_to_openai_body(obj, s, true);
        }
        if let Some(tools_raw) = req.tools_json.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
            let parsed: Value = serde_json::from_str(tools_raw)
                .map_err(|e| BenchError::Process(format!("Invalid tools JSON: {e}")))?;
            if !parsed.is_array() {
                return Err(BenchError::Process("Tools JSON must be an array of OpenAI tool definitions.".into()));
            }
            obj.insert("tools".into(), parsed);
        }
    }

    let started = Instant::now();
    let resp = post_chat_stream(endpoint_url, &body, extra_headers).await?;

    let mut stream = resp.bytes_stream();
    let mut byte_buf: Vec<u8> = Vec::new();
    let mut ttft_ms: Option<f64> = None;
    let mut reasoning_text = String::new();
    let mut output_text = String::new();
    let mut finish_reason: Option<String> = None;
    let mut tool_acc: BTreeMap<usize, PartialToolCall> = BTreeMap::new();
    let mut approx_tokens: u32 = 0;

    let mut usage = StreamUsage::default();

    while let Some(chunk) = stream.next().await {
        let bytes = chunk?;
        byte_buf.extend_from_slice(&bytes);

        for line in drain_complete_sse_lines(&mut byte_buf) {
            let Some(v) = sse_data_json(&line) else {
                continue;
            };

            if let Some(choice) = v
                .get("choices")
                .and_then(Value::as_array)
                .and_then(|arr| arr.first())
            {
                if let Some(fr) = choice.get("finish_reason").and_then(Value::as_str) {
                    finish_reason = Some(fr.to_string());
                }
                if let Some(delta) = choice.get("delta") {
                    let mut r_delta = String::new();
                    let mut c_delta = String::new();
                    if let Some(rc) = delta
                        .get("reasoning_content")
                        .or_else(|| delta.get("reasoning"))
                        .and_then(Value::as_str)
                        && !rc.is_empty()
                    {
                        if ttft_ms.is_none() {
                            ttft_ms = Some(started.elapsed().as_secs_f64() * 1000.0);
                        }
                        reasoning_text.push_str(rc);
                        r_delta = rc.to_string();
                        approx_tokens += 1;
                    }
                    if let Some(cc) = delta.get("content").and_then(Value::as_str)
                        && !cc.is_empty()
                    {
                        if ttft_ms.is_none() {
                            ttft_ms = Some(started.elapsed().as_secs_f64() * 1000.0);
                        }
                        output_text.push_str(cc);
                        c_delta = cc.to_string();
                        approx_tokens += 1;
                    }
                    if let Some(tcs) = delta.get("tool_calls").and_then(Value::as_array) {
                        if ttft_ms.is_none() {
                            ttft_ms = Some(started.elapsed().as_secs_f64() * 1000.0);
                        }
                        for tc in tcs {
                            let idx = tc.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                            let entry = tool_acc.entry(idx).or_default();
                            if let Some(id) = tc.get("id").and_then(Value::as_str) {
                                entry.id = id.to_string();
                            }
                            if let Some(tp) = tc.get("type").and_then(Value::as_str) {
                                entry.kind = tp.to_string();
                            }
                            if let Some(func) = tc.get("function") {
                                if let Some(n) = func.get("name").and_then(Value::as_str) {
                                    entry.name.push_str(n);
                                }
                                if let Some(args) = func.get("arguments").and_then(Value::as_str) {
                                    entry.arguments.push_str(args);
                                }
                            }
                        }
                    }
                    if !r_delta.is_empty() || !c_delta.is_empty() {
                        on_chunk(PlaygroundStreamChunk {
                            turn_id: turn_id.clone(),
                            reasoning_delta: r_delta,
                            content_delta: c_delta,
                            elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                            approx_tokens,
                        });
                    }
                }
            }

            usage.absorb(&v);
        }
    }

    let total_duration_ms = started.elapsed().as_secs_f64() * 1000.0;
    let ttft_final = ttft_ms.unwrap_or(total_duration_ms);
    let total_gen = if usage.completion_tokens > 0 {
        usage.completion_tokens
    } else {
        approx_tokens
    };
    let (_, _, prompt_eval_tok_s, generation_tok_s) =
        usage.rates(ttft_final, total_duration_ms, total_gen);

    let tool_calls: Vec<Value> = tool_acc
        .into_iter()
        .map(|(idx, tc)| {
            json!({
                "id": if tc.id.is_empty() { format!("call_{idx}") } else { tc.id },
                "type": if tc.kind.is_empty() { "function".to_string() } else { tc.kind },
                "function": {
                    "name": tc.name,
                    "arguments": tc.arguments,
                }
            })
        })
        .collect();

    Ok(PlaygroundTurnResult {
        turn_id,
        reasoning_text,
        output_text,
        tool_calls,
        finish_reason,
        prompt_tokens: usage.prompt_tokens,
        cached_tokens: usage.cached_tokens,
        completion_tokens: total_gen,
        ttft_ms: ttft_final,
        prompt_eval_tok_s,
        generation_tok_s,
        total_duration_ms,
    })
}
