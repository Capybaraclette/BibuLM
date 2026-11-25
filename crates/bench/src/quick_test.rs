use chrono::Utc;
use futures_util::StreamExt;
use bibulm_core::{
    BenchmarkRunRecord, LoadConfig, MemoryPlacementReport, SamplingParams,
    apply_sampling_to_openai_body,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::time::Instant;
use thiserror::Error;
use uuid::Uuid;

#[derive(Debug, Error)]
pub enum BenchError {
    #[error("HTTP error calling llama-server: {0}")]
    Http(#[from] reqwest::Error),
    #[error("Upstream returned HTTP {status}: {body}")]
    UpstreamStatus { status: u16, body: String },
    #[error("Benchmark process error: {0}")]
    Process(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuickTestRequest {
    pub prompt_preset: String,
    pub custom_prompt: Option<String>,
    pub max_tokens: u32,
    pub reasoning_effort: Option<String>,
    pub sampling: Option<SamplingParams>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuickTestStreamChunk {
    pub run_id: String,
    pub reasoning_delta: String,
    pub content_delta: String,
    pub elapsed_ms: f64,
    pub approx_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuickTestResult {
    pub record: BenchmarkRunRecord,
    pub reasoning_text: String,
    pub output_text: String,
    pub total_duration_ms: f64,
    pub prompt_eval_ms: f64,
    pub generation_ms: f64,
}

/// Expands a built-in prompt preset name into an actual benchmark prompt string.
pub fn resolve_preset_prompt(preset: &str, custom_prompt: Option<&str>) -> String {
    match preset {
        "Short QA" => "Explain in 3 concise bullet points how Mixture-of-Experts (MoE) sparse routing reduces active FLOPs per token compared to a dense transformer.".to_string(),
        "Code Gen" => "Write a clean, idiomatic Rust implementation of a lock-free ring buffer (SPSC) using std::sync::atomic with Acquire/Release memory ordering, plus a short unit test.".to_string(),
        "Long Prefill (4K)" => build_long_prefill_prompt(4096),
        "Long Prefill (8K)" => build_long_prefill_prompt(8192),
        "Reasoning Test" => "A datacenter has 3 GPU nodes. Node A has 2x the VRAM of Node B. Node C has 16 GB more VRAM than Node B. Together they total 144 GB of VRAM. If a 70B MoE model requires 42 GB for active attention+KV cache on one node and splits its remaining 46 GB of routed experts evenly across the remaining two nodes, determine each node's VRAM capacity and whether the placement fits without spillover. Show your step-by-step reasoning clearly.".to_string(),
        _ => custom_prompt
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or("Summarize the key architectural differences between KV cache quantization (Q8_0 vs F16) and Flash Attention in modern LLM inference.")
            .to_string(),
    }
}

fn build_long_prefill_prompt(approx_tokens: usize) -> String {
    let unit = "Telemetry log entry: node=gpu0 kv_quant=q8_0 flash_attn=on moe_cpu_layers=12 vram_mib=11840 ram_mib=6420 latency_ms=14.2 throughput_tps=68.4 status=nominal.\n";
    let tokens_per_unit = 36usize;
    let repeats = (approx_tokens / tokens_per_unit).max(16);
    let mut s = String::with_capacity(repeats * unit.len() + 256);
    s.push_str("You are analyzing a high-volume inference telemetry stream. Read all entries carefully and answer the final question in 2 sentences.\n\n");
    for i in 0..repeats {
        if i == repeats / 2 {
            s.push_str("ANOMALY_MARKER: node=gpu0 shared_wddm_spillover_mib=2450 throughput_tps=4.1 status=SPILLOVER_DETECTED.\n");
        } else {
            s.push_str(unit);
        }
    }
    s.push_str("\nQuestion: What anomaly occurred mid-stream, what was the spillover size in MiB, and what happened to throughput?");
    s
}

/// Executes a live streaming OpenAI `/v1/chat/completions` benchmark directly against
/// the managed `llama-server` loopback port and returns exact TTFT, pp/s, tg/s,
/// reasoning vs output token counts, and placement metadata.
#[allow(clippy::too_many_arguments)]
pub async fn run_quick_speed_test<F>(
    port: u16,
    model_id: String,
    runtime_tag: String,
    load_config: LoadConfig,
    placement: MemoryPlacementReport,
    peak_vram_mib: f64,
    peak_ram_mib: f64,
    req: QuickTestRequest,
    on_chunk: F,
) -> Result<QuickTestResult, BenchError>
where
    F: FnMut(QuickTestStreamChunk) + Send,
{
    let url = format!("http://127.0.0.1:{port}/v1/chat/completions");
    run_quick_speed_test_against_url(
        &url,
        &[],
        model_id.clone(),
        model_id,
        runtime_tag,
        load_config,
        placement,
        peak_vram_mib,
        peak_ram_mib,
        req,
        on_chunk,
    )
    .await
}

/// Executes a live streaming OpenAI `/v1/chat/completions` benchmark against an arbitrary
/// URL (local `llama-server` or a remote Bibulm Link peer via the local `tsnet` bridge).
#[allow(clippy::too_many_arguments)]
pub async fn run_quick_speed_test_against_url<F>(
    endpoint_url: &str,
    extra_headers: &[(&str, &str)],
    request_model_id: String,
    record_model_id: String,
    runtime_tag: String,
    load_config: LoadConfig,
    placement: MemoryPlacementReport,
    peak_vram_mib: f64,
    peak_ram_mib: f64,
    req: QuickTestRequest,
    mut on_chunk: F,
) -> Result<QuickTestResult, BenchError>
where
    F: FnMut(QuickTestStreamChunk) + Send,
{
    let run_id = Uuid::new_v4().to_string();
    let prompt_text = resolve_preset_prompt(&req.prompt_preset, req.custom_prompt.as_deref());

    let mut messages = Vec::new();
    if let Some(sp) = req
        .sampling
        .as_ref()
        .map(|s| s.system_prompt.trim())
        .filter(|s| !s.is_empty())
    {
        messages.push(json!({ "role": "system", "content": sp }));
    }
    messages.push(json!({ "role": "user", "content": prompt_text }));

    let mut body = json!({
        "model": request_model_id,
        "messages": messages,
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
    }

    let client = reqwest::Client::new();
    let started = Instant::now();

    let mut req_builder = client.post(endpoint_url).json(&body);
    for (k, v) in extra_headers {
        req_builder = req_builder.header(*k, *v);
    }
    let resp = req_builder.send().await?;
    if !resp.status().is_success() {
        return Err(BenchError::UpstreamStatus {
            status: resp.status().as_u16(),
            body: resp.text().await.unwrap_or_default(),
        });
    }

    let mut stream = resp.bytes_stream();
    let mut byte_buf: Vec<u8> = Vec::new();
    let mut ttft_ms: Option<f64> = None;
    let mut reasoning_text = String::new();
    let mut output_text = String::new();
    let mut reasoning_chunks: u32 = 0;
    let mut output_chunks: u32 = 0;

    let mut prompt_tokens: u32 = 0;
    let mut predicted_tokens: u32 = 0;
    let mut prompt_per_second: Option<f64> = None;
    let mut predicted_per_second: Option<f64> = None;
    let mut prompt_ms: Option<f64> = None;
    let mut predicted_ms: Option<f64> = None;

    while let Some(chunk) = stream.next().await {
        let bytes = chunk?;
        byte_buf.extend_from_slice(&bytes);

        while let Some(nl_pos) = byte_buf.iter().position(|&b| b == b'\n') {
            let line_bytes: Vec<u8> = byte_buf.drain(..=nl_pos).collect();
            let line = String::from_utf8_lossy(&line_bytes).trim().to_string();

            let Some(data) = line.strip_prefix("data:") else {
                continue;
            };
            let data = data.trim();
            if data.is_empty() || data == "[DONE]" {
                continue;
            }
            let Ok(v) = serde_json::from_str::<Value>(data) else {
                continue;
            };

            let mut r_delta = String::new();
            let mut c_delta = String::new();
            if let Some(delta) = v
                .get("choices")
                .and_then(Value::as_array)
                .and_then(|arr| arr.first())
                .and_then(|c| c.get("delta"))
            {
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
                    reasoning_chunks += 1;
                }
                if let Some(cc) = delta.get("content").and_then(Value::as_str)
                    && !cc.is_empty()
                {
                    if ttft_ms.is_none() {
                        ttft_ms = Some(started.elapsed().as_secs_f64() * 1000.0);
                    }
                    output_text.push_str(cc);
                    c_delta = cc.to_string();
                    output_chunks += 1;
                }
            }

            if !r_delta.is_empty() || !c_delta.is_empty() {
                on_chunk(QuickTestStreamChunk {
                    run_id: run_id.clone(),
                    reasoning_delta: r_delta,
                    content_delta: c_delta,
                    elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                    approx_tokens: reasoning_chunks + output_chunks,
                });
            }

            // Capture OpenAI `usage`
            if let Some(usage) = v.get("usage") {
                if let Some(pt) = usage.get("prompt_tokens").and_then(Value::as_u64) {
                    prompt_tokens = pt as u32;
                }
                if let Some(ct) = usage.get("completion_tokens").and_then(Value::as_u64) {
                    predicted_tokens = ct as u32;
                }
            }

            // Capture exact `llama-server` timings block
            if let Some(timings) = v.get("timings") {
                if let Some(pn) = timings.get("prompt_n").and_then(Value::as_u64) {
                    prompt_tokens = pn as u32;
                }
                if let Some(gn) = timings.get("predicted_n").and_then(Value::as_u64) {
                    predicted_tokens = gn as u32;
                }
                if let Some(ps) = timings.get("prompt_per_second").and_then(Value::as_f64) {
                    prompt_per_second = Some(ps);
                }
                if let Some(gs) = timings.get("predicted_per_second").and_then(Value::as_f64) {
                    predicted_per_second = Some(gs);
                }
                if let Some(pm) = timings.get("prompt_ms").and_then(Value::as_f64) {
                    prompt_ms = Some(pm);
                }
                if let Some(gm) = timings.get("predicted_ms").and_then(Value::as_f64) {
                    predicted_ms = Some(gm);
                }
            }
        }
    }

    let total_duration_ms = started.elapsed().as_secs_f64() * 1000.0;
    let ttft_final = ttft_ms.unwrap_or(total_duration_ms);
    let total_chunks = (reasoning_chunks + output_chunks).max(1);
    let total_gen_tokens = if predicted_tokens > 0 {
        predicted_tokens
    } else {
        reasoning_chunks + output_chunks
    };

    // Split total generated tokens accurately between reasoning vs answer.
    let (reasoning_tokens, output_tokens) = if reasoning_chunks == 0 {
        (0, total_gen_tokens)
    } else if output_chunks == 0 {
        (total_gen_tokens, 0)
    } else {
        let r = ((total_gen_tokens as f64) * (reasoning_chunks as f64 / total_chunks as f64))
            .round() as u32;
        (r.min(total_gen_tokens), total_gen_tokens.saturating_sub(r))
    };

    let gen_ms = predicted_ms.unwrap_or_else(|| (total_duration_ms - ttft_final).max(1.0));
    let pp_ms = prompt_ms.unwrap_or(ttft_final.max(1.0));

    let generation_tok_s = predicted_per_second
        .unwrap_or_else(|| (total_gen_tokens as f64) / (gen_ms / 1000.0).max(0.001));
    let prompt_eval_tok_s = prompt_per_second
        .unwrap_or_else(|| (prompt_tokens as f64) / (pp_ms / 1000.0).max(0.001));

    let record = BenchmarkRunRecord {
        id: run_id,
        created_at: Utc::now().to_rfc3339(),
        model_id: record_model_id,
        runtime_tag,
        kind: "quick_test".to_string(),
        prompt_preset: req.prompt_preset,
        prompt_tokens,
        reasoning_tokens,
        output_tokens,
        ttft_ms: ttft_final,
        prompt_eval_tok_s,
        generation_tok_s,
        peak_vram_mib,
        peak_ram_mib,
        placement,
        load_config,
        notes: None,
    };

    Ok(QuickTestResult {
        record,
        reasoning_text,
        output_text,
        total_duration_ms,
        prompt_eval_ms: pp_ms,
        generation_ms: gen_ms,
    })
}

/// Decodes complete `\n`-delimited lines from a raw byte buffer without splitting multi-byte UTF-8 sequences.
pub fn drain_complete_sse_lines(byte_buf: &mut Vec<u8>) -> Vec<String> {
    let mut lines = Vec::new();
    while let Some(nl_pos) = byte_buf.iter().position(|&b| b == b'\n') {
        let line_bytes: Vec<u8> = byte_buf.drain(..=nl_pos).collect();
        lines.push(String::from_utf8_lossy(&line_bytes).trim().to_string());
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_byte_sse_drain_preserves_split_multibyte_utf8() {
        let emoji_line = "data: {\"content\":\"Café 🚀\"}\n".as_bytes();
        // Split right in the middle of 'é' and '🚀'
        let split_idx = emoji_line.len() - 4;
        let mut buf = Vec::new();
        buf.extend_from_slice(&emoji_line[..split_idx]);
        assert!(drain_complete_sse_lines(&mut buf).is_empty());

        buf.extend_from_slice(&emoji_line[split_idx..]);
        let lines = drain_complete_sse_lines(&mut buf);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("Café 🚀"));
        assert!(!lines[0].contains('\u{FFFD}'));
    }
}
