use reqwest::Client;
use serde::{Deserialize, Serialize};

/// Live slot state returned by `llama-server GET /slots`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlamaSlotInfo {
    pub id: u32,
    pub is_processing: bool,
    pub n_ctx: u32,
    pub n_past: Option<u32>,
    pub n_decoded: Option<u32>,
    pub id_task: Option<i64>,
}

/// Combined live metrics + active slot breakdown scraped from `llama-server`
/// `/metrics` (Prometheus) and `/slots` (JSON).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LlamaServerMetrics {
    pub port: u16,
    pub slots_total: usize,
    pub slots_processing: usize,
    pub kv_cache_usage_ratio: Option<f64>,
    pub kv_cache_tokens: Option<u64>,
    pub prompt_tokens_total: Option<u64>,
    pub predicted_tokens_total: Option<u64>,
    pub prompt_tok_s: Option<f64>,
    pub predicted_tok_s: Option<f64>,
    pub requests_processing: Option<u64>,
    pub requests_deferred: Option<u64>,
    pub slots: Vec<LlamaSlotInfo>,
}

#[derive(Debug, Default, PartialEq)]
pub struct ParsedPrometheusMetrics {
    pub kv_cache_usage_ratio: Option<f64>,
    pub kv_cache_tokens: Option<u64>,
    pub prompt_tokens_total: Option<u64>,
    pub predicted_tokens_total: Option<u64>,
    pub prompt_tok_s: Option<f64>,
    pub predicted_tok_s: Option<f64>,
    pub requests_processing: Option<u64>,
    pub requests_deferred: Option<u64>,
}

/// Fetches `/slots` and `/metrics` from a running `llama-server` instance on `127.0.0.1:{port}`.
pub async fn fetch_llama_server_metrics(
    client: &Client,
    port: u16,
) -> Result<LlamaServerMetrics, String> {
    let slots_url = format!("http://127.0.0.1:{port}/slots");
    let metrics_url = format!("http://127.0.0.1:{port}/metrics");

    let (slots_res, metrics_res) = tokio::join!(
        client.get(&slots_url).send(),
        client.get(&metrics_url).send(),
    );

    let mut out = LlamaServerMetrics {
        port,
        ..Default::default()
    };
    let mut got_any = false;

    if let Ok(resp) = slots_res
        && resp.status().is_success()
        && let Ok(val) = resp.json::<serde_json::Value>().await
        && let Some(arr) = val.as_array()
    {
        got_any = true;
        out.slots_total = arr.len();
        for item in arr {
            let id = item.get("id").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            let is_processing = item
                .get("is_processing")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if is_processing {
                out.slots_processing += 1;
            }
            let n_ctx = item.get("n_ctx").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            let n_past = item
                .get("n_past")
                .or_else(|| item.get("next_token").and_then(|nt| nt.get("n_past")))
                .and_then(|v| v.as_u64())
                .map(|v| v as u32);
            let n_decoded = item
                .get("next_token")
                .and_then(|nt| nt.get("n_decoded"))
                .and_then(|v| v.as_u64())
                .map(|v| v as u32);
            let id_task = item.get("id_task").and_then(|v| v.as_i64());
            out.slots.push(LlamaSlotInfo {
                id,
                is_processing,
                n_ctx,
                n_past,
                n_decoded,
                id_task,
            });
        }
    }

    if let Ok(resp) = metrics_res
        && resp.status().is_success()
        && let Ok(text) = resp.text().await
    {
        got_any = true;
        let parsed = parse_prometheus_metrics(&text);
        out.kv_cache_usage_ratio = parsed.kv_cache_usage_ratio;
        out.kv_cache_tokens = parsed.kv_cache_tokens;
        out.prompt_tokens_total = parsed.prompt_tokens_total;
        out.predicted_tokens_total = parsed.predicted_tokens_total;
        out.prompt_tok_s = parsed.prompt_tok_s;
        out.predicted_tok_s = parsed.predicted_tok_s;
        out.requests_processing = parsed.requests_processing;
        out.requests_deferred = parsed.requests_deferred;
    }

    if !got_any {
        return Err(format!("Unable to query /slots or /metrics on port {port}"));
    }

    Ok(out)
}

/// Parses `llama-server`'s Prometheus text exposition payload into typed metrics.
pub fn parse_prometheus_metrics(text: &str) -> ParsedPrometheusMetrics {
    let mut m = ParsedPrometheusMetrics::default();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let mut parts = trimmed.split_whitespace();
        let Some(raw_key) = parts.next() else {
            continue;
        };
        let Some(raw_val) = parts.next() else {
            continue;
        };
        let Ok(val) = raw_val.parse::<f64>() else {
            continue;
        };
        let key = raw_key.split('{').next().unwrap_or(raw_key);
        match key {
            "llamacpp:kv_cache_usage_ratio" => m.kv_cache_usage_ratio = Some(val),
            "llamacpp:kv_cache_tokens" => m.kv_cache_tokens = Some(val.max(0.0) as u64),
            "llamacpp:prompt_tokens_total" => m.prompt_tokens_total = Some(val.max(0.0) as u64),
            "llamacpp:tokens_predicted_total" => {
                m.predicted_tokens_total = Some(val.max(0.0) as u64);
            }
            "llamacpp:prompt_tokens_seconds" => m.prompt_tok_s = Some(val),
            "llamacpp:predicted_tokens_seconds" => m.predicted_tok_s = Some(val),
            "llamacpp:requests_processing" => m.requests_processing = Some(val.max(0.0) as u64),
            "llamacpp:requests_deferred" => m.requests_deferred = Some(val.max(0.0) as u64),
            _ => {}
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_llama_server_prometheus_metrics() {
        let sample = r#"
# HELP llamacpp:kv_cache_usage_ratio KV-cache usage. 1 means 100 percent usage.
# TYPE llamacpp:kv_cache_usage_ratio gauge
llamacpp:kv_cache_usage_ratio 0.245
llamacpp:kv_cache_tokens 2010
llamacpp:prompt_tokens_total 4512
llamacpp:tokens_predicted_total 1290
llamacpp:prompt_tokens_seconds 1420.5
llamacpp:predicted_tokens_seconds 78.2
llamacpp:requests_processing 1
llamacpp:requests_deferred 0
"#;
        let parsed = parse_prometheus_metrics(sample);
        assert_eq!(parsed.kv_cache_usage_ratio, Some(0.245));
        assert_eq!(parsed.kv_cache_tokens, Some(2010));
        assert_eq!(parsed.prompt_tokens_total, Some(4512));
        assert_eq!(parsed.predicted_tokens_total, Some(1290));
        assert_eq!(parsed.prompt_tok_s, Some(1420.5));
        assert_eq!(parsed.predicted_tok_s, Some(78.2));
        assert_eq!(parsed.requests_processing, Some(1));
        assert_eq!(parsed.requests_deferred, Some(0));
    }
}
