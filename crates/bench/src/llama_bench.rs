use crate::quick_test::BenchError;
use bibulm_runtime::split_extra_args;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;
use tokio::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaBenchRequest {
    pub model_path: PathBuf,
    pub model_id: String,
    pub runtime_tag: String,
    pub prompt_sizes: Vec<u32>,
    pub gen_sizes: Vec<u32>,
    pub n_gpu_layers: Vec<u32>,
    pub flash_attn: Vec<bool>,
    #[serde(default)]
    pub cache_types_k: Vec<String>,
    #[serde(default)]
    pub cache_types_v: Vec<String>,
    pub threads: Option<u32>,
    pub batch_size: Option<u32>,
    pub ubatch_size: Option<u32>,
    pub repetitions: u32,
    #[serde(default)]
    pub extra_args: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlamaBenchRow {
    pub model_type: String,
    pub model_size: u64,
    pub n_prompt: u32,
    pub n_gen: u32,
    pub n_gpu_layers: u32,
    pub flash_attn: bool,
    pub type_k: String,
    pub type_v: String,
    pub avg_ts: f64,
    pub stddev_ts: f64,
    pub avg_ns: f64,
    pub test_label: String,
}

/// Builds deterministic `llama-bench` CLI arguments requesting JSON output.
pub fn build_llama_bench_args(req: &LlamaBenchRequest) -> Vec<String> {
    let mut args = vec![
        "-m".to_string(),
        req.model_path.to_string_lossy().into_owned(),
        "-o".to_string(),
        "json".to_string(),
        "-r".to_string(),
        req.repetitions.max(1).to_string(),
    ];

    if !req.prompt_sizes.is_empty() {
        args.push("-p".to_string());
        args.push(
            req.prompt_sizes
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(","),
        );
    }
    if !req.gen_sizes.is_empty() {
        args.push("-n".to_string());
        args.push(
            req.gen_sizes
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(","),
        );
    }
    if !req.n_gpu_layers.is_empty() {
        args.push("-ngl".to_string());
        args.push(
            req.n_gpu_layers
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(","),
        );
    }
    if !req.flash_attn.is_empty() {
        args.push("-fa".to_string());
        args.push(
            req.flash_attn
                .iter()
                .map(|b| if *b { "1" } else { "0" })
                .collect::<Vec<_>>()
                .join(","),
        );
    }
    if !req.cache_types_k.is_empty() {
        args.push("-ctk".to_string());
        args.push(req.cache_types_k.join(","));
    }
    if !req.cache_types_v.is_empty() {
        args.push("-ctv".to_string());
        args.push(req.cache_types_v.join(","));
    }
    if let Some(t) = req.threads {
        args.push("-t".to_string());
        args.push(t.to_string());
    }
    if let Some(b) = req.batch_size {
        args.push("-b".to_string());
        args.push(b.to_string());
    }
    if let Some(ub) = req.ubatch_size {
        args.push("-ub".to_string());
        args.push(ub.to_string());
    }
    for raw in &req.extra_args {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            args.extend(split_extra_args(trimmed));
        }
    }
    args
}

/// Parses `llama-bench -o json` stdout, tolerating any leading diagnostic lines before `[`.
pub fn parse_llama_bench_json(stdout: &str) -> Result<Vec<LlamaBenchRow>, BenchError> {
    let trimmed = stdout.trim();
    let json_slice = if trimmed.starts_with('[') {
        trimmed
    } else if let Some(idx) = trimmed.find('[') {
        &trimmed[idx..]
    } else {
        return Err(BenchError::Process(format!(
            "llama-bench did not emit a JSON array. Output: {}",
            trimmed.chars().take(400).collect::<String>()
        )));
    };

    let arr: Vec<Value> = serde_json::from_str(json_slice)
        .map_err(|e| BenchError::Process(format!("Failed parsing llama-bench JSON: {e}")))?;

    let mut rows = Vec::with_capacity(arr.len());
    for item in arr {
        let n_prompt = item.get("n_prompt").and_then(Value::as_u64).unwrap_or(0) as u32;
        let n_gen = item.get("n_gen").and_then(Value::as_u64).unwrap_or(0) as u32;
        let test_label = match (n_prompt, n_gen) {
            (p, 0) if p > 0 => format!("pp{p}"),
            (0, g) if g > 0 => format!("tg{g}"),
            (p, g) => format!("pp{p}+tg{g}"),
        };
        let flash_attn = match item.get("flash_attn") {
            Some(Value::Bool(b)) => *b,
            Some(Value::Number(n)) => n.as_u64().unwrap_or(0) != 0,
            _ => false,
        };
        rows.push(LlamaBenchRow {
            model_type: item
                .get("model_type")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_string(),
            model_size: item.get("model_size").and_then(Value::as_u64).unwrap_or(0),
            n_prompt,
            n_gen,
            n_gpu_layers: item.get("n_gpu_layers").and_then(Value::as_u64).unwrap_or(0) as u32,
            flash_attn,
            type_k: item
                .get("type_k")
                .and_then(Value::as_str)
                .unwrap_or("f16")
                .to_string(),
            type_v: item
                .get("type_v")
                .and_then(Value::as_str)
                .unwrap_or("f16")
                .to_string(),
            avg_ts: item.get("avg_ts").and_then(Value::as_f64).unwrap_or(0.0),
            stddev_ts: item.get("stddev_ts").and_then(Value::as_f64).unwrap_or(0.0),
            avg_ns: item.get("avg_ns").and_then(Value::as_f64).unwrap_or(0.0),
            test_label,
        });
    }

    Ok(rows)
}

/// Runs `llama-bench` as an isolated child process and parses its JSON sweep results.
pub async fn run_llama_bench(
    bench_exe: &Path,
    req: &LlamaBenchRequest,
) -> Result<Vec<LlamaBenchRow>, BenchError> {
    if !bench_exe.is_file() {
        return Err(BenchError::Process(format!(
            "llama-bench binary not found: {}",
            bench_exe.display()
        )));
    }
    let args = build_llama_bench_args(req);
    let mut cmd = Command::new(bench_exe);
    cmd.args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    if let Some(dir) = bench_exe.parent() {
        cmd.current_dir(dir);
    }
    #[cfg(target_os = "windows")]
    {
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    #[cfg(target_os = "linux")]
    unsafe {
        cmd.pre_exec(|| {
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }

    let output = tokio::time::timeout(Duration::from_secs(900), cmd.output())
        .await
        .map_err(|_| BenchError::Process("llama-bench timed out after 15 minutes".to_string()))?
        .map_err(|e| BenchError::Process(format!("Failed to spawn llama-bench: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(BenchError::Process(format!(
            "llama-bench exited with status {:?}: {}",
            output.status.code(),
            stderr.chars().take(600).collect::<String>()
        )));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_llama_bench_json(&stdout)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_args_and_parses_noisy_llama_bench_json() {
        let req = LlamaBenchRequest {
            model_path: PathBuf::from("C:/models/qwen.gguf"),
            model_id: "qwen".into(),
            runtime_tag: "b5200".into(),
            prompt_sizes: vec![512, 2048],
            gen_sizes: vec![128],
            n_gpu_layers: vec![99],
            flash_attn: vec![false, true],
            cache_types_k: vec!["f16".into(), "q8_0".into()],
            cache_types_v: vec![],
            threads: Some(8),
            batch_size: None,
            ubatch_size: None,
            repetitions: 3,
            extra_args: vec!["-ncmoe 12".into()],
        };
        let args = build_llama_bench_args(&req);
        assert!(args.windows(2).any(|w| w == ["-o", "json"]));
        assert!(args.windows(2).any(|w| w == ["-p", "512,2048"]));
        assert!(args.windows(2).any(|w| w == ["-fa", "0,1"]));
        assert!(args.windows(2).any(|w| w == ["-ctk", "f16,q8_0"]));
        assert!(args.windows(2).any(|w| w == ["-ncmoe", "12"]));

        let noisy_stdout = r#"
ggml_cuda_init: found 1 CUDA devices
[
  {
    "model_type": "qwen3moe 30B.A3B Q4_K - Medium",
    "model_size": 18000000000,
    "n_prompt": 512,
    "n_gen": 0,
    "n_gpu_layers": 99,
    "flash_attn": 1,
    "type_k": "q8_0",
    "type_v": "q8_0",
    "avg_ts": 1450.25,
    "stddev_ts": 12.1,
    "avg_ns": 353000000.0
  },
  {
    "model_type": "qwen3moe 30B.A3B Q4_K - Medium",
    "model_size": 18000000000,
    "n_prompt": 0,
    "n_gen": 128,
    "n_gpu_layers": 99,
    "flash_attn": true,
    "type_k": "q8_0",
    "type_v": "q8_0",
    "avg_ts": 74.8,
    "stddev_ts": 0.9,
    "avg_ns": 1711000000.0
  }
]
"#;
        let rows = parse_llama_bench_json(noisy_stdout).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].test_label, "pp512");
        assert!(rows[0].flash_attn);
        assert_eq!(rows[0].avg_ts, 1450.25);
        assert_eq!(rows[1].test_label, "tg128");
        assert_eq!(rows[1].type_k, "q8_0");
    }
}
