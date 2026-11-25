use crate::state::AppState;
use chrono::Utc;
use bibulm_bench::{
    LlamaBenchRequest, LlamaBenchRow, PlaygroundTurnRequest, PlaygroundTurnResult,
    QuickTestRequest, QuickTestResult, run_llama_bench, run_playground_turn,
    run_quick_speed_test, run_quick_speed_test_against_url,
};
use bibulm_core::{BenchmarkRunRecord, LoadConfig, MemoryPlacementReport};
use bibulm_link::LINK_SECRET_HEADER;
use bibulm_runtime::resolve_bench_exe;
use std::path::PathBuf;
use tauri::{AppHandle, Emitter, State};
use uuid::Uuid;

#[tauri::command]
pub async fn run_quick_benchmark(
    app: AppHandle,
    state: State<'_, AppState>,
    model_id: String,
    request: QuickTestRequest,
) -> Result<QuickTestResult, String> {
    if let Some(loaded) = state.supervisor.resolve_loaded(&model_id).await {
        let disabled = state.settings.read().await.disabled_gpu_ids.clone();
        let (vram_mib, ram_mib) = {
            let mut hw = state.hardware.lock().map_err(|_| "Hardware mutex poisoned")?;
            let snap = hw.sample(&disabled);
            (
                snap.vram_used_bytes as f64 / (1024.0 * 1024.0),
                snap.ram_used_bytes as f64 / (1024.0 * 1024.0),
            )
        };

        state.supervisor.begin_request(&loaded.model_id).await;
        let app_stream = app.clone();
        let result = run_quick_speed_test(
            loaded.port,
            loaded.model_id.clone(),
            loaded.runtime_tag.clone(),
            loaded.load_config.clone(),
            loaded.placement.clone(),
            vram_mib,
            ram_mib,
            request,
            move |chunk| {
                let _ = app_stream.emit("bench-stream-chunk", &chunk);
            },
        )
        .await;
        state.supervisor.end_request(&loaded.model_id).await;

        let res = result.map_err(|e| e.to_string())?;
        let _ = state.db.insert_benchmark_run(&res.record);
        return Ok(res);
    }

    // Fallback: benchmark a remote BibuLM Link peer model
    let (peer, remote_model_id) = state
        .link
        .resolve_remote_model(&model_id)
        .await
        .ok_or_else(|| format!("Model '{model_id}' is not loaded locally or on any Link peer."))?;

    let (endpoint_url, secret) = state
        .link
        .proxy_endpoint(&peer.tailscale_ip, "/v1/chat/completions")
        .await?;

    let app_stream = app.clone();
    let headers = [(LINK_SECRET_HEADER, secret.as_str())];
    let record_model_id = format!("{}/{}", peer.slug, remote_model_id);
    let result = run_quick_speed_test_against_url(
        &endpoint_url,
        &headers,
        remote_model_id,
        record_model_id,
        format!("link:{}", peer.device_name),
        LoadConfig::new(PathBuf::from(format!("link://{}", peer.slug))),
        MemoryPlacementReport::default(),
        peer.vram_used_mib,
        peer.ram_used_mib,
        request,
        move |chunk| {
            let _ = app_stream.emit("bench-stream-chunk", &chunk);
        },
    )
    .await;

    let res = result.map_err(|e| e.to_string())?;
    let _ = state.db.insert_benchmark_run(&res.record);
    Ok(res)
}

#[tauri::command]
pub async fn run_playground_turn_cmd(
    app: AppHandle,
    state: State<'_, AppState>,
    model_id: String,
    request: PlaygroundTurnRequest,
) -> Result<PlaygroundTurnResult, String> {
    if let Some(loaded) = state.supervisor.resolve_loaded(&model_id).await {
        let url = format!("http://127.0.0.1:{}/v1/chat/completions", loaded.port);
        state.supervisor.begin_request(&loaded.model_id).await;
        let app_stream = app.clone();
        let res = run_playground_turn(
            &url,
            &[],
            loaded.model_id.clone(),
            request,
            move |chunk| {
                let _ = app_stream.emit("playground-stream-chunk", &chunk);
            },
        )
        .await;
        state.supervisor.end_request(&loaded.model_id).await;
        return res.map_err(|e| e.to_string());
    }

    let (peer, remote_model_id) = state
        .link
        .resolve_remote_model(&model_id)
        .await
        .ok_or_else(|| format!("Model '{model_id}' is not loaded locally or on any Link peer."))?;
    let (endpoint_url, secret) = state
        .link
        .proxy_endpoint(&peer.tailscale_ip, "/v1/chat/completions")
        .await?;
    let headers = [(LINK_SECRET_HEADER, secret.as_str())];
    let app_stream = app.clone();
    run_playground_turn(
        &endpoint_url,
        &headers,
        remote_model_id,
        request,
        move |chunk| {
            let _ = app_stream.emit("playground-stream-chunk", &chunk);
        },
    )
    .await
    .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn run_llama_bench_matrix(
    state: State<'_, AppState>,
    runtime_id: Option<String>,
    request: LlamaBenchRequest,
) -> Result<Vec<LlamaBenchRow>, String> {
    let active_rt_id = state.settings.read().await.active_runtime_id.clone();
    let runtimes = state.db.list_runtimes().map_err(|e| e.to_string())?;
    let preferred = runtime_id.as_deref().or(active_rt_id.as_deref());
    let runtime = preferred
        .and_then(|id| runtimes.iter().find(|r| r.id == id))
        .or_else(|| runtimes.first())
        .ok_or_else(|| "No llama.cpp runtime installed.".to_string())?;

    let bench_exe = resolve_bench_exe(runtime).ok_or_else(|| {
        format!(
            "llama-bench executable was not found in runtime '{}' ({}).",
            runtime.tag, runtime.install_dir
        )
    })?;

    let rows = run_llama_bench(&bench_exe, &request)
        .await
        .map_err(|e| e.to_string())?;

    // Persist representative summary into SQLite benchmark history
    let best_pp = rows
        .iter()
        .filter(|r| r.n_prompt > 0 && r.n_gen == 0)
        .max_by(|a, b| a.avg_ts.total_cmp(&b.avg_ts));
    let best_tg = rows
        .iter()
        .filter(|r| r.n_gen > 0)
        .max_by(|a, b| a.avg_ts.total_cmp(&b.avg_ts));

    if best_pp.is_some() || best_tg.is_some() {
        let mut cfg = LoadConfig::new(&request.model_path);
        cfg.gpu_layers = request.n_gpu_layers.first().copied();
        let summary_preset = format!(
            "llama-bench matrix ({} configs)",
            rows.len()
        );
        let rec = BenchmarkRunRecord {
            id: Uuid::new_v4().to_string(),
            created_at: Utc::now().to_rfc3339(),
            model_id: request.model_id.clone(),
            runtime_tag: runtime.tag.clone(),
            kind: "llama_bench".to_string(),
            prompt_preset: summary_preset,
            prompt_tokens: best_pp.map(|r| r.n_prompt).unwrap_or(0),
            reasoning_tokens: 0,
            output_tokens: best_tg.map(|r| r.n_gen).unwrap_or(0),
            ttft_ms: 0.0,
            prompt_eval_tok_s: best_pp.map(|r| r.avg_ts).unwrap_or(0.0),
            generation_tok_s: best_tg.map(|r| r.avg_ts).unwrap_or(0.0),
            peak_vram_mib: 0.0,
            peak_ram_mib: 0.0,
            placement: MemoryPlacementReport::default(),
            load_config: cfg,
            notes: Some(format!(
                "pp={:.1} tok/s, tg={:.1} tok/s",
                best_pp.map(|r| r.avg_ts).unwrap_or(0.0),
                best_tg.map(|r| r.avg_ts).unwrap_or(0.0)
            )),
        };
        let _ = state.db.insert_benchmark_run(&rec);
    }

    Ok(rows)
}

#[tauri::command]
pub fn list_benchmark_history(
    state: State<'_, AppState>,
    limit: Option<usize>,
) -> Result<Vec<BenchmarkRunRecord>, String> {
    state
        .db
        .list_benchmark_runs(limit.unwrap_or(50))
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn delete_benchmark_run(state: State<'_, AppState>, id: String) -> Result<(), String> {
    state
        .db
        .delete_benchmark_run(&id)
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn export_benchmarks_csv(state: State<'_, AppState>) -> Result<String, String> {
    let runs = state
        .db
        .list_benchmark_runs(200)
        .map_err(|e| e.to_string())?;
    let mut csv = String::from(
        "created_at,model_id,runtime_tag,prompt_preset,placement,vram_mib,ram_mib,kv_vram_mib,prompt_tokens,reasoning_tokens,output_tokens,ttft_ms,prompt_eval_tok_s,generation_tok_s,peak_vram_mib,peak_ram_mib\n",
    );
    for r in runs {
        let b = r.placement.resolved_breakdown();
        csv.push_str(&format!(
            "{},{},{},{},{:?},{:.1},{:.1},{:.1},{},{},{},{:.1},{:.2},{:.2},{:.1},{:.1}\n",
            r.created_at,
            r.model_id.replace(',', "_"),
            r.runtime_tag,
            r.prompt_preset.replace(',', "_"),
            r.placement.status(),
            b.total_vram_mib(),
            b.total_ram_mib(),
            b.kv_cache_vram_mib,
            r.prompt_tokens,
            r.reasoning_tokens,
            r.output_tokens,
            r.ttft_ms,
            r.prompt_eval_tok_s,
            r.generation_tok_s,
            r.peak_vram_mib,
            r.peak_ram_mib
        ));
    }
    Ok(csv)
}
