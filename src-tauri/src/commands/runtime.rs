use super::hardware::compute_live_memory_context;
use crate::state::AppState;
use bibulm_core::{
    AppSettings, GuardrailLevel, InstalledRuntimeRecord, LoadConfig,
    estimate_memory_usage_with_context, parse_gguf_file,
};
use bibulm_models::DownloadJobState;
use bibulm_runtime::{
    DiscoveredCliFlag, LlamaRelease, LlamaServerMetrics, LoadedModelInfo,
    RuntimeBackendVariant, ServerLogEntry, discover_runtime_cli_flags, fetch_llama_cpp_releases,
    fetch_llama_server_metrics, install_runtime_variant, prune_old_runtimes,
    register_custom_runtime, uninstall_runtime,
};
use std::path::PathBuf;
use tauri::{AppHandle, Emitter, State};

#[tauri::command]
pub async fn fetch_github_releases(
    state: State<'_, AppState>,
) -> Result<Vec<LlamaRelease>, String> {
    let token = state.settings.read().await.github_token.clone();
    let hw_ctx = state.link_hw_snapshot.read().await.to_runtime_match_context();
    fetch_llama_cpp_releases(token.as_deref(), &hw_ctx)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn list_installed_runtimes(
    state: State<'_, AppState>,
) -> Result<Vec<InstalledRuntimeRecord>, String> {
    state.db.list_runtimes().map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn install_runtime(
    app: AppHandle,
    state: State<'_, AppState>,
    variant: RuntimeBackendVariant,
    release_notes: Option<String>,
) -> Result<InstalledRuntimeRecord, String> {
    let settings = state.settings.read().await.clone();
    let base_job = DownloadJobState::runtime_install(
        format!("llama.cpp {} ({})", variant.tag, variant.backend),
        settings.runtimes_dir.join(&variant.id).to_string_lossy().into_owned(),
        variant.total_size_bytes,
    );

    let app_clone = app.clone();
    let progress_base = base_job.clone();
    let res = install_runtime_variant(
        &state.db,
        &settings.runtimes_dir,
        &variant,
        release_notes,
        settings.github_token.as_deref(),
        move |prog| {
            let snap = progress_base.with_progress(prog.stage, prog.downloaded_bytes, prog.total_bytes);
            let _ = app_clone.emit("download-progress", &snap);
        },
    )
    .await;

    let outcome = res.as_ref().map(|rec| rec.install_dir.clone()).map_err(|e| e.to_string());
    state.downloads.record_external_job(base_job.finished(outcome)).await;
    let rec = res.map_err(|e| e.to_string())?;

    let in_use: Vec<String> = state
        .supervisor
        .list_loaded()
        .await
        .into_iter()
        .map(|m| m.runtime_id)
        .collect();
    let keep_old = {
        let mut s = state.settings.write().await;
        s.active_runtime_id = Some(rec.id.clone());
        let _ = state.db.save_settings(&s);
        s.old_runtimes_to_keep
    };
    let _ = prune_old_runtimes(&state.db, &rec, keep_old, &in_use);
    Ok(rec)
}

#[tauri::command]
pub async fn register_custom_runtime_cmd(
    state: State<'_, AppState>,
    display_name: String,
    path: String,
) -> Result<InstalledRuntimeRecord, String> {
    let rec = register_custom_runtime(&state.db, display_name, &PathBuf::from(path))
        .map_err(|e| e.to_string())?;
    let mut s = state.settings.write().await;
    s.active_runtime_id = Some(rec.id.clone());
    let _ = state.db.save_settings(&s);
    Ok(rec)
}

#[tauri::command]
pub async fn uninstall_runtime_cmd(
    state: State<'_, AppState>,
    id: String,
) -> Result<(), String> {
    uninstall_runtime(&state.db, &id).map_err(|e| e.to_string())?;
    let mut s = state.settings.write().await;
    if s.active_runtime_id.as_deref() == Some(id.as_str()) {
        s.active_runtime_id = state
            .db
            .list_runtimes()
            .ok()
            .and_then(|list| list.into_iter().next().map(|r| r.id));
        let _ = state.db.save_settings(&s);
    }
    Ok(())
}

#[tauri::command]
pub async fn set_active_runtime(
    state: State<'_, AppState>,
    runtime_id: String,
) -> Result<AppSettings, String> {
    let mut s = state.settings.write().await;
    s.active_runtime_id = Some(runtime_id);
    state.db.save_settings(&s).map_err(|e| e.to_string())?;
    Ok(s.clone())
}

#[tauri::command]
pub async fn discover_runtime_flags(
    state: State<'_, AppState>,
    runtime_id: Option<String>,
) -> Result<Vec<DiscoveredCliFlag>, String> {
    let active_id = state.settings.read().await.active_runtime_id.clone();
    let runtimes = state.db.list_runtimes().map_err(|e| e.to_string())?;
    let preferred = runtime_id.as_deref().or(active_id.as_deref());
    let runtime = preferred
        .and_then(|id| runtimes.iter().find(|r| r.id == id))
        .or_else(|| runtimes.first())
        .ok_or_else(|| "No llama.cpp runtime installed.".to_string())?;
    discover_runtime_cli_flags(&PathBuf::from(&runtime.server_exe)).await
}

#[tauri::command]
pub async fn get_loaded_model_metrics(
    state: State<'_, AppState>,
    model_id: String,
) -> Result<LlamaServerMetrics, String> {
    let loaded = state
        .supervisor
        .resolve_loaded(&model_id)
        .await
        .ok_or_else(|| format!("Model '{model_id}' is not currently loaded."))?;
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .map_err(|e| e.to_string())?;
    fetch_llama_server_metrics(&client, loaded.port).await
}

#[tauri::command]
pub async fn load_model_cmd(
    app: AppHandle,
    state: State<'_, AppState>,
    model_id: String,
    display_name: String,
    config: LoadConfig,
    remember_preset: bool,
    force_override: Option<bool>,
) -> Result<LoadedModelInfo, String> {
    if remember_preset {
        let _ = state.db.save_model_preset(&model_id, &config);
    }

    let settings = state.settings.read().await.clone();
    let runtimes = state.db.list_runtimes().map_err(|e| e.to_string())?;
    let preferred_rt = config
        .runtime_id
        .as_deref()
        .or(settings.active_runtime_id.as_deref());
    let runtime = preferred_rt
        .and_then(|id| runtimes.iter().find(|r| r.id == id))
        .or_else(|| runtimes.first())
        .cloned()
        .ok_or_else(|| {
            "No llama.cpp runtime installed. Open the Runtime tab to download an official ggml-org/llama.cpp build or add a custom llama-server binary."
                .to_string()
        })?;

    let live_ctx = compute_live_memory_context(
        &state,
        &settings.disabled_gpu_ids,
        Some(&model_id),
    )
    .await?;
    let model_path_clone = config.model_path.clone();
    let config_for_est = config.clone();
    let guardrail = settings.guardrail_level;
    let estimate = tokio::task::spawn_blocking(move || {
        parse_gguf_file(&model_path_clone).ok().map(|meta| {
            estimate_memory_usage_with_context(&meta, &config_for_est, live_ctx, guardrail)
        })
    })
    .await
    .ok()
    .flatten();

    if !force_override.unwrap_or(false)
        && let Some(ref est) = estimate
    {
        if settings.guardrail_level == GuardrailLevel::Strict
            && let Some(ref warn_msg) = est.guardrail_warning
        {
            return Err(format!("Strict memory guardrail blocked load: {warn_msg}"));
        }
        if settings.limit_offload_to_dedicated_vram
            && matches!(
                settings.guardrail_level,
                GuardrailLevel::Balanced | GuardrailLevel::Strict
            )
            && est.available_vram_mib > 0.0
            && est.total_vram_mib > est.available_vram_mib * 1.02
        {
            return Err(format!(
                "Dedicated VRAM guardrail blocked load ({:.0} MiB estimated > {:.0} MiB available). Adjust offload/context or override guardrail.",
                est.total_vram_mib, est.available_vram_mib
            ));
        }
    }

    let estimated_breakdown = estimate.map(|e| e.breakdown);

    let mut info = state
        .supervisor
        .load_model(
            model_id,
            display_name,
            runtime.id,
            runtime.tag,
            &PathBuf::from(&runtime.server_exe),
            config,
            estimated_breakdown,
            false,
            false,
        )
        .await
        .map_err(|e| e.to_string())?;

    // Reset rolling VRAM average to the new post-load baseline & immediately sample OS per-PID telemetry
    let disabled = settings.disabled_gpu_ids.clone();
    if let Some(pid) = info.pid {
        let (snap_opt, spillover, shared_mib, pid_mem) = {
            if let Ok(mut hw) = state.hardware.lock() {
                let snap = hw.reset_memory_average(&disabled);
                let pm = hw.sample_process_memory(&[pid], &disabled);
                (
                    Some(snap.clone()),
                    snap.wddm_spillover_warning,
                    snap.shared_gpu_mem_used_bytes as f64 / (1024.0 * 1024.0),
                    pm,
                )
            } else {
                (None, false, 0.0, std::collections::HashMap::new())
            }
        };
        if let Some(snap) = snap_opt {
            let _ = app.emit("telemetry-update", &snap);
        }
        let _ = state
            .supervisor
            .sync_live_telemetry(spillover, shared_mib, &pid_mem)
            .await;
        if let Some(refreshed) = state.supervisor.resolve_loaded(&info.model_id).await {
            info = refreshed;
        }
    }

    let loaded = state.supervisor.list_loaded().await;
    let _ = app.emit("loaded-models-changed", &loaded);
    state.link.notify_peers_state_changed();
    Ok(info)
}

#[tauri::command]
pub async fn unload_model_cmd(
    app: AppHandle,
    state: State<'_, AppState>,
    model_id: String,
) -> Result<bool, String> {
    let ok = state.supervisor.unload_model(&model_id).await;
    if ok {
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        let disabled = state.settings.read().await.disabled_gpu_ids.clone();
        let snap_opt = state
            .hardware
            .lock()
            .ok()
            .map(|mut hw| hw.reset_memory_average(&disabled));
        if let Some(snap) = snap_opt {
            let _ = app.emit("telemetry-update", &snap);
        }
    }
    let loaded = state.supervisor.list_loaded().await;
    let _ = app.emit("loaded-models-changed", &loaded);
    if ok {
        state.link.notify_peers_state_changed();
    }
    Ok(ok)
}

#[tauri::command]
pub async fn list_loaded_models(
    state: State<'_, AppState>,
) -> Result<Vec<LoadedModelInfo>, String> {
    Ok(state.supervisor.list_loaded().await)
}

#[tauri::command]
pub async fn get_server_logs(state: State<'_, AppState>) -> Result<Vec<ServerLogEntry>, String> {
    Ok(state.supervisor.recent_logs().await)
}

#[tauri::command]
pub async fn clear_server_logs(state: State<'_, AppState>) -> Result<(), String> {
    state.supervisor.clear_logs().await;
    Ok(())
}
