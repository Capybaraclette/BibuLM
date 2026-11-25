use super::hardware::compute_live_memory_context;
use crate::state::AppState;
use chrono::Utc;
use bibulm_core::{
    AppSettings, GuardrailLevel, InstalledRuntimeRecord, LoadConfig,
    estimate_memory_usage_with_context, parse_gguf_file,
};
use bibulm_models::{DownloadJobState, JobKind, JobStatus};
use bibulm_runtime::{
    DiscoveredCliFlag, HardwareMatchContext, LlamaRelease, LlamaServerMetrics, LoadedModelInfo,
    RuntimeBackendVariant, ServerLogEntry, discover_runtime_cli_flags, fetch_llama_cpp_releases,
    fetch_llama_server_metrics, install_runtime_variant, register_custom_runtime, uninstall_runtime,
};
use std::path::PathBuf;
use tauri::{AppHandle, Emitter, State};
use uuid::Uuid;

#[tauri::command]
pub async fn fetch_github_releases(
    state: State<'_, AppState>,
) -> Result<Vec<LlamaRelease>, String> {
    let token = state.settings.read().await.github_token.clone();
    let hw_ctx = {
        let hw = state.hardware.lock().map_err(|_| "Hardware mutex poisoned")?;
        let prof = hw.profile();
        let nv = prof.gpus.iter().find(|g| g.vendor == "NVIDIA");
        HardwareMatchContext {
            has_nvidia: nv.is_some(),
            nvidia_cuda_major: nv.and_then(|g| g.cuda_version_major),
            nvidia_cuda_minor: nv.and_then(|g| g.cuda_version_minor),
            has_amd_gpu: prof.gpus.iter().any(|g| g.vendor == "AMD"),
            has_intel_gpu: prof.gpus.iter().any(|g| g.vendor == "Intel"),
            has_any_gpu: !prof.gpus.is_empty(),
            has_avx512: prof.cpu.has_avx512,
            has_avx2: prof.cpu.has_avx2,
        }
    };
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
    let job_id = format!("rt-{}", Uuid::new_v4());
    let title = format!("llama.cpp {} ({})", variant.tag, variant.backend);
    let created_at = Utc::now().to_rfc3339();
    let total_bytes = variant.total_size_bytes;
    let dest_dir = settings.runtimes_dir.join(&variant.id);

    let downloads = state.downloads.clone();
    let app_clone = app.clone();
    let job_id_c = job_id.clone();
    let title_c = title.clone();
    let created_c = created_at.clone();
    let dest_str = dest_dir.to_string_lossy().into_owned();

    let res = install_runtime_variant(
        &state.db,
        &settings.runtimes_dir,
        &variant,
        release_notes,
        settings.github_token.as_deref(),
        move |prog| {
            let snap = DownloadJobState {
                id: job_id_c.clone(),
                kind: JobKind::Runtime,
                title: title_c.clone(),
                repo_id: Some("ggml-org/llama.cpp".to_string()),
                dest_dir: dest_str.clone(),
                files: Vec::new(),
                current_file: Some(prog.stage),
                total_bytes: prog.total_bytes,
                downloaded_bytes: prog.downloaded_bytes,
                speed_bytes_per_sec: 0.0,
                eta_seconds: None,
                status: JobStatus::Downloading,
                error: None,
                created_at: created_c.clone(),
                completed_at: None,
            };
            let _ = app_clone.emit("download-progress", &snap);
        },
    )
    .await;

    match res {
        Ok(rec) => {
            downloads
                .record_external_job(DownloadJobState {
                    id: job_id,
                    kind: JobKind::Runtime,
                    title,
                    repo_id: Some("ggml-org/llama.cpp".to_string()),
                    dest_dir: rec.install_dir.clone(),
                    files: Vec::new(),
                    current_file: None,
                    total_bytes,
                    downloaded_bytes: total_bytes,
                    speed_bytes_per_sec: 0.0,
                    eta_seconds: None,
                    status: JobStatus::Completed,
                    error: None,
                    created_at,
                    completed_at: Some(Utc::now().to_rfc3339()),
                })
                .await;

            // Automatically select as active runtime if none is set yet
            let mut s = state.settings.write().await;
            if s.active_runtime_id.is_none() {
                s.active_runtime_id = Some(rec.id.clone());
                let _ = state.db.save_settings(&s);
            }
            Ok(rec)
        }
        Err(e) => {
            downloads
                .record_external_job(DownloadJobState {
                    id: job_id,
                    kind: JobKind::Runtime,
                    title,
                    repo_id: Some("ggml-org/llama.cpp".to_string()),
                    dest_dir: dest_dir.to_string_lossy().into_owned(),
                    files: Vec::new(),
                    current_file: None,
                    total_bytes,
                    downloaded_bytes: 0,
                    speed_bytes_per_sec: 0.0,
                    eta_seconds: None,
                    status: JobStatus::Failed,
                    error: Some(e.to_string()),
                    created_at,
                    completed_at: None,
                })
                .await;
            Err(e.to_string())
        }
    }
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
    if s.active_runtime_id.is_none() {
        s.active_runtime_id = Some(rec.id.clone());
        let _ = state.db.save_settings(&s);
    }
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
            "No llama.cpp runtime installed. Open the Runtime tab to download an official ggml-org/llama.cpp build or add a custom llama-server.exe."
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
