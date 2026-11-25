use super::hardware::compute_live_memory_context;
use crate::state::AppState;
use bibulm_core::{
    DraftCompatibilityReport, ExternalModelSource, GgufMetadata, GuardrailLevel, LiveMemoryContext,
    LoadConfig, MemoryEstimate, MoeOffloadMode, auto_fit_moe, auto_fit_moe_with_sizes,
    check_draft_compatibility, default_data_dir, discover_external_model_sources,
    estimate_memory_usage_with_context, estimate_memory_usage_with_sizes, parse_gguf_file,
};
use bibulm_models::{
    DownloadJobState, HfDownloadFile, HfModelDetail, HfModelSummary, HfSortOrder, LocalModelEntry,
    delete_local_model, get_hf_model_detail, scan_local_models_with_external,
    search_hf_gguf_models,
};
use bibulm_runtime::{build_llama_server_args, format_cli_preview};
use std::path::PathBuf;
use tauri::State;

#[tauri::command]
pub async fn scan_local_library(state: State<'_, AppState>) -> Result<Vec<LocalModelEntry>, String> {
    state.gateway.invalidate_models_cache().await;
    let (dirs, auto_discover) = {
        let s = state.settings.read().await;
        (s.models_dirs.clone(), s.auto_discover_external_models)
    };
    tokio::task::spawn_blocking(move || scan_local_models_with_external(&dirs, auto_discover))
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn list_discovered_model_sources() -> Result<Vec<ExternalModelSource>, String> {
    tokio::task::spawn_blocking(discover_external_model_sources)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn inspect_gguf_file(path: String) -> Result<GgufMetadata, String> {
    let p = PathBuf::from(path);
    tokio::task::spawn_blocking(move || parse_gguf_file(&p))
        .await
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn check_draft_model_compatibility(
    main_path: String,
    draft_path: String,
) -> Result<DraftCompatibilityReport, String> {
    let main_p = PathBuf::from(main_path);
    let draft_p = PathBuf::from(draft_path);
    tokio::task::spawn_blocking(move || {
        let main_meta = parse_gguf_file(&main_p).map_err(|e| e.to_string())?;
        let draft_meta = parse_gguf_file(&draft_p).map_err(|e| e.to_string())?;
        Ok(check_draft_compatibility(&main_meta, &draft_meta))
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
pub async fn estimate_model_memory(
    state: State<'_, AppState>,
    gguf: GgufMetadata,
    config: LoadConfig,
    replacing_model_id: Option<String>,
) -> Result<MemoryEstimate, String> {
    let settings = state.settings.read().await.clone();
    let live_ctx = compute_live_memory_context(
        &state,
        &settings.disabled_gpu_ids,
        replacing_model_id.as_deref(),
    )
    .await?;
    Ok(estimate_memory_usage_with_context(
        &gguf,
        &config,
        live_ctx,
        settings.guardrail_level,
    ))
}

#[tauri::command]
pub async fn estimate_model_memory_with_context(
    gguf: GgufMetadata,
    config: LoadConfig,
    live_ctx: LiveMemoryContext,
    guardrail: GuardrailLevel,
    mmproj_size_bytes: Option<u64>,
    draft_size_bytes: Option<u64>,
) -> Result<MemoryEstimate, String> {
    Ok(estimate_memory_usage_with_sizes(
        &gguf,
        &config,
        live_ctx,
        guardrail,
        mmproj_size_bytes,
        draft_size_bytes,
    ))
}

#[tauri::command]
pub async fn compute_auto_fit_moe(
    state: State<'_, AppState>,
    gguf: GgufMetadata,
    config: LoadConfig,
    replacing_model_id: Option<String>,
) -> Result<MoeOffloadMode, String> {
    let disabled = state.settings.read().await.disabled_gpu_ids.clone();
    let live_ctx =
        compute_live_memory_context(&state, &disabled, replacing_model_id.as_deref()).await?;
    let headroom = if live_ctx.current_vram_used_mib > 0.0 {
        512.0
    } else {
        768.0
    };
    Ok(auto_fit_moe(
        &gguf,
        &config,
        live_ctx.available_vram_mib,
        headroom,
    ))
}

#[tauri::command]
pub async fn compute_auto_fit_moe_with_context(
    gguf: GgufMetadata,
    config: LoadConfig,
    live_ctx: LiveMemoryContext,
    mmproj_size_bytes: Option<u64>,
) -> Result<MoeOffloadMode, String> {
    let headroom = if live_ctx.current_vram_used_mib > 0.0 {
        512.0
    } else {
        768.0
    };
    Ok(auto_fit_moe_with_sizes(
        &gguf,
        &config,
        live_ctx.available_vram_mib,
        headroom,
        mmproj_size_bytes,
    ))
}

#[tauri::command]
pub async fn preview_llama_cli(
    state: State<'_, AppState>,
    config: LoadConfig,
) -> Result<String, String> {
    let active_rt_id = config.runtime_id.clone().or_else(|| {
        state
            .settings
            .try_read()
            .ok()
            .and_then(|s| s.active_runtime_id.clone())
    });
    let runtimes = state.db.list_runtimes().unwrap_or_default();
    let server_exe = active_rt_id
        .as_deref()
        .and_then(|id| runtimes.iter().find(|r| r.id == id))
        .or_else(|| runtimes.first())
        .map(|r| PathBuf::from(&r.server_exe));

    let args = build_llama_server_args(&config, "127.0.0.1", 51000);
    Ok(format_cli_preview(server_exe.as_deref(), &args))
}

#[tauri::command]
pub async fn delete_local_model_files(
    state: State<'_, AppState>,
    path: String,
) -> Result<usize, String> {
    let p = PathBuf::from(path);
    let deleted = tokio::task::spawn_blocking(move || delete_local_model(&p))
        .await
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())?;
    state.gateway.invalidate_models_cache().await;
    Ok(deleted)
}

#[tauri::command]
pub fn reveal_in_explorer(path: String) -> Result<(), String> {
    let p = PathBuf::from(&path);
    #[cfg(target_os = "windows")]
    {
        if p.is_file() {
            std::process::Command::new("explorer.exe")
                .arg(format!("/select,{}", p.to_string_lossy()))
                .spawn()
                .map_err(|e| e.to_string())?;
        } else {
            let dir = if p.is_dir() {
                p
            } else {
                p.parent().map(PathBuf::from).unwrap_or(p)
            };
            std::process::Command::new("explorer.exe")
                .arg(dir)
                .spawn()
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = p;
        Ok(())
    }
}

#[tauri::command]
pub async fn read_text_file_preview(
    state: State<'_, AppState>,
    path: String,
) -> Result<String, String> {
    use std::io::Read;
    let settings = state.settings.read().await.clone();
    let p = PathBuf::from(path);

    tokio::task::spawn_blocking(move || {
        let canon = std::fs::canonicalize(&p).map_err(|e| e.to_string())?;
        if !canon.is_file() {
            return Err("Target path is not a file.".to_string());
        }
        let ext_ok = canon
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|e| {
                matches!(
                    e.to_ascii_lowercase().as_str(),
                    "jinja" | "j2" | "txt" | "tmpl" | "json" | "gbnf"
                )
            });
        if !ext_ok {
            return Err("Only template/text files (.jinja, .j2, .txt, .tmpl, .json, .gbnf) can be previewed.".to_string());
        }

        let templates_dir = default_data_dir().join("templates");
        let mut allowed_roots = settings.models_dirs.clone();
        allowed_roots.push(templates_dir);
        if settings.auto_discover_external_models {
            for ext in discover_external_model_sources() {
                allowed_roots.push(ext.path);
            }
        }
        let in_allowed_root = bibulm_core::canonical_within(&canon, &allowed_roots).is_some();
        if !in_allowed_root {
            return Err("Access denied: file is outside configured model/template directories.".to_string());
        }

        let mut f = std::fs::File::open(&canon).map_err(|e| e.to_string())?;
        let mut buf = Vec::with_capacity(8192);
        f.by_ref()
            .take(64 * 1024)
            .read_to_end(&mut buf)
            .map_err(|e| e.to_string())?;
        Ok(String::from_utf8_lossy(&buf).into_owned())
    })
    .await
    .map_err(|e| e.to_string())?
}

// ========================
// HuggingFace Model Search & Downloads
// ========================

#[tauri::command]
pub async fn search_hf_models(
    state: State<'_, AppState>,
    query: String,
    sort: HfSortOrder,
    extra_filter: Option<String>,
) -> Result<Vec<HfModelSummary>, String> {
    let token = state.settings.read().await.hf_token.clone();
    search_hf_gguf_models(&query, sort, extra_filter.as_deref(), 30, token.as_deref())
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn get_hf_model_detail_cmd(
    state: State<'_, AppState>,
    repo_id: String,
    target_vram_bytes: Option<u64>,
    target_ram_bytes: Option<u64>,
) -> Result<HfModelDetail, String> {
    let settings = state.settings.read().await.clone();
    let (avail_vram, avail_ram) = if let (Some(v), Some(r)) = (target_vram_bytes, target_ram_bytes)
    {
        (v, r)
    } else {
        let hw = state.hardware.lock().map_err(|_| "Hardware mutex poisoned")?;
        let prof = hw.profile();
        let vram: u64 = prof
            .gpus
            .iter()
            .filter(|g| !settings.disabled_gpu_ids.contains(&g.device_id))
            .map(|g| g.vram_total_bytes)
            .sum();
        (vram, prof.ram_total_bytes)
    };
    get_hf_model_detail(&repo_id, settings.hf_token.as_deref(), avail_vram, avail_ram)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn enqueue_model_download(
    state: State<'_, AppState>,
    title: String,
    repo_id: String,
    files: Vec<HfDownloadFile>,
) -> Result<DownloadJobState, String> {
    let settings = state.settings.read().await.clone();
    let safe_sub: PathBuf = repo_id
        .split('/')
        .map(|seg| {
            seg.chars()
                .map(|c| {
                    if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                        c
                    } else {
                        '_'
                    }
                })
                .collect::<String>()
        })
        .collect();
    let dest_dir = settings.primary_models_dir().join(safe_sub);

    Ok(state
        .downloads
        .enqueue_model_download(title, repo_id, dest_dir, files, settings.hf_token)
        .await)
}

#[tauri::command]
pub async fn list_downloads(state: State<'_, AppState>) -> Result<Vec<DownloadJobState>, String> {
    Ok(state.downloads.list_jobs().await)
}

#[tauri::command]
pub async fn pause_download(state: State<'_, AppState>, id: String) -> Result<bool, String> {
    Ok(state.downloads.pause(&id).await)
}

#[tauri::command]
pub async fn resume_download(state: State<'_, AppState>, id: String) -> Result<bool, String> {
    let tok = state.settings.read().await.hf_token.clone();
    Ok(state.downloads.resume_or_retry(&id, tok).await)
}

#[tauri::command]
pub async fn cancel_download(state: State<'_, AppState>, id: String) -> Result<bool, String> {
    Ok(state.downloads.cancel(&id).await)
}

#[tauri::command]
pub async fn clear_completed_downloads(state: State<'_, AppState>) -> Result<(), String> {
    state.downloads.clear_completed().await;
    Ok(())
}
