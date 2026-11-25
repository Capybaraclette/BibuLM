use crate::state::AppState;
use crate::updater::{
    AppUpdateInfo, check_for_app_update, current_app_version, download_and_install_update,
};
use bibulm_core::{AppSettings, LoadConfig, NamedPresetRecord};
use tauri::{AppHandle, State};

#[tauri::command]
pub fn get_app_version() -> String {
    current_app_version().to_string()
}

#[tauri::command]
pub async fn check_app_update(state: State<'_, AppState>) -> Result<AppUpdateInfo, String> {
    let token = state.settings.read().await.github_token.clone();
    check_for_app_update(token.as_deref()).await
}

#[tauri::command]
pub async fn install_app_update(
    app: AppHandle,
    state: State<'_, AppState>,
    update_info: AppUpdateInfo,
) -> Result<(), String> {
    let token = state.settings.read().await.github_token.clone();
    download_and_install_update(&app, &state, update_info, token.as_deref()).await
}

#[tauri::command]
pub async fn get_settings(state: State<'_, AppState>) -> Result<AppSettings, String> {
    Ok(state.settings.read().await.clone())
}

#[tauri::command]
pub async fn save_settings(
    state: State<'_, AppState>,
    settings: AppSettings,
) -> Result<AppSettings, String> {
    let _ = std::fs::create_dir_all(settings.primary_models_dir());
    let _ = std::fs::create_dir_all(&settings.runtimes_dir);
    let _ = std::fs::create_dir_all(&settings.logs_dir);

    state.db.save_settings(&settings).map_err(|e| e.to_string())?;
    let prev_server = state.settings.read().await.server.clone();
    *state.settings.write().await = settings.clone();
    state.gateway.invalidate_models_cache().await;

    // Restart server automatically if port or LAN bind changed while server is running
    let status = state.gateway.status().await;
    if status.running
        && (prev_server.port != settings.server.port
            || prev_server.serve_on_lan != settings.server.serve_on_lan)
    {
        let _ = state.gateway.start().await;
    }

    // Keep running BibuLM Link sidecar synced with updated policy/host sharing/port
    let active_port = state.gateway.active_port().await;
    state.link.sync_config(&settings.link, active_port).await;

    Ok(settings)
}

#[tauri::command]
pub fn get_model_preset(
    state: State<'_, AppState>,
    model_id: String,
) -> Result<Option<LoadConfig>, String> {
    state
        .db
        .get_model_preset(&model_id)
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn list_named_presets(
    state: State<'_, AppState>,
    kind: Option<String>,
) -> Result<Vec<NamedPresetRecord>, String> {
    state
        .db
        .list_named_presets(kind.as_deref())
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn save_named_preset(
    state: State<'_, AppState>,
    id: Option<String>,
    name: String,
    kind: String,
    payload_json: String,
) -> Result<NamedPresetRecord, String> {
    state
        .db
        .upsert_named_preset(id, name, kind, payload_json)
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn delete_named_preset(state: State<'_, AppState>, id: String) -> Result<(), String> {
    state
        .db
        .delete_named_preset(&id)
        .map_err(|e| e.to_string())
}
