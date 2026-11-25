use crate::state::AppState;
use bibulm_core::ApiTokenRecord;
use bibulm_server::GatewayStatus;
use tauri::{AppHandle, Emitter, State};

#[tauri::command]
pub async fn get_server_status(state: State<'_, AppState>) -> Result<GatewayStatus, String> {
    Ok(state.gateway.status().await)
}

#[tauri::command]
pub async fn start_server(
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<GatewayStatus, String> {
    let status = state.gateway.start().await?;
    let _ = app.emit("server-status-changed", &status);
    Ok(status)
}

#[tauri::command]
pub async fn stop_server(
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<GatewayStatus, String> {
    let status = state.gateway.stop().await;
    let _ = app.emit("server-status-changed", &status);
    Ok(status)
}

#[tauri::command]
pub fn list_api_tokens(state: State<'_, AppState>) -> Result<Vec<ApiTokenRecord>, String> {
    state.db.list_api_tokens().map_err(|e| e.to_string())
}

#[tauri::command]
pub fn create_api_token(
    state: State<'_, AppState>,
    name: String,
) -> Result<ApiTokenRecord, String> {
    state.db.create_api_token(&name).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn delete_api_token(state: State<'_, AppState>, id: String) -> Result<(), String> {
    state.db.delete_api_token(&id).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn set_api_token_enabled(
    state: State<'_, AppState>,
    id: String,
    enabled: bool,
) -> Result<(), String> {
    state
        .db
        .set_api_token_enabled(&id, enabled)
        .map_err(|e| e.to_string())
}
