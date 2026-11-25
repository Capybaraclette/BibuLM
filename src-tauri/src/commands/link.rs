use crate::state::AppState;
use bibulm_core::LoadConfig;
use bibulm_link::{
    LinkStatus, RemoteActionResponse, RemoteInstallRuntimeResponse, RemoteLoadResponse,
    RemoteModelInspectResponse, RemotePeer, RemoteRuntimesResponse,
};
use bibulm_models::{DownloadJobState, HfDownloadFile};
use bibulm_runtime::{LlamaRelease, RuntimeBackendVariant, ServerLogEntry};
use tauri::{AppHandle, Emitter, State};

#[tauri::command]
pub async fn get_link_status(state: State<'_, AppState>) -> Result<LinkStatus, String> {
    Ok(state.link.status().await)
}

#[tauri::command]
pub async fn link_connect(
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<LinkStatus, String> {
    let gw_status = state.gateway.status().await;
    if !gw_status.running
        && let Ok(started) = state.gateway.start().await
    {
        let _ = app.emit("server-status-changed", &started);
    }
    let active_port = state.gateway.active_port().await;

    let link_cfg = {
        let mut s = state.settings.write().await;
        s.link.enabled = true;
        let _ = state.db.save_settings(&s);
        s.link.clone()
    };

    let status = state.link.connect(&link_cfg, active_port).await?;
    let _ = app.emit("link-status-changed", &status);
    Ok(status)
}

#[tauri::command]
pub async fn link_disconnect(
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<LinkStatus, String> {
    let link_cfg = {
        let mut s = state.settings.write().await;
        s.link.enabled = false;
        let _ = state.db.save_settings(&s);
        s.link.clone()
    };
    let status = state.link.disconnect(&link_cfg).await;
    let _ = app.emit("link-status-changed", &status);
    let _ = app.emit("link-peers-changed", Vec::<RemotePeer>::new());
    Ok(status)
}

#[tauri::command]
pub async fn link_logout(
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<LinkStatus, String> {
    let link_cfg = {
        let mut s = state.settings.write().await;
        s.link.enabled = false;
        let _ = state.db.save_settings(&s);
        s.link.clone()
    };
    let status = state.link.logout(&link_cfg).await?;
    let _ = app.emit("link-status-changed", &status);
    let _ = app.emit("link-peers-changed", Vec::<RemotePeer>::new());
    Ok(status)
}

#[tauri::command]
pub async fn list_link_peers(state: State<'_, AppState>) -> Result<Vec<RemotePeer>, String> {
    Ok(state.link.peers().await)
}

#[tauri::command]
pub async fn refresh_link_peers(
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<Vec<RemotePeer>, String> {
    let peers = state.link.refresh_peers().await?;
    let _ = app.emit("link-peers-changed", &peers);
    Ok(peers)
}

#[tauri::command]
pub async fn remote_inspect_model(
    state: State<'_, AppState>,
    peer_id: String,
    model_id: String,
) -> Result<RemoteModelInspectResponse, String> {
    state.link.remote_inspect_model(&peer_id, &model_id).await
}

#[tauri::command]
pub async fn remote_load_model(
    app: AppHandle,
    state: State<'_, AppState>,
    peer_id: String,
    model_id: String,
    config: Option<LoadConfig>,
    remember_preset: Option<bool>,
) -> Result<RemoteLoadResponse, String> {
    let res = state
        .link
        .remote_load_model(&peer_id, &model_id, config, remember_preset.unwrap_or(false))
        .await?;
    let peers = state.link.peers().await;
    let _ = app.emit("link-peers-changed", &peers);
    Ok(res)
}

#[tauri::command]
pub async fn remote_unload_model(
    app: AppHandle,
    state: State<'_, AppState>,
    peer_id: String,
    model_id: String,
) -> Result<RemoteLoadResponse, String> {
    let res = state.link.remote_unload_model(&peer_id, &model_id).await?;
    let peers = state.link.peers().await;
    let _ = app.emit("link-peers-changed", &peers);
    Ok(res)
}

// ---------------------------------------------------------------
// Remote runtime / model management on managed Link peers
// ---------------------------------------------------------------

#[tauri::command]
pub async fn remote_list_runtimes(
    state: State<'_, AppState>,
    peer_id: String,
) -> Result<RemoteRuntimesResponse, String> {
    state.link.remote_list_runtimes(&peer_id).await
}

#[tauri::command]
pub async fn remote_fetch_runtime_releases(
    state: State<'_, AppState>,
    peer_id: String,
) -> Result<Vec<LlamaRelease>, String> {
    state.link.remote_fetch_runtime_releases(&peer_id).await
}

#[tauri::command]
pub async fn remote_install_runtime(
    state: State<'_, AppState>,
    peer_id: String,
    variant: RuntimeBackendVariant,
    release_notes: Option<String>,
) -> Result<RemoteInstallRuntimeResponse, String> {
    state
        .link
        .remote_install_runtime(&peer_id, variant, release_notes)
        .await
}

#[tauri::command]
pub async fn remote_set_active_runtime(
    app: AppHandle,
    state: State<'_, AppState>,
    peer_id: String,
    runtime_id: String,
) -> Result<RemoteActionResponse, String> {
    let res = state
        .link
        .remote_set_active_runtime(&peer_id, &runtime_id)
        .await?;
    let _ = app.emit("link-peers-changed", &state.link.peers().await);
    Ok(res)
}

#[tauri::command]
pub async fn remote_uninstall_runtime(
    app: AppHandle,
    state: State<'_, AppState>,
    peer_id: String,
    runtime_id: String,
) -> Result<RemoteActionResponse, String> {
    let res = state
        .link
        .remote_uninstall_runtime(&peer_id, &runtime_id)
        .await?;
    let _ = app.emit("link-peers-changed", &state.link.peers().await);
    Ok(res)
}

#[tauri::command]
pub async fn remote_enqueue_model_download(
    state: State<'_, AppState>,
    peer_id: String,
    title: String,
    repo_id: String,
    files: Vec<HfDownloadFile>,
) -> Result<DownloadJobState, String> {
    state
        .link
        .remote_enqueue_model_download(&peer_id, title, repo_id, files)
        .await
}

#[tauri::command]
pub async fn remote_list_downloads(
    state: State<'_, AppState>,
    peer_id: String,
) -> Result<Vec<DownloadJobState>, String> {
    state.link.remote_list_downloads(&peer_id).await
}

#[tauri::command]
pub async fn remote_pause_download(
    state: State<'_, AppState>,
    peer_id: String,
    id: String,
) -> Result<bool, String> {
    state.link.remote_pause_download(&peer_id, &id).await
}

#[tauri::command]
pub async fn remote_resume_download(
    state: State<'_, AppState>,
    peer_id: String,
    id: String,
) -> Result<bool, String> {
    state.link.remote_resume_download(&peer_id, &id).await
}

#[tauri::command]
pub async fn remote_cancel_download(
    state: State<'_, AppState>,
    peer_id: String,
    id: String,
) -> Result<bool, String> {
    state.link.remote_cancel_download(&peer_id, &id).await
}

#[tauri::command]
pub async fn remote_clear_completed_downloads(
    state: State<'_, AppState>,
    peer_id: String,
) -> Result<(), String> {
    state.link.remote_clear_completed_downloads(&peer_id).await
}

#[tauri::command]
pub async fn remote_delete_model(
    app: AppHandle,
    state: State<'_, AppState>,
    peer_id: String,
    model_id: String,
) -> Result<RemoteActionResponse, String> {
    let res = state.link.remote_delete_model(&peer_id, &model_id).await?;
    let _ = app.emit("link-peers-changed", &state.link.peers().await);
    Ok(res)
}

#[tauri::command]
pub async fn remote_get_logs(
    state: State<'_, AppState>,
    peer_id: String,
) -> Result<Vec<ServerLogEntry>, String> {
    state.link.remote_get_logs(&peer_id).await
}
