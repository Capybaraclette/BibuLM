use crate::state::AppState;
use bibulm_core::LiveMemoryContext;
use bibulm_hardware::{LiveMemoryBudget, SystemHardwareProfile, TelemetrySnapshot};
use tauri::{AppHandle, Emitter, State};

#[tauri::command]
pub fn get_hardware_profile(state: State<'_, AppState>) -> Result<SystemHardwareProfile, String> {
    let hw = state.hardware.lock().map_err(|_| "Hardware mutex poisoned")?;
    Ok(hw.profile().clone())
}

#[tauri::command]
pub fn refresh_hardware_profile(
    state: State<'_, AppState>,
) -> Result<SystemHardwareProfile, String> {
    let mut hw = state.hardware.lock().map_err(|_| "Hardware mutex poisoned")?;
    Ok(hw.refresh_profile())
}

#[tauri::command]
pub async fn get_telemetry_snapshot(
    state: State<'_, AppState>,
) -> Result<TelemetrySnapshot, String> {
    let disabled = state.settings.read().await.disabled_gpu_ids.clone();
    let mut hw = state.hardware.lock().map_err(|_| "Hardware mutex poisoned")?;
    Ok(hw.sample(&disabled))
}

#[tauri::command]
pub fn format_hardware_summary(state: State<'_, AppState>) -> Result<String, String> {
    let hw = state.hardware.lock().map_err(|_| "Hardware mutex poisoned")?;
    Ok(hw.profile().format_summary())
}

#[tauri::command]
pub async fn reset_vram_usage_average(
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<LiveMemoryBudget, String> {
    let disabled = state.settings.read().await.disabled_gpu_ids.clone();
    let (snap, budget) = {
        let mut hw = state.hardware.lock().map_err(|_| "Hardware mutex poisoned")?;
        let snap = hw.reset_memory_average(&disabled);
        let budget = hw.memory_budget(&disabled);
        (snap, budget)
    };
    let _ = app.emit("telemetry-update", &snap);
    Ok(budget)
}

pub(crate) async fn compute_live_memory_context(
    state: &AppState,
    disabled_gpu_ids: &[u32],
    replacing_model_id: Option<&str>,
) -> Result<LiveMemoryContext, String> {
    let budget = {
        let mut hw = state.hardware.lock().map_err(|_| "Hardware mutex poisoned")?;
        hw.memory_budget(disabled_gpu_ids)
    };

    // If re-configuring a model that is already loaded, reloading it will unload the current
    // instance first — add its own VRAM/RAM back into the available headroom.
    let (reclaim_vram, reclaim_ram) = if let Some(mid) = replacing_model_id {
        if let Some(loaded) = state.supervisor.resolve_loaded(mid).await {
            let v = loaded
                .placement
                .process_vram_mib
                .filter(|&x| x > 1.0)
                .unwrap_or_else(|| loaded.placement.total_gpu_mib())
                .max(0.0);
            let r = loaded
                .placement
                .process_ram_mib
                .filter(|&x| x > 1.0)
                .unwrap_or_else(|| loaded.placement.total_cpu_mib())
                .max(0.0);
            (v, r)
        } else {
            (0.0, 0.0)
        }
    } else {
        (0.0, 0.0)
    };

    let effective_vram_used = (budget.vram_used_avg_mib - reclaim_vram).max(0.0);
    let effective_vram_avail = (budget.vram_total_mib - effective_vram_used)
        .clamp(0.0, budget.vram_total_mib);
    let effective_ram_used = (budget.ram_used_avg_mib - reclaim_ram).max(0.0);
    let effective_ram_avail = (budget.ram_total_mib - effective_ram_used)
        .clamp(0.0, budget.ram_total_mib);

    Ok(LiveMemoryContext {
        hardware_vram_total_mib: budget.vram_total_mib,
        current_vram_used_mib: effective_vram_used,
        available_vram_mib: effective_vram_avail,
        hardware_ram_total_mib: budget.ram_total_mib,
        current_ram_used_mib: effective_ram_used,
        available_ram_mib: effective_ram_avail,
        vram_sample_count: budget.sample_count,
    })
}
