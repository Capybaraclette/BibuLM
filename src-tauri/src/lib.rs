mod commands;
mod single_instance;
mod state;
mod updater;

use bibulm_core::default_data_dir;
use single_instance::SingleInstanceGuard;
use state::AppState;
use std::sync::atomic::Ordering;
use std::time::Duration;
use tauri::menu::{Menu, MenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{Emitter, Manager, RunEvent, WebviewUrl, WebviewWindowBuilder, WindowEvent};

fn ensure_main_window(app: &tauri::AppHandle) {
    if let Some(win) = app.get_webview_window("main") {
        apply_windows_taskbar_icon(&win);
        let _ = win.unminimize();
        let _ = win.show();
        let _ = win.set_focus();
        return;
    }

    let created = if let Some(cfg) = app.config().app.windows.first() {
        WebviewWindowBuilder::from_config(app, cfg).and_then(|b| b.visible(false).build())
    } else {
        WebviewWindowBuilder::new(app, "main", WebviewUrl::default())
            .title("BibuLM")
            .inner_size(1440.0, 900.0)
            .min_inner_size(1080.0, 700.0)
            .resizable(true)
            .decorations(false)
            .visible(false)
            .build()
    };

    match created {
        Ok(win) => {
            apply_windows_taskbar_icon(&win);
            let _ = win.show();
            let _ = win.set_focus();
        }
        Err(e) => {
            tracing::error!("Failed to create main WebView window: {e}");
        }
    }
}

#[cfg(windows)]
fn init_windows_app_user_model_id() {
    unsafe {
        let _ = windows::Win32::UI::Shell::SetCurrentProcessExplicitAppUserModelID(
            windows::core::w!("com.bibulm.app"),
        );
    }
}

#[cfg(not(windows))]
fn init_windows_app_user_model_id() {}

#[cfg(windows)]
const EMBEDDED_ICO: &[u8] = include_bytes!("../icons/icon.ico");

#[cfg(windows)]
fn select_ico_entry(ico: &[u8], target_w: i32, target_h: i32) -> Option<&[u8]> {
    if ico.len() < 6 {
        return None;
    }
    let reserved = u16::from_le_bytes([ico[0], ico[1]]);
    let ico_type = u16::from_le_bytes([ico[2], ico[3]]);
    let count = u16::from_le_bytes([ico[4], ico[5]]) as usize;
    if reserved != 0 || ico_type != 1 || count == 0 {
        return None;
    }
    let table_end = 6usize.checked_add(count.checked_mul(16)?)?;
    if ico.len() < table_end {
        return None;
    }

    let target = target_w.max(target_h).max(1);
    let mut best: Option<(u8, i32, usize, usize)> = None;

    for i in 0..count {
        let base = 6 + i * 16;
        let w = if ico[base] == 0 { 256 } else { ico[base] as i32 };
        let h = if ico[base + 1] == 0 { 256 } else { ico[base + 1] as i32 };
        let size = u32::from_le_bytes([ico[base + 8], ico[base + 9], ico[base + 10], ico[base + 11]]) as usize;
        let offset = u32::from_le_bytes([ico[base + 12], ico[base + 13], ico[base + 14], ico[base + 15]]) as usize;
        let Some(end) = offset.checked_add(size) else {
            continue;
        };
        if size == 0 || end > ico.len() {
            continue;
        }

        let dim = w.max(h);
        // Priority: 0 = exact match, 1 = smallest >= target, 2 = largest < target
        let candidate = if w == target_w && h == target_h {
            (0u8, 0, offset, end)
        } else if dim >= target {
            (1u8, dim - target, offset, end)
        } else {
            (2u8, target - dim, offset, end)
        };

        let is_better = match best {
            None => true,
            Some((bt, bd, _, _)) => (candidate.0, candidate.1) < (bt, bd),
        };
        if is_better {
            best = Some(candidate);
        }
    }

    best.map(|(_, _, start, end)| &ico[start..end])
}

#[cfg(windows)]
fn load_fallback_embedded_icon(w: i32, h: i32) -> Option<isize> {
    use windows::Win32::UI::WindowsAndMessaging::{CreateIconFromResourceEx, LR_DEFAULTCOLOR};

    let bytes = select_ico_entry(EMBEDDED_ICO, w, h)?;
    match unsafe { CreateIconFromResourceEx(bytes, true, 0x00030000, w, h, LR_DEFAULTCOLOR) } {
        Ok(hicon) if !hicon.is_invalid() && !hicon.0.is_null() => Some(hicon.0 as isize),
        Ok(_) => None,
        Err(e) => {
            tracing::warn!("CreateIconFromResourceEx fallback failed for {w}x{h}: {e}");
            None
        }
    }
}

#[cfg(windows)]
fn apply_windows_taskbar_icon(win: &tauri::WebviewWindow) {
    use std::sync::OnceLock;
    use windows::Win32::Foundation::{HINSTANCE, HWND, LPARAM, WPARAM};
    use windows::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows::Win32::UI::WindowsAndMessaging::{
        GetSystemMetrics, LoadImageW, SendMessageW, ICON_BIG, ICON_SMALL, IMAGE_ICON, LR_SHARED,
        SM_CXICON, SM_CXSMICON, SM_CYICON, SM_CYSMICON, WM_SETICON,
    };
    #[cfg(target_pointer_width = "64")]
    use windows::Win32::UI::WindowsAndMessaging::{GCLP_HICON, GCLP_HICONSM, SetClassLongPtrW};
    use windows::core::PCWSTR;

    static FALLBACK_BIG_HICON: OnceLock<Option<isize>> = OnceLock::new();
    static FALLBACK_SMALL_HICON: OnceLock<Option<isize>> = OnceLock::new();

    let Ok(raw_hwnd) = win.hwnd() else {
        tracing::warn!("Failed to get HWND for taskbar icon");
        return;
    };
    let hwnd = HWND(raw_hwnd.0);
    let hinst = match unsafe { GetModuleHandleW(PCWSTR::null()) } {
        Ok(m) => Some(HINSTANCE(m.0)),
        Err(e) => {
            tracing::warn!("Failed to get module handle for taskbar icon: {e}");
            None
        }
    };
    let res_id = PCWSTR(32512usize as *const u16);

    let big_w = unsafe { GetSystemMetrics(SM_CXICON) }.max(0);
    let big_h = unsafe { GetSystemMetrics(SM_CYICON) }.max(0);
    let (big_w, big_h) = (
        if big_w > 0 { big_w } else { 32 },
        if big_h > 0 { big_h } else { 32 },
    );
    let small_w = unsafe { GetSystemMetrics(SM_CXSMICON) }.max(0);
    let small_h = unsafe { GetSystemMetrics(SM_CYSMICON) }.max(0);
    let (small_w, small_h) = (
        if small_w > 0 { small_w } else { 16 },
        if small_h > 0 { small_h } else { 16 },
    );

    // LoadImageW with LR_SHARED and OnceLock-cached fallback handles live for the process lifetime;
    // do not call DestroyIcon on them while assigned to the window/class.
    let targets = [
        (ICON_BIG, GCLP_HICON, big_w, big_h, &FALLBACK_BIG_HICON, "ICON_BIG"),
        (ICON_SMALL, GCLP_HICONSM, small_w, small_h, &FALLBACK_SMALL_HICON, "ICON_SMALL"),
    ];
    for (icon_type, class_index, w, h, fallback_cache, label) in targets {
        let primary_hicon = hinst.and_then(|inst| {
            match unsafe { LoadImageW(inst, res_id, IMAGE_ICON, w, h, LR_SHARED) } {
                Ok(hicon) if !hicon.is_invalid() && !hicon.0.is_null() => Some(hicon.0 as isize),
                Ok(_) => {
                    tracing::warn!("LoadImageW returned invalid handle for {label}");
                    None
                }
                Err(e) => {
                    tracing::warn!("Failed to load embedded {label} resource: {e}");
                    None
                }
            }
        });

        let resolved_hicon = primary_hicon
            .or_else(|| *fallback_cache.get_or_init(|| load_fallback_embedded_icon(w, h)));

        if let Some(raw_hicon) = resolved_hicon {
            unsafe {
                #[cfg(target_pointer_width = "64")]
                {
                    let _ = SetClassLongPtrW(hwnd, class_index, raw_hicon);
                }
                let _ = SendMessageW(
                    hwnd,
                    WM_SETICON,
                    WPARAM(icon_type as usize),
                    LPARAM(raw_hicon),
                );
            }
        } else {
            tracing::warn!("Could not resolve {label} from resource or embedded ICO fallback");
        }
    }
}

#[cfg(not(windows))]
fn apply_windows_taskbar_icon(_win: &tauri::WebviewWindow) {}

#[cfg(windows)]
fn trim_working_set() {
    use windows::Win32::System::Threading::{GetCurrentProcess, SetProcessWorkingSetSize};
    // Passing usize::MAX (-1) trims cold init pages (e.g. NVML) out of the active working set.
    unsafe {
        let _ = SetProcessWorkingSetSize(GetCurrentProcess(), usize::MAX, usize::MAX);
    }
}

#[cfg(not(windows))]
fn trim_working_set() {}

fn show_or_create_main_window(app: &tauri::AppHandle) {
    let handle = app.clone();
    let _ = app.run_on_main_thread(move || ensure_main_window(&handle));
}

pub fn run() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info,bibulm=debug")
        .try_init();

    init_windows_app_user_model_id();

    // Enforce single running BibuLM instance: if one is already running, focus it and exit.
    let mut instance_guard = match SingleInstanceGuard::acquire(&default_data_dir()) {
        Some(g) => g,
        None => {
            tracing::info!("Another BibuLM instance is already running — focusing it and exiting.");
            return;
        }
    };

    let app_state = AppState::initialize().expect("Failed to initialize BibuLM core state");
    trim_working_set();

    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .manage(app_state)
        .setup(move |app| {
            let handle = app.handle().clone();

            // Wake and focus (or recreate) main window when a user tries to launch a second instance
            let focus_handle = handle.clone();
            instance_guard.spawn_wakeup_listener(move || {
                show_or_create_main_window(&focus_handle);
            });
            app.manage(instance_guard);

            // Setup System Tray (minimize to tray while OpenAI API server continues running)
            setup_system_tray(&handle)?;

            // Start background telemetry loop (every 1.5s) + per-PID VRAM/RAM & WDDM spillover sync
            let tel_handle = handle.clone();
            tauri::async_runtime::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_millis(1500));
                let mut last_memory_epoch = tel_handle.state::<AppState>().supervisor.memory_epoch();
                loop {
                    interval.tick().await;
                    let state = tel_handle.state::<AppState>();
                    let has_window = tel_handle.get_webview_window("main").is_some();
                    let cur_epoch = state.supervisor.memory_epoch();
                    let epoch_changed = cur_epoch != last_memory_epoch;
                    if epoch_changed {
                        last_memory_epoch = cur_epoch;
                    }
                    let (disabled, link_enabled) = {
                        let s = state.settings.read().await;
                        (s.disabled_gpu_ids.clone(), s.link.enabled)
                    };
                    let pids = state.supervisor.loaded_pids().await;
                    if !has_window && pids.is_empty() && !link_enabled && !epoch_changed {
                        continue;
                    }
                    let sample_res = {
                        match state.hardware.lock() {
                            Ok(mut hw) => {
                                let snap = if epoch_changed {
                                    hw.reset_memory_average(&disabled)
                                } else {
                                    hw.sample(&disabled)
                                };
                                let pid_mem = hw.sample_process_memory(&pids, &disabled);
                                let budget = hw.memory_budget(&disabled);
                                Some((snap, pid_mem, budget))
                            }
                            Err(_) => None,
                        }
                    };
                    if let Some((s, pid_mem, budget)) = sample_res {
                        let shared_mib = s.shared_gpu_mem_used_bytes as f64 / (1024.0 * 1024.0);
                        let changed = state
                            .supervisor
                            .sync_live_telemetry(s.wddm_spillover_warning, shared_mib, &pid_mem)
                            .await;
                        state.link_hw_snapshot.write().await.apply_sample(&s, &budget);
                        if tel_handle.get_webview_window("main").is_some() {
                            let _ = tel_handle.emit("telemetry-update", &s);
                            if changed || epoch_changed {
                                let loaded = state.supervisor.list_loaded().await;
                                let _ = tel_handle.emit("loaded-models-changed", &loaded);
                            }
                        }
                        if epoch_changed {
                            state.link.notify_peers_state_changed();
                        }
                    }
                }
            });

            // Automatic background app update check on launch (non-blocking)
            let upd_handle = handle.clone();
            tauri::async_runtime::spawn(async move {
                tokio::time::sleep(Duration::from_secs(4)).await;
                let state = upd_handle.state::<AppState>();
                let (auto_check, gh_token) = {
                    let s = state.settings.read().await;
                    (s.auto_check_app_updates, s.github_token.clone())
                };
                if auto_check
                    && let Ok(info) = updater::check_for_app_update(gh_token.as_deref()).await
                    && info.update_available
                {
                    let _ = upd_handle.emit("app-update-available", &info);
                }
            });

            // Forward real-time server & llama-server logs to UI + buffered async rotating daily log file
            let log_handle = handle.clone();
            tauri::async_runtime::spawn(async move {
                let state = log_handle.state::<AppState>();
                let mut rx = state.supervisor.subscribe_logs();
                let (logs_dir, retention_days) = {
                    let s = state.settings.read().await;
                    (s.logs_dir.clone(), s.log_retention_days)
                };
                let (file_tx, file_rx) =
                    tokio::sync::mpsc::channel::<bibulm_runtime::ServerLogEntry>(2048);
                let logs_dir_worker = logs_dir.clone();
                tokio::task::spawn_blocking(move || {
                    run_rotating_log_writer(logs_dir_worker, retention_days, file_rx);
                });
                while let Ok(entry) = rx.recv().await {
                    let _ = log_handle.emit("server-log", &entry);
                    let _ = file_tx.try_send(entry);
                }
            });

            // Forward real-time download progress events to UI
            let dl_handle = handle.clone();
            tauri::async_runtime::spawn(async move {
                let state = dl_handle.state::<AppState>();
                let mut rx = state.downloads.subscribe();
                while let Ok(job) = rx.recv().await {
                    if job.status == bibulm_models::JobStatus::Completed
                        && job.kind == bibulm_models::JobKind::Model {
                        state.gateway.invalidate_models_cache().await;
                        state.link.notify_peers_state_changed();
                    }
                    let _ = dl_handle.emit("download-progress", &job);
                }
            });

            // Forward real-time BibuLM Link status & peer events to UI
            let link_st_handle = handle.clone();
            tauri::async_runtime::spawn(async move {
                let state = link_st_handle.state::<AppState>();
                let mut rx = state.link.subscribe_status();
                while let Ok(st) = rx.recv().await {
                    let _ = link_st_handle.emit("link-status-changed", &st);
                }
            });

            let link_peers_handle = handle.clone();
            tauri::async_runtime::spawn(async move {
                let state = link_peers_handle.state::<AppState>();
                let mut rx = state.link.subscribe_peers();
                while let Ok(peers) = rx.recv().await {
                    let _ = link_peers_handle.emit("link-peers-changed", &peers);
                }
            });

            // Auto-start OpenAI API server if enabled, then auto-connect Link if enabled
            let srv_handle = handle.clone();
            tauri::async_runtime::spawn(async move {
                let state = srv_handle.state::<AppState>();
                let (auto_start, link_cfg) = {
                    let s = state.settings.read().await;
                    (s.auto_start_server, s.link.clone())
                };
                if (auto_start || (link_cfg.enabled && link_cfg.auto_connect))
                    && let Ok(status) = state.gateway.start().await
                {
                    let _ = srv_handle.emit("server-status-changed", &status);
                }
                if link_cfg.enabled && link_cfg.auto_connect {
                    let port = state.gateway.active_port().await;
                    if let Ok(st) = state.link.connect(&link_cfg, port).await {
                        let _ = srv_handle.emit("link-status-changed", &st);
                    }
                }
            });

            // Create main WebView window only if not configured to start minimized to tray
            let start_min = {
                let state = handle.state::<AppState>();
                state
                    .settings
                    .try_read()
                    .map(|s| s.start_minimized)
                    .unwrap_or(false)
            };
            if !start_min {
                ensure_main_window(&handle);
            }

            Ok(())
        })
        .on_window_event(|window, event| {
            match event {
                WindowEvent::CloseRequested { .. } => {
                    let state = window.app_handle().state::<AppState>();
                    let minimize_to_tray = state
                        .settings
                        .try_read()
                        .map(|s| s.minimize_to_tray)
                        .unwrap_or(true);
                    if minimize_to_tray {
                        // Allow the WebView window to close & terminate WebView2 processes;
                        // RunEvent::ExitRequested keeps the headless Rust backend alive in tray.
                        tracing::info!("Main window closed to tray — terminating WebView processes.");
                    } else {
                        state.allow_exit.store(true, Ordering::Release);
                        let sup = state.supervisor.clone();
                        let link = state.link.clone();
                        let link_cfg = state
                            .settings
                            .try_read()
                            .map(|s| s.link.clone())
                            .unwrap_or_default();
                        tauri::async_runtime::block_on(async move {
                            link.disconnect(&link_cfg).await;
                            sup.unload_all().await;
                        });
                    }
                }
                WindowEvent::Destroyed if window.label() == "main" => {
                    trim_working_set();
                }
                _ => {}
            }
        })
        .invoke_handler(tauri::generate_handler![
            commands::get_hardware_profile,
            commands::refresh_hardware_profile,
            commands::get_telemetry_snapshot,
            commands::format_hardware_summary,
            commands::reset_vram_usage_average,
            commands::get_app_version,
            commands::check_app_update,
            commands::install_app_update,
            commands::get_settings,
            commands::save_settings,
            commands::get_model_preset,
            commands::list_named_presets,
            commands::save_named_preset,
            commands::delete_named_preset,
            commands::scan_local_library,
            commands::list_discovered_model_sources,
            commands::check_draft_model_compatibility,
            commands::estimate_model_memory,
            commands::estimate_model_memory_with_context,
            commands::compute_auto_fit_moe,
            commands::compute_auto_fit_moe_with_context,
            commands::preview_llama_cli,
            commands::delete_local_model_files,
            commands::reveal_in_explorer,
            commands::search_hf_models,
            commands::get_hf_model_detail_cmd,
            commands::enqueue_model_download,
            commands::list_downloads,
            commands::pause_download,
            commands::resume_download,
            commands::cancel_download,
            commands::clear_completed_downloads,
            commands::fetch_github_releases,
            commands::list_installed_runtimes,
            commands::install_runtime,
            commands::register_custom_runtime_cmd,
            commands::uninstall_runtime_cmd,
            commands::set_active_runtime,
            commands::discover_runtime_flags,
            commands::get_loaded_model_metrics,
            commands::load_model_cmd,
            commands::read_text_file_preview,
            commands::unload_model_cmd,
            commands::list_loaded_models,
            commands::get_server_logs,
            commands::clear_server_logs,
            commands::get_server_status,
            commands::start_server,
            commands::stop_server,
            commands::list_api_tokens,
            commands::create_api_token,
            commands::delete_api_token,
            commands::set_api_token_enabled,
            commands::run_quick_benchmark,
            commands::run_playground_turn_cmd,
            commands::run_llama_bench_matrix,
            commands::list_benchmark_history,
            commands::delete_benchmark_run,
            commands::export_benchmarks_csv,
            commands::get_link_status,
            commands::link_connect,
            commands::link_disconnect,
            commands::link_logout,
            commands::list_link_peers,
            commands::refresh_link_peers,
            commands::remote_inspect_model,
            commands::remote_load_model,
            commands::remote_unload_model,
            commands::remote_list_runtimes,
            commands::remote_fetch_runtime_releases,
            commands::remote_install_runtime,
            commands::remote_set_active_runtime,
            commands::remote_uninstall_runtime,
            commands::remote_enqueue_model_download,
            commands::remote_list_downloads,
            commands::remote_pause_download,
            commands::remote_resume_download,
            commands::remote_cancel_download,
            commands::remote_clear_completed_downloads,
            commands::remote_delete_model,
            commands::remote_get_logs,
        ])
        .build(tauri::generate_context!())
        .expect("error while building BibuLM");

    app.run(|app_handle, event| {
        if let RunEvent::ExitRequested { api, code, .. } = event {
            let allow_exit = app_handle
                .state::<AppState>()
                .allow_exit
                .load(Ordering::Acquire);
            if code.is_none() && !allow_exit {
                api.prevent_exit();
            }
        }
    });
}

fn setup_system_tray(app: &tauri::AppHandle) -> tauri::Result<()> {
    let show_item = MenuItem::with_id(app, "show", "Open BibuLM", true, None::<&str>)?;
    let unload_item =
        MenuItem::with_id(app, "unload_all", "Unload All Models", true, None::<&str>)?;
    let quit_item = MenuItem::with_id(app, "quit", "Quit BibuLM", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&show_item, &unload_item, &quit_item])?;

    let mut builder = TrayIconBuilder::with_id("bibulm-tray")
        .menu(&menu)
        .tooltip("BibuLM — Local LLM Workbench")
        .show_menu_on_left_click(false)
        .on_menu_event(|app, event| match event.id().as_ref() {
            "show" => {
                show_or_create_main_window(app);
            }
            "unload_all" => {
                let handle = app.clone();
                tauri::async_runtime::spawn(async move {
                    let state = handle.state::<AppState>();
                    state.supervisor.unload_all().await;
                    state.link.notify_peers_state_changed();
                    tokio::time::sleep(Duration::from_millis(250)).await;
                    let disabled = state.settings.read().await.disabled_gpu_ids.clone();
                    let snap_opt = state
                        .hardware
                        .lock()
                        .ok()
                        .map(|mut hw| hw.reset_memory_average(&disabled));
                    if let Some(snap) = snap_opt {
                        let _ = handle.emit("telemetry-update", &snap);
                    }
                    let _ = handle.emit(
                        "loaded-models-changed",
                        Vec::<bibulm_runtime::LoadedModelInfo>::new(),
                    );
                });
            }
            "quit" => {
                let handle = app.clone();
                tauri::async_runtime::spawn(async move {
                    let state = handle.state::<AppState>();
                    state.allow_exit.store(true, Ordering::Release);
                    let link_cfg = state.settings.read().await.link.clone();
                    state.link.disconnect(&link_cfg).await;
                    state.gateway.stop().await;
                    state.supervisor.unload_all().await;
                    handle.exit(0);
                });
            }
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                show_or_create_main_window(tray.app_handle());
            }
        });

    if let Some(icon) = app.default_window_icon() {
        builder = builder.icon(icon.clone());
    }
    builder.build(app)?;
    Ok(())
}

fn run_rotating_log_writer(
    logs_dir: std::path::PathBuf,
    retention_days: u32,
    mut rx: tokio::sync::mpsc::Receiver<bibulm_runtime::ServerLogEntry>,
) {
    use std::io::{BufWriter, Write};

    prune_old_log_files(&logs_dir, retention_days);
    let mut current_date = String::new();
    let mut writer: Option<BufWriter<std::fs::File>> = None;

    while let Some(entry) = rx.blocking_recv() {
        let date = chrono::Utc::now().format("%Y-%m-%d").to_string();
        if date != current_date || writer.is_none() {
            if let Some(ref mut w) = writer {
                let _ = w.flush();
            }
            if !current_date.is_empty() {
                prune_old_log_files(&logs_dir, retention_days);
            }
            current_date = date.clone();
            let path = logs_dir.join(format!("bibulm-{date}.log"));
            writer = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .ok()
                .map(BufWriter::new);
        }

        if let Some(ref mut w) = writer {
            let _ = writeln!(
                w,
                "{} [{}] [{}] {}",
                entry.timestamp, entry.level, entry.source, entry.message
            );
            if rx.is_empty() {
                let _ = w.flush();
            }
        }
    }
    if let Some(ref mut w) = writer {
        let _ = w.flush();
    }
}

fn prune_old_log_files(logs_dir: &std::path::Path, retention_days: u32) {
    if retention_days == 0 || !logs_dir.is_dir() {
        return;
    }
    let cutoff = chrono::Utc::now().date_naive() - chrono::Duration::days(retention_days as i64);
    let Ok(entries) = std::fs::read_dir(logs_dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if let Some(date_str) = name
            .strip_prefix("bibulm-")
            .and_then(|s| s.strip_suffix(".log"))
            && let Ok(file_date) = chrono::NaiveDate::parse_from_str(date_str, "%Y-%m-%d")
            && file_date < cutoff
        {
            let _ = std::fs::remove_file(&path);
        }
    }
}
