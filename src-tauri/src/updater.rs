use crate::state::AppState;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use tauri::{AppHandle, Emitter};
use tokio::io::AsyncWriteExt;

const APP_RELEASES_API_URL: &str =
    "https://api.github.com/repos/Capybaraclette/bibulm/releases/latest";
const APP_RELEASES_PAGE_URL: &str = "https://github.com/Capybaraclette/bibulm/releases";

pub fn current_app_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppUpdateInfo {
    pub current_version: String,
    pub latest_version: String,
    pub update_available: bool,
    pub release_name: String,
    pub release_notes: String,
    pub published_at: String,
    pub html_url: String,
    pub asset_name: Option<String>,
    pub asset_url: Option<String>,
    pub asset_size_bytes: u64,
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppUpdateProgress {
    /// `"downloading" | "verifying" | "installing" | "error"`
    pub stage: String,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub percent: f32,
    pub message: String,
}

#[derive(Debug, Deserialize)]
struct GhRelease {
    tag_name: String,
    name: Option<String>,
    published_at: Option<String>,
    html_url: String,
    body: Option<String>,
    draft: bool,
    prerelease: bool,
    assets: Vec<GhAsset>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct GhAsset {
    name: String,
    browser_download_url: String,
    size: u64,
}

fn build_http_client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .user_agent(format!(
            "BibuLM/{} ({}; App Auto-Updater)",
            current_app_version(),
            std::env::consts::OS
        ))
        .build()
        .map_err(|e| format!("Failed to initialize HTTP client: {e}"))
}

/// Checks GitHub Releases for a newer version of BibuLM.
pub async fn check_for_app_update(github_token: Option<&str>) -> Result<AppUpdateInfo, String> {
    let client = build_http_client()?;
    let mut req = client
        .get(APP_RELEASES_API_URL)
        .header("Accept", "application/vnd.github+json");
    if let Some(tok) = github_token.map(str::trim).filter(|s| !s.is_empty()) {
        req = req.bearer_auth(tok);
    }

    let resp = req
        .send()
        .await
        .map_err(|e| format!("Unable to reach GitHub Releases: {e}"))?;
    let status = resp.status();

    // 404 simply means no GitHub Release has been published yet.
    if status.as_u16() == 404 {
        let cur = current_app_version().to_string();
        return Ok(AppUpdateInfo {
            current_version: cur.clone(),
            latest_version: cur,
            update_available: false,
            release_name: "No releases published yet".to_string(),
            release_notes: String::new(),
            published_at: String::new(),
            html_url: APP_RELEASES_PAGE_URL.to_string(),
            asset_name: None,
            asset_url: None,
            asset_size_bytes: 0,
            sha256: None,
        });
    }

    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("GitHub API error ({}): {}", status.as_u16(), body));
    }

    let gh: GhRelease = resp
        .json()
        .await
        .map_err(|e| format!("Invalid GitHub release payload: {e}"))?;

    let current = current_app_version().to_string();
    let latest_clean = normalize_version(&gh.tag_name);
    let update_available =
        !gh.draft && !gh.prerelease && is_newer_version(&current, &latest_clean);

    let chosen_asset = select_installer_asset(&gh.assets);
    let sha256 = if let Some(asset) = chosen_asset {
        fetch_release_asset_sha256(&client, &gh.assets, &asset.name, github_token).await
    } else {
        None
    };

    Ok(AppUpdateInfo {
        current_version: current,
        latest_version: latest_clean,
        update_available,
        release_name: gh.name.filter(|s| !s.trim().is_empty()).unwrap_or(gh.tag_name),
        release_notes: gh.body.unwrap_or_default(),
        published_at: gh.published_at.unwrap_or_default(),
        html_url: gh.html_url,
        asset_name: chosen_asset.map(|a| a.name.clone()),
        asset_url: chosen_asset.map(|a| a.browser_download_url.clone()),
        asset_size_bytes: chosen_asset.map(|a| a.size).unwrap_or(0),
        sha256,
    })
}

/// Downloads the verified platform installer, stops running models & server, launches
/// installation + automatic relaunch, and exits the current application process.
pub async fn download_and_install_update(
    app: &AppHandle,
    state: &AppState,
    _untrusted_info: AppUpdateInfo,
    github_token: Option<&str>,
) -> Result<(), String> {
    // Re-fetch release metadata from GitHub instead of trusting the webview payload.
    let info = check_for_app_update(github_token).await?;
    if !info.update_available {
        return Err("No newer official release is currently available.".to_string());
    }

    let asset_url = info
        .asset_url
        .as_deref()
        .filter(|u| !u.trim().is_empty())
        .ok_or_else(|| "This release does not contain a compatible installer for this platform.".to_string())?;
    let raw_name = info
        .asset_name
        .as_deref()
        .filter(|n| !n.trim().is_empty())
        .ok_or_else(|| "Missing installer filename in update metadata.".to_string())?;

    if !is_trusted_github_download_url(asset_url) {
        return Err(format!("Refusing untrusted update URL: {asset_url}"));
    }
    if info
        .sha256
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .is_none()
    {
        return Err(
            "Refusing to install update without a verified SHA-256 checksum from SHA256SUMS.txt."
                .to_string(),
        );
    }

    let safe_name = sanitize_installer_filename(raw_name)?;
    let update_dir = bibulm_core::default_data_dir().join("updates");
    std::fs::create_dir_all(&update_dir)
        .map_err(|e| format!("Failed to create private update folder: {e}"))?;
    let installer_path = update_dir.join(&safe_name);

    emit_progress(
        app,
        "downloading",
        0,
        info.asset_size_bytes,
        format!("Downloading {safe_name}..."),
    );

    let client = build_http_client()?;
    let mut req = client.get(asset_url);
    if let Some(tok) = github_token.map(str::trim).filter(|s| !s.is_empty()) {
        req = req.bearer_auth(tok);
    }

    let resp = req
        .send()
        .await
        .map_err(|e| format!("Failed to download update installer: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!(
            "Installer download failed with HTTP {}",
            resp.status().as_u16()
        ));
    }

    let total_bytes = resp.content_length().unwrap_or(info.asset_size_bytes);
    let mut file = tokio::fs::File::create(&installer_path)
        .await
        .map_err(|e| format!("Cannot create installer file: {e}"))?;
    let mut stream = resp.bytes_stream();
    let mut hasher = Sha256::new();
    let mut downloaded: u64 = 0;
    let mut last_emit = std::time::Instant::now();

    while let Some(chunk_res) = stream.next().await {
        let chunk = chunk_res.map_err(|e| format!("Download interrupted: {e}"))?;
        file.write_all(&chunk)
            .await
            .map_err(|e| format!("Failed writing installer chunk: {e}"))?;
        hasher.update(&chunk);
        downloaded = downloaded.saturating_add(chunk.len() as u64);

        if last_emit.elapsed().as_millis() >= 120 || downloaded == total_bytes {
            emit_progress(
                app,
                "downloading",
                downloaded,
                total_bytes,
                format!("Downloading {safe_name}..."),
            );
            last_emit = std::time::Instant::now();
        }
    }
    file.flush()
        .await
        .map_err(|e| format!("Failed flushing installer file: {e}"))?;
    drop(file);

    // Mandatory SHA-256 verification against SHA256SUMS.txt / <asset>.sha256
    emit_progress(
        app,
        "verifying",
        downloaded,
        total_bytes.max(downloaded),
        "Verifying SHA-256 integrity...",
    );
    let actual_hex = format!("{:x}", hasher.finalize());
    if let Err(msg) = verify_required_sha256(info.sha256.as_deref(), &actual_hex, &safe_name) {
        let _ = std::fs::remove_file(&installer_path);
        emit_progress(app, "error", downloaded, total_bytes, msg.clone());
        return Err(msg);
    }

    emit_progress(
        app,
        "installing",
        downloaded,
        total_bytes.max(downloaded),
        "Stopping models and launching installer...",
    );

    // Gracefully stop Link sidecar, local API gateway & unload all llama-server child processes before installing
    let link_cfg = state.settings.read().await.link.clone();
    state.link.disconnect(&link_cfg).await;
    state.gateway.stop().await;
    state.supervisor.unload_all().await;

    if let Err(msg) = install_and_relaunch(&installer_path).await {
        emit_progress(app, "error", downloaded, total_bytes, msg.clone());
        return Err(msg);
    }

    // Give the UI a brief beat to render "installing" state, then exit cleanly so the installer can replace binaries
    state
        .allow_exit
        .store(true, std::sync::atomic::Ordering::Release);
    let exit_handle = app.clone();
    tauri::async_runtime::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        exit_handle.exit(0);
    });

    Ok(())
}

fn emit_progress(
    app: &AppHandle,
    stage: &str,
    downloaded_bytes: u64,
    total_bytes: u64,
    message: impl Into<String>,
) {
    let percent = if total_bytes > 0 {
        ((downloaded_bytes as f64 / total_bytes as f64) * 100.0).clamp(0.0, 100.0) as f32
    } else if stage == "installing" || stage == "verifying" {
        100.0
    } else {
        0.0
    };
    let _ = app.emit(
        "app-update-progress",
        &AppUpdateProgress {
            stage: stage.to_string(),
            downloaded_bytes,
            total_bytes,
            percent,
            message: message.into(),
        },
    );
}

#[cfg(target_os = "windows")]
fn build_windows_update_script(
    current_pid: u32,
    installer_path: &std::path::Path,
    current_exe: &std::path::Path,
    default_nsis_exe: &std::path::Path,
) -> String {
    let is_msi = installer_path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("msi"));

    let esc = |p: &std::path::Path| p.to_string_lossy().replace('\'', "''");
    let installer_ps = esc(installer_path);
    let current_exe_ps = esc(current_exe);
    let default_exe_ps = esc(default_nsis_exe);

    let install_cmd = if is_msi {
        format!(
            "Start-Process -FilePath 'msiexec.exe' -ArgumentList @('/i', '{installer_ps}', '/passive', '/norestart') -Wait"
        )
    } else {
        // NSIS setup supports /S for silent update into the existing install directory.
        format!("Start-Process -FilePath '{installer_ps}' -ArgumentList '/S' -Wait")
    };

    // Leading UTF-8 BOM ensures Windows PowerShell 5.1 decodes non-ASCII user paths correctly.
    format!(
        "\u{FEFF}$ErrorActionPreference = 'SilentlyContinue'\n\
try {{\n\
    Wait-Process -Id {current_pid} -Timeout 15 -ErrorAction SilentlyContinue\n\
}} catch {{}}\n\
Start-Sleep -Milliseconds 600\n\
{install_cmd}\n\
try {{\n\
    Add-Type -Namespace BibuLMWin32 -Name ShellNotify -MemberDefinition '[System.Runtime.InteropServices.DllImport(\"shell32.dll\")] public static extern void SHChangeNotify(int wEventId, uint uFlags, System.IntPtr dwItem1, System.IntPtr dwItem2);'\n\
    [BibuLMWin32.ShellNotify]::SHChangeNotify(0x08000000, 0x0000, [System.IntPtr]::Zero, [System.IntPtr]::Zero)\n\
}} catch {{}}\n\
Start-Sleep -Milliseconds 500\n\
if ('{current_exe_ps}' -and (Test-Path '{current_exe_ps}')) {{\n\
    Start-Process -FilePath '{current_exe_ps}'\n\
}} elseif ('{default_exe_ps}' -and (Test-Path '{default_exe_ps}')) {{\n\
    Start-Process -FilePath '{default_exe_ps}'\n\
}}\n\
Remove-Item -Path '{installer_ps}' -Force -ErrorAction SilentlyContinue\n\
Remove-Item -Path $PSCommandPath -Force -ErrorAction SilentlyContinue\n"
    )
}

#[cfg(target_os = "windows")]
fn spawn_windows_updater_and_relaunch(installer_path: &std::path::Path) -> Result<(), String> {
    use std::os::windows::process::CommandExt;
    use std::process::Stdio;

    let current_pid = std::process::id();
    let current_exe = std::env::current_exe().unwrap_or_else(|_| PathBuf::new());
    let default_nsis_exe = std::env::var_os("LOCALAPPDATA")
        .map(|d| PathBuf::from(d).join("BibuLM").join("bibulm-app.exe"))
        .unwrap_or_default();

    let script_path = installer_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .join("apply-update.ps1");

    let ps_script =
        build_windows_update_script(current_pid, installer_path, &current_exe, &default_nsis_exe);

    std::fs::write(&script_path, ps_script)
        .map_err(|e| format!("Failed writing update helper script: {e}"))?;

    // Do NOT combine DETACHED_PROCESS with CREATE_NO_WINDOW: powershell.exe exits with 0 without running -File.
    const CREATE_NO_WINDOW: u32 = 0x08000000;

    std::process::Command::new("powershell.exe")
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-WindowStyle",
            "Hidden",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
        ])
        .arg(&script_path)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .creation_flags(CREATE_NO_WINDOW)
        .spawn()
        .map_err(|e| format!("Failed to launch update installer helper: {e}"))?;

    Ok(())
}

#[cfg(target_os = "windows")]
async fn install_and_relaunch(installer_path: &Path) -> Result<(), String> {
    spawn_windows_updater_and_relaunch(installer_path)
}

#[cfg(unix)]
fn sh_single_quote(p: &Path) -> String {
    format!("'{}'", p.to_string_lossy().replace('\'', "'\\''"))
}

#[cfg(any(unix, test))]
fn build_linux_appimage_update_script(
    current_pid: u32,
    installer_path: &Path,
    target_appimage: &Path,
) -> String {
    let esc = |p: &Path| format!("'{}'", p.to_string_lossy().replace('\'', "'\\''"));
    let src = esc(installer_path);
    let dst = esc(target_appimage);
    format!(
        "#!/bin/sh\n\
i=0\n\
while kill -0 {current_pid} 2>/dev/null && [ \"$i\" -lt 150 ]; do\n\
    sleep 0.1\n\
    i=$((i + 1))\n\
done\n\
mv -f {src} {dst} && chmod +x {dst} && nohup {dst} >/dev/null 2>&1 &\n\
rm -f \"$0\"\n"
    )
}

#[cfg(any(target_os = "linux", test))]
fn linux_package_install_command(is_rpm: bool) -> &'static str {
    if is_rpm {
        "if command -v dnf >/dev/null 2>&1; then \
             dnf install -y --nogpgcheck \"$1\"; \
         elif command -v zypper >/dev/null 2>&1; then \
             zypper --non-interactive install --allow-unsigned-rpm \"$1\"; \
         elif command -v rpm >/dev/null 2>&1; then \
             rpm -U \"$1\"; \
         else \
             echo 'No supported RPM package manager (dnf, zypper, rpm) found.' >&2; \
             exit 127; \
         fi"
    } else {
        "if command -v apt-get >/dev/null 2>&1; then \
             DEBIAN_FRONTEND=noninteractive apt-get install -y \"$1\"; \
         elif command -v dpkg >/dev/null 2>&1; then \
             dpkg -i \"$1\"; \
         else \
             echo 'No supported DEB package manager (apt-get, dpkg) found.' >&2; \
             exit 127; \
         fi"
    }
}

#[cfg(unix)]
fn spawn_posix_relaunch(target: &Path, use_open: bool) -> Result<(), String> {
    use std::process::{Command, Stdio};
    let pid = std::process::id().to_string();
    let launch_cmd = if use_open {
        "i=0; while kill -0 \"$1\" 2>/dev/null && [ \"$i\" -lt 150 ]; do sleep 0.1; i=$((i + 1)); done; open -n \"$2\""
    } else {
        "i=0; while kill -0 \"$1\" 2>/dev/null && [ \"$i\" -lt 150 ]; do sleep 0.1; i=$((i + 1)); done; nohup \"$2\" >/dev/null 2>&1 &"
    };
    Command::new("sh")
        .args(["-c", launch_cmd, "_", &pid])
        .arg(target)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("Failed to schedule application relaunch: {e}"))?;
    Ok(())
}

#[cfg(target_os = "linux")]
fn install_linux_pkg_blocking(installer_path: &Path, is_rpm: bool) -> Result<(), String> {
    use std::os::unix::fs::MetadataExt;
    use std::process::{Command, Stdio};

    let cmd = linux_package_install_command(is_rpm);
    let is_root = std::fs::metadata("/proc/self")
        .map(|m| m.uid() == 0)
        .unwrap_or(false);

    let output = if is_root {
        Command::new("sh")
            .args(["-c", cmd, "_"])
            .arg(installer_path)
            .stdin(Stdio::null())
            .output()
            .map_err(|e| format!("Failed running package installer: {e}"))?
    } else {
        match Command::new("pkexec")
            .args(["sh", "-c", cmd, "_"])
            .arg(installer_path)
            .stdin(Stdio::null())
            .output()
        {
            Ok(out) => out,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                let manual = if is_rpm {
                    format!("sudo rpm -U {}", sh_single_quote(installer_path))
                } else {
                    format!("sudo apt-get install -y {}", sh_single_quote(installer_path))
                };
                return Err(format!(
                    "Polkit (pkexec) was not found for graphical elevation. Run manually: {manual}"
                ));
            }
            Err(e) => return Err(format!("Failed launching pkexec: {e}")),
        }
    };

    if output.status.success() {
        return Ok(());
    }

    let code = output.status.code().unwrap_or(-1);
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    if code == 126 {
        return Err(format!(
            "Administrator authentication was cancelled. Verified package is at {}",
            installer_path.display()
        ));
    }
    let detail = if stderr.is_empty() {
        format!("exit code {code}")
    } else {
        stderr
    };
    Err(format!(
        "Package installation failed ({detail}). Verified package is at {}",
        installer_path.display()
    ))
}

#[cfg(target_os = "linux")]
async fn install_and_relaunch(installer_path: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    use std::process::{Command, Stdio};

    let is_appimage = installer_path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("appimage"));

    if is_appimage {
        let target_appimage = std::env::var_os("APPIMAGE")
            .map(PathBuf::from)
            .filter(|p| !p.as_os_str().is_empty())
            .ok_or_else(|| {
                format!(
                    "Cannot auto-replace AppImage because APPIMAGE is not set. Verified AppImage is at {}",
                    installer_path.display()
                )
            })?;

        let script_path = installer_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("apply-update.sh");
        let script = build_linux_appimage_update_script(
            std::process::id(),
            installer_path,
            &target_appimage,
        );
        std::fs::write(&script_path, script)
            .map_err(|e| format!("Failed writing update helper script: {e}"))?;
        let _ = std::fs::set_permissions(&script_path, std::fs::Permissions::from_mode(0o700));

        Command::new("sh")
            .arg(&script_path)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("Failed launching AppImage updater: {e}"))?;
        return Ok(());
    }

    let is_rpm = installer_path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("rpm"));
    // Capture executable path before rpm/dpkg unlinks the running binary on disk.
    let current_exe = std::env::current_exe().ok();
    let path_buf = installer_path.to_path_buf();

    tokio::task::spawn_blocking(move || install_linux_pkg_blocking(&path_buf, is_rpm))
        .await
        .map_err(|e| format!("Update installer task failed: {e}"))??;

    let _ = std::fs::remove_file(installer_path);
    let relaunch_exe = current_exe
        .filter(|p| p.exists())
        .unwrap_or_else(|| PathBuf::from("bibulm-app"));
    spawn_posix_relaunch(&relaunch_exe, false)
}

#[cfg(any(target_os = "macos", test))]
fn resolve_macos_target_app(current_exe: Option<&Path>) -> PathBuf {
    if let Some(exe) = current_exe
        && let Some(macos_dir) = exe.parent()
        && macos_dir.file_name().and_then(|n| n.to_str()) == Some("MacOS")
        && let Some(contents_dir) = macos_dir.parent()
        && contents_dir.file_name().and_then(|n| n.to_str()) == Some("Contents")
        && let Some(app_bundle) = contents_dir.parent()
        && app_bundle
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("app"))
    {
        return app_bundle.to_path_buf();
    }
    PathBuf::from("/Applications/BibuLM.app")
}

#[cfg(target_os = "macos")]
fn install_macos_dmg_blocking(dmg_path: &Path, target_app: &Path) -> Result<(), String> {
    use std::process::{Command, Stdio};

    let mount_dir = dmg_path
        .parent()
        .unwrap_or_else(|| Path::new("/tmp"))
        .join(format!("dmg-mount-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&mount_dir);
    std::fs::create_dir_all(&mount_dir)
        .map_err(|e| format!("Failed creating DMG mount folder: {e}"))?;

    let attach = Command::new("hdiutil")
        .arg("attach")
        .arg(dmg_path)
        .args(["-nobrowse", "-readonly", "-mountpoint"])
        .arg(&mount_dir)
        .output()
        .map_err(|e| format!("Failed running hdiutil attach: {e}"))?;

    if !attach.status.success() {
        let _ = std::fs::remove_dir_all(&mount_dir);
        let stderr = String::from_utf8_lossy(&attach.stderr).trim().to_string();
        return Err(format!("Failed mounting DMG image: {stderr}"));
    }

    let result = (|| -> Result<(), String> {
        let src_app = std::fs::read_dir(&mount_dir)
            .map_err(|e| format!("Cannot read mounted DMG: {e}"))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| {
                p.is_dir()
                    && p.extension()
                        .and_then(|ext| ext.to_str())
                        .is_some_and(|ext| ext.eq_ignore_ascii_case("app"))
            })
            .ok_or_else(|| "No .app bundle found inside the downloaded DMG.".to_string())?;

        if let Some(parent) = target_app.parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        let backup_app = target_app.with_extension("app.bibulm-backup");
        let _ = std::fs::remove_dir_all(&backup_app);
        let moved_old = target_app.exists() && std::fs::rename(target_app, &backup_app).is_ok();

        let ditto = Command::new("ditto")
            .arg(&src_app)
            .arg(target_app)
            .output()
            .map_err(|e| format!("Failed running ditto: {e}"))?;

        if ditto.status.success() {
            if moved_old {
                let _ = std::fs::remove_dir_all(&backup_app);
            }
            return Ok(());
        }

        if moved_old && !target_app.exists() {
            let _ = std::fs::rename(&backup_app, target_app);
        }

        let osa = Command::new("osascript")
            .args([
                "-e",
                "on run argv",
                "-e",
                "do shell script \"rm -rf \" & quoted form of (item 1 of argv) & \" && ditto \" & quoted form of (item 2 of argv) & \" \" & quoted form of (item 1 of argv) with administrator privileges",
                "-e",
                "end run",
            ])
            .arg(target_app)
            .arg(&src_app)
            .stdin(Stdio::null())
            .output()
            .map_err(|e| format!("Failed running osascript: {e}"))?;

        if osa.status.success() {
            Ok(())
        } else {
            let stderr = String::from_utf8_lossy(&osa.stderr).trim().to_string();
            Err(format!(
                "macOS application update failed: {stderr}. Verified DMG is at {}",
                dmg_path.display()
            ))
        }
    })();

    let _ = Command::new("hdiutil")
        .arg("detach")
        .arg(&mount_dir)
        .arg("-force")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    let _ = std::fs::remove_dir_all(&mount_dir);
    result
}

#[cfg(target_os = "macos")]
async fn install_and_relaunch(installer_path: &Path) -> Result<(), String> {
    let target_app = resolve_macos_target_app(std::env::current_exe().ok().as_deref());
    let dmg_buf = installer_path.to_path_buf();
    let target_buf = target_app.clone();

    tokio::task::spawn_blocking(move || install_macos_dmg_blocking(&dmg_buf, &target_buf))
        .await
        .map_err(|e| format!("Update installer task failed: {e}"))??;

    let _ = std::fs::remove_file(installer_path);
    spawn_posix_relaunch(&target_app, true)
}

#[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
async fn install_and_relaunch(_installer_path: &Path) -> Result<(), String> {
    Err("Automatic installer execution is not supported on this operating system.".to_string())
}

pub fn verify_required_sha256(
    expected_opt: Option<&str>,
    actual_hex: &str,
    file_name: &str,
) -> Result<(), String> {
    let expected_hex = expected_opt
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            format!("Missing required SHA-256 checksum for {file_name}.")
        })?;
    if !actual_hex.eq_ignore_ascii_case(expected_hex) {
        return Err(format!(
            "SHA-256 checksum mismatch for {file_name} (expected {expected_hex}, got {actual_hex})"
        ));
    }
    Ok(())
}

#[cfg_attr(not(any(target_os = "linux", test)), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LinuxPackageKind {
    AppImage,
    Rpm,
    Deb,
}

#[cfg(any(target_os = "linux", test))]
pub(crate) fn detect_linux_package_kind_from_hints(
    has_appimage_env: bool,
    rpm_owns_exe: bool,
    dpkg_owns_exe: bool,
    has_rpm_bin: bool,
    has_dpkg_bin: bool,
    os_release: &str,
) -> Option<LinuxPackageKind> {
    if has_appimage_env {
        return Some(LinuxPackageKind::AppImage);
    }
    if rpm_owns_exe {
        return Some(LinuxPackageKind::Rpm);
    }
    if dpkg_owns_exe {
        return Some(LinuxPackageKind::Deb);
    }
    if has_rpm_bin && !has_dpkg_bin {
        return Some(LinuxPackageKind::Rpm);
    }
    if has_dpkg_bin && !has_rpm_bin {
        return Some(LinuxPackageKind::Deb);
    }
    let lower = os_release.to_ascii_lowercase();
    if ["fedora", "rhel", "centos", "rocky", "almalinux", "opensuse", "suse"]
        .iter()
        .any(|id| lower.contains(id))
    {
        return Some(LinuxPackageKind::Rpm);
    }
    if ["debian", "ubuntu", "linuxmint", "pop"]
        .iter()
        .any(|id| lower.contains(id))
    {
        return Some(LinuxPackageKind::Deb);
    }
    None
}

#[cfg(target_os = "linux")]
fn detect_linux_package_kind() -> Option<LinuxPackageKind> {
    use std::process::{Command, Stdio};

    let has_appimage_env = std::env::var_os("APPIMAGE").is_some_and(|v| !v.is_empty());
    if has_appimage_env {
        return Some(LinuxPackageKind::AppImage);
    }

    let exe = std::env::current_exe()
        .ok()
        .map(|p| p.canonicalize().unwrap_or(p));
    let rpm_owns_exe = exe.as_ref().is_some_and(|p| {
        Command::new("rpm")
            .arg("-qf")
            .arg(p)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    });
    let dpkg_owns_exe = !rpm_owns_exe
        && exe.as_ref().is_some_and(|p| {
            Command::new("dpkg")
                .arg("-S")
                .arg(p)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .is_ok_and(|s| s.success())
        });

    let has_rpm_bin = Path::new("/usr/bin/rpm").exists() || Path::new("/bin/rpm").exists();
    let has_dpkg_bin = Path::new("/usr/bin/dpkg").exists() || Path::new("/bin/dpkg").exists();
    let os_release = std::fs::read_to_string("/etc/os-release").unwrap_or_default();

    detect_linux_package_kind_from_hints(
        false,
        rpm_owns_exe,
        dpkg_owns_exe,
        has_rpm_bin,
        has_dpkg_bin,
        &os_release,
    )
}

#[cfg(not(target_os = "linux"))]
fn detect_linux_package_kind() -> Option<LinuxPackageKind> {
    None
}

fn select_installer_asset(assets: &[GhAsset]) -> Option<&GhAsset> {
    select_installer_asset_for_platform(
        assets,
        std::env::consts::OS,
        std::env::consts::ARCH,
        detect_linux_package_kind(),
    )
}

pub(crate) fn select_installer_asset_for_platform<'a>(
    assets: &'a [GhAsset],
    os: &str,
    arch: &str,
    linux_pkg: Option<LinuxPackageKind>,
) -> Option<&'a GhAsset> {
    let want_arm64 = arch.eq_ignore_ascii_case("aarch64") || arch.eq_ignore_ascii_case("arm64");
    let matches_arch = |lower: &str| -> bool {
        let is_arm = lower.contains("arm64") || lower.contains("aarch64");
        if want_arm64 {
            is_arm
        } else {
            !is_arm && (lower.contains("x64") || lower.contains("x86_64") || lower.contains("amd64"))
        }
    };

    match os {
        "windows" => {
            // 1. Prefer NSIS setup .exe matching host CPU architecture
            if let Some(a) = assets.iter().find(|a| {
                let lower = a.name.to_ascii_lowercase();
                matches_arch(&lower)
                    && (lower.ends_with("-setup.exe")
                        || (lower.ends_with(".exe") && lower.contains("setup")))
            }) {
                return Some(a);
            }
            // 2. Fallback to Windows MSI installer matching host CPU architecture
            if let Some(a) = assets.iter().find(|a| {
                let lower = a.name.to_ascii_lowercase();
                matches_arch(&lower) && lower.ends_with(".msi")
            }) {
                return Some(a);
            }
            // 3. Fallback to any arch-matching .exe asset
            assets.iter().find(|a| {
                let lower = a.name.to_ascii_lowercase();
                matches_arch(&lower) && lower.ends_with(".exe")
            })
        }
        "linux" => {
            let ext = match linux_pkg? {
                LinuxPackageKind::Rpm => ".rpm",
                LinuxPackageKind::Deb => ".deb",
                LinuxPackageKind::AppImage => ".appimage",
            };
            assets.iter().find(|a| {
                let lower = a.name.to_ascii_lowercase();
                matches_arch(&lower) && lower.ends_with(ext)
            })
        }
        "macos" => {
            if let Some(a) = assets.iter().find(|a| {
                let lower = a.name.to_ascii_lowercase();
                matches_arch(&lower) && lower.ends_with(".dmg")
            }) {
                return Some(a);
            }
            assets.iter().find(|a| {
                let lower = a.name.to_ascii_lowercase();
                lower.contains("universal") && lower.ends_with(".dmg")
            })
        }
        _ => None,
    }
}

#[cfg(test)]
pub(crate) fn select_windows_installer_asset_for_arch<'a>(
    assets: &'a [GhAsset],
    arch: &str,
) -> Option<&'a GhAsset> {
    select_installer_asset_for_platform(assets, "windows", arch, None)
}

async fn fetch_release_asset_sha256(
    client: &reqwest::Client,
    assets: &[GhAsset],
    target_asset_name: &str,
    github_token: Option<&str>,
) -> Option<String> {
    let checksum_asset = assets.iter().find(|a| {
        let lower = a.name.to_ascii_lowercase();
        lower == "sha256sums.txt"
            || lower == "sha256sums"
            || lower == format!("{}.sha256", target_asset_name.to_ascii_lowercase())
    })?;

    if !is_trusted_github_download_url(&checksum_asset.browser_download_url) {
        return None;
    }

    let mut req = client.get(&checksum_asset.browser_download_url);
    if let Some(tok) = github_token.map(str::trim).filter(|s| !s.is_empty()) {
        req = req.bearer_auth(tok);
    }
    let text = req.send().await.ok()?.text().await.ok()?;
    parse_sha256_for_file(&text, target_asset_name)
}

pub fn parse_sha256_for_file(checksums_text: &str, file_name: &str) -> Option<String> {
    let target_lower = file_name.to_ascii_lowercase();
    for line in checksums_text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let mut parts = trimmed.split_whitespace();
        let hash = parts.next()?;
        if hash.len() != 64 || !hash.chars().all(|c| c.is_ascii_hexdigit()) {
            continue;
        }
        match parts.next() {
            Some(name_part) => {
                let clean = name_part.trim_start_matches('*').trim_start_matches("./");
                if clean.to_ascii_lowercase() == target_lower
                    || clean.to_ascii_lowercase().ends_with(&format!("/{target_lower}"))
                {
                    return Some(hash.to_ascii_lowercase());
                }
            }
            // Single-hash `<file>.sha256` containing only the digest
            None => return Some(hash.to_ascii_lowercase()),
        }
    }
    None
}

pub fn normalize_version(raw: &str) -> String {
    raw.trim().trim_start_matches(['v', 'V']).to_string()
}

pub fn is_newer_version(current: &str, candidate: &str) -> bool {
    let parse = |s: &str| -> Option<(u64, u64, u64)> {
        let core = normalize_version(s);
        let core = core.split(['-', '+']).next().unwrap_or(&core);
        let mut it = core.split('.');
        let maj = it.next()?.parse::<u64>().ok()?;
        let min = it.next().unwrap_or("0").parse::<u64>().ok()?;
        let pat = it.next().unwrap_or("0").parse::<u64>().ok()?;
        Some((maj, min, pat))
    };
    match (parse(current), parse(candidate)) {
        (Some(cur), Some(cand)) => cand > cur,
        _ => false,
    }
}

pub fn is_trusted_github_download_url(url: &str) -> bool {
    let lower = url.to_ascii_lowercase();
    // GitHub serves the repo as `Capybaraclette/BibuLM`; owner and repo names are case-insensitive.
    lower.starts_with("https://github.com/capybaraclette/bibulm/releases/download/")
        && !lower.contains("..")
        && !lower.contains("%2e")
        && !lower.contains("%5c")
        && !url.contains('\\')
        && !url.contains('@')
        && !url.contains('?')
        && !url.contains('#')
}

fn sanitize_installer_filename(raw: &str) -> Result<String, String> {
    sanitize_installer_filename_for_os(raw, std::env::consts::OS)
}

fn sanitize_installer_filename_for_os(raw: &str, os: &str) -> Result<String, String> {
    if raw.is_empty()
        || raw.contains('/')
        || raw.contains('\\')
        || raw.contains("..")
        || raw.contains('\0')
    {
        return Err(format!("Unsafe installer filename: {raw}"));
    }
    let lower = raw.to_ascii_lowercase();
    let valid_ext = match os {
        "windows" => lower.ends_with(".exe") || lower.ends_with(".msi"),
        "linux" => lower.ends_with(".rpm") || lower.ends_with(".deb") || lower.ends_with(".appimage"),
        "macos" => lower.ends_with(".dmg"),
        _ => false,
    };
    if !valid_ext {
        return Err(format!("Unsupported installer file type for {os}: {raw}"));
    }
    Ok(raw.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semver_comparison_handles_v_prefix_and_patch_bumps() {
        assert!(is_newer_version("0.1.0", "v0.1.1"));
        assert!(is_newer_version("0.1.0", "0.2.0"));
        assert!(is_newer_version("0.9.9", "1.0.0"));
        assert!(!is_newer_version("0.2.0", "v0.2.0"));
        assert!(!is_newer_version("0.2.0", "v0.1.9"));
    }

    #[test]
    fn checksum_parser_matches_target_installer() {
        let hash = "a".repeat(64);
        let text = format!(
            "{}  BibuLM_0.3.0_x64-setup.exe\n{} *BibuLM_0.3.0_x64_en-US.msi\n",
            hash,
            "b".repeat(64)
        );
        assert_eq!(
            parse_sha256_for_file(&text, "BibuLM_0.3.0_x64-setup.exe"),
            Some(hash)
        );
        assert_eq!(
            parse_sha256_for_file(&text, "BibuLM_0.3.0_x64_en-US.msi"),
            Some("b".repeat(64))
        );
    }

    #[test]
    fn trusted_url_validator_blocks_non_github_hosts() {
        assert!(is_trusted_github_download_url(
            "https://github.com/Capybaraclette/BibuLM/releases/download/v1.0.0/Bibulm_1.0.0_x64-setup.exe"
        ));
        assert!(is_trusted_github_download_url(
            "https://github.com/Capybaraclette/bibulm/releases/download/v0.3.0/BibuLM_0.3.0_x64-setup.exe"
        ));
        assert!(!is_trusted_github_download_url(
            "http://github.com/Capybaraclette/bibulm/releases/download/v0.3.0/setup.exe"
        ));
        assert!(!is_trusted_github_download_url(
            "https://github.com/evil/repo/releases/download/v0.3.0/setup.exe"
        ));
        assert!(!is_trusted_github_download_url(
            "https://github.com/Capybaraclette/bibulm/releases/download/../../evil/repo/releases/download/v1/setup.exe"
        ));
        assert!(!is_trusted_github_download_url(
            "https://github.com/Capybaraclette/bibulm/releases/download/%2e%2e/evil/setup.exe"
        ));
        assert!(!is_trusted_github_download_url(
            "https://evil.example.com/setup.exe"
        ));
    }

    #[test]
    fn arch_aware_installer_selection_and_mandatory_sha256() {
        let assets = vec![
            GhAsset {
                name: "BibuLM_0.3.1_arm64-setup.exe".into(),
                browser_download_url: "https://github.com/a/b/releases/download/v0.3.1/arm64.exe".into(),
                size: 100,
            },
            GhAsset {
                name: "BibuLM_0.3.1_x64-setup.exe".into(),
                browser_download_url: "https://github.com/a/b/releases/download/v0.3.1/x64.exe".into(),
                size: 100,
            },
        ];
        assert_eq!(
            select_windows_installer_asset_for_arch(&assets, "x86_64")
                .unwrap()
                .name,
            "BibuLM_0.3.1_x64-setup.exe"
        );
        assert_eq!(
            select_windows_installer_asset_for_arch(&assets, "aarch64")
                .unwrap()
                .name,
            "BibuLM_0.3.1_arm64-setup.exe"
        );
        assert!(verify_required_sha256(None, &"a".repeat(64), "x.exe").is_err());
        assert!(verify_required_sha256(Some(&"a".repeat(64)), &"a".repeat(64), "x.exe").is_ok());
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn windows_update_script_prefers_current_exe_and_escapes_quotes() {
        let script = build_windows_update_script(
            4242,
            std::path::Path::new(r"C:\Users\O'Neil\updates\BibuLM_1.0.3_x64-setup.exe"),
            std::path::Path::new(r"C:\Program Files\BibuLM\bibulm-app.exe"),
            std::path::Path::new(r"C:\Users\O'Neil\AppData\Local\BibuLM\bibulm-app.exe"),
        );
        assert!(script.starts_with('\u{FEFF}'));
        assert!(script.contains("Wait-Process -Id 4242"));
        assert!(script.contains(r"C:\Users\O''Neil\updates\BibuLM_1.0.3_x64-setup.exe"));
        let cur_pos = script.find(r"C:\Program Files\BibuLM\bibulm-app.exe").unwrap();
        let def_pos = script.find(r"C:\Users\O''Neil\AppData\Local\BibuLM\bibulm-app.exe").unwrap();
        assert!(cur_pos < def_pos);
    }

    #[test]
    fn linux_and_macos_installer_selection_and_sanitization() {
        let assets = vec![
            GhAsset {
                name: "BibuLM_1.0.5_x64-setup.exe".into(),
                browser_download_url: "https://github.com/a/b/releases/download/v1.0.5/win.exe".into(),
                size: 100,
            },
            GhAsset {
                name: "BibuLM-1.0.5-1.x86_64.rpm".into(),
                browser_download_url: "https://github.com/a/b/releases/download/v1.0.5/lin.rpm".into(),
                size: 100,
            },
            GhAsset {
                name: "BibuLM_1.0.5_amd64.deb".into(),
                browser_download_url: "https://github.com/a/b/releases/download/v1.0.5/lin.deb".into(),
                size: 100,
            },
            GhAsset {
                name: "BibuLM_1.0.5_amd64.AppImage".into(),
                browser_download_url: "https://github.com/a/b/releases/download/v1.0.5/lin.AppImage".into(),
                size: 100,
            },
            GhAsset {
                name: "BibuLM_1.0.5_aarch64.dmg".into(),
                browser_download_url: "https://github.com/a/b/releases/download/v1.0.5/mac.dmg".into(),
                size: 100,
            },
        ];

        assert_eq!(
            select_installer_asset_for_platform(&assets, "linux", "x86_64", Some(LinuxPackageKind::Rpm))
                .unwrap()
                .name,
            "BibuLM-1.0.5-1.x86_64.rpm"
        );
        assert_eq!(
            select_installer_asset_for_platform(&assets, "linux", "x86_64", Some(LinuxPackageKind::Deb))
                .unwrap()
                .name,
            "BibuLM_1.0.5_amd64.deb"
        );
        assert_eq!(
            select_installer_asset_for_platform(
                &assets,
                "linux",
                "x86_64",
                Some(LinuxPackageKind::AppImage)
            )
            .unwrap()
            .name,
            "BibuLM_1.0.5_amd64.AppImage"
        );
        assert!(select_installer_asset_for_platform(&assets, "linux", "x86_64", None).is_none());
        assert_eq!(
            select_installer_asset_for_platform(&assets, "macos", "aarch64", None)
                .unwrap()
                .name,
            "BibuLM_1.0.5_aarch64.dmg"
        );
        assert!(select_installer_asset_for_platform(&assets, "macos", "x86_64", None).is_none());

        assert!(sanitize_installer_filename_for_os("BibuLM-1.0.5-1.x86_64.rpm", "linux").is_ok());
        assert!(sanitize_installer_filename_for_os("BibuLM_1.0.5_amd64.deb", "linux").is_ok());
        assert!(sanitize_installer_filename_for_os("BibuLM_1.0.5_amd64.AppImage", "linux").is_ok());
        assert!(sanitize_installer_filename_for_os("BibuLM_1.0.5_x64-setup.exe", "linux").is_err());
        assert!(sanitize_installer_filename_for_os("BibuLM_1.0.5_aarch64.dmg", "macos").is_ok());
        assert!(sanitize_installer_filename_for_os("../evil.rpm", "linux").is_err());
    }

    #[test]
    fn linux_package_kind_detection_handles_rpm_deb_and_appimage() {
        assert_eq!(
            detect_linux_package_kind_from_hints(true, true, true, true, true, "ID=fedora"),
            Some(LinuxPackageKind::AppImage)
        );
        assert_eq!(
            detect_linux_package_kind_from_hints(false, true, false, true, false, ""),
            Some(LinuxPackageKind::Rpm)
        );
        assert_eq!(
            detect_linux_package_kind_from_hints(false, false, true, false, true, ""),
            Some(LinuxPackageKind::Deb)
        );
        assert_eq!(
            detect_linux_package_kind_from_hints(false, false, false, false, false, "ID=fedora"),
            Some(LinuxPackageKind::Rpm)
        );
        assert_eq!(
            detect_linux_package_kind_from_hints(false, false, false, false, false, "ID=ubuntu"),
            Some(LinuxPackageKind::Deb)
        );
        assert_eq!(
            detect_linux_package_kind_from_hints(false, false, false, false, false, "ID=arch"),
            None
        );
    }

    #[test]
    fn linux_and_macos_script_and_bundle_helpers() {
        let appimage_script = build_linux_appimage_update_script(
            999,
            Path::new("/tmp/BibuLM's.AppImage"),
            Path::new("/home/u/BibuLM.AppImage"),
        );
        assert!(appimage_script.contains("kill -0 999"));
        assert!(appimage_script.contains("'/tmp/BibuLM'\\''s.AppImage'"));

        assert!(linux_package_install_command(true).contains("dnf install -y --nogpgcheck"));
        assert!(linux_package_install_command(true).contains("rpm -U"));
        assert!(linux_package_install_command(false).contains("apt-get install -y"));

        assert_eq!(
            resolve_macos_target_app(Some(Path::new(
                "/Users/alex/Applications/BibuLM.app/Contents/MacOS/bibulm-app"
            ))),
            PathBuf::from("/Users/alex/Applications/BibuLM.app")
        );
        assert_eq!(
            resolve_macos_target_app(Some(Path::new("/usr/local/bin/bibulm-app"))),
            PathBuf::from("/Applications/BibuLM.app")
        );
    }
}
