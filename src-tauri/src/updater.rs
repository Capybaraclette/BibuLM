use crate::state::AppState;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
#[cfg(target_os = "windows")]
use std::path::PathBuf;
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
            "BibuLM/{} (Windows; App Auto-Updater)",
            current_app_version()
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

    let chosen_asset = select_windows_installer_asset(&gh.assets);
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

/// Downloads the verified Windows installer, stops running models & server, launches silent
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
        .ok_or_else(|| "This release does not contain a Windows installer (.exe / .msi).".to_string())?;
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

    spawn_windows_updater_and_relaunch(&installer_path)?;

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
fn spawn_windows_updater_and_relaunch(installer_path: &std::path::Path) -> Result<(), String> {
    use std::os::windows::process::CommandExt;

    let current_pid = std::process::id();
    let current_exe = std::env::current_exe().unwrap_or_else(|_| PathBuf::new());
    let default_nsis_exe = std::env::var_os("LOCALAPPDATA")
        .map(|d| PathBuf::from(d).join("BibuLM").join("BibuLM.exe"))
        .unwrap_or_default();

    let is_msi = installer_path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("msi"));

    let script_path = installer_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .join("apply-update.ps1");

    let esc = |p: &std::path::Path| p.to_string_lossy().replace('\'', "''");
    let installer_ps = esc(installer_path);
    let current_exe_ps = esc(&current_exe);
    let default_exe_ps = esc(&default_nsis_exe);

    let install_cmd = if is_msi {
        format!(
            "Start-Process -FilePath 'msiexec.exe' -ArgumentList @('/i', '{installer_ps}', '/passive', '/norestart') -Wait"
        )
    } else {
        // NSIS setup supports /S for silent update into the existing install directory.
        format!("Start-Process -FilePath '{installer_ps}' -ArgumentList '/S' -Wait")
    };

    let ps_script = format!(
        r#"$ErrorActionPreference = 'SilentlyContinue'
try {{
    Wait-Process -Id {current_pid} -Timeout 15 -ErrorAction SilentlyContinue
}} catch {{}}
Start-Sleep -Milliseconds 600
{install_cmd}
try {{
    Add-Type -Namespace BibuLMWin32 -Name ShellNotify -MemberDefinition '[System.Runtime.InteropServices.DllImport("shell32.dll")] public static extern void SHChangeNotify(int wEventId, uint uFlags, System.IntPtr dwItem1, System.IntPtr dwItem2);'
    [BibuLMWin32.ShellNotify]::SHChangeNotify(0x08000000, 0x0000, [System.IntPtr]::Zero, [System.IntPtr]::Zero)
}} catch {{}}
Start-Sleep -Milliseconds 500
if ('{default_exe_ps}' -and (Test-Path '{default_exe_ps}')) {{
    Start-Process -FilePath '{default_exe_ps}'
}} elseif ('{current_exe_ps}' -and (Test-Path '{current_exe_ps}')) {{
    Start-Process -FilePath '{current_exe_ps}'
}}
Remove-Item -Path '{installer_ps}' -Force -ErrorAction SilentlyContinue
Remove-Item -Path $PSCommandPath -Force -ErrorAction SilentlyContinue
"#
    );

    std::fs::write(&script_path, ps_script)
        .map_err(|e| format!("Failed writing update helper script: {e}"))?;

    const CREATE_NO_WINDOW: u32 = 0x08000000;
    const DETACHED_PROCESS: u32 = 0x00000008;

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
        .creation_flags(CREATE_NO_WINDOW | DETACHED_PROCESS)
        .spawn()
        .map_err(|e| format!("Failed to launch update installer helper: {e}"))?;

    Ok(())
}

#[cfg(not(target_os = "windows"))]
fn spawn_windows_updater_and_relaunch(_installer_path: &std::path::Path) -> Result<(), String> {
    Err("Automatic installer execution is currently implemented for Windows (.exe / .msi).".to_string())
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

fn select_windows_installer_asset(assets: &[GhAsset]) -> Option<&GhAsset> {
    select_windows_installer_asset_for_arch(assets, std::env::consts::ARCH)
}

pub(crate) fn select_windows_installer_asset_for_arch<'a>(
    assets: &'a [GhAsset],
    arch: &str,
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

    // 1. Prefer NSIS setup .exe matching host CPU architecture
    if let Some(a) = assets.iter().find(|a| {
        let lower = a.name.to_ascii_lowercase();
        matches_arch(&lower)
            && (lower.ends_with("-setup.exe") || (lower.ends_with(".exe") && lower.contains("setup")))
    }) {
        return Some(a);
    }
    // 2. Fallback to Windows MSI installer matching host CPU architecture
    if let Some(a) = assets
        .iter()
        .find(|a| {
            let lower = a.name.to_ascii_lowercase();
            matches_arch(&lower) && lower.ends_with(".msi")
        })
    {
        return Some(a);
    }
    // 3. Fallback to any arch-matching .exe asset
    assets.iter().find(|a| {
        let lower = a.name.to_ascii_lowercase();
        matches_arch(&lower) && lower.ends_with(".exe")
    })
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
    if raw.contains('/') || raw.contains('\\') || raw.contains("..") {
        return Err(format!("Unsafe installer filename: {raw}"));
    }
    let lower = raw.to_ascii_lowercase();
    if !lower.ends_with(".exe") && !lower.ends_with(".msi") {
        return Err(format!("Unsupported installer file type: {raw}"));
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
}
