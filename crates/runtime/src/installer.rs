use crate::github::RuntimeBackendVariant;
use chrono::Utc;
use flate2::read::GzDecoder;
use futures_util::StreamExt;
use bibulm_core::{Database, DbError, InstalledRuntimeRecord};
use std::fs::{self, File};
use std::io;
use std::path::{Component, Path, PathBuf};
use tar::Archive;
use thiserror::Error;
use tokio::io::AsyncWriteExt;
use uuid::Uuid;
use zip::ZipArchive;

#[derive(Debug, Error)]
pub enum InstallerError {
    #[error("HTTP download failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("Zip extraction error: {0}")]
    Zip(#[from] zip::result::ZipError),
    #[error("Database error: {0}")]
    Db(#[from] DbError),
    #[error("HTTP status {0} downloading {1}")]
    HttpStatus(u16, String),
    #[error("Installed archive did not contain llama-server executable")]
    ServerBinaryNotFound,
    #[error("Invalid custom runtime executable path: {0}")]
    InvalidCustomPath(String),
}

#[derive(Debug, Clone)]
pub struct InstallProgress {
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub stage: String,
}

/// Downloads all zip archives in `variant` (e.g. main binary + companion `cudart` DLLs),
/// extracts them into `<runtimes_dir>/<variant.id>/`, locates `llama-server.exe` & `llama-bench.exe`,
/// and registers the runtime in SQLite.
pub async fn install_runtime_variant<F>(
    db: &Database,
    runtimes_dir: &Path,
    variant: &RuntimeBackendVariant,
    release_notes: Option<String>,
    github_token: Option<&str>,
    mut on_progress: F,
) -> Result<InstalledRuntimeRecord, InstallerError>
where
    F: FnMut(InstallProgress) + Send,
{
    fs::create_dir_all(runtimes_dir)?;
    let install_dir = runtimes_dir.join(sanitize_folder_name(&variant.id));
    fs::create_dir_all(&install_dir)?;

    let temp_dir = runtimes_dir.join(format!(".tmp-{}", Uuid::new_v4()));
    fs::create_dir_all(&temp_dir)?;

    let client = reqwest::Client::builder()
        .user_agent("BibuLM/0.3 (llama.cpp runtime installer)")
        .build()?;

    let total_bytes = variant.total_size_bytes.max(1);
    let mut cumulative_downloaded: u64 = 0;
    let mut downloaded_zips = Vec::new();

    let download_result: Result<(), InstallerError> = async {
        for asset in &variant.assets {
            on_progress(InstallProgress {
                downloaded_bytes: cumulative_downloaded,
                total_bytes,
                stage: format!("Downloading {}", asset.name),
            });

            let mut req = client.get(&asset.download_url);
            if let Some(tok) = github_token.map(str::trim).filter(|s| !s.is_empty()) {
                req = req.bearer_auth(tok);
            }
            let resp = req.send().await?;
            if !resp.status().is_success() {
                return Err(InstallerError::HttpStatus(
                    resp.status().as_u16(),
                    asset.name.clone(),
                ));
            }

            let archive_path = temp_dir.join(&asset.name);
            let mut file = tokio::fs::File::create(&archive_path).await?;
            let mut stream = resp.bytes_stream();
            while let Some(chunk) = stream.next().await {
                let bytes = chunk?;
                file.write_all(&bytes).await?;
                cumulative_downloaded = cumulative_downloaded.saturating_add(bytes.len() as u64);
                on_progress(InstallProgress {
                    downloaded_bytes: cumulative_downloaded.min(total_bytes),
                    total_bytes,
                    stage: format!("Downloading {}", asset.name),
                });
            }
            file.flush().await?;
            downloaded_zips.push(archive_path);
        }
        Ok(())
    }
    .await;

    if let Err(e) = download_result {
        let _ = fs::remove_dir_all(&temp_dir);
        return Err(e);
    }

    on_progress(InstallProgress {
        downloaded_bytes: total_bytes,
        total_bytes,
        stage: "Extracting binaries & shared libraries...".to_string(),
    });

    // Extract every downloaded archive off the async executor thread.
    let install_dir_clone = install_dir.clone();
    let temp_dir_clone = temp_dir.clone();
    let extract_res = tokio::task::spawn_blocking(move || -> Result<(), InstallerError> {
        for archive_path in &downloaded_zips {
            extract_archive_safely(archive_path, &install_dir_clone)?;
        }
        let _ = fs::remove_dir_all(&temp_dir_clone);
        Ok(())
    })
    .await
    .map_err(|e| io::Error::other(e.to_string()))?;

    let _ = fs::remove_dir_all(&temp_dir);
    extract_res?;

    let server_exe = find_executable(&install_dir, &["llama-server.exe", "llama-server"])
        .ok_or(InstallerError::ServerBinaryNotFound)?;
    let bench_exe = find_executable(&install_dir, &["llama-bench.exe", "llama-bench"]);

    // If llama-server was inside a nested folder while companion libs extracted to root, copy them alongside.
    if let Some(server_dir) = server_exe.parent()
        && server_dir != install_dir
    {
        copy_sibling_dlls(&install_dir, server_dir);
    }

    ensure_executable_permissions(&server_exe);
    if let Some(ref b) = bench_exe {
        ensure_executable_permissions(b);
    }

    let size_bytes = dir_size_bytes(&install_dir);
    let record = InstalledRuntimeRecord {
        id: variant.id.clone(),
        tag: variant.tag.clone(),
        backend: variant.backend.clone(),
        display_name: variant.display_name.clone(),
        install_dir: install_dir.to_string_lossy().into_owned(),
        server_exe: server_exe.to_string_lossy().into_owned(),
        bench_exe: bench_exe.map(|p| p.to_string_lossy().into_owned()),
        is_custom: false,
        installed_at: Utc::now().to_rfc3339(),
        release_notes,
        size_bytes,
    };

    db.upsert_runtime(&record)?;
    on_progress(InstallProgress {
        downloaded_bytes: total_bytes,
        total_bytes,
        stage: "Installed".to_string(),
    });
    Ok(record)
}

/// Registers a user-provided local `llama-server.exe` binary (or folder containing it).
pub fn register_custom_runtime(
    db: &Database,
    display_name: String,
    path: &Path,
) -> Result<InstalledRuntimeRecord, InstallerError> {
    let server_exe = if path.is_file() {
        path.to_path_buf()
    } else if path.is_dir() {
        find_executable(path, &["llama-server.exe", "llama-server"]).ok_or_else(|| {
            InstallerError::InvalidCustomPath(format!(
                "No llama-server executable found in {}",
                path.display()
            ))
        })?
    } else {
        return Err(InstallerError::InvalidCustomPath(
            path.to_string_lossy().into_owned(),
        ));
    };

    let install_dir = server_exe
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let bench_exe = find_executable(&install_dir, &["llama-bench.exe", "llama-bench"])
        .map(|p| p.to_string_lossy().into_owned());
    let id = format!("custom-{}", &Uuid::new_v4().to_string()[..8]);

    let record = InstalledRuntimeRecord {
        id,
        tag: "custom".to_string(),
        backend: "custom".to_string(),
        display_name: if display_name.trim().is_empty() {
            format!(
                "Custom ({})",
                install_dir
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("local")
            )
        } else {
            display_name.trim().to_string()
        },
        install_dir: install_dir.to_string_lossy().into_owned(),
        server_exe: server_exe.to_string_lossy().into_owned(),
        bench_exe,
        is_custom: true,
        installed_at: Utc::now().to_rfc3339(),
        release_notes: None,
        size_bytes: fs::metadata(&server_exe).map(|m| m.len()).unwrap_or(0),
    };
    db.upsert_runtime(&record)?;
    Ok(record)
}

/// Resolves `llama-bench` executable path for an installed runtime record, falling back to
/// scanning `install_dir` / `server_exe` parent for runtimes installed before `bench_exe` discovery.
pub fn resolve_bench_exe(rec: &InstalledRuntimeRecord) -> Option<PathBuf> {
    if let Some(ref b) = rec.bench_exe {
        let p = PathBuf::from(b);
        if p.is_file() {
            return Some(p);
        }
    }
    if let Some(parent) = Path::new(&rec.server_exe).parent()
        && let Some(found) = find_executable(parent, &["llama-bench.exe", "llama-bench"])
    {
        return Some(found);
    }
    let dir = Path::new(&rec.install_dir);
    if dir.is_dir() {
        return find_executable(dir, &["llama-bench.exe", "llama-bench"]);
    }
    None
}

/// Normalizes a runtime backend identifier to its runtime family so versioned backends
/// (e.g. `cuda-12.4` and `cuda-13.1`, or `cpu-avx2` and `cpu`) count as the same runtime.
pub fn runtime_family(backend: &str) -> &str {
    let b = backend.trim();
    if b.eq_ignore_ascii_case("cuda")
        || b.get(..5).is_some_and(|p| p.eq_ignore_ascii_case("cuda-"))
    {
        "cuda"
    } else if b.eq_ignore_ascii_case("cpu")
        || b.get(..4).is_some_and(|p| p.eq_ignore_ascii_case("cpu-"))
    {
        "cpu"
    } else {
        b
    }
}

/// Removes older managed runtimes in the same runtime family as `installed`, keeping at most
/// `keep_old` previous builds alongside `installed` (and never removing runtimes in `in_use_ids`).
/// Returns the IDs of removed runtimes.
pub fn prune_old_runtimes(
    db: &Database,
    installed: &InstalledRuntimeRecord,
    keep_old: u32,
    in_use_ids: &[String],
) -> Result<Vec<String>, InstallerError> {
    if installed.is_custom {
        return Ok(Vec::new());
    }
    let target_family = runtime_family(&installed.backend);
    // `list_runtimes()` is ordered newest-first (`installed_at DESC`).
    let mut kept_old = 0usize;
    let mut removed = Vec::new();
    for rt in db.list_runtimes()? {
        if rt.id == installed.id || rt.is_custom {
            continue;
        }
        if !runtime_family(&rt.backend).eq_ignore_ascii_case(target_family) {
            continue;
        }
        if kept_old < keep_old as usize {
            kept_old += 1;
            continue;
        }
        if in_use_ids.iter().any(|id| id == &rt.id) {
            continue;
        }
        uninstall_runtime(db, &rt.id)?;
        removed.push(rt.id);
    }
    Ok(removed)
}

/// Removes a runtime from the database, deleting its managed folder if not a custom binary.
pub fn uninstall_runtime(db: &Database, id: &str) -> Result<(), InstallerError> {
    let runtimes = db.list_runtimes()?;
    if let Some(rt) = runtimes.into_iter().find(|r| r.id == id) {
        if !rt.is_custom {
            let dir = PathBuf::from(&rt.install_dir);
            if dir.is_dir() {
                let _ = fs::remove_dir_all(&dir);
            }
        }
        db.delete_runtime(id)?;
    }
    Ok(())
}

fn sanitize_folder_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn extract_archive_safely(archive_path: &Path, dest_dir: &Path) -> Result<(), InstallerError> {
    let lower = archive_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if lower.ends_with(".tar.gz") || lower.ends_with(".tgz") {
        extract_targz_safely(archive_path, dest_dir)
    } else {
        extract_zip_safely(archive_path, dest_dir)
    }
}

fn is_safe_relative_path(path: &Path) -> bool {
    !path.as_os_str().is_empty()
        && !path
            .components()
            .any(|c| matches!(c, Component::ParentDir | Component::RootDir | Component::Prefix(_)))
}

fn extract_targz_safely(archive_path: &Path, dest_dir: &Path) -> Result<(), InstallerError> {
    let file = File::open(archive_path)?;
    let gz = GzDecoder::new(file);
    let mut archive = Archive::new(gz);
    for entry_res in archive.entries()? {
        let mut entry = entry_res?;
        let rel = entry.path()?.into_owned();
        if !is_safe_relative_path(&rel) {
            continue;
        }
        let kind = entry.header().entry_type();
        if kind.is_symlink() || kind.is_hard_link() {
            continue;
        }
        let out_path = dest_dir.join(&rel);
        if kind.is_dir() {
            fs::create_dir_all(&out_path)?;
        } else if kind.is_file() {
            if let Some(parent) = out_path.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut out_file = File::create(&out_path)?;
            io::copy(&mut entry, &mut out_file)?;
            #[cfg(unix)]
            if let Ok(mode) = entry.header().mode() {
                use std::os::unix::fs::PermissionsExt;
                let _ = fs::set_permissions(&out_path, fs::Permissions::from_mode(mode & 0o777));
            }
        }
    }
    Ok(())
}

fn extract_zip_safely(zip_path: &Path, dest_dir: &Path) -> Result<(), InstallerError> {
    let file = File::open(zip_path)?;
    let mut archive = ZipArchive::new(file)?;
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let Some(enclosed) = entry.enclosed_name() else {
            continue;
        };
        if !is_safe_relative_path(&enclosed) {
            continue;
        }
        let out_path = dest_dir.join(&enclosed);
        if entry.is_dir() {
            fs::create_dir_all(&out_path)?;
        } else {
            if let Some(parent) = out_path.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut out_file = File::create(&out_path)?;
            io::copy(&mut entry, &mut out_file)?;
        }
    }
    Ok(())
}

fn ensure_executable_permissions(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(meta) = fs::metadata(path) {
            let mut perms = meta.permissions();
            perms.set_mode(perms.mode() | 0o755);
            let _ = fs::set_permissions(path, perms);
        }
    }
    #[cfg(not(unix))]
    let _ = path;
}

fn find_executable(dir: &Path, candidates: &[&str]) -> Option<PathBuf> {
    for c in candidates {
        let direct = dir.join(c);
        if direct.is_file() {
            return Some(direct);
        }
    }
    // Check 1-2 subdirectories deep (some releases extract into `build/bin/`).
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir()
                && let Some(found) = find_executable(&p, candidates)
            {
                return Some(found);
            }
        }
    }
    None
}

fn copy_sibling_dlls(from_dir: &Path, to_dir: &Path) {
    if let Ok(entries) = fs::read_dir(from_dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_file()
                && p.extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| e.eq_ignore_ascii_case("dll"))
                && let Some(name) = p.file_name()
            {
                let target = to_dir.join(name);
                if !target.exists() {
                    let _ = fs::copy(&p, &target);
                }
            }
        }
    }
}

fn dir_size_bytes(dir: &Path) -> u64 {
    let mut sum = 0u64;
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                sum = sum.saturating_add(dir_size_bytes(&p));
            } else if let Ok(meta) = p.metadata() {
                sum = sum.saturating_add(meta.len());
            }
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::Compression;
    use flate2::write::GzEncoder;

    #[test]
    fn extracts_targz_and_zip_safely_and_finds_llama_bench() {
        let tmp = std::env::temp_dir().join(format!("bibulm_inst_{}", Uuid::new_v4()));
        let dest = tmp.join("out");
        fs::create_dir_all(&dest).unwrap();

        // 1. Build a tar.gz with safe files and a path-traversal entry
        let tgz_path = tmp.join("test.tar.gz");
        {
            let file = File::create(&tgz_path).unwrap();
            let enc = GzEncoder::new(file, Compression::default());
            let mut builder = tar::Builder::new(enc);

            let server_bytes = b"#!/bin/sh\necho ok\n";
            let mut h1 = tar::Header::new_gnu();
            h1.set_size(server_bytes.len() as u64);
            h1.set_mode(0o755);
            h1.set_cksum();
            builder
                .append_data(&mut h1, "build/bin/llama-server", &server_bytes[..])
                .unwrap();

            let bench_bytes = b"#!/bin/sh\necho bench\n";
            let mut h2 = tar::Header::new_gnu();
            h2.set_size(bench_bytes.len() as u64);
            h2.set_mode(0o755);
            h2.set_cksum();
            builder
                .append_data(&mut h2, "build/bin/llama-bench", &bench_bytes[..])
                .unwrap();

            builder.finish().unwrap();
        }

        extract_archive_safely(&tgz_path, &dest).unwrap();
        assert!(find_executable(&dest, &["llama-server"]).is_some());
        assert!(find_executable(&dest, &["llama-bench"]).is_some());

        // 2. Build a zip archive and verify extraction
        let zip_path = tmp.join("test.zip");
        {
            let f = File::create(&zip_path).unwrap();
            let mut zw = zip::ZipWriter::new(f);
            let opts = zip::write::SimpleFileOptions::default();
            zw.start_file("llama-bench.exe", opts).unwrap();
            io::Write::write_all(&mut zw, b"MZ").unwrap();
            zw.finish().unwrap();
        }
        let zip_dest = tmp.join("zip_out");
        fs::create_dir_all(&zip_dest).unwrap();
        extract_archive_safely(&zip_path, &zip_dest).unwrap();
        assert!(find_executable(&zip_dest, &["llama-bench.exe"]).is_some());

        let _ = fs::remove_dir_all(&tmp);
    }

    fn make_record(id: &str, backend: &str, dir: &Path, installed_at: &str, is_custom: bool) -> InstalledRuntimeRecord {
        if !is_custom {
            fs::create_dir_all(dir).unwrap();
            fs::write(dir.join("llama-server.exe"), b"stub").unwrap();
        }
        InstalledRuntimeRecord {
            id: id.to_string(),
            tag: id.split('-').next().unwrap_or("b0").to_string(),
            backend: backend.to_string(),
            display_name: id.to_string(),
            install_dir: dir.to_string_lossy().into_owned(),
            server_exe: dir.join("llama-server.exe").to_string_lossy().into_owned(),
            bench_exe: None,
            is_custom,
            installed_at: installed_at.to_string(),
            release_notes: None,
            size_bytes: 4,
        }
    }

    #[test]
    fn runtime_family_groups_versioned_backends() {
        assert_eq!(runtime_family("cuda-12.4"), "cuda");
        assert_eq!(runtime_family("cuda-13.1"), "cuda");
        assert_eq!(runtime_family("cuda"), "cuda");
        assert_eq!(runtime_family("cpu"), "cpu");
        assert_eq!(runtime_family("cpu-avx2"), "cpu");
        assert_eq!(runtime_family("cpu-avx512"), "cpu");
        assert_eq!(runtime_family("vulkan"), "vulkan");
        assert_eq!(runtime_family("rocm"), "rocm");
    }

    #[test]
    fn prune_old_runtimes_respects_keep_count_family_and_in_use() {
        let tmp = std::env::temp_dir().join(format!("bibulm_prune_{}", Uuid::new_v4()));
        fs::create_dir_all(&tmp).unwrap();
        let db = Database::open_in_memory().unwrap();

        let cuda_1 = make_record("b5100-cuda-12.4", "cuda-12.4", &tmp.join("b5100-cuda-12.4"), "2026-01-01T00:00:00Z", false);
        let cuda_2 = make_record("b5200-cuda-12.4", "cuda-12.4", &tmp.join("b5200-cuda-12.4"), "2026-01-02T00:00:00Z", false);
        let cuda_3 = make_record("b5300-cuda-13.1", "cuda-13.1", &tmp.join("b5300-cuda-13.1"), "2026-01-03T00:00:00Z", false);
        let vulkan = make_record("b5100-vulkan", "vulkan", &tmp.join("b5100-vulkan"), "2026-01-01T00:00:00Z", false);
        let custom = make_record("custom-1", "custom", &tmp.join("custom-1"), "2026-01-01T00:00:00Z", true);

        for r in [&cuda_1, &cuda_2, &cuda_3, &vulkan, &custom] {
            db.upsert_runtime(r).unwrap();
        }

        // keep_old = 1 keeps b5200-cuda-12.4 and removes b5100-cuda-12.4; leaves vulkan and custom intact.
        let removed = prune_old_runtimes(&db, &cuda_3, 1, &[]).unwrap();
        assert_eq!(removed, vec!["b5100-cuda-12.4"]);
        assert!(!tmp.join("b5100-cuda-12.4").exists());
        assert!(tmp.join("b5200-cuda-12.4").exists());
        assert!(tmp.join("b5300-cuda-13.1").exists());
        assert!(tmp.join("b5100-vulkan").exists());

        // keep_old = 0 skips in-use build, then removes it once no longer in use.
        let removed_protected = prune_old_runtimes(&db, &cuda_3, 0, &["b5200-cuda-12.4".to_string()]).unwrap();
        assert!(removed_protected.is_empty());
        assert!(tmp.join("b5200-cuda-12.4").exists());

        let removed_all_old = prune_old_runtimes(&db, &cuda_3, 0, &[]).unwrap();
        assert_eq!(removed_all_old, vec!["b5200-cuda-12.4"]);
        assert!(!tmp.join("b5200-cuda-12.4").exists());

        let remaining: Vec<String> = db.list_runtimes().unwrap().into_iter().map(|r| r.id).collect();
        assert!(remaining.contains(&"b5300-cuda-13.1".to_string()));
        assert!(remaining.contains(&"b5100-vulkan".to_string()));
        assert!(remaining.contains(&"custom-1".to_string()));
        assert_eq!(remaining.len(), 3);

        let _ = fs::remove_dir_all(&tmp);
    }
}
