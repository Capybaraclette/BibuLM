use crate::hf_client::HfDownloadFile;
use chrono::Utc;
use futures_util::StreamExt;
use bibulm_core::{Database, DownloadRecord};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use thiserror::Error;
use tokio::sync::{Mutex, Semaphore, broadcast};
use uuid::Uuid;

#[derive(Debug, Error)]
pub enum DownloadError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("HTTP status {status} downloading {url}")]
    HttpStatus { status: u16, url: String },
    #[error("SHA-256 verification mismatch for {file}: expected {expected}, got {actual}")]
    ChecksumMismatch {
        file: String,
        expected: String,
        actual: String,
    },
    #[error("Download paused")]
    Paused,
    #[error("Download cancelled")]
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadJobState {
    pub id: String,
    pub kind: String,
    pub title: String,
    pub repo_id: Option<String>,
    pub dest_dir: String,
    pub files: Vec<HfDownloadFile>,
    pub current_file: Option<String>,
    pub total_bytes: u64,
    pub downloaded_bytes: u64,
    pub speed_bytes_per_sec: f64,
    pub eta_seconds: Option<u64>,
    /// `"queued"`, `"downloading"`, `"verifying"`, `"paused"`, `"completed"`, `"failed"`, `"cancelled"`
    pub status: String,
    pub error: Option<String>,
    pub created_at: String,
    pub completed_at: Option<String>,
}

struct ActiveControl {
    pause_flag: Arc<AtomicBool>,
    cancel_flag: Arc<AtomicBool>,
    hf_token: Option<String>,
}

/// Manages concurrent resumable HTTP Range downloads with pause/resume/cancel/retry,
/// SHA-256 verification, live progress events, and SQLite history persistence.
pub struct DownloadManager {
    db: Database,
    jobs: Arc<Mutex<HashMap<String, DownloadJobState>>>,
    controls: Arc<Mutex<HashMap<String, ActiveControl>>>,
    semaphore: Arc<Semaphore>,
    events_tx: broadcast::Sender<DownloadJobState>,
}

impl DownloadManager {
    pub fn new(db: Database, max_concurrent: usize) -> Self {
        let (events_tx, _) = broadcast::channel(256);
        let mut initial = HashMap::new();
        if let Ok(records) = db.list_downloads() {
            for r in records {
                // Any interrupted in-progress download from a prior run starts as paused/failed safely.
                let status = if r.status == "downloading" || r.status == "queued" {
                    "paused".to_string()
                } else {
                    r.status
                };
                let files: Vec<HfDownloadFile> = r
                    .files_json
                    .as_deref()
                    .and_then(|s| serde_json::from_str(s).ok())
                    .unwrap_or_default();
                initial.insert(
                    r.id.clone(),
                    DownloadJobState {
                        id: r.id,
                        kind: r.kind,
                        title: r.title,
                        repo_id: r.repo_id,
                        dest_dir: r.dest_dir,
                        files,
                        current_file: None,
                        total_bytes: r.total_bytes,
                        downloaded_bytes: r.downloaded_bytes,
                        speed_bytes_per_sec: 0.0,
                        eta_seconds: None,
                        status,
                        error: r.error,
                        created_at: r.created_at,
                        completed_at: r.completed_at,
                    },
                );
            }
        }
        Self {
            db,
            jobs: Arc::new(Mutex::new(initial)),
            controls: Arc::new(Mutex::new(HashMap::new())),
            semaphore: Arc::new(Semaphore::new(max_concurrent.max(1))),
            events_tx,
        }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<DownloadJobState> {
        self.events_tx.subscribe()
    }

    pub async fn list_jobs(&self) -> Vec<DownloadJobState> {
        let mut list: Vec<DownloadJobState> = self.jobs.lock().await.values().cloned().collect();
        list.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        list
    }

    /// Enqueues a model (single or multi-part GGUF + optional mmproj) for resumable download.
    pub async fn enqueue_model_download(
        &self,
        title: String,
        repo_id: String,
        dest_dir: PathBuf,
        files: Vec<HfDownloadFile>,
        hf_token: Option<String>,
    ) -> DownloadJobState {
        let id = Uuid::new_v4().to_string();
        let total_bytes: u64 = files.iter().map(|f| f.size_bytes).sum();
        let job = DownloadJobState {
            id: id.clone(),
            kind: "model".to_string(),
            title,
            repo_id: Some(repo_id),
            dest_dir: dest_dir.to_string_lossy().into_owned(),
            files,
            current_file: None,
            total_bytes,
            downloaded_bytes: 0,
            speed_bytes_per_sec: 0.0,
            eta_seconds: None,
            status: "queued".to_string(),
            error: None,
            created_at: Utc::now().to_rfc3339(),
            completed_at: None,
        };

        self.persist_and_emit(&job).await;
        self.jobs.lock().await.insert(id.clone(), job.clone());
        self.spawn_worker(id, hf_token).await;
        job
    }

    /// Records a runtime installation download in the unified Downloads list & emits progress.
    pub async fn record_external_job(&self, job: DownloadJobState) {
        self.jobs.lock().await.insert(job.id.clone(), job.clone());
        self.persist_and_emit(&job).await;
    }

    /// Updates an in-memory external job's progress and broadcasts it without writing to SQLite.
    pub async fn update_external_progress(&self, job: DownloadJobState) {
        self.jobs.lock().await.insert(job.id.clone(), job.clone());
        let _ = self.events_tx.send(job);
    }

    pub async fn pause(&self, id: &str) -> bool {
        if let Some(ctrl) = self.controls.lock().await.get(id) {
            ctrl.pause_flag.store(true, Ordering::SeqCst);
            return true;
        }
        false
    }

    pub async fn cancel(&self, id: &str) -> bool {
        if let Some(ctrl) = self.controls.lock().await.get(id) {
            ctrl.cancel_flag.store(true, Ordering::SeqCst);
        }
        let mut jobs = self.jobs.lock().await;
        if let Some(job) = jobs.get_mut(id) {
            job.status = "cancelled".to_string();
            job.speed_bytes_per_sec = 0.0;
            job.eta_seconds = None;
            let snap = job.clone();
            drop(jobs);
            self.persist_and_emit(&snap).await;
            return true;
        }
        false
    }

    pub async fn resume_or_retry(&self, id: &str, hf_token_override: Option<String>) -> bool {
        let token = {
            let existing_tok = self
                .controls
                .lock()
                .await
                .get(id)
                .and_then(|c| c.hf_token.clone());
            hf_token_override.or(existing_tok)
        };
        let can_restart = {
            let mut jobs = self.jobs.lock().await;
            if let Some(job) = jobs.get_mut(id) {
                if job.files.is_empty() {
                    false
                } else {
                    job.status = "queued".to_string();
                    job.error = None;
                    let snap = job.clone();
                    drop(jobs);
                    self.persist_and_emit(&snap).await;
                    true
                }
            } else {
                false
            }
        };
        if can_restart {
            self.spawn_worker(id.to_string(), token).await;
        }
        can_restart
    }

    pub async fn clear_completed(&self) {
        let _ = self.db.clear_completed_downloads();
        let mut jobs = self.jobs.lock().await;
        jobs.retain(|_, j| {
            !matches!(
                j.status.as_str(),
                "completed" | "failed" | "cancelled"
            )
        });
    }

    async fn spawn_worker(&self, id: String, hf_token: Option<String>) {
        let pause_flag = Arc::new(AtomicBool::new(false));
        let cancel_flag = Arc::new(AtomicBool::new(false));
        self.controls.lock().await.insert(
            id.clone(),
            ActiveControl {
                pause_flag: pause_flag.clone(),
                cancel_flag: cancel_flag.clone(),
                hf_token: hf_token.clone(),
            },
        );

        let jobs = self.jobs.clone();
        let db = self.db.clone();
        let sem = self.semaphore.clone();
        let tx = self.events_tx.clone();

        tokio::spawn(async move {
            let _permit = match sem.acquire_owned().await {
                Ok(p) => p,
                Err(_) => return,
            };

            let (files, dest_dir) = {
                let mut map = jobs.lock().await;
                let Some(job) = map.get_mut(&id) else {
                    return;
                };
                job.status = "downloading".to_string();
                let snap = job.clone();
                let _ = db.upsert_download(&to_record(&snap));
                let _ = tx.send(snap);
                (job.files.clone(), PathBuf::from(&job.dest_dir))
            };

            let result = run_resumable_job(
                &id,
                &files,
                &dest_dir,
                hf_token.as_deref(),
                &pause_flag,
                &cancel_flag,
                &jobs,
                &tx,
            )
            .await;

            let final_snap = {
                let mut map = jobs.lock().await;
                let Some(job) = map.get_mut(&id) else {
                    return;
                };
                job.speed_bytes_per_sec = 0.0;
                job.eta_seconds = None;
                match result {
                    Ok(()) => {
                        job.status = "completed".to_string();
                        job.downloaded_bytes = job.total_bytes;
                        job.completed_at = Some(Utc::now().to_rfc3339());
                        job.error = None;
                    }
                    Err(DownloadError::Paused) => {
                        job.status = "paused".to_string();
                    }
                    Err(DownloadError::Cancelled) => {
                        job.status = "cancelled".to_string();
                    }
                    Err(e) => {
                        job.status = "failed".to_string();
                        job.error = Some(e.to_string());
                    }
                }
                job.clone()
            };
            let _ = db.upsert_download(&to_record(&final_snap));
            let _ = tx.send(final_snap);
        });
    }

    async fn persist_and_emit(&self, job: &DownloadJobState) {
        let _ = self.db.upsert_download(&to_record(job));
        let _ = self.events_tx.send(job.clone());
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_resumable_job(
    id: &str,
    files: &[HfDownloadFile],
    dest_dir: &Path,
    hf_token: Option<&str>,
    pause_flag: &AtomicBool,
    cancel_flag: &AtomicBool,
    jobs: &Arc<Mutex<HashMap<String, DownloadJobState>>>,
    tx: &broadcast::Sender<DownloadJobState>,
) -> Result<(), DownloadError> {
    fs::create_dir_all(dest_dir)?;
    let client = reqwest::Client::builder()
        .user_agent("Bibulm/0.3 (Resumable GGUF Downloader)")
        .build()?;

    let mut completed_prefix_bytes: u64 = 0;

    for f in files {
        if cancel_flag.load(Ordering::SeqCst) {
            return Err(DownloadError::Cancelled);
        }
        if pause_flag.load(Ordering::SeqCst) {
            return Err(DownloadError::Paused);
        }

        if f.file_name.is_empty()
            || f.file_name == "."
            || f.file_name == ".."
            || f.file_name.contains("..")
            || f.file_name.contains('/')
            || f.file_name.contains('\\')
            || f.file_name.contains('\0')
        {
            return Err(DownloadError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Invalid download file name: '{}'", f.file_name),
            )));
        }

        let final_path = dest_dir.join(&f.file_name);
        let part_path = dest_dir.join(format!("{}.part", f.file_name));

        // Skip file if already fully downloaded on disk
        if final_path.is_file() {
            let existing_len = fs::metadata(&final_path).map(|m| m.len()).unwrap_or(0);
            if f.size_bytes == 0 || existing_len == f.size_bytes {
                completed_prefix_bytes = completed_prefix_bytes.saturating_add(existing_len);
                continue;
            }
        }

        let mut existing_part = if part_path.is_file() {
            fs::metadata(&part_path).map(|m| m.len()).unwrap_or(0)
        } else {
            0
        };

        let mut req = client.get(&f.url);
        if let Some(tok) = hf_token.map(str::trim).filter(|s| !s.is_empty()) {
            req = req.bearer_auth(tok);
        }
        if existing_part > 0 {
            req = req.header(reqwest::header::RANGE, format!("bytes={existing_part}-"));
        }

        let resp = req.send().await?;
        let status = resp.status();
        let mut out_file = if status == reqwest::StatusCode::PARTIAL_CONTENT && existing_part > 0 {
            let mut file = OpenOptions::new().write(true).open(&part_path)?;
            file.seek(SeekFrom::End(0))?;
            file
        } else if status.is_success() {
            existing_part = 0;
            File::create(&part_path)?
        } else {
            return Err(DownloadError::HttpStatus {
                status: status.as_u16(),
                url: f.url.clone(),
            });
        };

        let mut file_downloaded = existing_part;
        let mut last_emit = Instant::now();
        let mut speed_window_start = Instant::now();
        let mut speed_window_bytes: u64 = 0;
        let mut stream = resp.bytes_stream();

        while let Some(chunk) = stream.next().await {
            if cancel_flag.load(Ordering::SeqCst) {
                drop(out_file);
                let _ = fs::remove_file(&part_path);
                return Err(DownloadError::Cancelled);
            }
            if pause_flag.load(Ordering::SeqCst) {
                let _ = out_file.flush();
                return Err(DownloadError::Paused);
            }

            let bytes = chunk?;
            out_file.write_all(&bytes)?;
            let n = bytes.len() as u64;
            file_downloaded = file_downloaded.saturating_add(n);
            speed_window_bytes = speed_window_bytes.saturating_add(n);

            if last_emit.elapsed().as_millis() >= 250 {
                let dt = speed_window_start.elapsed().as_secs_f64().max(0.05);
                let speed = speed_window_bytes as f64 / dt;
                if dt >= 1.5 {
                    speed_window_start = Instant::now();
                    speed_window_bytes = 0;
                }
                let total_done = completed_prefix_bytes.saturating_add(file_downloaded);
                let mut map = jobs.lock().await;
                if let Some(job) = map.get_mut(id) {
                    job.current_file = Some(f.file_name.clone());
                    job.downloaded_bytes = total_done;
                    job.speed_bytes_per_sec = speed;
                    if speed > 1.0 && job.total_bytes > total_done {
                        job.eta_seconds = Some(((job.total_bytes - total_done) as f64 / speed) as u64);
                    } else {
                        job.eta_seconds = None;
                    }
                    let _ = tx.send(job.clone());
                }
                last_emit = Instant::now();
            }
        }
        out_file.flush()?;
        drop(out_file);

        // Verify SHA-256 if HuggingFace LFS hash is available
        if let Some(expected_sha) = &f.sha256 {
            {
                let mut map = jobs.lock().await;
                if let Some(job) = map.get_mut(id) {
                    job.status = "verifying".to_string();
                    job.current_file = Some(format!("Verifying SHA-256: {}", f.file_name));
                    let _ = tx.send(job.clone());
                }
            }
            let part_clone = part_path.clone();
            let actual_sha = tokio::task::spawn_blocking(move || compute_sha256(&part_clone))
                .await
                .map_err(|e| std::io::Error::other(e.to_string()))??;
            if !actual_sha.eq_ignore_ascii_case(expected_sha) {
                let _ = fs::remove_file(&part_path);
                return Err(DownloadError::ChecksumMismatch {
                    file: f.file_name.clone(),
                    expected: expected_sha.clone(),
                    actual: actual_sha,
                });
            }
        }

        if final_path.exists() {
            let _ = fs::remove_file(&final_path);
        }
        fs::rename(&part_path, &final_path)?;
        completed_prefix_bytes = completed_prefix_bytes.saturating_add(file_downloaded);
    }

    Ok(())
}

fn compute_sha256(path: &Path) -> std::io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(64);
    for b in digest {
        hex.push_str(&format!("{b:02x}"));
    }
    Ok(hex)
}

fn to_record(j: &DownloadJobState) -> DownloadRecord {
    let files_json = if j.files.is_empty() {
        None
    } else {
        serde_json::to_string(&j.files).ok()
    };
    DownloadRecord {
        id: j.id.clone(),
        kind: j.kind.clone(),
        title: j.title.clone(),
        repo_id: j.repo_id.clone(),
        dest_dir: j.dest_dir.clone(),
        total_bytes: j.total_bytes,
        downloaded_bytes: j.downloaded_bytes,
        status: j.status.clone(),
        error: j.error.clone(),
        files_json,
        created_at: j.created_at.clone(),
        completed_at: j.completed_at.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn persists_download_files_across_manager_restart() {
        let db = Database::open_in_memory().unwrap();
        let job_id = "dl-restart-test".to_string();
        let files = vec![HfDownloadFile {
            path: "model-q4_k_m.gguf".into(),
            file_name: "model-q4_k_m.gguf".into(),
            url: "https://huggingface.co/org/repo/resolve/main/model-q4_k_m.gguf".into(),
            size_bytes: 1_000_000,
            sha256: Some("abc".into()),
        }];
        let job = DownloadJobState {
            id: job_id.clone(),
            kind: "model".into(),
            title: "Test Model".into(),
            repo_id: Some("org/repo".into()),
            dest_dir: std::env::temp_dir().to_string_lossy().into_owned(),
            files: files.clone(),
            current_file: None,
            total_bytes: 1_000_000,
            downloaded_bytes: 250_000,
            speed_bytes_per_sec: 0.0,
            eta_seconds: None,
            status: "downloading".into(),
            error: None,
            created_at: Utc::now().to_rfc3339(),
            completed_at: None,
        };
        db.upsert_download(&to_record(&job)).unwrap();

        let mgr = DownloadManager::new(db, 2);
        let jobs = mgr.list_jobs().await;
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].status, "paused");
        assert_eq!(jobs[0].files.len(), 1);
        assert_eq!(jobs[0].files[0].file_name, "model-q4_k_m.gguf");
    }
}
