use crate::secrets::{constant_time_eq, decrypt_secret, encrypt_secret, mask_api_token, sha256_hex};
use crate::{AppSettings, LoadConfig, MemoryPlacementReport};
use chrono::Utc;
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::{Arc, Mutex};
use thiserror::Error;
use uuid::Uuid;

#[derive(Debug, Error)]
pub enum DbError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Database mutex poisoned")]
    LockPoisoned,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApiTokenRecord {
    pub id: String,
    pub name: String,
    pub token: String,
    pub created_at: String,
    pub last_used_at: Option<String>,
    pub enabled: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NamedPresetRecord {
    pub id: String,
    pub name: String,
    /// `"sampling"` or `"load"`
    pub kind: String,
    pub payload_json: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstalledRuntimeRecord {
    pub id: String,
    pub tag: String,
    pub backend: String,
    pub display_name: String,
    pub install_dir: String,
    pub server_exe: String,
    pub bench_exe: Option<String>,
    pub is_custom: bool,
    pub installed_at: String,
    pub release_notes: Option<String>,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DownloadRecord {
    pub id: String,
    /// `"model"` or `"runtime"`
    pub kind: String,
    pub title: String,
    pub repo_id: Option<String>,
    pub dest_dir: String,
    pub total_bytes: u64,
    pub downloaded_bytes: u64,
    /// `"queued"`, `"downloading"`, `"paused"`, `"completed"`, `"failed"`, `"cancelled"`
    pub status: String,
    pub error: Option<String>,
    #[serde(default)]
    pub files_json: Option<String>,
    pub created_at: String,
    pub completed_at: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkRunRecord {
    pub id: String,
    pub created_at: String,
    pub model_id: String,
    pub runtime_tag: String,
    /// `"quick_test"` or `"llama_bench"`
    pub kind: String,
    pub prompt_preset: String,
    pub prompt_tokens: u32,
    pub reasoning_tokens: u32,
    pub output_tokens: u32,
    pub ttft_ms: f64,
    pub prompt_eval_tok_s: f64,
    pub generation_tok_s: f64,
    pub peak_vram_mib: f64,
    pub peak_ram_mib: f64,
    pub placement: MemoryPlacementReport,
    pub load_config: LoadConfig,
    pub notes: Option<String>,
}

/// Thread-safe SQLite store backing all persistent state in BibuLM.
#[derive(Clone)]
pub struct Database {
    conn: Arc<Mutex<Connection>>,
}

impl Database {
    pub fn open(path: &Path) -> Result<Self, DbError> {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let conn = Connection::open(path)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
        }
        let db = Self {
            conn: Arc::new(Mutex::new(conn)),
        };
        db.init_schema()?;
        Ok(db)
    }

    pub fn open_in_memory() -> Result<Self, DbError> {
        let conn = Connection::open_in_memory()?;
        let db = Self {
            conn: Arc::new(Mutex::new(conn)),
        };
        db.init_schema()?;
        Ok(db)
    }

    fn init_schema(&self) -> Result<(), DbError> {
        let conn = self.conn.lock().map_err(|_| DbError::LockPoisoned)?;
        conn.execute_batch(
            r#"
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS kv_settings (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS model_presets (
                model_id TEXT PRIMARY KEY,
                config_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS named_presets (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS api_tokens (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                token TEXT NOT NULL,
                token_hash TEXT,
                token_prefix TEXT,
                created_at TEXT NOT NULL,
                last_used_at TEXT,
                enabled INTEGER NOT NULL DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS installed_runtimes (
                id TEXT PRIMARY KEY,
                tag TEXT NOT NULL,
                backend TEXT NOT NULL,
                display_name TEXT NOT NULL,
                install_dir TEXT NOT NULL,
                server_exe TEXT NOT NULL,
                bench_exe TEXT,
                is_custom INTEGER NOT NULL DEFAULT 0,
                installed_at TEXT NOT NULL,
                release_notes TEXT,
                size_bytes INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS download_history (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                title TEXT NOT NULL,
                repo_id TEXT,
                dest_dir TEXT NOT NULL,
                total_bytes INTEGER NOT NULL DEFAULT 0,
                downloaded_bytes INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL,
                error TEXT,
                files_json TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS benchmark_runs (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                model_id TEXT NOT NULL,
                runtime_tag TEXT NOT NULL,
                kind TEXT NOT NULL,
                prompt_preset TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                reasoning_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                ttft_ms REAL NOT NULL,
                prompt_eval_tok_s REAL NOT NULL,
                generation_tok_s REAL NOT NULL,
                peak_vram_mib REAL NOT NULL,
                peak_ram_mib REAL NOT NULL,
                placement_json TEXT NOT NULL,
                load_config_json TEXT NOT NULL,
                notes TEXT
            );
            "#,
        )?;

        Self::migrate_api_tokens(&conn)?;
        Self::migrate_download_history(&conn)?;
        Ok(())
    }

    fn table_has_column(conn: &Connection, table: &str, column: &str) -> Result<bool, DbError> {
        let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
        let cols = stmt.query_map([], |r| r.get::<_, String>(1))?;
        for c in cols {
            if c? == column {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn migrate_api_tokens(conn: &Connection) -> Result<(), DbError> {
        if !Self::table_has_column(conn, "api_tokens", "token_hash")? {
            conn.execute("ALTER TABLE api_tokens ADD COLUMN token_hash TEXT", [])?;
        }
        if !Self::table_has_column(conn, "api_tokens", "token_prefix")? {
            conn.execute("ALTER TABLE api_tokens ADD COLUMN token_prefix TEXT", [])?;
        }
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_api_tokens_hash ON api_tokens(token_hash)",
            [],
        )?;

        let mut stmt = conn.prepare("SELECT id, token FROM api_tokens WHERE token_hash IS NULL")?;
        let pending: Vec<(String, String)> = stmt
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
            .filter_map(Result::ok)
            .collect();
        for (id, raw_tok) in pending {
            let hash = sha256_hex(&raw_tok);
            let masked = mask_api_token(&raw_tok);
            conn.execute(
                "UPDATE api_tokens SET token_hash = ?1, token_prefix = ?2, token = ?2 WHERE id = ?3",
                params![hash, masked, id],
            )?;
        }
        Ok(())
    }

    fn migrate_download_history(conn: &Connection) -> Result<(), DbError> {
        if !Self::table_has_column(conn, "download_history", "files_json")? {
            conn.execute("ALTER TABLE download_history ADD COLUMN files_json TEXT", [])?;
        }
        Ok(())
    }

    // --- App Settings ---

    pub fn load_settings(&self) -> Result<AppSettings, DbError> {
        let conn = self.conn.lock().map_err(|_| DbError::LockPoisoned)?;
        let raw: Option<String> = conn
            .query_row(
                "SELECT value_json FROM kv_settings WHERE key = 'app_settings'",
                [],
                |r| r.get(0),
            )
            .optional()?;
        match raw {
            Some(json) => {
                let mut settings: AppSettings = serde_json::from_str(&json)?;
                if let Some(tok) = settings.hf_token.as_mut() {
                    *tok = decrypt_secret(tok);
                }
                if let Some(tok) = settings.github_token.as_mut() {
                    *tok = decrypt_secret(tok);
                }
                Ok(settings)
            }
            None => Ok(AppSettings::default()),
        }
    }

    pub fn save_settings(&self, settings: &AppSettings) -> Result<(), DbError> {
        let conn = self.conn.lock().map_err(|_| DbError::LockPoisoned)?;
        let mut stored = settings.clone();
        if let Some(tok) = stored.hf_token.as_mut() {
            *tok = encrypt_secret(tok);
        }
        if let Some(tok) = stored.github_token.as_mut() {
            *tok = encrypt_secret(tok);
        }
        let json = serde_json::to_string(&stored)?;
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO kv_settings (key, value_json, updated_at) VALUES ('app_settings', ?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json, updated_at = excluded.updated_at",
            params![json, now],
        )?;
        Ok(())
    }

    // --- Per-Model Presets ---

    pub fn get_model_preset(&self, model_id: &str) -> Result<Option<LoadConfig>, DbError> {
        let conn = self.conn.lock().map_err(|_| DbError::LockPoisoned)?;
        let raw: Option<String> = conn
            .query_row(
                "SELECT config_json FROM model_presets WHERE model_id = ?1",
                params![model_id],
                |r| r.get(0),
            )
            .optional()?;
        match raw {
            Some(json) => Ok(Some(serde_json::from_str(&json)?)),
            None => Ok(None),
        }
    }

    pub fn save_model_preset(&self, model_id: &str, config: &LoadConfig) -> Result<(), DbError> {
        let conn = self.conn.lock().map_err(|_| DbError::LockPoisoned)?;
        let json = serde_json::to_string(config)?;
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO model_presets (model_id, config_json, updated_at) VALUES (?1, ?2, ?3)
             ON CONFLICT(model_id) DO UPDATE SET config_json = excluded.config_json, updated_at = excluded.updated_at",
            params![model_id, json, now],
        )?;
        Ok(())
    }

    // --- Named Presets ---

    pub fn list_named_presets(&self, kind: Option<&str>) -> Result<Vec<NamedPresetRecord>, DbError> {
        let conn = self.conn.lock().map_err(|_| DbError::LockPoisoned)?;
        let mut out = Vec::new();
        if let Some(k) = kind {
            let mut stmt = conn.prepare(
                "SELECT id, name, kind, payload_json, updated_at FROM named_presets WHERE kind = ?1 ORDER BY updated_at DESC",
            )?;
            let rows = stmt.query_map(params![k], |r| {
                Ok(NamedPresetRecord {
                    id: r.get(0)?,
                    name: r.get(1)?,
                    kind: r.get(2)?,
                    payload_json: r.get(3)?,
                    updated_at: r.get(4)?,
                })
            })?;
            for row in rows {
                out.push(row?);
            }
        } else {
            let mut stmt = conn.prepare(
                "SELECT id, name, kind, payload_json, updated_at FROM named_presets ORDER BY updated_at DESC",
            )?;
            let rows = stmt.query_map([], |r| {
                Ok(NamedPresetRecord {
                    id: r.get(0)?,
                    name: r.get(1)?,
                    kind: r.get(2)?,
                    payload_json: r.get(3)?,
                    updated_at: r.get(4)?,
                })
            })?;
            for row in rows {
                out.push(row?);
            }
        }
        Ok(out)
    }

    pub fn upsert_named_preset(
        &self,
        id: Option<String>,
        name: String,
        kind: String,
        payload_json: String,
    ) -> Result<NamedPresetRecord, DbError> {
        let conn = self.conn.lock().map_err(|_| DbError::LockPoisoned)?;
        let rec = NamedPresetRecord {
            id: id.unwrap_or_else(|| Uuid::new_v4().to_string()),
            name,
            kind,
            payload_json,
            updated_at: Utc::now().to_rfc3339(),
        };
        conn.execute(
            "INSERT INTO named_presets (id, name, kind, payload_json, updated_at) VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(id) DO UPDATE SET name = excluded.name, payload_json = excluded.payload_json, updated_at = excluded.updated_at",
            params![rec.id, rec.name, rec.kind, rec.payload_json, rec.updated_at],
        )?;
        Ok(rec)
    }

    pub fn delete_named_preset(&self, id: &str) -> Result<(), DbError> {
        let conn = self.conn.lock().map_err(|_| DbError::LockPoisoned)?;
        conn.execute("DELETE FROM named_presets WHERE id = ?1", params![id])?;
        Ok(())
    }

    // --- API Tokens ---

    pub fn create_api_token(&self, name: &str) -> Result<ApiTokenRecord, DbError> {
        let conn = self.conn.lock().map_err(|_| DbError::LockPoisoned)?;
        let id = Uuid::new_v4().to_string();
        let raw_token = format!(
            "rs-{}",
            Uuid::new_v4().to_string().replace('-', "")
        );
        let token_hash = sha256_hex(&raw_token);
        let masked = mask_api_token(&raw_token);
        let created_at = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO api_tokens (id, name, token, token_hash, token_prefix, created_at, enabled) VALUES (?1, ?2, ?3, ?4, ?3, ?5, 1)",
            params![id, name, masked, token_hash, created_at],
        )?;
        Ok(ApiTokenRecord {
            id,
            name: name.to_string(),
            token: raw_token,
            created_at,
            last_used_at: None,
            enabled: true,
        })
    }

    pub fn list_api_tokens(&self) -> Result<Vec<ApiTokenRecord>, DbError> {
        let conn = self.conn.lock().map_err(|_| DbError::LockPoisoned)?;
        let mut stmt = conn.prepare(
            "SELECT id, name, COALESCE(token_prefix, token), created_at, last_used_at, enabled FROM api_tokens ORDER BY created_at DESC",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(ApiTokenRecord {
                id: r.get(0)?,
                name: r.get(1)?,
                token: r.get(2)?,
                created_at: r.get(3)?,
                last_used_at: r.get(4)?,
                enabled: r.get::<_, i64>(5)? != 0,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    pub fn validate_api_token(&self, bearer: &str) -> Result<bool, DbError> {
        let incoming = bearer.trim();
        if incoming.is_empty() {
            return Ok(false);
        }
        let incoming_hash = sha256_hex(incoming);
        let conn = self.conn.lock().map_err(|_| DbError::LockPoisoned)?;
        let mut stmt = conn.prepare(
            "SELECT id, token_hash, token FROM api_tokens WHERE enabled = 1",
        )?;
        let rows: Vec<(String, Option<String>, String)> = stmt
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
            .filter_map(Result::ok)
            .collect();
        drop(stmt);

        for (id, hash_opt, tok_fallback) in rows {
            let matched = match hash_opt {
                Some(ref h) => constant_time_eq(h.as_bytes(), incoming_hash.as_bytes()),
                None => constant_time_eq(tok_fallback.as_bytes(), incoming.as_bytes()),
            };
            if matched {
                let now = Utc::now().to_rfc3339();
                let _ = conn.execute(
                    "UPDATE api_tokens SET last_used_at = ?1 WHERE id = ?2",
                    params![now, id],
                );
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub fn delete_api_token(&self, id: &str) -> Result<(), DbError> {
        let conn = self.conn.lock().map_err(|_| DbError::LockPoisoned)?;
        conn.execute("DELETE FROM api_tokens WHERE id = ?1", params![id])?;
        Ok(())
    }

    pub fn set_api_token_enabled(&self, id: &str, enabled: bool) -> Result<(), DbError> {
        let conn = self.conn.lock().map_err(|_| DbError::LockPoisoned)?;
        conn.execute(
            "UPDATE api_tokens SET enabled = ?1 WHERE id = ?2",
            params![if enabled { 1 } else { 0 }, id],
        )?;
        Ok(())
    }

    // --- Installed Runtimes ---

    pub fn upsert_runtime(&self, rec: &InstalledRuntimeRecord) -> Result<(), DbError> {
        let conn = self.conn.lock().map_err(|_| DbError::LockPoisoned)?;
        conn.execute(
            "INSERT INTO installed_runtimes
             (id, tag, backend, display_name, install_dir, server_exe, bench_exe, is_custom, installed_at, release_notes, size_bytes)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
             ON CONFLICT(id) DO UPDATE SET
               tag = excluded.tag,
               backend = excluded.backend,
               display_name = excluded.display_name,
               install_dir = excluded.install_dir,
               server_exe = excluded.server_exe,
               bench_exe = excluded.bench_exe,
               release_notes = excluded.release_notes,
               size_bytes = excluded.size_bytes",
            params![
                rec.id,
                rec.tag,
                rec.backend,
                rec.display_name,
                rec.install_dir,
                rec.server_exe,
                rec.bench_exe,
                if rec.is_custom { 1 } else { 0 },
                rec.installed_at,
                rec.release_notes,
                rec.size_bytes as i64,
            ],
        )?;
        Ok(())
    }

    pub fn list_runtimes(&self) -> Result<Vec<InstalledRuntimeRecord>, DbError> {
        let conn = self.conn.lock().map_err(|_| DbError::LockPoisoned)?;
        let mut stmt = conn.prepare(
            "SELECT id, tag, backend, display_name, install_dir, server_exe, bench_exe, is_custom, installed_at, release_notes, size_bytes
             FROM installed_runtimes ORDER BY installed_at DESC",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(InstalledRuntimeRecord {
                id: r.get(0)?,
                tag: r.get(1)?,
                backend: r.get(2)?,
                display_name: r.get(3)?,
                install_dir: r.get(4)?,
                server_exe: r.get(5)?,
                bench_exe: r.get(6)?,
                is_custom: r.get::<_, i64>(7)? != 0,
                installed_at: r.get(8)?,
                release_notes: r.get(9)?,
                size_bytes: r.get::<_, i64>(10).unwrap_or(0).max(0) as u64,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    pub fn delete_runtime(&self, id: &str) -> Result<(), DbError> {
        let conn = self.conn.lock().map_err(|_| DbError::LockPoisoned)?;
        conn.execute("DELETE FROM installed_runtimes WHERE id = ?1", params![id])?;
        Ok(())
    }

    // --- Download History ---

    pub fn upsert_download(&self, rec: &DownloadRecord) -> Result<(), DbError> {
        let conn = self.conn.lock().map_err(|_| DbError::LockPoisoned)?;
        conn.execute(
            "INSERT INTO download_history
             (id, kind, title, repo_id, dest_dir, total_bytes, downloaded_bytes, status, error, files_json, created_at, completed_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
             ON CONFLICT(id) DO UPDATE SET
               total_bytes = excluded.total_bytes,
               downloaded_bytes = excluded.downloaded_bytes,
               status = excluded.status,
               error = excluded.error,
               files_json = COALESCE(excluded.files_json, download_history.files_json),
               completed_at = excluded.completed_at",
            params![
                rec.id,
                rec.kind,
                rec.title,
                rec.repo_id,
                rec.dest_dir,
                rec.total_bytes as i64,
                rec.downloaded_bytes as i64,
                rec.status,
                rec.error,
                rec.files_json,
                rec.created_at,
                rec.completed_at,
            ],
        )?;
        Ok(())
    }

    pub fn list_downloads(&self) -> Result<Vec<DownloadRecord>, DbError> {
        let conn = self.conn.lock().map_err(|_| DbError::LockPoisoned)?;
        let mut stmt = conn.prepare(
            "SELECT id, kind, title, repo_id, dest_dir, total_bytes, downloaded_bytes, status, error, files_json, created_at, completed_at
             FROM download_history ORDER BY created_at DESC LIMIT 100",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(DownloadRecord {
                id: r.get(0)?,
                kind: r.get(1)?,
                title: r.get(2)?,
                repo_id: r.get(3)?,
                dest_dir: r.get(4)?,
                total_bytes: r.get::<_, i64>(5).unwrap_or(0).max(0) as u64,
                downloaded_bytes: r.get::<_, i64>(6).unwrap_or(0).max(0) as u64,
                status: r.get(7)?,
                error: r.get(8)?,
                files_json: r.get(9)?,
                created_at: r.get(10)?,
                completed_at: r.get(11)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    pub fn clear_completed_downloads(&self) -> Result<(), DbError> {
        let conn = self.conn.lock().map_err(|_| DbError::LockPoisoned)?;
        conn.execute(
            "DELETE FROM download_history WHERE status IN ('completed', 'failed', 'cancelled')",
            [],
        )?;
        Ok(())
    }

    // --- Benchmark Runs ---

    pub fn insert_benchmark_run(&self, rec: &BenchmarkRunRecord) -> Result<(), DbError> {
        let conn = self.conn.lock().map_err(|_| DbError::LockPoisoned)?;
        let placement_json = serde_json::to_string(&rec.placement)?;
        let load_config_json = serde_json::to_string(&rec.load_config)?;
        conn.execute(
            "INSERT INTO benchmark_runs
             (id, created_at, model_id, runtime_tag, kind, prompt_preset, prompt_tokens, reasoning_tokens, output_tokens, ttft_ms, prompt_eval_tok_s, generation_tok_s, peak_vram_mib, peak_ram_mib, placement_json, load_config_json, notes)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)",
            params![
                rec.id,
                rec.created_at,
                rec.model_id,
                rec.runtime_tag,
                rec.kind,
                rec.prompt_preset,
                rec.prompt_tokens,
                rec.reasoning_tokens,
                rec.output_tokens,
                rec.ttft_ms,
                rec.prompt_eval_tok_s,
                rec.generation_tok_s,
                rec.peak_vram_mib,
                rec.peak_ram_mib,
                placement_json,
                load_config_json,
                rec.notes,
            ],
        )?;
        Ok(())
    }

    pub fn list_benchmark_runs(&self, limit: usize) -> Result<Vec<BenchmarkRunRecord>, DbError> {
        let conn = self.conn.lock().map_err(|_| DbError::LockPoisoned)?;
        let mut stmt = conn.prepare(
            "SELECT id, created_at, model_id, runtime_tag, kind, prompt_preset, prompt_tokens, reasoning_tokens, output_tokens, ttft_ms, prompt_eval_tok_s, generation_tok_s, peak_vram_mib, peak_ram_mib, placement_json, load_config_json, notes
             FROM benchmark_runs ORDER BY created_at DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit as i64], |r| {
            let placement_str: String = r.get(14)?;
            let config_str: String = r.get(15)?;
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, String>(4)?,
                r.get::<_, String>(5)?,
                r.get::<_, u32>(6)?,
                r.get::<_, u32>(7)?,
                r.get::<_, u32>(8)?,
                r.get::<_, f64>(9)?,
                r.get::<_, f64>(10)?,
                r.get::<_, f64>(11)?,
                r.get::<_, f64>(12)?,
                r.get::<_, f64>(13)?,
                placement_str,
                config_str,
                r.get::<_, Option<String>>(16)?,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (
                id,
                created_at,
                model_id,
                runtime_tag,
                kind,
                prompt_preset,
                prompt_tokens,
                reasoning_tokens,
                output_tokens,
                ttft_ms,
                prompt_eval_tok_s,
                generation_tok_s,
                peak_vram_mib,
                peak_ram_mib,
                placement_str,
                config_str,
                notes,
            ) = row?;
            let placement = serde_json::from_str(&placement_str).unwrap_or_default();
            let load_config = serde_json::from_str(&config_str).unwrap_or_default();
            out.push(BenchmarkRunRecord {
                id,
                created_at,
                model_id,
                runtime_tag,
                kind,
                prompt_preset,
                prompt_tokens,
                reasoning_tokens,
                output_tokens,
                ttft_ms,
                prompt_eval_tok_s,
                generation_tok_s,
                peak_vram_mib,
                peak_ram_mib,
                placement,
                load_config,
                notes,
            });
        }
        Ok(out)
    }

    pub fn delete_benchmark_run(&self, id: &str) -> Result<(), DbError> {
        let conn = self.conn.lock().map_err(|_| DbError::LockPoisoned)?;
        conn.execute("DELETE FROM benchmark_runs WHERE id = ?1", params![id])?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MoeOffloadMode;

    #[test]
    fn sqlite_roundtrips_settings_presets_tokens_and_benchmarks() {
        let db = Database::open_in_memory().unwrap();

        let mut settings = AppSettings::default();
        settings.server.port = 8089;
        db.save_settings(&settings).unwrap();
        assert_eq!(db.load_settings().unwrap().server.port, 8089);

        let mut cfg = LoadConfig::new("C:/models/test.gguf");
        cfg.moe_offload = MoeOffloadMode::FirstNLayersExpertsCpu(12);
        db.save_model_preset("test-model", &cfg).unwrap();
        let loaded = db.get_model_preset("test-model").unwrap().unwrap();
        assert_eq!(loaded.moe_offload, MoeOffloadMode::FirstNLayersExpertsCpu(12));

        let tok = db.create_api_token("IDE").unwrap();
        assert!(tok.token.starts_with("rs-"));
        assert!(db.validate_api_token(&tok.token).unwrap());
        assert!(!db.validate_api_token("rs-invalid").unwrap());
        let listed = db.list_api_tokens().unwrap();
        assert_eq!(listed.len(), 1);
        assert!(listed[0].token.contains("••••"));
        assert_ne!(listed[0].token, tok.token);

        let run = BenchmarkRunRecord {
            id: Uuid::new_v4().to_string(),
            created_at: Utc::now().to_rfc3339(),
            model_id: "test-model".into(),
            runtime_tag: "b5120".into(),
            kind: "quick_test".into(),
            prompt_preset: "Code Gen".into(),
            prompt_tokens: 512,
            reasoning_tokens: 200,
            output_tokens: 180,
            ttft_ms: 145.2,
            prompt_eval_tok_s: 1840.5,
            generation_tok_s: 72.4,
            peak_vram_mib: 12400.0,
            peak_ram_mib: 4200.0,
            placement: MemoryPlacementReport::default(),
            load_config: cfg,
            notes: None,
        };
        db.insert_benchmark_run(&run).unwrap();
        let history = db.list_benchmark_runs(10).unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].generation_tok_s, 72.4);
    }

    #[test]
    fn migrates_legacy_plaintext_tokens_and_persists_download_files_json() {
        let db = Database::open_in_memory().unwrap();
        {
            let conn = db.conn.lock().unwrap();
            conn.execute(
                "INSERT INTO api_tokens (id, name, token, created_at, enabled) VALUES ('legacy-1', 'Old Key', 'rs-legacyplaintext1234567890', '2026-01-01T00:00:00Z', 1)",
                [],
            )
            .unwrap();
            Database::migrate_api_tokens(&conn).unwrap();
        }
        assert!(db.validate_api_token("rs-legacyplaintext1234567890").unwrap());
        let listed = db.list_api_tokens().unwrap();
        assert!(listed[0].token.contains("••••"));

        let dl = DownloadRecord {
            id: "dl-1".into(),
            kind: "model".into(),
            title: "Test Model".into(),
            repo_id: Some("org/repo".into()),
            dest_dir: "C:/models".into(),
            total_bytes: 1024,
            downloaded_bytes: 512,
            status: "paused".into(),
            error: None,
            files_json: Some("[{\"file_name\":\"m.gguf\"}]".into()),
            created_at: Utc::now().to_rfc3339(),
            completed_at: None,
        };
        db.upsert_download(&dl).unwrap();
        let dls = db.list_downloads().unwrap();
        assert_eq!(dls.len(), 1);
        assert_eq!(dls[0].files_json.as_deref(), Some("[{\"file_name\":\"m.gguf\"}]"));
    }
}
