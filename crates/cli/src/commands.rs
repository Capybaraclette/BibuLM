//! Implementations of the `bibulm` subcommands.

use crate::args::{LoginArgs, ModelsCmd, RuntimeCmd, ServeArgs, SystemdArgs};
use crate::node::{HeadlessNode, shutdown_signal};
use anyhow::{Context, anyhow, bail};
use bibulm_core::AppSettings;
use bibulm_link::{LinkConnectionState, LinkStatus};
use bibulm_models::{HfSortOrder, scan_local_models_with_external};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

pub fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut v = bytes as f64;
    let mut i = 0;
    while v >= 1024.0 && i < UNITS.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{bytes} B")
    } else {
        format!("{v:.2} {}", UNITS[i])
    }
}

fn print_json<T: serde::Serialize>(value: &T) {
    match serde_json::to_string_pretty(value) {
        Ok(s) => println!("{s}"),
        Err(e) => eprintln!("failed to serialise output: {e}"),
    }
}

fn print_login_box(url: &str) {
    let line = "=".repeat(url.len().max(60) + 4);
    println!("\n{line}");
    println!("  Sign in to Tailscale to link this server to your BibuLM devices:\n");
    println!("  {url}\n");
    println!("  (or restart with --authkey / TS_AUTHKEY for unattended setup)");
    println!("{line}\n");
}

fn print_connected(status: &LinkStatus, gateway_url: &str, runtimes: usize, models: usize, manage: bool) {
    println!("\n[link] connected");
    println!("  device     : {}", status.hostname);
    if let Some(ip) = &status.tailscale_ip {
        println!("  tailscale  : {ip}");
    }
    if let Some(dns) = &status.magic_dns {
        println!("  magicdns   : {}", dns.trim_end_matches('.'));
    }
    if let Some(acc) = &status.account_login {
        println!("  account    : {acc}");
    }
    println!("  gateway    : {gateway_url}");
    println!("  runtimes   : {runtimes}   models: {models}");
    println!(
        "  remote mgmt: {}",
        if manage { "enabled" } else { "disabled" }
    );
    println!("  -> open BibuLM Desktop > Link to manage this server.\n");
}

async fn save_settings(node: &HeadlessNode, s: &AppSettings) -> anyhow::Result<()> {
    node.db.save_settings(s).map_err(|e| anyhow!(e.to_string()))
}

// ---------------------------------------------------------------- serve

pub async fn cmd_serve(node: &HeadlessNode, a: ServeArgs) -> anyhow::Result<()> {
    let (link_cfg, require_auth_created) = {
        let mut s = node.settings.write().await;
        if let Some(p) = a.port {
            s.server.port = p;
        }
        s.server.serve_on_lan = a.lan;
        s.server.jit_loading = true;
        if let Some(n) = a.name.clone() {
            s.link.device_name = Some(n);
        }
        if let Some(m) = a.mesh_port {
            s.link.mesh_port = m;
        }
        if let Some(p) = a.policy {
            s.link.access_policy = p;
        }
        if let Some(u) = a.control_url.clone() {
            s.link.control_url = Some(u);
        }
        if let Some(d) = a.models_dir.clone() {
            std::fs::create_dir_all(&d)?;
            s.models_dirs.retain(|p| p != &d);
            s.models_dirs.insert(0, d);
        }
        if let Some(d) = a.runtimes_dir.clone() {
            std::fs::create_dir_all(&d)?;
            s.runtimes_dir = d;
        }
        if a.hf_token.is_some() {
            s.hf_token = a.hf_token.clone();
        }
        if a.github_token.is_some() {
            s.github_token = a.github_token.clone();
        }
        s.link.enabled = !a.no_link;
        s.link.expose_local_models = true;
        s.link.allow_remote_load = !a.no_remote_load;
        s.link.allow_remote_manage = !a.no_remote_manage;

        // A LAN-exposed API must never be unauthenticated.
        let mut created: Option<String> = None;
        if s.server.serve_on_lan {
            s.server.require_auth = true;
            let tokens = node.db.list_api_tokens().unwrap_or_default();
            if !tokens.iter().any(|t| t.enabled) {
                let rec = node
                    .db
                    .create_api_token("headless-default")
                    .map_err(|e| anyhow!(e.to_string()))?;
                created = Some(rec.token);
            }
        }
        std::fs::create_dir_all(s.primary_models_dir())?;
        save_settings(node, &s).await?;
        (s.link.clone(), created)
    };

    node.spawn_background_tasks(true);
    let status = node
        .gateway
        .start()
        .await
        .map_err(|e| anyhow!("failed to start gateway: {e}"))?;
    println!("bibulm {} - headless server", env!("CARGO_PKG_VERSION"));
    println!("  data dir : {}", node.data_dir.display());
    println!("  gateway  : {} (bind {})", status.reachable_url, status.bind_address);
    if let Some(tok) = require_auth_created {
        println!("  api token: {tok}   (created because --lan exposes the API; shown once)");
    }

    let gateway_url = status.reachable_url.clone();
    if link_cfg.enabled {
        let mut st_rx = node.link.subscribe_status();
        let mut peers_rx = node.link.subscribe_peers();
        let db = node.db.clone();
        let settings = node.settings.clone();
        tokio::spawn(async move {
            let mut last: Option<(LinkConnectionState, Option<String>)> = None;
            while let Ok(st) = st_rx.recv().await {
                let key = (st.state, st.auth_url.clone());
                if last.as_ref() == Some(&key) {
                    continue;
                }
                last = Some(key);
                match st.state {
                    LinkConnectionState::NeedsLogin => {
                        if let Some(url) = st.auth_url.as_deref() {
                            print_login_box(url);
                        }
                    }
                    LinkConnectionState::Connected => {
                        let runtimes = db.list_runtimes().map(|r| r.len()).unwrap_or(0);
                        let (dirs, auto_discover, manage) = {
                            let s = settings.read().await;
                            (
                                s.models_dirs.clone(),
                                s.auto_discover_external_models,
                                s.link.allow_remote_manage,
                            )
                        };
                        let models = tokio::task::spawn_blocking(move || {
                            scan_local_models_with_external(&dirs, auto_discover).len()
                        })
                        .await
                        .unwrap_or(0);
                        print_connected(&st, &gateway_url, runtimes, models, manage);
                    }
                    LinkConnectionState::Error => {
                        eprintln!(
                            "[link] error: {}",
                            st.error.as_deref().unwrap_or("unknown error")
                        );
                    }
                    _ => {}
                }
            }
        });
        tokio::spawn(async move {
            let mut last_names: Vec<String> = Vec::new();
            while let Ok(peers) = peers_rx.recv().await {
                let mut names: Vec<String> = peers.iter().map(|p| p.device_name.clone()).collect();
                names.sort();
                if names != last_names {
                    println!("[link] {} peer(s): {}", names.len(), names.join(", "));
                    last_names = names;
                }
            }
        });

        let link = node.link.clone();
        let cfg = link_cfg.clone();
        let port = node.gateway.active_port().await;
        let authkey = a.authkey.clone();
        tokio::spawn(async move {
            if let Err(e) = link.connect_with_auth_key(&cfg, port, authkey.as_deref()).await {
                eprintln!("[link] could not start BibuLM Link: {e}");
                eprintln!("[link] the local OpenAI API stays available.");
            }
        });
    } else {
        println!("  link     : disabled (--no-link)");
    }

    println!("Press Ctrl+C to stop.");
    shutdown_signal().await;
    println!("\nShutting down...");
    node.link.disconnect(&link_cfg).await;
    node.gateway.stop().await;
    node.supervisor.unload_all().await;
    Ok(())
}

// ---------------------------------------------------------------- login / logout

async fn wait_for_link(
    node: &HeadlessNode,
    timeout: Duration,
    stop_on_needs_login: bool,
) -> anyhow::Result<LinkStatus> {
    let deadline = Instant::now() + timeout;
    let mut shown_url: Option<String> = None;
    loop {
        let st = node.link.status().await;
        match st.state {
            LinkConnectionState::Connected => return Ok(st),
            LinkConnectionState::Error => {
                bail!(st.error.unwrap_or_else(|| "link error".to_string()));
            }
            LinkConnectionState::NeedsLogin => {
                if let Some(url) = st.auth_url.clone()
                    && shown_url.as_deref() != Some(url.as_str())
                {
                    print_login_box(&url);
                    shown_url = Some(url);
                }
                if stop_on_needs_login {
                    return Ok(st);
                }
            }
            _ => {}
        }
        if Instant::now() >= deadline {
            bail!("timed out waiting for Tailscale connection");
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

pub async fn cmd_login(node: &HeadlessNode, a: LoginArgs) -> anyhow::Result<()> {
    let (cfg, port) = {
        let mut s = node.settings.write().await;
        if let Some(n) = a.name.clone() {
            s.link.device_name = Some(n);
        }
        if let Some(u) = a.control_url.clone() {
            s.link.control_url = Some(u);
        }
        if let Some(p) = a.policy {
            s.link.access_policy = p;
        }
        s.link.enabled = true;
        save_settings(node, &s).await?;
        (s.link.clone(), s.server.port)
    };
    println!("Starting BibuLM Link...");
    node.link
        .connect_with_auth_key(&cfg, port, a.authkey.as_deref())
        .await
        .map_err(|e| anyhow!(e))?;
    let result = wait_for_link(node, Duration::from_secs(300), false).await;
    let status = match result {
        Ok(st) => st,
        Err(e) => {
            node.link.disconnect(&cfg).await;
            return Err(e);
        }
    };
    println!("Signed in as {}.", status.account_login.as_deref().unwrap_or("unknown"));
    if let Some(ip) = &status.tailscale_ip {
        println!("This server is reachable on your tailnet at {ip} as '{}'.", status.hostname);
    }
    node.link.disconnect(&cfg).await;
    println!("Identity saved. Start the server with `bibulm serve`.");
    Ok(())
}

pub async fn cmd_logout(node: &HeadlessNode) -> anyhow::Result<()> {
    let port = node.settings.read().await.server.port;
    let health_url = format!("http://127.0.0.1:{port}/health");
    if let Ok(c) = reqwest::Client::builder().timeout(Duration::from_secs(1)).build()
        && c.get(&health_url).send().await.is_ok_and(|r| r.status().is_success())
    {
        bail!("a `bibulm serve` instance is currently running on port {port}; stop it before logging out");
    }

    let (cfg, port) = {
        let mut s = node.settings.write().await;
        s.link.enabled = false;
        save_settings(node, &s).await?;
        (s.link.clone(), s.server.port)
    };
    // Reconnect briefly with the saved identity so Tailscale can revoke this node server-side.
    if node.link.connect(&cfg, port).await.is_ok() {
        let _ = wait_for_link(node, Duration::from_secs(20), true).await;
    }
    node.link.logout(&cfg).await.map_err(|e| anyhow!(e))?;
    println!("Logged out. This device was removed from your tailnet and local identity was deleted.");
    Ok(())
}

// ---------------------------------------------------------------- status / hardware

pub async fn cmd_status(node: &HeadlessNode, json_out: bool) -> anyhow::Result<()> {
    let settings = node.settings.read().await.clone();
    let runtimes = node.db.list_runtimes().unwrap_or_default();
    let tokens = node.db.list_api_tokens().unwrap_or_default();
    let downloads = node.downloads.list_jobs().await;
    let dirs = settings.models_dirs.clone();
    let auto_discover = settings.auto_discover_external_models;
    let models = tokio::task::spawn_blocking(move || scan_local_models_with_external(&dirs, auto_discover))
        .await
        .unwrap_or_default();
    let sidecar = bibulm_link::manager::locate_sidecar_binary();
    let hw_summary = node
        .hardware
        .lock()
        .map(|h| h.profile().format_summary())
        .unwrap_or_default();

    let health_url = format!("http://127.0.0.1:{}/health", settings.server.port);
    let health = match reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
    {
        Ok(c) => match c.get(&health_url).send().await {
            Ok(r) if r.status().is_success() => r.json::<serde_json::Value>().await.ok(),
            _ => None,
        },
        Err(_) => None,
    };

    if json_out {
        print_json(&json!({
            "version": env!("CARGO_PKG_VERSION"),
            "data_dir": node.data_dir,
            "server": { "port": settings.server.port, "lan": settings.server.serve_on_lan, "running": health.is_some(), "health": health },
            "link": {
                "enabled": settings.link.enabled,
                "device_name": settings.link.device_name,
                "allow_remote_load": settings.link.allow_remote_load,
                "allow_remote_manage": settings.link.allow_remote_manage,
                "sidecar": sidecar,
            },
            "hardware": hw_summary,
            "runtimes": runtimes,
            "active_runtime_id": settings.active_runtime_id,
            "models": models.iter().map(|m| json!({"id": m.id, "size_bytes": m.size_bytes, "quantization": m.quantization})).collect::<Vec<_>>(),
            "downloads": downloads,
            "api_tokens": tokens.iter().map(|t| json!({"name": t.name, "token": t.token, "enabled": t.enabled})).collect::<Vec<_>>(),
        }));
        return Ok(());
    }

    println!("bibulm {}", env!("CARGO_PKG_VERSION"));
    println!("data dir : {}", node.data_dir.display());
    println!(
        "server   : port {} ({}) - {}",
        settings.server.port,
        if settings.server.serve_on_lan { "LAN" } else { "localhost" },
        if health.is_some() { "RUNNING" } else { "not running" }
    );
    println!(
        "link     : {} | sidecar: {} | remote load: {} | remote manage: {}",
        if settings.link.enabled { "enabled" } else { "disabled" },
        sidecar
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "NOT FOUND".to_string()),
        settings.link.allow_remote_load,
        settings.link.allow_remote_manage
    );
    println!("\nhardware:\n{hw_summary}\n");
    println!("runtimes ({}):", runtimes.len());
    for r in &runtimes {
        let active = settings.active_runtime_id.as_deref() == Some(r.id.as_str());
        println!("  {} {}  [{}]  {}", if active { "*" } else { " " }, r.id, r.backend, format_bytes(r.size_bytes));
    }
    println!("models ({}):", models.len());
    for m in &models {
        println!("    {}  {}  {}", m.id, m.quantization.as_deref().unwrap_or("-"), format_bytes(m.size_bytes));
    }
    let active: Vec<_> = downloads
        .iter()
        .filter(|d| matches!(d.status.as_str(), "queued" | "downloading" | "verifying" | "paused"))
        .collect();
    println!("downloads in progress: {}", active.len());
    println!("api tokens: {}", tokens.len());
    for t in &tokens {
        println!("    {} {}{}", t.name, t.token, if t.enabled { "" } else { " (disabled)" });
    }
    Ok(())
}

pub fn cmd_hardware(node: &HeadlessNode, json_out: bool) -> anyhow::Result<()> {
    let hw = node
        .hardware
        .lock()
        .map_err(|_| anyhow!("hardware monitor lock poisoned"))?;
    if json_out {
        print_json(hw.profile());
    } else {
        println!("{}", hw.profile().format_summary());
    }
    Ok(())
}

// ---------------------------------------------------------------- runtime

pub async fn cmd_runtime(node: &HeadlessNode, cmd: RuntimeCmd, json_out: bool) -> anyhow::Result<()> {
    let settings = node.settings.read().await.clone();
    match cmd {
        RuntimeCmd::List => {
            let list = node.db.list_runtimes().map_err(|e| anyhow!(e.to_string()))?;
            if json_out {
                print_json(&list);
            } else if list.is_empty() {
                println!("No runtime installed. Run `bibulm runtime install`.");
            } else {
                for r in &list {
                    let active = settings.active_runtime_id.as_deref() == Some(r.id.as_str());
                    println!(
                        "{} {}  tag={} backend={} size={}",
                        if active { "*" } else { " " },
                        r.id,
                        r.tag,
                        r.backend,
                        format_bytes(r.size_bytes)
                    );
                }
            }
        }
        RuntimeCmd::Releases => {
            let hw = node.hardware_match_context();
            let releases =
                bibulm_runtime::fetch_llama_cpp_releases(settings.github_token.as_deref(), &hw)
                    .await
                    .map_err(|e| anyhow!(e.to_string()))?;
            if json_out {
                print_json(&releases);
            } else {
                for rel in releases.iter().take(5) {
                    println!("{}  ({})", rel.tag_name, rel.published_at);
                    for v in &rel.variants {
                        println!(
                            "    {:<14} {:>10}  {}{}{}",
                            v.backend,
                            format_bytes(v.total_size_bytes),
                            v.display_name,
                            if v.recommended { "  [recommended]" } else { "" },
                            if v.compatible { "" } else { "  [incompatible]" }
                        );
                    }
                }
            }
        }
        RuntimeCmd::Install { tag, backend } => {
            let hw = node.hardware_match_context();
            let releases =
                bibulm_runtime::fetch_llama_cpp_releases(settings.github_token.as_deref(), &hw)
                    .await
                    .map_err(|e| anyhow!(e.to_string()))?;
            let release = match tag.as_deref() {
                Some(t) => releases
                    .iter()
                    .find(|r| r.tag_name == t)
                    .ok_or_else(|| anyhow!("release tag '{t}' not found in the latest releases"))?,
                None => releases
                    .first()
                    .ok_or_else(|| anyhow!("no compatible llama.cpp release found for this platform"))?,
            };
            let variant = match backend.as_deref() {
                Some(b) => release
                    .variants
                    .iter()
                    .find(|v| v.backend.eq_ignore_ascii_case(b))
                    .or_else(|| {
                        release
                            .variants
                            .iter()
                            .find(|v| v.backend.to_ascii_lowercase().starts_with(&b.to_ascii_lowercase()))
                    })
                    .ok_or_else(|| {
                        anyhow!(
                            "backend '{b}' not available in {}. Options: {}",
                            release.tag_name,
                            release.variants.iter().map(|v| v.backend.as_str()).collect::<Vec<_>>().join(", ")
                        )
                    })?,
                None => release
                    .variants
                    .iter()
                    .find(|v| v.recommended)
                    .or_else(|| release.variants.iter().find(|v| v.compatible))
                    .or_else(|| release.variants.first())
                    .ok_or_else(|| anyhow!("release has no installable variants"))?,
            };
            println!(
                "Installing {} ({}) - {}",
                variant.display_name,
                release.tag_name,
                format_bytes(variant.total_size_bytes)
            );
            let mut last_pct = u64::MAX;
            let mut last_stage = String::new();
            let rec = bibulm_runtime::install_runtime_variant(
                &node.db,
                &settings.runtimes_dir,
                variant,
                Some(release.release_notes.clone()),
                settings.github_token.as_deref(),
                move |p| {
                    let pct = (p.downloaded_bytes * 100)
                        .checked_div(p.total_bytes)
                        .unwrap_or(0);
                    if pct != last_pct || p.stage != last_stage {
                        last_pct = pct;
                        last_stage = p.stage.clone();
                        eprint!("\r  {} {pct:>3}%   ", p.stage);
                    }
                },
            )
            .await
            .map_err(|e| anyhow!(e.to_string()))?;
            eprintln!();
            let mut s = node.settings.write().await;
            if s.active_runtime_id.is_none() {
                s.active_runtime_id = Some(rec.id.clone());
                save_settings(node, &s).await?;
            }
            println!("Installed runtime '{}' at {}", rec.id, rec.install_dir);
        }
        RuntimeCmd::Use(id) => {
            let list = node.db.list_runtimes().map_err(|e| anyhow!(e.to_string()))?;
            if !list.iter().any(|r| r.id == id) {
                bail!("runtime '{id}' is not installed (see `bibulm runtime list`)");
            }
            let mut s = node.settings.write().await;
            s.active_runtime_id = Some(id.clone());
            save_settings(node, &s).await?;
            println!("Active runtime: {id}");
        }
        RuntimeCmd::Remove(id) => {
            bibulm_runtime::uninstall_runtime(&node.db, &id).map_err(|e| anyhow!(e.to_string()))?;
            let mut s = node.settings.write().await;
            if s.active_runtime_id.as_deref() == Some(id.as_str()) {
                s.active_runtime_id = node
                    .db
                    .list_runtimes()
                    .ok()
                    .and_then(|l| l.into_iter().next().map(|r| r.id));
                save_settings(node, &s).await?;
            }
            println!("Removed runtime '{id}'.");
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- models

pub async fn cmd_models(node: &HeadlessNode, cmd: ModelsCmd, json_out: bool) -> anyhow::Result<()> {
    let settings = node.settings.read().await.clone();
    match cmd {
        ModelsCmd::List => {
            let dirs = settings.models_dirs.clone();
            let auto_discover = settings.auto_discover_external_models;
            let models = tokio::task::spawn_blocking(move || scan_local_models_with_external(&dirs, auto_discover))
                .await
                .context("model scan failed")?;
            if json_out {
                print_json(&models);
            } else if models.is_empty() {
                println!("No model found. Try `bibulm models pull <REPO_ID>`.");
            } else {
                for m in &models {
                    println!(
                        "{}  {}  {}  {}",
                        m.id,
                        m.architecture.as_deref().unwrap_or("-"),
                        m.quantization.as_deref().unwrap_or("-"),
                        format_bytes(m.size_bytes)
                    );
                }
            }
        }
        ModelsCmd::Search(query) => {
            let results = bibulm_models::search_hf_gguf_models(
                &query,
                HfSortOrder::BestMatch,
                None,
                20,
                settings.hf_token.as_deref(),
            )
            .await
            .map_err(|e| anyhow!(e.to_string()))?;
            if json_out {
                print_json(&results);
            } else {
                for r in &results {
                    println!(
                        "{}  downloads={} likes={} {}",
                        r.repo_id,
                        r.downloads,
                        r.likes,
                        r.param_count_label.as_deref().unwrap_or("")
                    );
                }
            }
        }
        ModelsCmd::Pull { repo, quant, mmproj } => {
            let (vram, ram) = node.total_memory_bytes(&settings.disabled_gpu_ids);
            let detail = bibulm_models::get_hf_model_detail(&repo, settings.hf_token.as_deref(), vram, ram)
                .await
                .map_err(|e| anyhow!(e.to_string()))?;
            if detail.quant_options.is_empty() {
                bail!("repository '{repo}' has no GGUF files");
            }
            let wanted = quant.unwrap_or_else(|| "Q4_K_M".to_string()).to_ascii_lowercase();
            let option = detail
                .quant_options
                .iter()
                .find(|q| q.quantization.to_ascii_lowercase() == wanted)
                .or_else(|| {
                    detail
                        .quant_options
                        .iter()
                        .find(|q| q.label.to_ascii_lowercase().contains(&wanted))
                })
                .or_else(|| {
                    detail
                        .quant_options
                        .iter()
                        .find(|q| q.quantization.eq_ignore_ascii_case("Q4_K_M"))
                })
                .unwrap_or(&detail.quant_options[0]);
            let mut files = option.files.clone();
            if mmproj && let Some(mm) = detail.mmproj_files.first() {
                files.push(mm.clone());
            }
            let sub = bibulm_server::sanitize_repo_subpath(&repo)
                .ok_or_else(|| anyhow!("invalid repository id '{repo}'"))?;
            let dest = settings.primary_models_dir().join(sub);
            let title = format!("{repo} ({})", option.quantization);
            println!(
                "Downloading {title} - {} ({} file(s), fit: {:?})",
                format_bytes(files.iter().map(|f| f.size_bytes).sum()),
                files.len(),
                option.fit_verdict
            );
            let mut rx = node.downloads.subscribe();
            let job = node
                .downloads
                .enqueue_model_download(title, repo.clone(), dest, files, settings.hf_token.clone())
                .await;
            let mut last_print: Option<Instant> = None;
            let interrupted = shutdown_signal();
            tokio::pin!(interrupted);
            loop {
                tokio::select! {
                    _ = &mut interrupted => {
                        node.downloads.pause(&job.id).await;
                        let _ = tokio::time::timeout(Duration::from_secs(2), async {
                            while let Ok(j) = rx.recv().await {
                                if j.id == job.id && j.status == "paused" {
                                    break;
                                }
                            }
                        })
                        .await;
                        eprintln!("\nInterrupted. Re-run the same command to resume.");
                        return Ok(());
                    }
                    ev = rx.recv() => {
                        match ev {
                            Ok(j) if j.id == job.id => {
                                match j.status.as_str() {
                                    "completed" => {
                                        eprintln!();
                                        println!("Done: {}", j.dest_dir);
                                        return Ok(());
                                    }
                                    "failed" => bail!(j.error.unwrap_or_else(|| "download failed".into())),
                                    "cancelled" => bail!("download cancelled"),
                                    _ => {
                                        if last_print.is_none_or(|t| t.elapsed() >= Duration::from_millis(500)) {
                                            last_print = Some(Instant::now());
                                            let pct = (j.downloaded_bytes * 100).checked_div(j.total_bytes).unwrap_or(0);
                                            eprint!(
                                                "\r  {pct:>3}%  {} / {}  {}/s   ",
                                                format_bytes(j.downloaded_bytes),
                                                format_bytes(j.total_bytes),
                                                format_bytes(j.speed_bytes_per_sec as u64)
                                            );
                                        }
                                    }
                                }
                            }
                            Ok(_) => {}
                            Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {}
                            Err(tokio::sync::broadcast::error::RecvError::Closed) => bail!("download channel closed"),
                        }
                    }
                }
            }
        }
        ModelsCmd::Remove(id) => {
            let dirs = settings.models_dirs.clone();
            let auto_discover = settings.auto_discover_external_models;
            let models = tokio::task::spawn_blocking(move || scan_local_models_with_external(&dirs, auto_discover))
                .await
                .context("model scan failed")?;
            let q = id.to_ascii_lowercase();
            let model = models
                .iter()
                .find(|m| m.id.to_ascii_lowercase() == q || m.display_name.to_ascii_lowercase() == q)
                .ok_or_else(|| anyhow!("model '{id}' not found"))?;
            let n = bibulm_models::delete_local_model(&model.path)?;
            println!("Deleted {n} file(s) for '{}'.", model.id);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- service systemd

pub struct UnitParams {
    pub exe: String,
    pub data_dir: String,
    pub serve_args: Vec<String>,
    pub user: Option<String>,
    pub user_unit: bool,
}

fn quote_arg(a: &str) -> String {
    if a.is_empty() || a.chars().any(|c| c.is_whitespace() || c == '"' || c == '\\') {
        format!("\"{}\"", a.replace('\\', "\\\\").replace('"', "\\\""))
    } else {
        a.to_string()
    }
}

pub fn render_systemd_unit(p: &UnitParams) -> String {
    let mut exec = vec![quote_arg(&p.exe), "serve".to_string()];
    exec.extend(p.serve_args.iter().map(|a| quote_arg(a)));
    let env_file = if p.user_unit {
        "%h/.config/bibulm/bibulm.env"
    } else {
        "/etc/bibulm/bibulm.env"
    };
    let mut out = String::new();
    out.push_str("[Unit]\nDescription=BibuLM headless inference server\n");
    out.push_str("After=network-online.target\nWants=network-online.target\n\n[Service]\nType=simple\n");
    out.push_str(&format!("ExecStart={}\n", exec.join(" ")));
    out.push_str(&format!("Environment=BIBULM_DATA_DIR={}\n", quote_arg(&p.data_dir)));
    out.push_str(&format!("EnvironmentFile=-{env_file}\n"));
    if let Some(u) = &p.user
        && !p.user_unit
    {
        out.push_str(&format!("User={u}\n"));
    }
    out.push_str("Restart=on-failure\nRestartSec=5\nKillSignal=SIGTERM\nTimeoutStopSec=30\nLimitNOFILE=65536\n\n[Install]\n");
    out.push_str(if p.user_unit {
        "WantedBy=default.target\n"
    } else {
        "WantedBy=multi-user.target\n"
    });
    out
}

pub fn cmd_service_systemd(node_data_dir: &Path, a: SystemdArgs) -> anyhow::Result<()> {
    let exe = std::env::current_exe()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| "/usr/local/bin/bibulm".to_string());
    let mut serve_args = Vec::new();
    if let Some(n) = &a.name {
        serve_args.push("--name".to_string());
        serve_args.push(n.clone());
    }
    if let Some(p) = a.port {
        serve_args.push("--port".to_string());
        serve_args.push(p.to_string());
    }
    if a.lan {
        serve_args.push("--lan".to_string());
    }
    let unit = render_systemd_unit(&UnitParams {
        exe,
        data_dir: node_data_dir.to_string_lossy().into_owned(),
        serve_args,
        user: a.user.clone(),
        user_unit: a.user_unit,
    });
    if !a.install {
        print!("{unit}");
        eprintln!("\n# Save as bibulm.service, or re-run with --install.");
        return Ok(());
    }
    let path: PathBuf = if a.user_unit {
        let home = std::env::var("HOME").context("HOME is not set")?;
        PathBuf::from(home).join(".config/systemd/user/bibulm.service")
    } else {
        PathBuf::from("/etc/systemd/system/bibulm.service")
    };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("cannot create {} (try sudo?)", parent.display()))?;
    }
    std::fs::write(&path, unit).with_context(|| format!("cannot write {} (try sudo?)", path.display()))?;
    println!("Wrote {}", path.display());
    let ctl = if a.user_unit { "systemctl --user" } else { "sudo systemctl" };
    println!("Next:\n  {ctl} daemon-reload\n  {ctl} enable --now bibulm\n  journalctl {} -u bibulm -f", if a.user_unit { "--user" } else { "" });
    println!("Put secrets such as TS_AUTHKEY or HF_TOKEN in the EnvironmentFile (mode 600).");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn systemd_unit_contains_expected_directives() {
        let unit = render_systemd_unit(&UnitParams {
            exe: "/usr/local/bin/bibulm".into(),
            data_dir: "/var/lib/my bibulm".into(),
            serve_args: vec!["--name".into(), "gpu-box".into(), "--lan".into()],
            user: Some("bibulm".into()),
            user_unit: false,
        });
        assert!(unit.contains("ExecStart=/usr/local/bin/bibulm serve --name gpu-box --lan\n"));
        assert!(unit.contains("Environment=BIBULM_DATA_DIR=\"/var/lib/my bibulm\"\n"));
        assert!(unit.contains("User=bibulm\n"));
        assert!(unit.contains("KillSignal=SIGTERM"));
        assert!(unit.contains("WantedBy=multi-user.target"));

        let user_unit = render_systemd_unit(&UnitParams {
            exe: "bibulm".into(),
            data_dir: "/d".into(),
            serve_args: vec![],
            user: Some("ignored".into()),
            user_unit: true,
        });
        assert!(!user_unit.contains("User="));
        assert!(user_unit.contains("WantedBy=default.target"));
    }

    #[test]
    fn formats_bytes() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
    }
}
