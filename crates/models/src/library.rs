use bibulm_core::{
    ExternalSourceKind, GgufMetadata, discover_external_model_sources, discover_split_shards,
    infer_quant_from_filename, is_secondary_split_shard, parse_gguf_file,
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Represents a locally installed GGUF model inspected natively in Rust.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LocalModelEntry {
    pub id: String,
    pub display_name: String,
    pub path: PathBuf,
    #[serde(default)]
    pub source: Option<String>,
    pub shard_count: usize,
    pub mmproj_path: Option<PathBuf>,
    pub mmproj_size_bytes: Option<u64>,
    pub size_bytes: u64,
    pub architecture: Option<String>,
    pub quantization: Option<String>,
    pub parameter_count: Option<u64>,
    pub is_moe: bool,
    pub expert_count: Option<u32>,
    pub expert_used_count: Option<u32>,
    pub routed_expert_layers: u32,
    pub has_mtp: bool,
    pub mtp_layer_count: u32,
    pub mtp_bytes: u64,
    pub layer_count: Option<u32>,
    pub head_count: Option<u32>,
    pub head_count_kv: Option<u32>,
    pub max_context_length: Option<u32>,
    pub supports_reasoning: bool,
    pub supports_reasoning_effort: bool,
    pub supports_enable_thinking: bool,
    pub supports_tools: bool,
    pub supports_vision: bool,
    pub is_embedding: bool,
    pub has_chat_template: bool,
    pub chat_template_preview: Option<String>,
    pub gguf: Option<GgufMetadata>,
}

/// Recursively scans all configured model directories and parses GGUF headers natively.
pub fn scan_local_models(dirs: &[PathBuf]) -> Vec<LocalModelEntry> {
    scan_local_models_with_external(dirs, false)
}

/// Recursively scans configured model directories and, when `auto_discover` is `true`,
/// also scans model folders managed by external apps (Ollama, LM Studio, Jan, GPT4All,
/// llama.cpp cache, HuggingFace Hub cache, AnythingLLM, Msty, Backyard AI, etc.).
pub fn scan_local_models_with_external(
    dirs: &[PathBuf],
    auto_discover: bool,
) -> Vec<LocalModelEntry> {
    let mut entries = Vec::new();
    let mut seen_paths = HashSet::new();
    let mut seen_roots = HashSet::new();

    for root in dirs {
        if !root.is_dir() {
            continue;
        }
        let canonical_root = fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
        seen_roots.insert(canonical_root);

        if root.join("manifests").is_dir() && root.join("blobs").is_dir() {
            scan_ollama_dir(root, None, &mut seen_paths, &mut entries);
        }
        scan_gguf_dir(root, None, &mut seen_paths, &mut entries);
    }

    if auto_discover {
        for src in discover_external_model_sources() {
            if !src.path.is_dir() {
                continue;
            }
            let canonical_root =
                fs::canonicalize(&src.path).unwrap_or_else(|_| src.path.clone());
            if !seen_roots.insert(canonical_root) {
                continue;
            }
            match src.kind {
                ExternalSourceKind::Ollama => {
                    scan_ollama_dir(&src.path, Some(src.app_name), &mut seen_paths, &mut entries);
                }
                ExternalSourceKind::GgufDir => {
                    scan_gguf_dir(&src.path, Some(src.app_name), &mut seen_paths, &mut entries);
                }
            }
        }
    }

    entries.sort_by(|a, b| {
        a.display_name
            .to_ascii_lowercase()
            .cmp(&b.display_name.to_ascii_lowercase())
    });
    entries
}

fn scan_gguf_dir(
    root: &Path,
    source: Option<String>,
    seen_paths: &mut HashSet<PathBuf>,
    entries: &mut Vec<LocalModelEntry>,
) {
    for entry in WalkDir::new(root)
        .follow_links(true)
        .max_depth(8)
        .into_iter()
        .filter_entry(|e| {
            if e.depth() > 0
                && e.file_type().is_dir()
                && let Some(name) = e.file_name().to_str()
                && (name == ".git" || name == "node_modules" || name == "blobs")
            {
                return false;
            }
            true
        })
        .flatten()
    {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(file_name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        let lower = file_name.to_ascii_lowercase();
        if !lower.ends_with(".gguf") {
            continue;
        }
        // Skip vision projectors (paired alongside main models) and secondary split shards.
        if lower.contains("mmproj") || is_secondary_split_shard(file_name) {
            continue;
        }

        let canonical = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        if !seen_paths.insert(canonical) {
            continue;
        }

        let mmproj_path = find_sibling_mmproj(path);
        let mmproj_size_bytes = mmproj_path
            .as_deref()
            .and_then(|p| fs::metadata(p).ok())
            .map(|m| m.len());
        let id = build_model_id(root, path);
        let fallback_name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model")
            .to_string();

        entries.push(build_entry_from_gguf(
            id,
            None,
            path.to_path_buf(),
            source.clone(),
            mmproj_path,
            mmproj_size_bytes,
            fallback_name,
        ));
    }
}

#[derive(Debug, Deserialize)]
struct OllamaManifest {
    #[serde(default)]
    layers: Vec<OllamaLayer>,
}

#[derive(Debug, Deserialize)]
struct OllamaLayer {
    #[serde(rename = "mediaType", default)]
    media_type: String,
    #[serde(default)]
    digest: String,
}

fn resolve_ollama_blob(blobs_dir: &Path, digest: &str) -> Option<PathBuf> {
    let trimmed = digest.trim();
    if trimmed.is_empty() {
        return None;
    }
    let dash_name = trimmed.replace(':', "-");
    let p1 = blobs_dir.join(&dash_name);
    if p1.is_file() {
        return Some(p1);
    }
    let p2 = blobs_dir.join(trimmed);
    if p2.is_file() {
        return Some(p2);
    }
    None
}

fn format_ollama_manifest_name(manifests_dir: &Path, manifest_path: &Path) -> (String, String) {
    let rel = manifest_path.strip_prefix(manifests_dir).unwrap_or(manifest_path);
    let parts: Vec<String> = rel
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    let tag_name = match parts.as_slice() {
        [registry, ns, model, tag]
            if registry.eq_ignore_ascii_case("registry.ollama.ai")
                && ns.eq_ignore_ascii_case("library") =>
        {
            format!("{model}:{tag}")
        }
        [registry, ns, model, tag] if registry.eq_ignore_ascii_case("registry.ollama.ai") => {
            format!("{ns}/{model}:{tag}")
        }
        [registry, ns, model, tag] => format!("{registry}/{ns}/{model}:{tag}"),
        [.., model, tag] => format!("{model}:{tag}"),
        [single] => single.clone(),
        [] => "ollama-model".to_string(),
    };
    let id = format!("ollama/{}", tag_name.to_ascii_lowercase());
    (id, tag_name)
}

fn scan_ollama_dir(
    root: &Path,
    source: Option<String>,
    seen_paths: &mut HashSet<PathBuf>,
    entries: &mut Vec<LocalModelEntry>,
) {
    let manifests_dir = root.join("manifests");
    let blobs_dir = root.join("blobs");
    if !manifests_dir.is_dir() || !blobs_dir.is_dir() {
        return;
    }

    let effective_source = source.or_else(|| Some("Ollama".to_string()));

    for entry in WalkDir::new(&manifests_dir)
        .follow_links(true)
        .max_depth(6)
        .into_iter()
        .flatten()
    {
        let manifest_path = entry.path();
        if !manifest_path.is_file() {
            continue;
        }
        let Ok(meta) = fs::metadata(manifest_path) else {
            continue;
        };
        if meta.len() > 1024 * 1024 {
            continue;
        }
        let Ok(raw) = fs::read_to_string(manifest_path) else {
            continue;
        };
        let Ok(manifest) = serde_json::from_str::<OllamaManifest>(&raw) else {
            continue;
        };

        let Some(model_layer) = manifest
            .layers
            .iter()
            .find(|l| l.media_type == "application/vnd.ollama.image.model")
        else {
            continue;
        };
        let Some(blob_path) = resolve_ollama_blob(&blobs_dir, &model_layer.digest) else {
            continue;
        };

        // Ensure the Ollama blob is actually a GGUF file before registering it.
        let Ok(gguf_meta) = parse_gguf_file(&blob_path) else {
            continue;
        };

        let canonical = fs::canonicalize(&blob_path).unwrap_or_else(|_| blob_path.clone());
        if !seen_paths.insert(canonical) {
            continue;
        }

        let mmproj_path = manifest
            .layers
            .iter()
            .find(|l| l.media_type == "application/vnd.ollama.image.projector")
            .and_then(|l| resolve_ollama_blob(&blobs_dir, &l.digest));
        let mmproj_size_bytes = mmproj_path
            .as_deref()
            .and_then(|p| fs::metadata(p).ok())
            .map(|m| m.len());

        let (id, ollama_name) = format_ollama_manifest_name(&manifests_dir, manifest_path);
        entries.push(build_entry_from_parsed_gguf(
            id,
            ollama_name.clone(),
            blob_path,
            effective_source.clone(),
            mmproj_path,
            mmproj_size_bytes,
            gguf_meta,
        ));
    }
}

fn build_entry_from_gguf(
    id: String,
    preferred_name: Option<String>,
    path: PathBuf,
    source: Option<String>,
    mmproj_path: Option<PathBuf>,
    mmproj_size_bytes: Option<u64>,
    fallback_name: String,
) -> LocalModelEntry {
    match parse_gguf_file(&path) {
        Ok(meta) => {
            let display_name = preferred_name.unwrap_or_else(|| {
                meta.name
                    .clone()
                    .filter(|s| !s.trim().is_empty())
                    .unwrap_or_else(|| fallback_name.clone())
            });
            build_entry_from_parsed_gguf(
                id,
                display_name,
                path,
                source,
                mmproj_path,
                mmproj_size_bytes,
                meta,
            )
        }
        Err(_) => {
            let size_bytes = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            let quantization = infer_quant_from_filename(&path);
            LocalModelEntry {
                id,
                display_name: preferred_name.unwrap_or(fallback_name),
                path,
                source,
                shard_count: 1,
                mmproj_path: mmproj_path.clone(),
                mmproj_size_bytes,
                size_bytes,
                architecture: None,
                quantization,
                parameter_count: None,
                is_moe: false,
                expert_count: None,
                expert_used_count: None,
                routed_expert_layers: 0,
                has_mtp: false,
                mtp_layer_count: 0,
                mtp_bytes: 0,
                layer_count: None,
                head_count: None,
                head_count_kv: None,
                max_context_length: None,
                supports_reasoning: false,
                supports_reasoning_effort: false,
                supports_enable_thinking: false,
                supports_tools: false,
                supports_vision: mmproj_path.is_some(),
                is_embedding: false,
                has_chat_template: false,
                chat_template_preview: None,
                gguf: None,
            }
        }
    }
}

fn build_entry_from_parsed_gguf(
    id: String,
    display_name: String,
    path: PathBuf,
    source: Option<String>,
    mmproj_path: Option<PathBuf>,
    mmproj_size_bytes: Option<u64>,
    meta: GgufMetadata,
) -> LocalModelEntry {
    let shard_count = meta.shard_paths.len().max(1);
    let supports_vision = mmproj_path.is_some();
    let chat_template_preview = meta
        .chat_template
        .as_deref()
        .map(|t| t.chars().take(4000).collect());
    let mtp_layer_count = meta
        .nextn_predict_layers
        .unwrap_or_else(|| meta.tensor_summary.mtp_layer_count());
    let mtp_bytes = meta.tensor_summary.total_mtp_bytes();

    LocalModelEntry {
        id,
        display_name,
        path,
        source,
        shard_count,
        mmproj_path,
        mmproj_size_bytes,
        size_bytes: meta.total_size_bytes,
        architecture: meta.architecture.clone(),
        quantization: meta.quantization.clone(),
        parameter_count: meta.parameter_count,
        is_moe: meta.is_moe,
        expert_count: meta.expert_count,
        expert_used_count: meta.expert_used_count,
        routed_expert_layers: meta.tensor_summary.routed_expert_layer_count(),
        has_mtp: meta.has_mtp,
        mtp_layer_count,
        mtp_bytes,
        layer_count: meta.effective_layer_count(),
        head_count: meta.head_count,
        head_count_kv: meta.head_count_kv,
        max_context_length: meta.context_length,
        supports_reasoning: meta.supports_reasoning,
        supports_reasoning_effort: meta.supports_reasoning_effort,
        supports_enable_thinking: meta.supports_enable_thinking,
        supports_tools: meta.supports_tools,
        supports_vision,
        is_embedding: meta.is_embedding,
        has_chat_template: meta.chat_template.is_some(),
        chat_template_preview,
        gguf: Some(meta),
    }
}

/// Deletes a model GGUF and all of its sibling split shards from disk.
pub fn delete_local_model(primary_path: &Path) -> std::io::Result<usize> {
    let shards = discover_split_shards(primary_path);
    let mut deleted = 0usize;
    for shard in shards {
        if shard.is_file() {
            fs::remove_file(&shard)?;
            deleted += 1;
        }
    }
    Ok(deleted)
}

fn find_sibling_mmproj(model_path: &Path) -> Option<PathBuf> {
    let parent = model_path.parent()?;
    let mut candidates = Vec::new();
    if let Ok(read) = fs::read_dir(parent) {
        for entry in read.flatten() {
            let p = entry.path();
            if p.is_file()
                && let Some(name) = p.file_name().and_then(|s| s.to_str())
            {
                let low = name.to_ascii_lowercase();
                if low.ends_with(".gguf") && low.contains("mmproj") {
                    candidates.push(p);
                }
            }
        }
    }
    // Prefer BF16/F16 mmproj if multiple exist
    candidates.sort();
    candidates
        .iter()
        .find(|p| {
            let n = p.to_string_lossy().to_ascii_lowercase();
            n.contains("bf16") || n.contains("f16")
        })
        .cloned()
        .or_else(|| candidates.into_iter().next())
}

fn build_model_id(root: &Path, path: &Path) -> String {
    let rel = path.strip_prefix(root).unwrap_or(path);
    let mut parts: Vec<String> = rel
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    if let Some(last) = parts.last_mut() {
        if let Some(stripped) = last.strip_suffix(".gguf").or_else(|| last.strip_suffix(".GGUF")) {
            *last = stripped.to_string();
        }
        if let Some((prefix, _)) = bibulm_core::parse_split_shard_pattern(&format!("{last}.gguf")) {
            *last = prefix;
        }
    }

    // Normalize HuggingFace Hub cache paths:
    // models--<org>--<repo>/snapshots/<hash>/<rest> -> hf/<org>/<repo>/<rest>
    if parts.len() >= 4
        && let Some(repo_slug) = parts[0].strip_prefix("models--")
        && parts[1] == "snapshots"
    {
        let mut norm = vec!["hf".to_string()];
        for seg in repo_slug.split("--") {
            if !seg.is_empty() {
                norm.push(seg.to_string());
            }
        }
        norm.extend(parts.into_iter().skip(3));
        return norm.join("/").to_ascii_lowercase();
    }

    parts.join("/").to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_gguf_bytes() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes()); // version 3
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
        buf.extend_from_slice(&0u64.to_le_bytes()); // kv_count = 0
        buf
    }

    #[test]
    fn scans_ollama_manifests_blobs_and_projector() {
        let tmp = std::env::temp_dir().join(format!("bibulm_ollama_test_{}", uuid::Uuid::new_v4()));
        let manifests = tmp
            .join("manifests")
            .join("registry.ollama.ai")
            .join("library")
            .join("qwen2.5");
        let blobs = tmp.join("blobs");
        fs::create_dir_all(&manifests).unwrap();
        fs::create_dir_all(&blobs).unwrap();

        let model_digest = "sha256:1111111111111111";
        let mmproj_digest = "sha256:2222222222222222";
        fs::write(blobs.join("sha256-1111111111111111"), minimal_gguf_bytes()).unwrap();
        fs::write(blobs.join("sha256-2222222222222222"), minimal_gguf_bytes()).unwrap();

        let manifest_json = serde_json::json!({
            "schemaVersion": 2,
            "layers": [
                { "mediaType": "application/vnd.ollama.image.model", "digest": model_digest },
                { "mediaType": "application/vnd.ollama.image.projector", "digest": mmproj_digest }
            ]
        });
        fs::write(manifests.join("7b"), manifest_json.to_string()).unwrap();

        let results = scan_local_models(std::slice::from_ref(&tmp));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "ollama/qwen2.5:7b");
        assert_eq!(results[0].display_name, "qwen2.5:7b");
        assert_eq!(results[0].source.as_deref(), Some("Ollama"));
        assert!(results[0].supports_vision);
        assert!(results[0].mmproj_path.is_some());

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn normalizes_hf_cache_ids_and_skips_blobs_dir() {
        let tmp = std::env::temp_dir().join(format!("bibulm_hf_test_{}", uuid::Uuid::new_v4()));
        let snap = tmp
            .join("models--bartowski--Llama-3.2-1B-GGUF")
            .join("snapshots")
            .join("abcdef1234");
        let blobs = tmp
            .join("models--bartowski--Llama-3.2-1B-GGUF")
            .join("blobs");
        fs::create_dir_all(&snap).unwrap();
        fs::create_dir_all(&blobs).unwrap();

        fs::write(snap.join("Llama-3.2-1B-Q4_K_M.gguf"), minimal_gguf_bytes()).unwrap();
        // Should be ignored because it lives under blobs/
        fs::write(blobs.join("ignored.gguf"), minimal_gguf_bytes()).unwrap();

        let results = scan_local_models(std::slice::from_ref(&tmp));
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].id,
            "hf/bartowski/llama-3.2-1b-gguf/llama-3.2-1b-q4_k_m"
        );

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn deduplicates_identical_files_across_duplicate_roots() {
        let tmp = std::env::temp_dir().join(format!("bibulm_dedup_test_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&tmp).unwrap();
        fs::write(tmp.join("test-model.gguf"), minimal_gguf_bytes()).unwrap();

        let results = scan_local_models(&[tmp.clone(), tmp.clone()]);
        assert_eq!(results.len(), 1);

        let _ = fs::remove_dir_all(&tmp);
    }
}
