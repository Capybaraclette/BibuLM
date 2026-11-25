use bibulm_core::{
    FitVerdict, estimate_hf_file_fit, infer_quant_from_filename, parse_split_shard_pattern,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;
use thiserror::Error;

const HF_API_BASE: &str = "https://huggingface.co/api";
const HF_WEB_BASE: &str = "https://huggingface.co";

#[derive(Debug, Error)]
pub enum HfError {
    #[error("HTTP error contacting HuggingFace: {0}")]
    Http(#[from] reqwest::Error),
    #[error("HuggingFace API status {status}: {body}")]
    ApiStatus { status: u16, body: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HfSortOrder {
    BestMatch,
    MostDownloads,
    MostLikes,
    RecentlyUpdated,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCapabilities {
    pub vision: bool,
    pub tool_use: bool,
    pub reasoning: bool,
    pub moe: bool,
    pub embedding: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfModelSummary {
    pub repo_id: String,
    pub author: String,
    pub name: String,
    pub downloads: u64,
    pub likes: u64,
    pub last_modified: String,
    pub tags: Vec<String>,
    pub gated: bool,
    pub capabilities: ModelCapabilities,
    pub architecture: Option<String>,
    pub param_count_label: Option<String>,
    pub context_length: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfDownloadFile {
    pub path: String,
    pub file_name: String,
    pub url: String,
    pub size_bytes: u64,
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfQuantOption {
    pub id: String,
    pub label: String,
    pub quantization: String,
    pub total_size_bytes: u64,
    pub shard_count: usize,
    pub files: Vec<HfDownloadFile>,
    pub fit_verdict: FitVerdict,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfModelDetail {
    pub summary: HfModelSummary,
    pub readme_markdown: String,
    pub quant_options: Vec<HfQuantOption>,
    pub mmproj_files: Vec<HfDownloadFile>,
}

#[derive(Debug, Deserialize)]
struct RawHfModel {
    id: String,
    #[serde(default)]
    downloads: u64,
    #[serde(default)]
    likes: u64,
    #[serde(rename = "lastModified", default)]
    last_modified: String,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    gated: serde_json::Value,
    #[serde(default)]
    gguf: Option<RawHfGgufInfo>,
}

#[derive(Debug, Deserialize)]
struct RawHfGgufInfo {
    #[serde(default)]
    total: Option<u64>,
    #[serde(default)]
    architecture: Option<String>,
    #[serde(default)]
    context_length: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct RawHfTreeEntry {
    #[serde(rename = "type")]
    entry_type: String,
    path: String,
    #[serde(default)]
    size: u64,
    #[serde(default)]
    lfs: Option<RawHfLfs>,
}

#[derive(Debug, Deserialize)]
struct RawHfLfs {
    #[serde(default)]
    oid: Option<String>,
    #[serde(default)]
    size: Option<u64>,
}

fn build_client() -> Result<reqwest::Client, HfError> {
    Ok(reqwest::Client::builder()
        .user_agent("Bibulm/0.3 (HuggingFace GGUF Client)")
        .build()?)
}

/// Searches HuggingFace Hub for GGUF models.
pub async fn search_hf_gguf_models(
    query: &str,
    sort: HfSortOrder,
    extra_filter: Option<&str>,
    limit: usize,
    hf_token: Option<&str>,
) -> Result<Vec<HfModelSummary>, HfError> {
    let client = build_client()?;
    let mut url = format!(
        "{HF_API_BASE}/models?filter=gguf&limit={}&full=true",
        limit.clamp(1, 50)
    );
    let q = query.trim();
    if !q.is_empty() {
        url.push_str("&search=");
        url.push_str(&percent_encode(q));
    }
    if let Some(f) = extra_filter.map(str::trim).filter(|s| !s.is_empty()) {
        url.push_str("&filter=");
        url.push_str(&percent_encode(f));
    }
    match sort {
        HfSortOrder::BestMatch => {
            if q.is_empty() {
                url.push_str("&sort=trendingScore&direction=-1");
            }
        }
        HfSortOrder::MostDownloads => url.push_str("&sort=downloads&direction=-1"),
        HfSortOrder::MostLikes => url.push_str("&sort=likes&direction=-1"),
        HfSortOrder::RecentlyUpdated => url.push_str("&sort=lastModified&direction=-1"),
    }

    let mut req = client.get(&url);
    if let Some(tok) = hf_token.map(str::trim).filter(|s| !s.is_empty()) {
        req = req.bearer_auth(tok);
    }
    let resp = req.send().await?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(HfError::ApiStatus {
            status: status.as_u16(),
            body,
        });
    }

    let raw: Vec<RawHfModel> = resp.json().await?;
    Ok(raw.into_iter().map(raw_to_summary).collect())
}

/// Fetches full HuggingFace repo details, README markdown, grouped GGUF quantizations,
/// vision `mmproj` files, and hardware fit estimates.
pub async fn get_hf_model_detail(
    repo_id: &str,
    hf_token: Option<&str>,
    avail_vram_bytes: u64,
    avail_ram_bytes: u64,
) -> Result<HfModelDetail, HfError> {
    let client = build_client()?;
    let auth = hf_token.map(str::trim).filter(|s| !s.is_empty());

    // 1. Model metadata
    let meta_url = format!("{HF_API_BASE}/models/{repo_id}");
    let mut meta_req = client.get(&meta_url);
    if let Some(t) = auth {
        meta_req = meta_req.bearer_auth(t);
    }
    let meta_resp = meta_req.send().await?;
    if !meta_resp.status().is_success() {
        return Err(HfError::ApiStatus {
            status: meta_resp.status().as_u16(),
            body: meta_resp.text().await.unwrap_or_default(),
        });
    }
    let raw_meta: RawHfModel = meta_resp.json().await?;
    let mut summary = raw_to_summary(raw_meta);

    // 2. Recursive file tree
    let tree_url = format!("{HF_API_BASE}/models/{repo_id}/tree/main?recursive=true");
    let mut tree_req = client.get(&tree_url);
    if let Some(t) = auth {
        tree_req = tree_req.bearer_auth(t);
    }
    let tree_entries: Vec<RawHfTreeEntry> = match tree_req.send().await {
        Ok(r) if r.status().is_success() => r.json().await.unwrap_or_default(),
        _ => Vec::new(),
    };

    // 3. README.md (best-effort)
    let readme_url = format!("{HF_WEB_BASE}/{repo_id}/resolve/main/README.md");
    let mut readme_req = client.get(&readme_url);
    if let Some(t) = auth {
        readme_req = readme_req.bearer_auth(t);
    }
    let readme_markdown = match readme_req.send().await {
        Ok(r) if r.status().is_success() => strip_yaml_frontmatter(&r.text().await.unwrap_or_default()),
        _ => String::new(),
    };

    // Enrich capabilities from README text if not obvious from tags
    let readme_low = readme_markdown.to_ascii_lowercase();
    if readme_low.contains("mixture-of-experts") || readme_low.contains("mixture of experts") {
        summary.capabilities.moe = true;
    }
    if readme_low.contains("reasoning_effort")
        || readme_low.contains("<think>")
        || readme_low.contains("chain-of-thought")
    {
        summary.capabilities.reasoning = true;
    }

    let (quant_options, mmproj_files) = group_gguf_tree(
        repo_id,
        &tree_entries,
        summary.capabilities.moe,
        avail_vram_bytes,
        avail_ram_bytes,
    );

    if !mmproj_files.is_empty() {
        summary.capabilities.vision = true;
    }

    Ok(HfModelDetail {
        summary,
        readme_markdown,
        quant_options,
        mmproj_files,
    })
}

fn raw_to_summary(raw: RawHfModel) -> HfModelSummary {
    let (author, name) = raw
        .id
        .split_once('/')
        .map(|(a, n)| (a.to_string(), n.to_string()))
        .unwrap_or_else(|| ("huggingface".to_string(), raw.id.clone()));

    let gated = match &raw.gated {
        serde_json::Value::Bool(b) => *b,
        serde_json::Value::String(s) => !s.is_empty() && s != "false",
        _ => false,
    };

    let architecture = raw.gguf.as_ref().and_then(|g| g.architecture.clone());
    let context_length = raw.gguf.as_ref().and_then(|g| g.context_length);
    let param_count_label = raw
        .gguf
        .as_ref()
        .and_then(|g| g.total)
        .map(format_param_count)
        .or_else(|| infer_param_label_from_name(&name));

    let capabilities = detect_capabilities(&raw.id, &raw.tags, architecture.as_deref());

    HfModelSummary {
        repo_id: raw.id,
        author,
        name,
        downloads: raw.downloads,
        likes: raw.likes,
        last_modified: raw.last_modified,
        tags: raw.tags,
        gated,
        capabilities,
        architecture,
        param_count_label,
        context_length,
    }
}

fn detect_capabilities(repo_id: &str, tags: &[String], arch: Option<&str>) -> ModelCapabilities {
    let combined = format!(
        "{} {} {}",
        repo_id.to_ascii_lowercase(),
        tags.join(" ").to_ascii_lowercase(),
        arch.unwrap_or("").to_ascii_lowercase()
    );
    ModelCapabilities {
        vision: combined.contains("vision")
            || combined.contains("image-text-to-text")
            || combined.contains("multimodal")
            || combined.contains("-vl")
            || combined.contains("omni")
            || combined.contains("llava")
            || combined.contains("pixtral"),
        tool_use: combined.contains("tool")
            || combined.contains("function-calling")
            || combined.contains("agent")
            || combined.contains("qwen")
            || combined.contains("llama-3")
            || combined.contains("mistral")
            || combined.contains("gemma-4"),
        reasoning: combined.contains("reasoning")
            || combined.contains("thinking")
            || combined.contains("deepseek-r1")
            || combined.contains("qwq")
            || combined.contains("qwen3")
            || combined.contains("gpt-oss")
            || combined.contains("gemma-4"),
        moe: combined.contains("moe")
            || combined.contains("mixtral")
            || combined.contains("deepseek")
            || combined.contains("-a3b")
            || combined.contains("-a4b")
            || combined.contains("a22b")
            || combined.contains("8x7b")
            || combined.contains("8x22b"),
        embedding: combined.contains("embedding")
            || combined.contains("embed")
            || combined.contains("feature-extraction")
            || combined.contains("bge-")
            || combined.contains("nomic-embed"),
    }
}

fn group_gguf_tree(
    repo_id: &str,
    entries: &[RawHfTreeEntry],
    is_moe: bool,
    avail_vram_bytes: u64,
    avail_ram_bytes: u64,
) -> (Vec<HfQuantOption>, Vec<HfDownloadFile>) {
    let mut mmproj_files = Vec::new();
    let mut grouped: BTreeMap<String, Vec<HfDownloadFile>> = BTreeMap::new();

    for entry in entries {
        if entry.entry_type != "file" {
            continue;
        }
        let low = entry.path.to_ascii_lowercase();
        if !low.ends_with(".gguf") {
            continue;
        }
        let file_name = Path::new(&entry.path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(&entry.path)
            .to_string();

        let size_bytes = entry
            .lfs
            .as_ref()
            .and_then(|l| l.size)
            .unwrap_or(entry.size);
        let sha256 = entry.lfs.as_ref().and_then(|l| l.oid.clone());
        let url = format!(
            "{HF_WEB_BASE}/{repo_id}/resolve/main/{}",
            encode_path_segments(&entry.path)
        );
        let dl_file = HfDownloadFile {
            path: entry.path.clone(),
            file_name: file_name.clone(),
            url,
            size_bytes,
            sha256,
        };

        if file_name.to_ascii_lowercase().contains("mmproj") {
            mmproj_files.push(dl_file);
            continue;
        }

        // Group split shards (`foo-00001-of-00003.gguf`) under their shared key
        let group_key = if let Some((prefix, total)) = parse_split_shard_pattern(&file_name) {
            let parent = Path::new(&entry.path)
                .parent()
                .and_then(|p| p.to_str())
                .unwrap_or("");
            if parent.is_empty() {
                format!("{prefix} ({total} parts)")
            } else {
                format!("{parent}/{prefix} ({total} parts)")
            }
        } else {
            entry.path.clone()
        };

        grouped.entry(group_key).or_default().push(dl_file);
    }

    let mut quants = Vec::new();
    for (group_key, mut files) in grouped {
        files.sort_by(|a, b| a.file_name.cmp(&b.file_name));
        let total_size_bytes: u64 = files.iter().map(|f| f.size_bytes).sum();
        let shard_count = files.len();
        let primary_name = &files[0].file_name;
        let quantization = infer_quant_from_filename(Path::new(primary_name))
            .unwrap_or_else(|| "GGUF".to_string());
        let fit_verdict =
            estimate_hf_file_fit(total_size_bytes, is_moe, avail_vram_bytes, avail_ram_bytes);

        quants.push(HfQuantOption {
            id: group_key.clone(),
            label: if shard_count > 1 {
                format!("{quantization} · {shard_count} shards")
            } else {
                primary_name.clone()
            },
            quantization,
            total_size_bytes,
            shard_count,
            files,
            fit_verdict,
        });
    }

    // Sort by size ascending so users see smallest to largest clearly
    quants.sort_by_key(|q| q.total_size_bytes);
    mmproj_files.sort_by_key(|f| f.size_bytes);

    (quants, mmproj_files)
}

fn format_param_count(total: u64) -> String {
    if total >= 1_000_000_000 {
        format!("{:.1}B", total as f64 / 1e9)
    } else if total >= 1_000_000 {
        format!("{:.0}M", total as f64 / 1e6)
    } else {
        format!("{total}")
    }
}

fn infer_param_label_from_name(name: &str) -> Option<String> {
    for tok in name.split(['-', '_', ' ']) {
        let up = tok.to_ascii_uppercase();
        if (up.ends_with('B') || up.ends_with('M'))
            && up.len() >= 2
            && up[..up.len() - 1]
                .chars()
                .all(|c| c.is_ascii_digit() || c == '.' || c == 'X')
        {
            return Some(up);
        }
    }
    None
}

fn strip_yaml_frontmatter(md: &str) -> String {
    let trimmed = md.trim_start();
    if let Some(rest) = trimmed.strip_prefix("---")
        && let Some(idx) = rest.find("\n---")
    {
        return rest[idx + 4..].trim_start().to_string();
    }
    md.to_string()
}

fn percent_encode(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for b in input.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

fn encode_path_segments(path: &str) -> String {
    path.split('/')
        .map(percent_encode)
        .collect::<Vec<_>>()
        .join("/")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_split_ggufs_and_separates_mmproj() {
        let entries = vec![
            RawHfTreeEntry {
                entry_type: "file".into(),
                path: "Q4_K_M/model-Q4_K_M-00001-of-00002.gguf".into(),
                size: 4_000_000_000,
                lfs: None,
            },
            RawHfTreeEntry {
                entry_type: "file".into(),
                path: "Q4_K_M/model-Q4_K_M-00002-of-00002.gguf".into(),
                size: 3_500_000_000,
                lfs: None,
            },
            RawHfTreeEntry {
                entry_type: "file".into(),
                path: "mmproj-F16.gguf".into(),
                size: 800_000_000,
                lfs: None,
            },
        ];

        let (quants, mmprojs) = group_gguf_tree(
            "org/model-GGUF",
            &entries,
            false,
            16 * 1024 * 1024 * 1024,
            32 * 1024 * 1024 * 1024,
        );
        assert_eq!(quants.len(), 1);
        assert_eq!(quants[0].shard_count, 2);
        assert_eq!(quants[0].total_size_bytes, 7_500_000_000);
        assert_eq!(quants[0].quantization, "Q4_K_M");
        assert_eq!(quants[0].fit_verdict, FitVerdict::FullGpu);
        assert_eq!(mmprojs.len(), 1);
        assert_eq!(mmprojs[0].file_name, "mmproj-F16.gguf");
    }
}
