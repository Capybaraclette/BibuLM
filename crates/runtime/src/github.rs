use serde::{Deserialize, Serialize};
use thiserror::Error;

const GITHUB_RELEASES_URL: &str =
    "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=25";

#[derive(Debug, Error)]
pub enum GitHubError {
    #[error("HTTP error querying GitHub Releases: {0}")]
    Http(#[from] reqwest::Error),
    #[error("GitHub API returned status {status}: {body}")]
    ApiStatus { status: u16, body: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReleaseAssetInfo {
    pub name: String,
    pub download_url: String,
    pub size_bytes: u64,
}

/// Categorized downloadable `llama.cpp` Windows x64 backend variant from a GitHub release.
/// For CUDA builds, `assets` automatically pairs both `llama-<tag>-bin-win-cuda-*.zip`
/// and the companion `cudart-llama-bin-win-cuda-*.zip`.
/// Target OS + CPU architecture used to select matching `ggml-org/llama.cpp` release archives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetPlatform {
    WinX64,
    WinArm64,
    LinuxX64,
    LinuxArm64,
    MacosArm64,
    MacosX64,
}

impl TargetPlatform {
    pub fn current() -> Self {
        match (std::env::consts::OS, std::env::consts::ARCH) {
            ("windows", "aarch64") => Self::WinArm64,
            ("windows", _) => Self::WinX64,
            ("linux", "aarch64") => Self::LinuxArm64,
            ("linux", _) => Self::LinuxX64,
            ("macos", "aarch64") => Self::MacosArm64,
            ("macos", _) => Self::MacosX64,
            _ => Self::LinuxX64,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeBackendVariant {
    pub id: String,
    pub tag: String,
    /// e.g. `"cuda-12.4"`, `"cuda-13.1"`, `"vulkan"`, `"cpu"`, `"rocm"`, `"sycl"`
    pub backend: String,
    pub display_name: String,
    pub description: String,
    pub cuda_version: Option<String>,
    pub assets: Vec<ReleaseAssetInfo>,
    pub total_size_bytes: u64,
    pub compatible: bool,
    pub recommended: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlamaRelease {
    pub tag_name: String,
    pub name: String,
    pub published_at: String,
    pub html_url: String,
    pub release_notes: String,
    pub variants: Vec<RuntimeBackendVariant>,
}

#[derive(Debug, Deserialize)]
struct GhRelease {
    tag_name: String,
    name: Option<String>,
    published_at: Option<String>,
    html_url: String,
    body: Option<String>,
    draft: bool,
    #[allow(dead_code)]
    prerelease: bool,
    assets: Vec<GhAsset>,
}

#[derive(Debug, Deserialize)]
struct GhAsset {
    name: String,
    browser_download_url: String,
    size: u64,
}

#[derive(Debug, Clone, Default)]
pub struct HardwareMatchContext {
    pub has_nvidia: bool,
    pub nvidia_cuda_major: Option<u32>,
    pub nvidia_cuda_minor: Option<u32>,
    pub has_amd_gpu: bool,
    pub has_intel_gpu: bool,
    pub has_any_gpu: bool,
    pub has_avx512: bool,
    pub has_avx2: bool,
}

/// Fetches official releases directly from `ggml-org/llama.cpp` on GitHub.
pub async fn fetch_llama_cpp_releases(
    github_token: Option<&str>,
    hw: &HardwareMatchContext,
) -> Result<Vec<LlamaRelease>, GitHubError> {
    let client = reqwest::Client::builder()
        .user_agent("Bibulm/0.3 (ggml-org/llama.cpp manager)")
        .build()?;

    let mut req = client
        .get(GITHUB_RELEASES_URL)
        .header("Accept", "application/vnd.github+json");
    if let Some(tok) = github_token.map(str::trim).filter(|s| !s.is_empty()) {
        req = req.bearer_auth(tok);
    }

    let resp = req.send().await?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(GitHubError::ApiStatus {
            status: status.as_u16(),
            body,
        });
    }

    let raw: Vec<GhRelease> = resp.json().await?;
    let mut releases = Vec::new();
    for gh in raw {
        // Note: ggml-org/llama.cpp marks rolling `bXXXX` builds as `prerelease: true`,
        // so only skip unpublished drafts.
        if gh.draft {
            continue;
        }
        let assets: Vec<ReleaseAssetInfo> = gh
            .assets
            .into_iter()
            .map(|a| ReleaseAssetInfo {
                name: a.name,
                download_url: a.browser_download_url,
                size_bytes: a.size,
            })
            .collect();
        let platform = TargetPlatform::current();
        let variants = categorize_platform_assets(&gh.tag_name, &assets, hw, platform);
        if variants.is_empty() {
            continue;
        }
        releases.push(LlamaRelease {
            name: gh.name.filter(|s| !s.is_empty()).unwrap_or_else(|| gh.tag_name.clone()),
            tag_name: gh.tag_name,
            published_at: gh.published_at.unwrap_or_default(),
            html_url: gh.html_url,
            release_notes: gh.body.unwrap_or_default(),
            variants,
        });
    }
    Ok(releases)
}

/// Categorizes Windows x64 assets from a `ggml-org/llama.cpp` release into downloadable backend variants,
/// automatically pairing companion `cudart` zips for CUDA builds.
pub fn categorize_win_x64_assets(
    tag: &str,
    assets: &[ReleaseAssetInfo],
    hw: &HardwareMatchContext,
) -> Vec<RuntimeBackendVariant> {
    categorize_platform_assets(tag, assets, hw, TargetPlatform::WinX64)
}

fn is_archive_name(lower: &str) -> bool {
    lower.ends_with(".zip") || lower.ends_with(".tar.gz") || lower.ends_with(".tgz")
}

fn matches_target_platform(lower: &str, platform: TargetPlatform) -> bool {
    if !is_archive_name(lower) || lower.starts_with("cudart-") {
        return false;
    }
    match platform {
        TargetPlatform::WinX64 => {
            lower.ends_with(".zip") && lower.contains("-bin-win-") && lower.contains("x64")
        }
        TargetPlatform::WinArm64 => {
            lower.ends_with(".zip")
                && lower.contains("-bin-win-")
                && (lower.contains("arm64") || lower.contains("aarch64"))
        }
        TargetPlatform::LinuxX64 => {
            (lower.contains("-bin-ubuntu-") || lower.contains("-bin-linux-"))
                && (lower.contains("x64") || lower.contains("x86_64"))
        }
        TargetPlatform::LinuxArm64 => {
            (lower.contains("-bin-ubuntu-") || lower.contains("-bin-linux-"))
                && (lower.contains("arm64") || lower.contains("aarch64"))
        }
        TargetPlatform::MacosArm64 => lower.contains("-bin-macos-") && lower.contains("arm64"),
        TargetPlatform::MacosX64 => {
            lower.contains("-bin-macos-") && (lower.contains("x64") || lower.contains("x86_64"))
        }
    }
}

fn platform_label(platform: TargetPlatform) -> &'static str {
    match platform {
        TargetPlatform::WinX64 => "Windows x64",
        TargetPlatform::WinArm64 => "Windows ARM64",
        TargetPlatform::LinuxX64 => "Linux x64",
        TargetPlatform::LinuxArm64 => "Linux ARM64",
        TargetPlatform::MacosArm64 => "macOS arm64",
        TargetPlatform::MacosX64 => "macOS x64",
    }
}

/// Categorizes assets from a `ggml-org/llama.cpp` release for any supported OS/arch.
pub fn categorize_platform_assets(
    tag: &str,
    assets: &[ReleaseAssetInfo],
    hw: &HardwareMatchContext,
    platform: TargetPlatform,
) -> Vec<RuntimeBackendVariant> {
    let mut variants = Vec::new();
    let plat_label = platform_label(platform);

    // Index cudart companion archives (Windows CUDA builds).
    let cudart_assets: Vec<&ReleaseAssetInfo> = if platform == TargetPlatform::WinX64 {
        assets
            .iter()
            .filter(|a| {
                let low = a.name.to_ascii_lowercase();
                low.starts_with("cudart-llama-bin-win-")
                    && low.ends_with(".zip")
                    && low.contains("x64")
            })
            .collect()
    } else {
        Vec::new()
    };

    for asset in assets {
        let low = asset.name.to_ascii_lowercase();
        if !matches_target_platform(&low, platform) {
            continue;
        }

        // 1. NVIDIA CUDA builds
        if let Some(cuda_ver) = extract_cuda_version(&low) {
            let mut paired = vec![asset.clone()];
            if let Some(cudart) = find_matching_cudart(&cudart_assets, &cuda_ver) {
                paired.push(cudart.clone());
            }
            let total_size_bytes: u64 = paired.iter().map(|a| a.size_bytes).sum();
            let compatible = is_cuda_compatible(&cuda_ver, hw);
            let backend = format!("cuda-{cuda_ver}");
            variants.push(RuntimeBackendVariant {
                id: format!("{tag}-{backend}"),
                tag: tag.to_string(),
                backend,
                display_name: format!("CUDA {cuda_ver} llama.cpp ({plat_label})"),
                description: if paired.len() > 1 {
                    format!("NVIDIA CUDA {cuda_ver} accelerated build (+ bundled CUDART DLLs)")
                } else {
                    format!("NVIDIA CUDA {cuda_ver} accelerated build")
                },
                cuda_version: Some(cuda_ver),
                assets: paired,
                total_size_bytes,
                compatible,
                recommended: false,
            });
            continue;
        }

        // 2. Vulkan builds
        if low.contains("-vulkan-") || low.contains("-vulkan.") {
            variants.push(RuntimeBackendVariant {
                id: format!("{tag}-vulkan"),
                tag: tag.to_string(),
                backend: "vulkan".to_string(),
                display_name: format!("Vulkan llama.cpp ({plat_label})"),
                description: "Cross-vendor Vulkan GPU build (NVIDIA / AMD / Intel)".to_string(),
                cuda_version: None,
                assets: vec![asset.clone()],
                total_size_bytes: asset.size_bytes,
                compatible: hw.has_any_gpu || matches!(platform, TargetPlatform::LinuxX64 | TargetPlatform::LinuxArm64),
                recommended: false,
            });
            continue;
        }

        // 3. ROCm / HIP Radeon builds
        if low.contains("-hip-") || low.contains("-rocm-") {
            variants.push(RuntimeBackendVariant {
                id: format!("{tag}-rocm"),
                tag: tag.to_string(),
                backend: "rocm".to_string(),
                display_name: format!("ROCm / HIP llama.cpp ({plat_label})"),
                description: "AMD Radeon ROCm/HIP GPU build".to_string(),
                cuda_version: None,
                assets: vec![asset.clone()],
                total_size_bytes: asset.size_bytes,
                compatible: hw.has_amd_gpu,
                recommended: false,
            });
            continue;
        }

        // 4. Intel SYCL & OpenVINO builds
        if low.contains("-sycl-") {
            variants.push(RuntimeBackendVariant {
                id: format!("{tag}-sycl"),
                tag: tag.to_string(),
                backend: "sycl".to_string(),
                display_name: format!("SYCL llama.cpp ({plat_label})"),
                description: "Intel Arc / oneAPI SYCL GPU build".to_string(),
                cuda_version: None,
                assets: vec![asset.clone()],
                total_size_bytes: asset.size_bytes,
                compatible: hw.has_intel_gpu,
                recommended: false,
            });
            continue;
        }

        if low.contains("-openvino-") {
            variants.push(RuntimeBackendVariant {
                id: format!("{tag}-openvino"),
                tag: tag.to_string(),
                backend: "openvino".to_string(),
                display_name: format!("OpenVINO llama.cpp ({plat_label})"),
                description: "Intel OpenVINO accelerated build (CPU / iGPU / NPU)".to_string(),
                cuda_version: None,
                assets: vec![asset.clone()],
                total_size_bytes: asset.size_bytes,
                compatible: true,
                recommended: false,
            });
            continue;
        }

        // 5. macOS Apple Silicon / Intel Metal builds
        if matches!(platform, TargetPlatform::MacosArm64 | TargetPlatform::MacosX64)
            && !low.contains("-cpu-")
        {
            variants.push(RuntimeBackendVariant {
                id: format!("{tag}-metal"),
                tag: tag.to_string(),
                backend: "metal".to_string(),
                display_name: format!("Metal llama.cpp ({plat_label})"),
                description: "Apple Metal GPU-accelerated build".to_string(),
                cuda_version: None,
                assets: vec![asset.clone()],
                total_size_bytes: asset.size_bytes,
                compatible: true,
                recommended: false,
            });
            continue;
        }

        // 6. CPU / generic OS builds (`-cpu-`, `-avx512-`, `-avx2-`, or `llama-bXXXX-bin-ubuntu-x64.tar.gz`)
        let is_explicit_cpu = low.contains("-cpu-")
            || low.contains("-avx512-")
            || low.contains("-avx2-")
            || low.contains("-avx-")
            || low.contains("-noavx-");
        let is_linux_or_arm_default = matches!(
            platform,
            TargetPlatform::LinuxX64 | TargetPlatform::LinuxArm64 | TargetPlatform::WinArm64
        );
        if is_explicit_cpu || is_linux_or_arm_default {
            let (backend, label, compat) = if low.contains("-avx512-") {
                (
                    "cpu-avx512".to_string(),
                    format!("CPU AVX-512 llama.cpp ({plat_label})"),
                    hw.has_avx512,
                )
            } else if low.contains("-avx2-") {
                (
                    "cpu-avx2".to_string(),
                    format!("CPU AVX2 llama.cpp ({plat_label})"),
                    hw.has_avx2,
                )
            } else if low.contains("-cpu-") || is_linux_or_arm_default {
                (
                    "cpu".to_string(),
                    format!("CPU llama.cpp ({plat_label})"),
                    true,
                )
            } else {
                (
                    "cpu-generic".to_string(),
                    format!("CPU Generic llama.cpp ({plat_label})"),
                    true,
                )
            };
            variants.push(RuntimeBackendVariant {
                id: format!("{tag}-{backend}"),
                tag: tag.to_string(),
                backend,
                display_name: label,
                description: "CPU-only llama.cpp engine (universal fallback)".to_string(),
                cuda_version: None,
                assets: vec![asset.clone()],
                total_size_bytes: asset.size_bytes,
                compatible: compat,
                recommended: false,
            });
        }
    }

    // Mark the single best compatible variant as recommended.
    mark_recommended_variant(&mut variants, hw);

    // Sort: recommended first, then compatible, then CUDA highest version, Metal, Vulkan, ROCm, SYCL, CPU.
    variants.sort_by(|a, b| {
        b.recommended
            .cmp(&a.recommended)
            .then(b.compatible.cmp(&a.compatible))
            .then(backend_priority(&b.backend).cmp(&backend_priority(&a.backend)))
    });

    variants
}

fn mark_recommended_variant(variants: &mut [RuntimeBackendVariant], hw: &HardwareMatchContext) {
    // Prefer highest compatible CUDA version when NVIDIA GPU is detected.
    if hw.has_nvidia {
        let best_cuda = variants
            .iter()
            .enumerate()
            .filter(|(_, v)| v.compatible && v.cuda_version.is_some())
            .max_by_key(|(_, v)| parse_ver_tuple(v.cuda_version.as_deref().unwrap_or("0.0")))
            .map(|(i, _)| i);
        if let Some(idx) = best_cuda {
            variants[idx].recommended = true;
            return;
        }
    }
    if hw.has_amd_gpu
        && let Some(v) = variants.iter_mut().find(|v| v.compatible && v.backend == "rocm")
    {
        v.recommended = true;
        return;
    }
    if let Some(v) = variants.iter_mut().find(|v| v.compatible && v.backend == "metal") {
        v.recommended = true;
        return;
    }
    if hw.has_any_gpu
        && let Some(v) = variants.iter_mut().find(|v| v.compatible && v.backend == "vulkan")
    {
        v.recommended = true;
        return;
    }
    if let Some(v) = variants.iter_mut().find(|v| v.compatible && v.backend.starts_with("cpu")) {
        v.recommended = true;
    }
}

fn backend_priority(backend: &str) -> u32 {
    if let Some(ver) = backend.strip_prefix("cuda-") {
        let (maj, min) = parse_ver_tuple(ver);
        return 1000 + maj * 20 + min;
    }
    match backend {
        "metal" => 600,
        "vulkan" => 500,
        "rocm" => 400,
        "sycl" => 350,
        "openvino" => 300,
        "cpu" => 200,
        "cpu-avx512" => 190,
        "cpu-avx2" => 180,
        _ => 100,
    }
}

fn extract_cuda_version(lower_name: &str) -> Option<String> {
    let after = lower_name
        .split("-cuda-")
        .nth(1)
        .or_else(|| lower_name.split("-cu").nth(1))?;
    let token = after.split('-').next()?;
    let clean = token.strip_prefix("cu").unwrap_or(token);
    if clean.chars().next()?.is_ascii_digit() {
        Some(clean.to_string())
    } else {
        None
    }
}

fn find_matching_cudart<'a>(
    cudart_assets: &[&'a ReleaseAssetInfo],
    cuda_ver: &str,
) -> Option<&'a ReleaseAssetInfo> {
    let major = cuda_ver.split('.').next().unwrap_or(cuda_ver);
    cudart_assets
        .iter()
        .copied()
        .find(|a| {
            let low = a.name.to_ascii_lowercase();
            low.contains(&format!("cuda-{cuda_ver}")) || low.contains(&format!("cu{cuda_ver}"))
        })
        .or_else(|| {
            cudart_assets.iter().copied().find(|a| {
                let low = a.name.to_ascii_lowercase();
                low.contains(&format!("cuda-{major}")) || low.contains(&format!("cu{major}"))
            })
        })
}

fn parse_ver_tuple(ver: &str) -> (u32, u32) {
    let mut parts = ver.split('.');
    let maj = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let min = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    (maj, min)
}

fn is_cuda_compatible(cuda_ver: &str, hw: &HardwareMatchContext) -> bool {
    if !hw.has_nvidia {
        return false;
    }
    let (req_maj, req_min) = parse_ver_tuple(cuda_ver);
    match (hw.nvidia_cuda_major, hw.nvidia_cuda_minor) {
        (Some(drv_maj), Some(drv_min)) => (drv_maj, drv_min) >= (req_maj, req_min),
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairs_cuda_zip_with_matching_cudart_and_picks_recommended() {
        let assets = vec![
            ReleaseAssetInfo {
                name: "llama-b5200-bin-win-cuda-12.4-x64.zip".into(),
                download_url: "https://example/cuda124.zip".into(),
                size_bytes: 95_000_000,
            },
            ReleaseAssetInfo {
                name: "cudart-llama-bin-win-cuda-12.4-x64.zip".into(),
                download_url: "https://example/cudart124.zip".into(),
                size_bytes: 185_000_000,
            },
            ReleaseAssetInfo {
                name: "llama-b5200-bin-win-cuda-13.1-x64.zip".into(),
                download_url: "https://example/cuda131.zip".into(),
                size_bytes: 100_000_000,
            },
            ReleaseAssetInfo {
                name: "cudart-llama-bin-win-cuda-13.1-x64.zip".into(),
                download_url: "https://example/cudart131.zip".into(),
                size_bytes: 190_000_000,
            },
            ReleaseAssetInfo {
                name: "llama-b5200-bin-win-vulkan-x64.zip".into(),
                download_url: "https://example/vulkan.zip".into(),
                size_bytes: 25_000_000,
            },
            ReleaseAssetInfo {
                name: "llama-b5200-bin-win-cpu-x64.zip".into(),
                download_url: "https://example/cpu.zip".into(),
                size_bytes: 12_000_000,
            },
        ];

        let hw = HardwareMatchContext {
            has_nvidia: true,
            nvidia_cuda_major: Some(12),
            nvidia_cuda_minor: Some(8),
            has_any_gpu: true,
            has_avx2: true,
            ..Default::default()
        };

        let variants = categorize_win_x64_assets("b5200", &assets, &hw);
        assert_eq!(variants.len(), 4);

        // Best compatible CUDA for driver 12.8 is 12.4 (not 13.1)
        assert_eq!(variants[0].backend, "cuda-12.4");
        assert!(variants[0].recommended);
        assert!(variants[0].compatible);
        assert_eq!(variants[0].assets.len(), 2);
        assert_eq!(variants[0].total_size_bytes, 280_000_000);

        let cuda13 = variants.iter().find(|v| v.backend == "cuda-13.1").unwrap();
        assert!(!cuda13.compatible);
        assert_eq!(cuda13.assets.len(), 2);
    }

    #[test]
    fn categorizes_linux_arm64_win_arm64_and_macos_assets() {
        let assets = vec![
            ReleaseAssetInfo {
                name: "llama-b5200-bin-ubuntu-x64.tar.gz".into(),
                download_url: "https://example/lin-x64.tar.gz".into(),
                size_bytes: 30_000_000,
            },
            ReleaseAssetInfo {
                name: "llama-b5200-bin-ubuntu-vulkan-x64.tar.gz".into(),
                download_url: "https://example/lin-vk.tar.gz".into(),
                size_bytes: 40_000_000,
            },
            ReleaseAssetInfo {
                name: "llama-b5200-bin-ubuntu-arm64.tar.gz".into(),
                download_url: "https://example/lin-arm64.tar.gz".into(),
                size_bytes: 28_000_000,
            },
            ReleaseAssetInfo {
                name: "llama-b5200-bin-win-arm64.zip".into(),
                download_url: "https://example/win-arm64.zip".into(),
                size_bytes: 22_000_000,
            },
            ReleaseAssetInfo {
                name: "llama-b5200-bin-macos-arm64.tar.gz".into(),
                download_url: "https://example/mac-arm64.tar.gz".into(),
                size_bytes: 25_000_000,
            },
        ];
        let hw = HardwareMatchContext {
            has_any_gpu: true,
            ..Default::default()
        };

        let lin_x64 = categorize_platform_assets("b5200", &assets, &hw, TargetPlatform::LinuxX64);
        assert_eq!(lin_x64.len(), 2);
        assert!(lin_x64.iter().any(|v| v.backend == "vulkan" && v.recommended));
        assert!(lin_x64.iter().any(|v| v.backend == "cpu"));

        let lin_arm = categorize_platform_assets("b5200", &assets, &hw, TargetPlatform::LinuxArm64);
        assert_eq!(lin_arm.len(), 1);
        assert_eq!(lin_arm[0].backend, "cpu");

        let win_arm = categorize_platform_assets("b5200", &assets, &hw, TargetPlatform::WinArm64);
        assert_eq!(win_arm.len(), 1);
        assert_eq!(win_arm[0].backend, "cpu");

        let mac_arm = categorize_platform_assets("b5200", &assets, &hw, TargetPlatform::MacosArm64);
        assert_eq!(mac_arm.len(), 1);
        assert_eq!(mac_arm[0].backend, "metal");
        assert!(mac_arm[0].recommended);
    }
}
