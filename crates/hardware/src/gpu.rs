#[cfg(any(target_os = "linux", target_os = "windows"))]
pub use nvml_wrapper::Nvml;

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
pub struct Nvml;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuDeviceInfo {
    pub device_id: u32,
    pub name: String,
    pub vendor: String,
    pub backend: String,
    pub vram_total_bytes: u64,
    pub shared_mem_total_bytes: u64,
    pub driver_version: Option<String>,
    pub cuda_version: Option<String>,
    pub cuda_version_major: Option<u32>,
    pub cuda_version_minor: Option<u32>,
}

#[derive(Debug, Clone, Default)]
pub struct GpuDynamicSample {
    pub vram_used_bytes: u64,
    pub vram_total_bytes: u64,
    pub shared_gpu_mem_used_bytes: u64,
    pub gpu_utilization_pct: Option<f32>,
    pub gpu_temp_celsius: Option<u32>,
}

/// Enumerates GPUs using NVIDIA NVML (primary for NVIDIA + CUDA driver version)
/// augmented with Windows DXGI (for shared memory capacity and non-NVIDIA GPUs).
pub fn detect_gpus(nvml: Option<&Nvml>) -> Vec<GpuDeviceInfo> {
    let dxgi_adapters = enumerate_dxgi_adapters();
    let mut gpus: Vec<GpuDeviceInfo> = Vec::new();

    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    let _ = nvml;

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    if let Some(nvml) = nvml {
        let driver_version = nvml.sys_driver_version().ok();
        let cuda_raw = nvml.sys_cuda_driver_version().ok();
        let (cuda_version, cuda_major, cuda_minor) = match cuda_raw {
            Some(v) if v > 0 => {
                let maj = (v / 1000) as u32;
                let min = ((v % 1000) / 10) as u32;
                (Some(format!("{maj}.{min}")), Some(maj), Some(min))
            }
            _ => (None, None, None),
        };

        if let Ok(count) = nvml.device_count() {
            for idx in 0..count {
                if let Ok(dev) = nvml.device_by_index(idx) {
                    let name = dev
                        .name()
                        .unwrap_or_else(|_| format!("NVIDIA GPU #{idx}"));
                    let mem = dev.memory_info().ok();
                    let vram_total = mem.map(|m| m.total).unwrap_or(0);

                    // Match DXGI adapter to obtain Windows shared memory capacity
                    let shared_total = dxgi_adapters
                        .iter()
                        .find(|a| a.vendor_id == 0x10DE && names_roughly_match(&a.name, &name))
                        .or_else(|| dxgi_adapters.iter().find(|a| a.vendor_id == 0x10DE))
                        .map(|a| a.shared_system_memory)
                        .unwrap_or(0);

                    gpus.push(GpuDeviceInfo {
                        device_id: idx,
                        name,
                        vendor: "NVIDIA".to_string(),
                        backend: "CUDA".to_string(),
                        vram_total_bytes: vram_total,
                        shared_mem_total_bytes: shared_total,
                        driver_version: driver_version.clone(),
                        cuda_version: cuda_version.clone(),
                        cuda_version_major: cuda_major,
                        cuda_version_minor: cuda_minor,
                    });
                }
            }
        }
    }

    // Add non-NVIDIA discrete/integrated GPUs from DXGI (or NVIDIA fallback if NVML unavailable).
    for adapter in dxgi_adapters {
        if adapter.dedicated_video_memory < 256 * 1024 * 1024 {
            // Skip tiny display-only / virtual adapters (< 256 MiB)
            continue;
        }
        let already_present = gpus
            .iter()
            .any(|g| names_roughly_match(&g.name, &adapter.name));
        if already_present {
            continue;
        }
        let (vendor, backend) = match adapter.vendor_id {
            0x10DE => ("NVIDIA", "CUDA"),
            0x1002 => ("AMD", "Vulkan"),
            0x8086 => ("Intel", "Vulkan"),
            _ => ("Unknown", "Vulkan"),
        };
        let next_id = gpus.len() as u32;
        gpus.push(GpuDeviceInfo {
            device_id: next_id,
            name: adapter.name,
            vendor: vendor.to_string(),
            backend: backend.to_string(),
            vram_total_bytes: adapter.dedicated_video_memory,
            shared_mem_total_bytes: adapter.shared_system_memory,
            driver_version: None,
            cuda_version: None,
            cuda_version_major: None,
            cuda_version_minor: None,
        });
    }

    #[cfg(target_os = "linux")]
    {
        gpus.extend(detect_linux_drm_gpus(&gpus));
    }

    gpus
}

/// Samples live dedicated VRAM, Windows WDDM shared GPU memory, GPU utilization, and temperature.
pub fn sample_gpu_dynamic(nvml: Option<&Nvml>, disabled_ids: &[u32]) -> GpuDynamicSample {
    let mut sample = GpuDynamicSample::default();

    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    let _ = (nvml, disabled_ids);

    // Query NVML for dedicated VRAM, utilization, and temperature.
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    if let Some(nvml) = nvml
        && let Ok(count) = nvml.device_count()
    {
        let mut util_sum = 0.0f32;
        let mut util_cnt = 0u32;
        let mut max_temp: Option<u32> = None;

        for idx in 0..count {
            if disabled_ids.contains(&idx) {
                continue;
            }
            if let Ok(dev) = nvml.device_by_index(idx) {
                if let Ok(mem) = dev.memory_info() {
                    sample.vram_used_bytes = sample.vram_used_bytes.saturating_add(mem.used);
                    sample.vram_total_bytes = sample.vram_total_bytes.saturating_add(mem.total);
                }
                if let Ok(util) = dev.utilization_rates() {
                    util_sum += util.gpu as f32;
                    util_cnt += 1;
                }
                if let Ok(temp) = dev
                    .temperature(nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu)
                {
                    max_temp = Some(max_temp.map_or(temp, |m| m.max(temp)));
                }
            }
        }
        if util_cnt > 0 {
            sample.gpu_utilization_pct = Some(util_sum / util_cnt as f32);
        }
        sample.gpu_temp_celsius = max_temp;
    }

    // Query DXGI for Windows WDDM Shared GPU Memory usage (and VRAM fallback when NVML unavailable).
    let dxgi_usage = query_dxgi_video_memory();
    if sample.vram_total_bytes == 0 && dxgi_usage.local_budget > 0 {
        sample.vram_used_bytes = dxgi_usage.local_usage;
        sample.vram_total_bytes = dxgi_usage.local_budget;
    }
    sample.shared_gpu_mem_used_bytes = dxgi_usage.non_local_usage;

    #[cfg(target_os = "linux")]
    if sample.vram_total_bytes == 0 {
        let drm = sample_linux_drm_telemetry(disabled_ids);
        if drm.vram_total_bytes > 0 {
            sample.vram_used_bytes = drm.vram_used_bytes;
            sample.vram_total_bytes = drm.vram_total_bytes;
            if sample.gpu_utilization_pct.is_none() {
                sample.gpu_utilization_pct = drm.gpu_utilization_pct;
            }
            if sample.gpu_temp_celsius.is_none() {
                sample.gpu_temp_celsius = drm.gpu_temp_celsius;
            }
        }
    }

    sample
}

/// Queries NVML for dedicated VRAM usage (in MiB) attributed to a specific OS process ID
/// across all enabled NVIDIA GPUs.
#[cfg(any(target_os = "linux", target_os = "windows"))]
pub fn sample_process_vram_mib(
    nvml: Option<&Nvml>,
    pid: u32,
    disabled_ids: &[u32],
) -> Option<f64> {
    use nvml_wrapper::enums::device::UsedGpuMemory;

    let nvml = nvml?;
    let count = nvml.device_count().ok()?;
    let mut total_bytes: u64 = 0;
    let mut found = false;

    for idx in 0..count {
        if disabled_ids.contains(&idx) {
            continue;
        }
        let Ok(dev) = nvml.device_by_index(idx) else {
            continue;
        };
        let mut dev_bytes: u64 = 0;
        let mut dev_found = false;

        for procs in [
            dev.running_compute_processes(),
            dev.running_graphics_processes(),
        ]
        .into_iter()
        .flatten()
        {
            for p in procs {
                if p.pid == pid
                    && let UsedGpuMemory::Used(b) = p.used_gpu_memory
                {
                    dev_bytes = dev_bytes.max(b);
                    dev_found = true;
                }
            }
        }
        if dev_found {
            total_bytes = total_bytes.saturating_add(dev_bytes);
            found = true;
        }
    }

    if found {
        Some(total_bytes as f64 / (1024.0 * 1024.0))
    } else {
        None
    }
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
pub fn sample_process_vram_mib(
    _nvml: Option<&Nvml>,
    _pid: u32,
    _disabled_ids: &[u32],
) -> Option<f64> {
    None
}

fn names_roughly_match(a: &str, b: &str) -> bool {
    let al = a.to_ascii_lowercase();
    let bl = b.to_ascii_lowercase();
    al == bl || al.contains(&bl) || bl.contains(&al)
}

#[derive(Debug, Clone)]
struct DxgiAdapterSummary {
    name: String,
    vendor_id: u32,
    dedicated_video_memory: u64,
    shared_system_memory: u64,
}

#[derive(Debug, Clone, Default)]
struct DxgiMemoryUsage {
    local_usage: u64,
    local_budget: u64,
    non_local_usage: u64,
}

#[cfg(target_os = "windows")]
fn enumerate_dxgi_adapters() -> Vec<DxgiAdapterSummary> {
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE, IDXGIFactory1,
    };

    let mut out = Vec::new();
    let Ok(factory) = (unsafe { CreateDXGIFactory1::<IDXGIFactory1>() }) else {
        return out;
    };

    let mut idx = 0u32;
    while let Ok(adapter) = unsafe { factory.EnumAdapters1(idx) } {
        idx += 1;
        if let Ok(desc) = unsafe { adapter.GetDesc1() } {
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0 {
                continue;
            }
            let nul = desc
                .Description
                .iter()
                .position(|&c| c == 0)
                .unwrap_or(desc.Description.len());
            let name = String::from_utf16_lossy(&desc.Description[..nul])
                .trim()
                .to_string();
            out.push(DxgiAdapterSummary {
                name,
                vendor_id: desc.VendorId,
                dedicated_video_memory: desc.DedicatedVideoMemory as u64,
                shared_system_memory: desc.SharedSystemMemory as u64,
            });
        }
    }
    out
}

#[cfg(not(target_os = "windows"))]
fn enumerate_dxgi_adapters() -> Vec<DxgiAdapterSummary> {
    Vec::new()
}

#[cfg(target_os = "windows")]
fn query_dxgi_video_memory() -> DxgiMemoryUsage {
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
        DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, DXGI_QUERY_VIDEO_MEMORY_INFO, IDXGIAdapter3,
        IDXGIFactory1,
    };
    use windows::core::Interface;

    let mut usage = DxgiMemoryUsage::default();
    let Ok(factory) = (unsafe { CreateDXGIFactory1::<IDXGIFactory1>() }) else {
        return usage;
    };

    let mut idx = 0u32;
    while let Ok(adapter) = unsafe { factory.EnumAdapters1(idx) } {
        idx += 1;
        let Ok(desc) = (unsafe { adapter.GetDesc1() }) else {
            continue;
        };
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0 {
            continue;
        }
        if desc.DedicatedVideoMemory < 256 * 1024 * 1024 {
            continue;
        }
        if let Ok(adapter3) = adapter.cast::<IDXGIAdapter3>() {
            let mut local = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
            if unsafe {
                adapter3.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut local)
            }
            .is_ok()
            {
                usage.local_usage = usage.local_usage.saturating_add(local.CurrentUsage);
                usage.local_budget = usage
                    .local_budget
                    .saturating_add(desc.DedicatedVideoMemory as u64);
            }
            let mut non_local = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
            if unsafe {
                adapter3.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &mut non_local)
            }
            .is_ok()
            {
                usage.non_local_usage = usage.non_local_usage.saturating_add(non_local.CurrentUsage);
            }
        }
    }
    usage
}

#[cfg(not(target_os = "windows"))]
fn query_dxgi_video_memory() -> DxgiMemoryUsage {
    DxgiMemoryUsage::default()
}

#[cfg(target_os = "linux")]
fn is_drm_card_dir(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("card") else {
        return false;
    };
    !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit())
}

#[cfg(target_os = "linux")]
fn read_sysfs_trimmed(path: &std::path::Path) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

#[cfg(target_os = "linux")]
fn read_sysfs_u64(path: &std::path::Path) -> Option<u64> {
    read_sysfs_trimmed(path)?.parse::<u64>().ok()
}

#[cfg(target_os = "linux")]
fn detect_linux_drm_gpus(existing: &[GpuDeviceInfo]) -> Vec<GpuDeviceInfo> {
    let drm_root = std::path::Path::new("/sys/class/drm");
    let Ok(entries) = std::fs::read_dir(drm_root) else {
        return Vec::new();
    };

    let mut cards: Vec<(String, std::path::PathBuf)> = entries
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            if is_drm_card_dir(&name) {
                Some((name, e.path()))
            } else {
                None
            }
        })
        .collect();
    cards.sort_by(|a, b| a.0.cmp(&b.0));

    let has_nvml_nvidia = existing
        .iter()
        .any(|g| g.vendor.eq_ignore_ascii_case("NVIDIA"));
    let mut next_id = existing.len() as u32;
    let mut out = Vec::new();

    for (card_name, card_path) in cards {
        let dev_dir = card_path.join("device");
        let Some(vendor_hex) = read_sysfs_trimmed(&dev_dir.join("vendor")) else {
            continue;
        };
        let vendor_low = vendor_hex.to_ascii_lowercase();
        if vendor_low == "0x10de" && has_nvml_nvidia {
            continue;
        }
        let (vendor, backend) = match vendor_low.as_str() {
            "0x1002" => ("AMD", "ROCm/Vulkan"),
            "0x8086" => ("Intel", "Vulkan"),
            "0x10de" => ("NVIDIA", "CUDA"),
            _ => continue,
        };

        let vram_total_bytes = read_sysfs_u64(&dev_dir.join("mem_info_vram_total")).unwrap_or(0);
        if vram_total_bytes == 0 && vendor != "Intel" {
            // Skip headless/display-only virtual nodes without dedicated VRAM unless Intel iGPU
            continue;
        }

        let name = read_sysfs_trimmed(&dev_dir.join("product_name"))
            .or_else(|| {
                read_sysfs_trimmed(&dev_dir.join("uevent")).and_then(|u| {
                    u.lines()
                        .find_map(|l| l.strip_prefix("PCI_SLOT_NAME="))
                        .map(|slot| format!("{vendor} GPU ({card_name} @ {slot})"))
                })
            })
            .unwrap_or_else(|| format!("{vendor} GPU ({card_name})"));

        out.push(GpuDeviceInfo {
            device_id: next_id,
            name,
            vendor: vendor.to_string(),
            backend: backend.to_string(),
            vram_total_bytes,
            shared_mem_total_bytes: 0,
            driver_version: None,
            cuda_version: None,
            cuda_version_major: None,
            cuda_version_minor: None,
        });
        next_id += 1;
    }

    out
}

#[cfg(target_os = "linux")]
fn sample_linux_drm_telemetry(disabled_ids: &[u32]) -> GpuDynamicSample {
    let mut sample = GpuDynamicSample::default();
    let drm_root = std::path::Path::new("/sys/class/drm");
    let Ok(entries) = std::fs::read_dir(drm_root) else {
        return sample;
    };
    let mut cards: Vec<(String, std::path::PathBuf)> = entries
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            if is_drm_card_dir(&name) {
                Some((name, e.path()))
            } else {
                None
            }
        })
        .collect();
    cards.sort_by(|a, b| a.0.cmp(&b.0));

    let mut util_sum = 0.0f32;
    let mut util_cnt = 0u32;
    let mut max_temp: Option<u32> = None;

    for (idx, (_name, card_path)) in cards.into_iter().enumerate() {
        if disabled_ids.contains(&(idx as u32)) {
            continue;
        }
        let dev_dir = card_path.join("device");
        let Some(total) = read_sysfs_u64(&dev_dir.join("mem_info_vram_total")) else {
            continue;
        };
        if total == 0 {
            continue;
        }
        sample.vram_total_bytes = sample.vram_total_bytes.saturating_add(total);
        if let Some(used) = read_sysfs_u64(&dev_dir.join("mem_info_vram_used")) {
            sample.vram_used_bytes = sample.vram_used_bytes.saturating_add(used);
        }
        if let Some(busy) = read_sysfs_u64(&dev_dir.join("gpu_busy_percent")) {
            util_sum += busy as f32;
            util_cnt += 1;
        }
        if let Ok(hwmon_entries) = std::fs::read_dir(dev_dir.join("hwmon")) {
            for hw in hwmon_entries.flatten() {
                if let Some(milli) = read_sysfs_u64(&hw.path().join("temp1_input")) {
                    let c = (milli / 1000) as u32;
                    max_temp = Some(max_temp.map_or(c, |m| m.max(c)));
                    break;
                }
            }
        }
    }

    if util_cnt > 0 {
        sample.gpu_utilization_pct = Some(util_sum / util_cnt as f32);
    }
    sample.gpu_temp_celsius = max_temp;
    sample
}
