use crate::cpu::{CpuInfo, detect_cpu};
use crate::gpu::{
    GpuDeviceInfo, Nvml, detect_gpus, sample_gpu_dynamic, sample_process_vram_mib,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use sysinfo::{
    CpuRefreshKind, MemoryRefreshKind, Pid, ProcessRefreshKind, ProcessesToUpdate, RefreshKind,
    System,
};

const MEMORY_AVG_WINDOW: usize = 8;
const MIB: f64 = 1024.0 * 1024.0;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LiveMemoryBudget {
    pub vram_total_mib: f64,
    pub vram_used_avg_mib: f64,
    pub vram_used_current_mib: f64,
    pub vram_available_mib: f64,
    pub ram_total_mib: f64,
    pub ram_used_avg_mib: f64,
    pub ram_available_mib: f64,
    pub sample_count: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TelemetrySnapshot {
    pub ram_used_bytes: u64,
    pub ram_total_bytes: u64,
    pub vram_used_bytes: u64,
    pub vram_total_bytes: u64,
    pub shared_gpu_mem_used_bytes: u64,
    pub cpu_utilization_pct: f32,
    pub gpu_utilization_pct: Option<f32>,
    pub gpu_temp_celsius: Option<u32>,
    pub wddm_spillover_warning: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SystemHardwareProfile {
    pub cpu: CpuInfo,
    pub ram_total_bytes: u64,
    pub gpus: Vec<GpuDeviceInfo>,
    pub total_vram_bytes: u64,
    pub recommended_backend: String,
    pub os_name: String,
}

impl SystemHardwareProfile {
    /// Generates a formatted hardware summary suitable for clipboard export / bug reports.
    pub fn format_summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("OS: {}", self.os_name));
        let mut isa = vec![self.cpu.arch.as_str()];
        if self.cpu.has_avx {
            isa.push("AVX");
        }
        if self.cpu.has_avx2 {
            isa.push("AVX2");
        }
        if self.cpu.has_avx512 {
            isa.push("AVX-512");
        }
        if self.cpu.has_fma {
            isa.push("FMA");
        }
        lines.push(format!(
            "CPU: {} ({}C/{}T) [{}]",
            self.cpu.name,
            self.cpu.physical_cores,
            self.cpu.logical_cores,
            isa.join(", ")
        ));
        lines.push(format!(
            "RAM: {:.2} GB",
            self.ram_total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        ));
        if self.gpus.is_empty() {
            lines.push("GPUs: None detected (CPU backend)".to_string());
        } else {
            for gpu in &self.gpus {
                let cuda_str = gpu
                    .cuda_version
                    .as_deref()
                    .map(|v| format!(", CUDA {v}"))
                    .unwrap_or_default();
                let drv_str = gpu
                    .driver_version
                    .as_deref()
                    .map(|v| format!(", Driver {v}"))
                    .unwrap_or_default();
                lines.push(format!(
                    "GPU #{}: {} — {:.2} GB VRAM ({}{}{})",
                    gpu.device_id,
                    gpu.name,
                    gpu.vram_total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                    gpu.backend,
                    cuda_str,
                    drv_str
                ));
            }
        }
        lines.push(format!("Recommended llama.cpp backend: {}", self.recommended_backend));
        lines.join("\n")
    }
}

/// Persistent hardware probe & live resource monitor.
pub struct HardwareMonitor {
    sys: System,
    nvml: Option<Nvml>,
    profile: SystemHardwareProfile,
    /// Rolling window of dedicated VRAM used (bytes) for smoothed pre-load headroom estimation.
    vram_used_samples: VecDeque<u64>,
    /// Rolling window of system RAM used (bytes) for smoothed pre-load headroom estimation.
    ram_used_samples: VecDeque<u64>,
    /// Last `disabled_gpu_ids` list used when recording rolling samples; changing it resets the window.
    last_disabled_gpu_ids: Vec<u32>,
    last_vram_total_bytes: u64,
    last_ram_total_bytes: u64,
}

impl HardwareMonitor {
    pub fn new() -> Self {
        let mut sys = System::new_with_specifics(
            RefreshKind::nothing()
                .with_cpu(CpuRefreshKind::everything())
                .with_memory(MemoryRefreshKind::everything()),
        );
        sys.refresh_cpu_all();
        sys.refresh_memory();

        #[cfg(any(target_os = "linux", target_os = "windows"))]
        let nvml = Nvml::init().ok();
        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        let nvml: Option<Nvml> = None;
        let cpu = detect_cpu(&sys);
        let ram_total_bytes = sys.total_memory();
        let gpus = detect_gpus(nvml.as_ref());
        let total_vram_bytes = gpus.iter().map(|g| g.vram_total_bytes).sum();
        let recommended_backend = recommend_backend(&gpus);

        let os_name = format!(
            "{} {}",
            System::name().unwrap_or_else(|| std::env::consts::OS.to_string()),
            System::os_version().unwrap_or_default()
        )
        .trim()
        .to_string();

        let profile = SystemHardwareProfile {
            cpu,
            ram_total_bytes,
            gpus,
            total_vram_bytes,
            recommended_backend,
            os_name,
        };

        Self {
            sys,
            nvml,
            profile,
            vram_used_samples: VecDeque::with_capacity(MEMORY_AVG_WINDOW),
            ram_used_samples: VecDeque::with_capacity(MEMORY_AVG_WINDOW),
            last_disabled_gpu_ids: Vec::new(),
            last_vram_total_bytes: total_vram_bytes,
            last_ram_total_bytes: ram_total_bytes,
        }
    }

    pub fn profile(&self) -> &SystemHardwareProfile {
        &self.profile
    }

    pub fn refresh_profile(&mut self) -> SystemHardwareProfile {
        self.sys.refresh_memory();
        let gpus = detect_gpus(self.nvml.as_ref());
        self.profile.ram_total_bytes = self.sys.total_memory();
        self.profile.total_vram_bytes = gpus.iter().map(|g| g.vram_total_bytes).sum();
        self.profile.recommended_backend = recommend_backend(&gpus);
        self.profile.gpus = gpus;
        self.profile.clone()
    }

    /// Captures a live telemetry snapshot (CPU%, RAM, Dedicated VRAM, WDDM Shared GPU Mem, GPU util & temp).
    pub fn sample(&mut self, disabled_gpu_ids: &[u32]) -> TelemetrySnapshot {
        self.sys.refresh_cpu_usage();
        self.sys.refresh_memory();

        let ram_total_bytes = self.sys.total_memory();
        let ram_used_bytes = self.sys.used_memory();
        let cpu_utilization_pct = self.sys.global_cpu_usage();

        let gpu = sample_gpu_dynamic(self.nvml.as_ref(), disabled_gpu_ids);
        let vram_total = if gpu.vram_total_bytes > 0 {
            gpu.vram_total_bytes
        } else {
            self.profile
                .gpus
                .iter()
                .filter(|g| !disabled_gpu_ids.contains(&g.device_id))
                .map(|g| g.vram_total_bytes)
                .sum()
        };

        let wddm_spillover_warning = detect_wddm_spillover(
            gpu.vram_used_bytes,
            vram_total,
            gpu.shared_gpu_mem_used_bytes,
        );

        let snap = TelemetrySnapshot {
            ram_used_bytes,
            ram_total_bytes,
            vram_used_bytes: gpu.vram_used_bytes,
            vram_total_bytes: vram_total,
            shared_gpu_mem_used_bytes: gpu.shared_gpu_mem_used_bytes,
            cpu_utilization_pct,
            gpu_utilization_pct: gpu.gpu_utilization_pct,
            gpu_temp_celsius: gpu.gpu_temp_celsius,
            wddm_spillover_warning,
        };

        self.record_memory_sample(
            disabled_gpu_ids,
            snap.vram_used_bytes,
            snap.vram_total_bytes,
            snap.ram_used_bytes,
            snap.ram_total_bytes,
        );

        snap
    }

    fn record_memory_sample(
        &mut self,
        disabled_gpu_ids: &[u32],
        vram_used_bytes: u64,
        vram_total_bytes: u64,
        ram_used_bytes: u64,
        ram_total_bytes: u64,
    ) {
        if self.last_disabled_gpu_ids != disabled_gpu_ids {
            self.vram_used_samples.clear();
            self.ram_used_samples.clear();
            self.last_disabled_gpu_ids = disabled_gpu_ids.to_vec();
        }
        if self.vram_used_samples.len() >= MEMORY_AVG_WINDOW {
            self.vram_used_samples.pop_front();
        }
        self.vram_used_samples.push_back(vram_used_bytes);
        if self.ram_used_samples.len() >= MEMORY_AVG_WINDOW {
            self.ram_used_samples.pop_front();
        }
        self.ram_used_samples.push_back(ram_used_bytes);
        self.last_vram_total_bytes = vram_total_bytes;
        self.last_ram_total_bytes = ram_total_bytes;
    }

    /// Clears the rolling VRAM/RAM consumption average (e.g., right after unloading or loading a model)
    /// and immediately captures a fresh telemetry snapshot to seed the new baseline.
    pub fn reset_memory_average(&mut self, disabled_gpu_ids: &[u32]) -> TelemetrySnapshot {
        self.vram_used_samples.clear();
        self.ram_used_samples.clear();
        self.last_disabled_gpu_ids = disabled_gpu_ids.to_vec();
        self.sample(disabled_gpu_ids)
    }

    /// Returns the live VRAM and RAM budget based on the rolling average of recent telemetry samples.
    /// If no samples have been recorded yet (or GPU filter changed), captures a fresh sample first.
    pub fn memory_budget(&mut self, disabled_gpu_ids: &[u32]) -> LiveMemoryBudget {
        if self.vram_used_samples.is_empty() || self.last_disabled_gpu_ids != disabled_gpu_ids {
            self.reset_memory_average(disabled_gpu_ids);
        }

        let vram_total_bytes = if self.last_vram_total_bytes > 0 {
            self.last_vram_total_bytes
        } else {
            self.profile
                .gpus
                .iter()
                .filter(|g| !disabled_gpu_ids.contains(&g.device_id))
                .map(|g| g.vram_total_bytes)
                .sum()
        };
        let ram_total_bytes = if self.last_ram_total_bytes > 0 {
            self.last_ram_total_bytes
        } else {
            self.profile.ram_total_bytes
        };

        let vram_avg_bytes = mean_u64(&self.vram_used_samples);
        let vram_cur_bytes = self
            .vram_used_samples
            .back()
            .copied()
            .map(|v| v as f64)
            .unwrap_or(0.0);
        let ram_avg_bytes = mean_u64(&self.ram_used_samples);

        let vram_total_mib = vram_total_bytes as f64 / MIB;
        let vram_used_avg_mib = (vram_avg_bytes / MIB).min(vram_total_mib);
        let vram_used_current_mib = (vram_cur_bytes / MIB).min(vram_total_mib);
        let vram_available_mib = (vram_total_mib - vram_used_avg_mib).max(0.0);

        let ram_total_mib = ram_total_bytes as f64 / MIB;
        let ram_used_avg_mib = (ram_avg_bytes / MIB).min(ram_total_mib);
        let ram_available_mib = (ram_total_mib - ram_used_avg_mib).max(0.0);

        LiveMemoryBudget {
            vram_total_mib,
            vram_used_avg_mib,
            vram_used_current_mib,
            vram_available_mib,
            ram_total_mib,
            ram_used_avg_mib,
            ram_available_mib,
            sample_count: self.vram_used_samples.len() as u32,
        }
    }

    /// Samples per-PID dedicated VRAM (via NVML) and system RAM RSS (via sysinfo) in MiB.
    pub fn sample_process_memory(
        &mut self,
        pids: &[u32],
        disabled_gpu_ids: &[u32],
    ) -> HashMap<u32, (Option<f64>, Option<f64>)> {
        let mut out = HashMap::new();
        if pids.is_empty() {
            return out;
        }
        let sys_pids: Vec<Pid> = pids.iter().map(|&p| Pid::from_u32(p)).collect();
        self.sys.refresh_processes_specifics(
            ProcessesToUpdate::Some(&sys_pids),
            true,
            ProcessRefreshKind::nothing().with_memory(),
        );
        for &pid in pids {
            let vram_mib = sample_process_vram_mib(self.nvml.as_ref(), pid, disabled_gpu_ids);
            let ram_mib = self
                .sys
                .process(Pid::from_u32(pid))
                .map(|p| p.memory() as f64 / (1024.0 * 1024.0));
            out.insert(pid, (vram_mib, ram_mib));
        }
        out
    }
}

impl Default for HardwareMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Detects silent Windows WDDM Shared GPU Memory spillover when VRAM is saturated.
pub fn detect_wddm_spillover(
    vram_used_bytes: u64,
    vram_total_bytes: u64,
    shared_used_bytes: u64,
) -> bool {
    if vram_total_bytes == 0 {
        return false;
    }
    let vram_ratio = vram_used_bytes as f64 / vram_total_bytes as f64;
    let shared_mib = shared_used_bytes as f64 / (1024.0 * 1024.0);
    (vram_ratio >= 0.92 && shared_mib >= 512.0) || (vram_ratio >= 0.85 && shared_mib >= 1024.0)
}

fn mean_u64(samples: &VecDeque<u64>) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: u128 = samples.iter().map(|&v| v as u128).sum();
    (sum as f64) / (samples.len() as f64)
}

fn recommend_backend(gpus: &[GpuDeviceInfo]) -> String {
    if let Some(nv) = gpus.iter().find(|g| g.vendor == "NVIDIA") {
        if let (Some(maj), Some(min)) = (nv.cuda_version_major, nv.cuda_version_minor) {
            return format!("cuda-{maj}.{min}");
        }
        return "cuda".to_string();
    }
    if gpus.iter().any(|g| g.vram_total_bytes >= 1024 * 1024 * 1024) {
        return "vulkan".to_string();
    }
    "cpu".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spillover_detector_triggers_only_when_vram_saturated_and_shared_elevated() {
        let gib = 1024 * 1024 * 1024u64;
        assert!(!detect_wddm_spillover(10 * gib, 16 * gib, 600 * 1024 * 1024));
        assert!(!detect_wddm_spillover(15 * gib, 16 * gib, 200 * 1024 * 1024));
        assert!(detect_wddm_spillover(15 * gib, 16 * gib, 800 * 1024 * 1024));
    }

    #[test]
    fn monitor_detects_local_hardware_without_panicking() {
        let mut mon = HardwareMonitor::new();
        assert!(mon.profile().cpu.logical_cores >= 1);
        assert!(mon.profile().ram_total_bytes > 0);
        let snap = mon.sample(&[]);
        assert!(snap.ram_total_bytes > 0);
    }

    #[test]
    fn rolling_memory_average_updates_and_resets_cleanly() {
        let mut mon = HardwareMonitor::new();
        let gib = 1024 * 1024 * 1024u64;
        mon.record_memory_sample(&[], 10 * gib, 16 * gib, 12 * gib, 32 * gib);
        mon.record_memory_sample(&[], 12 * gib, 16 * gib, 14 * gib, 32 * gib);

        let b1 = mon.memory_budget(&[]);
        assert_eq!(b1.sample_count, 2);
        assert!((b1.vram_used_avg_mib - 11.0 * 1024.0).abs() < 1.0);
        assert!((b1.vram_available_mib - 5.0 * 1024.0).abs() < 1.0);

        // Simulate model unload -> reset rolling average and record post-unload baseline
        mon.vram_used_samples.clear();
        mon.ram_used_samples.clear();
        mon.record_memory_sample(&[], 2 * gib, 16 * gib, 8 * gib, 32 * gib);

        let b2 = mon.memory_budget(&[]);
        assert_eq!(b2.sample_count, 1);
        assert!((b2.vram_used_avg_mib - 2.0 * 1024.0).abs() < 1.0);
        assert!((b2.vram_available_mib - 14.0 * 1024.0).abs() < 1.0);
    }
}
