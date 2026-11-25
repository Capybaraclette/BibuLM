//! System hardware discovery (CPU/AVX, RAM, NVIDIA NVML / Windows DXGI GPUs)
//! and live VRAM + Windows WDDM shared-memory spillover telemetry.

pub mod cpu;
pub mod gpu;
pub mod monitor;

pub use cpu::CpuInfo;
pub use gpu::GpuDeviceInfo;
pub use monitor::{
    HardwareMonitor, LiveMemoryBudget, SystemHardwareProfile, TelemetrySnapshot,
    detect_wddm_spillover,
};
