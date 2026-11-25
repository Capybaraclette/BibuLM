use serde::{Deserialize, Serialize};

/// High-level verdict describing where a loaded model actually resides in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlacementStatus {
    /// All model weights and KV cache reside in dedicated GPU VRAM.
    GpuOnly,
    /// Model weights and/or KV cache are split across GPU VRAM and System RAM (CPU).
    HybridGpuCpu,
    /// llama.cpp reports GPU buffers, but OS telemetry indicates Windows WDDM Shared GPU Memory spillover.
    GpuSpilloverWarning,
    /// No layers/weights are resident on GPU; inference runs strictly on CPU + System RAM.
    CpuOnly,
    /// Insufficient log lines parsed yet.
    Unknown,
}

/// Buffer allocation reported by `llama-server` for a single backend (e.g., `CUDA0`, `Vulkan0`, `CPU_Mapped`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BufferAllocation {
    pub backend: String,
    pub is_gpu: bool,
    pub size_mib: f64,
}

/// Detailed per-component VRAM vs System RAM breakdown (in MiB).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MemoryComponentBreakdown {
    /// Dense / core weights (embeddings, output head, attention, norms, dense FFN, shared experts, routers).
    pub dense_vram_mib: f64,
    pub dense_ram_mib: f64,
    /// Routed MoE expert tensors (0 for dense models).
    pub moe_experts_vram_mib: f64,
    pub moe_experts_ram_mib: f64,
    /// Multi-Token Prediction (MTP / NextN) heads or speculative draft model.
    pub mtp_vram_mib: f64,
    pub mtp_ram_mib: f64,
    /// Vision / multimodal projector (`mmproj`) weights + vision scratch.
    pub vision_vram_mib: f64,
    pub vision_ram_mib: f64,
    /// KV cache (+ hybrid SSM/DeltaNet recurrent state).
    pub kv_cache_vram_mib: f64,
    pub kv_cache_ram_mib: f64,
    /// Compute / activation / output scratch buffers.
    pub compute_vram_mib: f64,
    pub compute_ram_mib: f64,
    /// Source of totals: `"verified_logs"`, `"telemetry_reconciled"`, or `"estimated_gguf"`.
    pub source: String,
}

impl MemoryComponentBreakdown {
    pub fn total_vram_mib(&self) -> f64 {
        self.dense_vram_mib
            + self.moe_experts_vram_mib
            + self.mtp_vram_mib
            + self.vision_vram_mib
            + self.kv_cache_vram_mib
            + self.compute_vram_mib
    }

    pub fn total_ram_mib(&self) -> f64 {
        self.dense_ram_mib
            + self.moe_experts_ram_mib
            + self.mtp_ram_mib
            + self.vision_ram_mib
            + self.kv_cache_ram_mib
            + self.compute_ram_mib
    }

    pub fn weights_vram_mib(&self) -> f64 {
        self.dense_vram_mib + self.moe_experts_vram_mib + self.mtp_vram_mib + self.vision_vram_mib
    }

    pub fn weights_ram_mib(&self) -> f64 {
        self.dense_ram_mib + self.moe_experts_ram_mib + self.mtp_ram_mib + self.vision_ram_mib
    }
}

/// Structured memory & execution placement summary extracted from `llama-server` startup logs,
/// native GGUF inspection, and live OS hardware telemetry.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MemoryPlacementReport {
    pub offloaded_layers: Option<u32>,
    pub total_layers: Option<u32>,
    pub cpu_moe_layers: Option<u32>,
    pub model_buffers: Vec<BufferAllocation>,
    pub vision_buffers: Vec<BufferAllocation>,
    pub mtp_buffers: Vec<BufferAllocation>,
    pub kv_buffers: Vec<BufferAllocation>,
    pub compute_buffers: Vec<BufferAllocation>,
    pub tensor_overrides_applied: Vec<String>,
    pub wddm_shared_spillover_detected: bool,
    pub shared_gpu_mem_mib: Option<f64>,
    /// Live dedicated VRAM usage of the child `llama-server` PID reported by NVML (MiB).
    pub process_vram_mib: Option<f64>,
    /// Live system RAM usage (RSS) of the child `llama-server` PID reported by OS telemetry (MiB).
    pub process_ram_mib: Option<f64>,
    /// Component-level breakdown precomputed from native GGUF + `mmproj` + `LoadConfig`.
    pub estimated_breakdown: Option<MemoryComponentBreakdown>,
}

impl MemoryPlacementReport {
    pub fn gpu_model_mib(&self) -> f64 {
        self.model_buffers
            .iter()
            .filter(|b| b.is_gpu)
            .map(|b| b.size_mib)
            .sum()
    }

    pub fn cpu_model_mib(&self) -> f64 {
        self.model_buffers
            .iter()
            .filter(|b| !b.is_gpu)
            .map(|b| b.size_mib)
            .sum()
    }

    pub fn gpu_vision_mib(&self) -> f64 {
        self.vision_buffers
            .iter()
            .filter(|b| b.is_gpu)
            .map(|b| b.size_mib)
            .sum()
    }

    pub fn cpu_vision_mib(&self) -> f64 {
        self.vision_buffers
            .iter()
            .filter(|b| !b.is_gpu)
            .map(|b| b.size_mib)
            .sum()
    }

    pub fn gpu_mtp_mib(&self) -> f64 {
        self.mtp_buffers
            .iter()
            .filter(|b| b.is_gpu)
            .map(|b| b.size_mib)
            .sum()
    }

    pub fn cpu_mtp_mib(&self) -> f64 {
        self.mtp_buffers
            .iter()
            .filter(|b| !b.is_gpu)
            .map(|b| b.size_mib)
            .sum()
    }

    pub fn gpu_kv_mib(&self) -> f64 {
        self.kv_buffers
            .iter()
            .filter(|b| b.is_gpu)
            .map(|b| b.size_mib)
            .sum()
    }

    pub fn cpu_kv_mib(&self) -> f64 {
        self.kv_buffers
            .iter()
            .filter(|b| !b.is_gpu)
            .map(|b| b.size_mib)
            .sum()
    }

    pub fn gpu_compute_mib(&self) -> f64 {
        self.compute_buffers
            .iter()
            .filter(|b| b.is_gpu)
            .map(|b| b.size_mib)
            .sum()
    }

    pub fn cpu_compute_mib(&self) -> f64 {
        self.compute_buffers
            .iter()
            .filter(|b| !b.is_gpu)
            .map(|b| b.size_mib)
            .sum()
    }

    pub fn has_verified_log_buffers(&self) -> bool {
        !self.model_buffers.is_empty()
            || !self.kv_buffers.is_empty()
            || !self.compute_buffers.is_empty()
            || !self.vision_buffers.is_empty()
            || !self.mtp_buffers.is_empty()
    }

    pub fn total_gpu_mib(&self) -> f64 {
        if self.has_verified_log_buffers() {
            self.gpu_model_mib()
                + self.gpu_vision_mib()
                + self.gpu_mtp_mib()
                + self.gpu_kv_mib()
                + self.gpu_compute_mib()
        } else {
            self.resolved_breakdown().total_vram_mib()
        }
    }

    pub fn total_cpu_mib(&self) -> f64 {
        if self.has_verified_log_buffers() {
            self.cpu_model_mib()
                + self.cpu_vision_mib()
                + self.cpu_mtp_mib()
                + self.cpu_kv_mib()
                + self.cpu_compute_mib()
        } else {
            self.resolved_breakdown().total_ram_mib()
        }
    }

    /// Reconciles verified `llama-server` log buffers, live per-PID OS telemetry, and native GGUF
    /// component metadata into a complete per-component VRAM/RAM breakdown.
    pub fn resolved_breakdown(&self) -> MemoryComponentBreakdown {
        let est = self.estimated_breakdown.as_ref();

        if self.has_verified_log_buffers() {
            let gpu_model_total = self.gpu_model_mib();
            let cpu_model_total = self.cpu_model_mib();

            // Split verified GPU/CPU model buffers across Dense, MoE Experts, and (if not separately logged) MTP.
            let has_explicit_mtp = !self.mtp_buffers.is_empty();
            let (dense_vram, moe_vram, mtp_from_model_vram) = split_model_pool(
                gpu_model_total,
                est.map(|e| e.dense_vram_mib).unwrap_or(gpu_model_total),
                est.map(|e| e.moe_experts_vram_mib).unwrap_or(0.0),
                if has_explicit_mtp {
                    0.0
                } else {
                    est.map(|e| e.mtp_vram_mib).unwrap_or(0.0)
                },
            );
            let (dense_ram, moe_ram, mtp_from_model_ram) = split_model_pool(
                cpu_model_total,
                est.map(|e| e.dense_ram_mib).unwrap_or(cpu_model_total),
                est.map(|e| e.moe_experts_ram_mib).unwrap_or(0.0),
                if has_explicit_mtp {
                    0.0
                } else {
                    est.map(|e| e.mtp_ram_mib).unwrap_or(0.0)
                },
            );

            let vision_vram = if !self.vision_buffers.is_empty() {
                self.gpu_vision_mib()
            } else {
                est.map(|e| e.vision_vram_mib).unwrap_or(0.0)
            };
            let vision_ram = if !self.vision_buffers.is_empty() {
                self.cpu_vision_mib()
            } else {
                est.map(|e| e.vision_ram_mib).unwrap_or(0.0)
            };

            let kv_vram = if !self.kv_buffers.is_empty() {
                self.gpu_kv_mib()
            } else {
                est.map(|e| e.kv_cache_vram_mib).unwrap_or(0.0)
            };
            let kv_ram = if !self.kv_buffers.is_empty() {
                self.cpu_kv_mib()
            } else {
                est.map(|e| e.kv_cache_ram_mib).unwrap_or(0.0)
            };

            let compute_vram = if !self.compute_buffers.is_empty() {
                self.gpu_compute_mib()
            } else {
                est.map(|e| e.compute_vram_mib).unwrap_or(0.0)
            };
            let compute_ram = if !self.compute_buffers.is_empty() {
                self.cpu_compute_mib()
            } else {
                est.map(|e| e.compute_ram_mib).unwrap_or(0.0)
            };

            return MemoryComponentBreakdown {
                dense_vram_mib: dense_vram,
                dense_ram_mib: dense_ram,
                moe_experts_vram_mib: moe_vram,
                moe_experts_ram_mib: moe_ram,
                mtp_vram_mib: if has_explicit_mtp {
                    self.gpu_mtp_mib()
                } else {
                    mtp_from_model_vram
                },
                mtp_ram_mib: if has_explicit_mtp {
                    self.cpu_mtp_mib()
                } else {
                    mtp_from_model_ram
                },
                vision_vram_mib: vision_vram,
                vision_ram_mib: vision_ram,
                kv_cache_vram_mib: kv_vram,
                kv_cache_ram_mib: kv_ram,
                compute_vram_mib: compute_vram,
                compute_ram_mib: compute_ram,
                source: "verified_logs".to_string(),
            };
        }

        // Fallback when custom llama-server did not print standard buffer size lines:
        // reconcile GGUF component estimate with live per-PID OS VRAM/RAM telemetry when available.
        if let Some(est_b) = est {
            let mut b = est_b.clone();
            let has_proc_vram = self.process_vram_mib.is_some_and(|v| v > 1.0);
            let has_proc_ram = self.process_ram_mib.is_some_and(|v| v > 1.0);

            if has_proc_vram || has_proc_ram {
                if let Some(proc_vram) = self.process_vram_mib.filter(|&v| v > 1.0) {
                    let est_vram = b.total_vram_mib();
                    if est_vram > 1.0 {
                        let scale = proc_vram / est_vram;
                        // Keep weights/KV/vision proportional; absorb CUDA context overhead gently
                        if scale > 0.25 && scale < 3.0 {
                            b.dense_vram_mib *= scale;
                            b.moe_experts_vram_mib *= scale;
                            b.mtp_vram_mib *= scale;
                            b.vision_vram_mib *= scale;
                            b.kv_cache_vram_mib *= scale;
                            b.compute_vram_mib *= scale;
                        } else {
                            let extra = (proc_vram - est_vram).max(0.0);
                            b.compute_vram_mib += extra;
                        }
                    } else {
                        b.dense_vram_mib = proc_vram;
                    }
                }
                if let Some(proc_ram) = self.process_ram_mib.filter(|&v| v > 1.0) {
                    let est_ram = b.total_ram_mib();
                    if est_ram > 1.0 {
                        let scale = proc_ram / est_ram;
                        if scale > 0.25 && scale < 3.0 {
                            b.dense_ram_mib *= scale;
                            b.moe_experts_ram_mib *= scale;
                            b.mtp_ram_mib *= scale;
                            b.vision_ram_mib *= scale;
                            b.kv_cache_ram_mib *= scale;
                            b.compute_ram_mib *= scale;
                        } else if proc_ram > est_ram {
                            b.compute_ram_mib += proc_ram - est_ram;
                        }
                    } else {
                        b.compute_ram_mib = proc_ram;
                    }
                }
                b.source = "telemetry_reconciled".to_string();
            } else {
                b.source = "estimated_gguf".to_string();
            }
            return b;
        }

        // Last resort: pure OS process VRAM/RAM when neither logs nor GGUF estimate are present
        let proc_vram = self.process_vram_mib.unwrap_or(0.0);
        let proc_ram = self.process_ram_mib.unwrap_or(0.0);
        if proc_vram > 0.0 || proc_ram > 0.0 {
            return MemoryComponentBreakdown {
                dense_vram_mib: proc_vram,
                dense_ram_mib: proc_ram,
                source: "telemetry_reconciled".to_string(),
                ..Default::default()
            };
        }

        MemoryComponentBreakdown::default()
    }

    /// Fraction of model weights placed on GPU VRAM (`0.0..=1.0`).
    pub fn gpu_weight_ratio(&self) -> Option<f64> {
        let b = self.resolved_breakdown();
        let gpu = b.weights_vram_mib();
        let total = gpu + b.weights_ram_mib();
        if total <= 0.0 {
            None
        } else {
            Some(gpu / total)
        }
    }

    pub fn status(&self) -> PlacementStatus {
        let partial_layers = matches!(
            (self.offloaded_layers, self.total_layers),
            (Some(off), Some(tot)) if off < tot
        );

        if self.has_verified_log_buffers() {
            let gpu_weights = self.gpu_model_mib() + self.gpu_vision_mib() + self.gpu_mtp_mib();
            let cpu_weights = self.cpu_model_mib() + self.cpu_vision_mib() + self.cpu_mtp_mib();
            let gpu_kv = self.gpu_kv_mib();
            let cpu_kv = self.cpu_kv_mib();

            if gpu_weights == 0.0 && gpu_kv == 0.0 {
                return PlacementStatus::CpuOnly;
            }
            // Note: llama.cpp often places small CPU-only metadata buffers in CPU_Mapped;
            // treat > 64 MiB of CPU model/KV weights (or explicit partial layers) as hybrid.
            // Preserve strict > 1.0 MiB check when CPU model/KV buffers were explicitly populated.
            if cpu_weights > 1.0 || cpu_kv > 1.0 || partial_layers {
                return PlacementStatus::HybridGpuCpu;
            }
            if self.wddm_shared_spillover_detected {
                return PlacementStatus::GpuSpilloverWarning;
            }
            return PlacementStatus::GpuOnly;
        }

        // Fall back to resolved breakdown (OS telemetry + GGUF estimate) so UI never stalls on "Placement Pending".
        let b = self.resolved_breakdown();
        let gpu_total = b.total_vram_mib();
        let cpu_weights = b.weights_ram_mib();
        let cpu_kv = b.kv_cache_ram_mib;
        let cpu_total = b.total_ram_mib();

        if gpu_total <= 0.0 && cpu_total <= 0.0 && self.offloaded_layers.is_none() {
            return PlacementStatus::Unknown;
        }
        if gpu_total <= 1.0 {
            return PlacementStatus::CpuOnly;
        }
        if cpu_weights > 1.0 || cpu_kv > 1.0 || partial_layers {
            return PlacementStatus::HybridGpuCpu;
        }
        if self.wddm_shared_spillover_detected {
            return PlacementStatus::GpuSpilloverWarning;
        }
        PlacementStatus::GpuOnly
    }
}

fn split_model_pool(total: f64, est_dense: f64, est_moe: f64, est_mtp: f64) -> (f64, f64, f64) {
    if total <= 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let est_sum = est_dense + est_moe + est_mtp;
    if est_sum <= 0.0 {
        return (total, 0.0, 0.0);
    }
    (
        total * (est_dense / est_sum),
        total * (est_moe / est_sum),
        total * (est_mtp / est_sum),
    )
}
