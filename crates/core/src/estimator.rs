use crate::{
    GgufMetadata, GuardrailLevel, KvQuantType, LoadConfig, MemoryComponentBreakdown, MoeOffloadMode,
};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

const MIB: f64 = 1024.0 * 1024.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FitVerdict {
    FullGpu,
    MoeHybrid,
    PartialOffload,
    CpuOnly,
    TooLarge,
}

/// Live memory headroom snapshot supplied to the estimator so pre-load fit checks
/// account for VRAM/RAM already occupied by the OS, desktop apps, or other loaded models.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct LiveMemoryContext {
    pub hardware_vram_total_mib: f64,
    pub current_vram_used_mib: f64,
    pub available_vram_mib: f64,
    pub hardware_ram_total_mib: f64,
    pub current_ram_used_mib: f64,
    pub available_ram_mib: f64,
    pub vram_sample_count: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryEstimate {
    pub weights_vram_mib: f64,
    pub weights_ram_mib: f64,
    pub kv_cache_vram_mib: f64,
    pub kv_cache_ram_mib: f64,
    pub compute_vram_mib: f64,
    pub compute_ram_mib: f64,
    pub total_vram_mib: f64,
    pub total_ram_mib: f64,
    pub vram_utilization_pct: f64,
    pub ram_utilization_pct: f64,
    pub fit_verdict: FitVerdict,
    pub recommended_n_cpu_moe: Option<u32>,
    pub guardrail_warning: Option<String>,
    /// Detailed per-component VRAM & RAM breakdown (Dense, MoE Experts, MTP Heads, Vision, KV Cache, Compute).
    pub breakdown: MemoryComponentBreakdown,
    /// Total Vision (`mmproj`) footprint in MiB if enabled (shown even when Vision is disabled so user sees savings).
    pub vision_available_mib: f64,
    /// Total MTP / NextN head weights in the GGUF in MiB (shown even when MTP is disabled).
    pub mtp_available_mib: f64,
    /// Total dedicated GPU VRAM capacity (MiB) across enabled GPUs.
    #[serde(default)]
    pub hardware_vram_total_mib: f64,
    /// Rolling average VRAM currently consumed (MiB) by desktop/apps/other loaded models.
    #[serde(default)]
    pub current_vram_used_mib: f64,
    /// Real available VRAM (MiB) at model load time (`hardware_vram_total_mib - current_vram_used_mib`).
    #[serde(default)]
    pub available_vram_mib: f64,
    /// Total System RAM capacity (MiB).
    #[serde(default)]
    pub hardware_ram_total_mib: f64,
    /// Rolling average System RAM currently consumed (MiB).
    #[serde(default)]
    pub current_ram_used_mib: f64,
    /// Real available System RAM (MiB) at model load time.
    #[serde(default)]
    pub available_ram_mib: f64,
    /// Number of telemetry samples currently in the rolling VRAM average window.
    #[serde(default)]
    pub vram_sample_count: u32,
}

/// Computes accurate KV cache (+ hybrid SSM/DeltaNet recurrent state) bytes from GGUF metadata and LoadConfig.
pub fn estimate_kv_cache_bytes(meta: &GgufMetadata, config: &LoadConfig) -> f64 {
    let n_layers = meta.effective_layer_count().unwrap_or(32) as f64;
    let n_kv_heads = meta.head_count_kv.or(meta.head_count).unwrap_or(8) as f64;
    let head_dim_k = meta
        .key_length
        .or_else(|| {
            let embd = meta.embedding_length?;
            let heads = meta.head_count?;
            (heads > 0).then_some(embd / heads)
        })
        .unwrap_or(128) as f64;
    let head_dim_v = meta.value_length.map(|v| v as f64).unwrap_or(head_dim_k);

    let k_bpe = config
        .cache_type_k
        .unwrap_or(KvQuantType::F16)
        .bytes_per_element();
    let v_bpe = config
        .cache_type_v
        .unwrap_or(KvQuantType::F16)
        .bytes_per_element();

    let ctx = config.ctx_size as f64;
    let slots = if config.unified_kv_cache {
        1.0
    } else {
        config.parallel_slots.max(1) as f64
    };

    let per_attn_layer_bytes = ctx * slots * n_kv_heads * (head_dim_k * k_bpe + head_dim_v * v_bpe);

    let mut total_bytes = if let Some(interval) = meta.full_attention_interval
        && interval > 1
    {
        // Hybrid SSM / Gated DeltaNet + periodic full-attention architecture (e.g. Qwen3.5)
        let n_attn_layers = (n_layers / interval as f64).max(1.0);
        let n_ssm_layers = (n_layers - n_attn_layers).max(0.0);
        let ssm_inner = meta
            .ssm_inner_size
            .unwrap_or_else(|| meta.embedding_length.unwrap_or(4096)) as f64;
        let ssm_state = meta.ssm_state_size.unwrap_or(128) as f64;
        let ssm_conv = meta.ssm_conv_kernel.unwrap_or(4).saturating_sub(1).max(1) as f64;
        // Recurrent state matrix (S) + short conv state per SSM layer (F32), constant wrt context length
        let per_ssm_layer_bytes =
            slots * (ssm_inner * ssm_state * 4.0 + ssm_inner * 3.0 * ssm_conv * 4.0);
        n_attn_layers * per_attn_layer_bytes + n_ssm_layers * per_ssm_layer_bytes
    } else {
        n_layers * per_attn_layer_bytes
    };

    // MTP draft context adds a small KV slice for the NextN prediction layers when active.
    if config.mtp_enabled && meta.has_mtp {
        let mtp_layers = meta
            .nextn_predict_layers
            .or_else(|| {
                let c = meta.tensor_summary.mtp_layer_count();
                (c > 0).then_some(c)
            })
            .unwrap_or(1) as f64;
        total_bytes += mtp_layers * per_attn_layer_bytes;
    }

    total_bytes
}

/// Estimates Vision projector (`mmproj`) footprint (weights + vision graph scratch) from byte size in MiB.
pub fn estimate_vision_mib_from_bytes(bytes: u64) -> f64 {
    if bytes == 0 {
        return 0.0;
    }
    let weights_mib = bytes as f64 / MIB;
    let scratch_mib = (weights_mib * 0.18).clamp(48.0, 384.0);
    weights_mib + scratch_mib
}

/// Estimates Vision projector (`mmproj`) footprint (weights + vision graph scratch) in MiB.
pub fn estimate_vision_mib(mmproj_path: Option<&Path>) -> f64 {
    let Some(path) = mmproj_path.filter(|p| !p.as_os_str().is_empty()) else {
        return 0.0;
    };
    let bytes = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    estimate_vision_mib_from_bytes(bytes)
}

/// Estimates compute / scratch buffer overhead in MiB.
pub fn estimate_compute_buffer_mib(meta: &GgufMetadata, config: &LoadConfig) -> f64 {
    let embd = meta.embedding_length.unwrap_or(4096) as f64;
    let batch = config.batch_size.max(512) as f64;
    let ubatch = config.ubatch_size.max(256) as f64;
    // Conservative empirical estimate matching ggml-cuda/vulkan compute graph allocation.
    let raw_mib = (embd * (batch + ubatch) * 4.0) / MIB * 1.35 + 220.0;
    raw_mib.clamp(256.0, 2048.0)
}

/// Estimates VRAM vs RAM usage for a GGUF model under a specific `LoadConfig`,
/// including a detailed per-component breakdown (Dense, MoE Experts, MTP Heads, Vision, KV Cache, Compute).
pub fn estimate_memory_usage(
    meta: &GgufMetadata,
    config: &LoadConfig,
    avail_vram_mib: f64,
    avail_ram_mib: f64,
    guardrail: GuardrailLevel,
) -> MemoryEstimate {
    estimate_memory_usage_with_context(
        meta,
        config,
        LiveMemoryContext {
            hardware_vram_total_mib: avail_vram_mib,
            current_vram_used_mib: 0.0,
            available_vram_mib: avail_vram_mib,
            hardware_ram_total_mib: avail_ram_mib,
            current_ram_used_mib: 0.0,
            available_ram_mib: avail_ram_mib,
            vram_sample_count: 0,
        },
        guardrail,
    )
}

/// Estimates VRAM vs RAM usage against real-time available VRAM/RAM headroom
/// (accounting for OS/app/other-model consumption measured via rolling telemetry average).
pub fn estimate_memory_usage_with_context(
    meta: &GgufMetadata,
    config: &LoadConfig,
    live_ctx: LiveMemoryContext,
    guardrail: GuardrailLevel,
) -> MemoryEstimate {
    estimate_memory_usage_with_sizes(meta, config, live_ctx, guardrail, None, None)
}

/// Estimates VRAM vs RAM usage with optional pre-resolved `mmproj` and `draft` byte sizes
/// (used when estimating for a remote Bibulm Link peer whose paths are not local).
pub fn estimate_memory_usage_with_sizes(
    meta: &GgufMetadata,
    config: &LoadConfig,
    live_ctx: LiveMemoryContext,
    guardrail: GuardrailLevel,
    mmproj_size_bytes: Option<u64>,
    draft_size_bytes: Option<u64>,
) -> MemoryEstimate {
    let avail_vram_mib = live_ctx.available_vram_mib;
    let avail_ram_mib = live_ctx.available_ram_mib;
    let total_layers = meta.effective_layer_count().unwrap_or(32).max(1);
    let ngl = config.gpu_layers.unwrap_or(total_layers).min(total_layers);

    let (dense_vram_bytes, dense_ram_bytes, moe_vram_bytes, moe_ram_bytes) =
        if meta.tensor_summary.total_tensor_bytes > 0 {
            let mut dense_gpu = 0.0;
            let mut dense_cpu = 0.0;
            let mut moe_gpu = 0.0;
            let mut moe_cpu = 0.0;

            // Embeddings / output head / final norm
            if ngl >= total_layers {
                dense_gpu += meta.tensor_summary.non_layer_hot_bytes as f64;
            } else if ngl == 0 {
                dense_cpu += meta.tensor_summary.non_layer_hot_bytes as f64;
            } else {
                dense_cpu += meta.tensor_summary.non_layer_hot_bytes as f64 * 0.5;
                dense_gpu += meta.tensor_summary.non_layer_hot_bytes as f64 * 0.5;
            }

            // Per-layer dense/core weights (attention + shared experts + norms + dense FFN + routers)
            let first_gpu_layer = total_layers.saturating_sub(ngl);
            for (&layer_idx, &bytes) in &meta.tensor_summary.per_layer_hot_bytes {
                if layer_idx >= first_gpu_layer && ngl > 0 {
                    dense_gpu += bytes as f64;
                } else {
                    dense_cpu += bytes as f64;
                }
            }

            // Routed MoE expert tensors
            let moe_layers: Vec<(u32, u64)> = meta
                .tensor_summary
                .per_layer_routed_expert_bytes
                .iter()
                .map(|(&k, &v)| (k, v))
                .collect();

            match &config.moe_offload {
                MoeOffloadMode::FullGpu => {
                    for (layer_idx, bytes) in moe_layers {
                        if layer_idx >= first_gpu_layer && ngl > 0 {
                            moe_gpu += bytes as f64;
                        } else {
                            moe_cpu += bytes as f64;
                        }
                    }
                }
                MoeOffloadMode::AllExpertsCpu => {
                    for (_, bytes) in moe_layers {
                        moe_cpu += bytes as f64;
                    }
                }
                MoeOffloadMode::FirstNLayersExpertsCpu(n_cpu) => {
                    for (idx, (layer_idx, bytes)) in moe_layers.into_iter().enumerate() {
                        if (idx as u32) < *n_cpu || ngl == 0 || layer_idx < first_gpu_layer {
                            moe_cpu += bytes as f64;
                        } else {
                            moe_gpu += bytes as f64;
                        }
                    }
                }
                MoeOffloadMode::CustomOverrides(rules) => {
                    let any_cpu_moe = rules.iter().any(|r| {
                        r.buffer.to_ascii_uppercase().starts_with("CPU")
                            && (r.pattern.contains("exp") || r.pattern.contains("ffn"))
                    });
                    for (layer_idx, bytes) in moe_layers {
                        if any_cpu_moe || ngl == 0 || layer_idx < first_gpu_layer {
                            moe_cpu += bytes as f64;
                        } else {
                            moe_gpu += bytes as f64;
                        }
                    }
                }
            }

            (dense_gpu, dense_cpu, moe_gpu, moe_cpu)
        } else {
            let total_fallback = meta.total_size_bytes as f64;
            let ratio = (ngl as f64) / (total_layers as f64);
            (total_fallback * ratio, total_fallback * (1.0 - ratio), 0.0, 0.0)
        };

    // MTP heads / Speculative draft model accounting
    let mtp_gguf_mib = meta.tensor_summary.total_mtp_bytes() as f64 / MIB;
    let mtp_available_mib = if mtp_gguf_mib > 0.0 {
        mtp_gguf_mib
    } else if meta.has_mtp {
        // Fallback estimate (~1 transformer block) if GGUF declared nextn_predict_layers without distinct tensors
        let avg_layer_mib = if !meta.tensor_summary.per_layer_hot_bytes.is_empty() {
            let sum: u64 = meta.tensor_summary.per_layer_hot_bytes.values().copied().sum();
            (sum as f64 / meta.tensor_summary.per_layer_hot_bytes.len() as f64) / MIB
        } else {
            0.0
        };
        avg_layer_mib * meta.nextn_predict_layers.unwrap_or(1) as f64
    } else {
        0.0
    };

    let (mtp_vram_mib, mtp_ram_mib) = if config.mtp_enabled && mtp_available_mib > 0.0 {
        let with_scratch = mtp_available_mib + 48.0;
        if ngl > 0 {
            (with_scratch, 0.0)
        } else {
            (0.0, with_scratch)
        }
    } else if !config.mtp_enabled
        && let Some(draft_path) = config
            .draft_model_path
            .as_deref()
            .filter(|p| !p.as_os_str().is_empty())
    {
        let draft_mib = draft_size_bytes
            .unwrap_or_else(|| fs::metadata(draft_path).map(|m| m.len()).unwrap_or(0))
            as f64
            / MIB;
        if draft_mib > 0.0 {
            let draft_on_gpu = config.draft_gpu_layers.unwrap_or(99) > 0 && ngl > 0;
            if draft_on_gpu {
                (draft_mib, 0.0)
            } else {
                (0.0, draft_mib)
            }
        } else {
            (0.0, 0.0)
        }
    } else {
        (0.0, 0.0)
    };

    // Vision projector (`mmproj`) accounting
    let vision_available_mib = if config
        .mmproj_path
        .as_deref()
        .is_some_and(|p| !p.as_os_str().is_empty())
    {
        mmproj_size_bytes
            .map(estimate_vision_mib_from_bytes)
            .unwrap_or_else(|| estimate_vision_mib(config.mmproj_path.as_deref()))
    } else {
        0.0
    };
    let (vision_vram_mib, vision_ram_mib) = if config.enable_vision && vision_available_mib > 0.0 {
        if config.offload_mmproj && ngl > 0 {
            (vision_available_mib, 0.0)
        } else {
            (0.0, vision_available_mib)
        }
    } else {
        (0.0, 0.0)
    };

    let kv_bytes = estimate_kv_cache_bytes(meta, config);
    let (kv_vram_mib, kv_ram_mib) = if config.offload_kv_cache && ngl > 0 {
        let kv_gpu_ratio = if ngl >= total_layers {
            1.0
        } else {
            (ngl as f64) / (total_layers as f64)
        };
        (
            (kv_bytes * kv_gpu_ratio) / MIB,
            (kv_bytes * (1.0 - kv_gpu_ratio)) / MIB,
        )
    } else {
        (0.0, kv_bytes / MIB)
    };

    let dense_vram_mib = dense_vram_bytes / MIB;
    let dense_ram_mib = dense_ram_bytes / MIB;
    let moe_experts_vram_mib = moe_vram_bytes / MIB;
    let moe_experts_ram_mib = moe_ram_bytes / MIB;

    let weights_vram_mib = dense_vram_mib + moe_experts_vram_mib + mtp_vram_mib + vision_vram_mib;
    let weights_ram_mib = dense_ram_mib + moe_experts_ram_mib + mtp_ram_mib + vision_ram_mib;

    let compute_mib = estimate_compute_buffer_mib(meta, config);
    let (compute_vram_mib, compute_ram_mib) = if ngl > 0 {
        let cpu_side = if weights_ram_mib > 1.0 { 180.0 } else { 48.0 };
        (compute_mib, cpu_side)
    } else {
        (0.0, compute_mib)
    };

    let breakdown = MemoryComponentBreakdown {
        dense_vram_mib,
        dense_ram_mib,
        moe_experts_vram_mib,
        moe_experts_ram_mib,
        mtp_vram_mib,
        mtp_ram_mib,
        vision_vram_mib,
        vision_ram_mib,
        kv_cache_vram_mib: kv_vram_mib,
        kv_cache_ram_mib: kv_ram_mib,
        compute_vram_mib,
        compute_ram_mib,
        source: "estimated_gguf".to_string(),
    };

    let total_vram_mib = breakdown.total_vram_mib();
    let total_ram_mib = breakdown.total_ram_mib();

    let vram_utilization_pct = if avail_vram_mib > 0.0 {
        (total_vram_mib / avail_vram_mib) * 100.0
    } else {
        0.0
    };
    let ram_utilization_pct = if avail_ram_mib > 0.0 {
        (total_ram_mib / avail_ram_mib) * 100.0
    } else {
        0.0
    };

    let recommended_n_cpu_moe = if meta.is_moe && avail_vram_mib > 0.0 {
        let headroom = if live_ctx.current_vram_used_mib > 0.0 {
            512.0
        } else {
            768.0
        };
        match auto_fit_moe_with_sizes(meta, config, avail_vram_mib, headroom, mmproj_size_bytes) {
            MoeOffloadMode::FirstNLayersExpertsCpu(n) => Some(n),
            MoeOffloadMode::AllExpertsCpu => {
                Some(meta.tensor_summary.routed_expert_layer_count().max(total_layers))
            }
            _ => Some(0),
        }
    } else {
        None
    };

    let fit_verdict = if avail_vram_mib <= 0.0 || ngl == 0 {
        if total_ram_mib > avail_ram_mib && avail_ram_mib > 0.0 {
            FitVerdict::TooLarge
        } else {
            FitVerdict::CpuOnly
        }
    } else if total_vram_mib > avail_vram_mib * 1.02 || total_ram_mib > avail_ram_mib * 0.98 {
        FitVerdict::TooLarge
    } else if weights_ram_mib <= 1.0 && kv_ram_mib <= 1.0 {
        FitVerdict::FullGpu
    } else if meta.is_moe && ngl >= total_layers {
        FitVerdict::MoeHybrid
    } else {
        FitVerdict::PartialOffload
    };

    let guardrail_warning = guardrail.thresholds().and_then(|(vram_max, ram_max)| {
        if avail_vram_mib > 0.0 && total_vram_mib > avail_vram_mib * vram_max {
            if live_ctx.current_vram_used_mib > 1.0 && live_ctx.hardware_vram_total_mib > 0.0 {
                Some(format!(
                    "Estimated VRAM ({:.2} GB) exceeds {:.0}% of currently available VRAM ({:.2} GB free of {:.2} GB total; {:.2} GB in use) — risk of WDDM shared memory spillover or OOM.",
                    total_vram_mib / 1024.0,
                    vram_max * 100.0,
                    avail_vram_mib / 1024.0,
                    live_ctx.hardware_vram_total_mib / 1024.0,
                    live_ctx.current_vram_used_mib / 1024.0,
                ))
            } else {
                Some(format!(
                    "Estimated VRAM ({:.2} GB) exceeds {:.0}% of dedicated VRAM ({:.2} GB) — risk of WDDM shared memory spillover or OOM.",
                    total_vram_mib / 1024.0,
                    vram_max * 100.0,
                    avail_vram_mib / 1024.0
                ))
            }
        } else if avail_ram_mib > 0.0 && total_ram_mib > avail_ram_mib * ram_max {
            if live_ctx.current_ram_used_mib > 1.0 && live_ctx.hardware_ram_total_mib > 0.0 {
                Some(format!(
                    "Estimated System RAM ({:.2} GB) exceeds {:.0}% of available RAM ({:.2} GB free of {:.2} GB total).",
                    total_ram_mib / 1024.0,
                    ram_max * 100.0,
                    avail_ram_mib / 1024.0,
                    live_ctx.hardware_ram_total_mib / 1024.0,
                ))
            } else {
                Some(format!(
                    "Estimated System RAM ({:.2} GB) exceeds {:.0}% of total RAM ({:.2} GB).",
                    total_ram_mib / 1024.0,
                    ram_max * 100.0,
                    avail_ram_mib / 1024.0
                ))
            }
        } else {
            None
        }
    });

    MemoryEstimate {
        weights_vram_mib,
        weights_ram_mib,
        kv_cache_vram_mib: kv_vram_mib,
        kv_cache_ram_mib: kv_ram_mib,
        compute_vram_mib,
        compute_ram_mib,
        total_vram_mib,
        total_ram_mib,
        vram_utilization_pct,
        ram_utilization_pct,
        fit_verdict,
        recommended_n_cpu_moe,
        guardrail_warning,
        breakdown,
        vision_available_mib,
        mtp_available_mib,
        hardware_vram_total_mib: live_ctx.hardware_vram_total_mib,
        current_vram_used_mib: live_ctx.current_vram_used_mib,
        available_vram_mib: live_ctx.available_vram_mib,
        hardware_ram_total_mib: live_ctx.hardware_ram_total_mib,
        current_ram_used_mib: live_ctx.current_ram_used_mib,
        available_ram_mib: live_ctx.available_ram_mib,
        vram_sample_count: live_ctx.vram_sample_count,
    }
}

/// Computes the optimal `MoeOffloadMode` so attention, shared experts, embeddings, vision, MTP,
/// KV cache, compute buffer, and the maximum possible number of routed MoE expert layers fit inside
/// `avail_vram_mib - headroom_mib`.
pub fn auto_fit_moe(
    meta: &GgufMetadata,
    config: &LoadConfig,
    avail_vram_mib: f64,
    headroom_mib: f64,
) -> MoeOffloadMode {
    auto_fit_moe_with_sizes(meta, config, avail_vram_mib, headroom_mib, None)
}

/// Variant of [`auto_fit_moe`] accepting an optional pre-resolved `mmproj` byte size (for remote peers).
pub fn auto_fit_moe_with_sizes(
    meta: &GgufMetadata,
    config: &LoadConfig,
    avail_vram_mib: f64,
    headroom_mib: f64,
    mmproj_size_bytes: Option<u64>,
) -> MoeOffloadMode {
    let moe_layers: Vec<u64> = meta
        .tensor_summary
        .per_layer_routed_expert_bytes
        .values()
        .copied()
        .collect();
    if moe_layers.is_empty() {
        return MoeOffloadMode::FullGpu;
    }

    let target_vram_mib = (avail_vram_mib - headroom_mib).max(512.0);
    let hot_mib = meta.tensor_summary.total_hot_bytes() as f64 / MIB;
    let vision_mib = if config.enable_vision
        && config.offload_mmproj
        && config
            .mmproj_path
            .as_deref()
            .is_some_and(|p| !p.as_os_str().is_empty())
    {
        mmproj_size_bytes
            .map(estimate_vision_mib_from_bytes)
            .unwrap_or_else(|| estimate_vision_mib(config.mmproj_path.as_deref()))
    } else {
        0.0
    };
    let mtp_mib = if config.mtp_enabled && meta.tensor_summary.total_mtp_bytes() > 0 {
        meta.tensor_summary.total_mtp_bytes() as f64 / MIB + 48.0
    } else {
        0.0
    };
    let kv_mib = if config.offload_kv_cache {
        estimate_kv_cache_bytes(meta, config) / MIB
    } else {
        0.0
    };
    let compute_mib = estimate_compute_buffer_mib(meta, config);
    let fixed_vram_mib = hot_mib + vision_mib + mtp_mib + kv_mib + compute_mib;

    let total_experts_mib = meta.tensor_summary.total_routed_expert_bytes() as f64 / MIB;
    if fixed_vram_mib + total_experts_mib <= target_vram_mib {
        return MoeOffloadMode::FullGpu;
    }

    if fixed_vram_mib >= target_vram_mib {
        return MoeOffloadMode::AllExpertsCpu;
    }

    // Remove routed expert layers starting from layer 0 (`--n-cpu-moe N`) until remaining fits.
    let mut current_vram = fixed_vram_mib + total_experts_mib;
    let mut n_cpu: u32 = 0;
    for layer_bytes in &moe_layers {
        if current_vram <= target_vram_mib {
            break;
        }
        current_vram -= (*layer_bytes as f64) / MIB;
        n_cpu += 1;
    }

    if n_cpu == 0 {
        MoeOffloadMode::FullGpu
    } else if n_cpu as usize >= moe_layers.len() {
        MoeOffloadMode::AllExpertsCpu
    } else {
        MoeOffloadMode::FirstNLayersExpertsCpu(n_cpu)
    }
}

/// Fast fit heuristic for HuggingFace GGUF quant files before downloading.
pub fn estimate_hf_file_fit(
    size_bytes: u64,
    is_moe: bool,
    avail_vram_bytes: u64,
    avail_ram_bytes: u64,
) -> FitVerdict {
    let size = size_bytes as f64;
    let vram = avail_vram_bytes as f64;
    let ram = avail_ram_bytes as f64;
    // Account for ~15% KV + compute overhead.
    let with_overhead = size * 1.15 + 512.0 * MIB;

    if vram > 0.0 && with_overhead <= vram * 0.94 {
        FitVerdict::FullGpu
    } else if is_moe && vram > 0.0 && size <= (vram * 0.85 + ram * 0.80) {
        FitVerdict::MoeHybrid
    } else if vram > 0.0 && size <= (vram * 0.85 + ram * 0.80) {
        FitVerdict::PartialOffload
    } else if size <= ram * 0.82 {
        FitVerdict::CpuOnly
    } else {
        FitVerdict::TooLarge
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TensorMemorySummary;
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    fn sample_moe_meta() -> GgufMetadata {
        let mut per_layer_hot = BTreeMap::new();
        let mut per_layer_exp = BTreeMap::new();
        // 48 layers: ~40 MiB hot/layer (~1.92 GiB), ~320 MiB routed experts/layer (~15.36 GiB)
        for i in 0..48 {
            per_layer_hot.insert(i, 40 * 1024 * 1024);
            per_layer_exp.insert(i, 320 * 1024 * 1024);
        }
        let non_layer_hot = 600 * 1024 * 1024; // 600 MiB
        let total = non_layer_hot
            + per_layer_hot.values().sum::<u64>()
            + per_layer_exp.values().sum::<u64>();
        GgufMetadata {
            path: PathBuf::from("qwen3-30b-a3b.gguf"),
            shard_paths: vec![PathBuf::from("qwen3-30b-a3b.gguf")],
            gguf_version: 3,
            total_size_bytes: total,
            architecture: Some("qwen3moe".into()),
            name: Some("Qwen3-30B-A3B".into()),
            size_label: Some("30B-A3B".into()),
            file_type: Some(15),
            quantization: Some("Q4_K_M".into()),
            parameter_count: Some(30_000_000_000),
            block_count: Some(48),
            effective_block_count: Some(48),
            context_length: Some(131072),
            embedding_length: Some(2048),
            head_count: Some(32),
            head_count_kv: Some(4),
            key_length: Some(128),
            value_length: Some(128),
            rope_freq_base: Some(1_000_000.0),
            expert_count: Some(128),
            expert_used_count: Some(8),
            is_moe: true,
            full_attention_interval: None,
            ssm_state_size: None,
            ssm_conv_kernel: None,
            ssm_inner_size: None,
            ssm_group_count: None,
            ssm_time_step_rank: None,
            nextn_predict_layers: None,
            has_mtp: false,
            chat_template: None,
            supports_reasoning: true,
            supports_reasoning_effort: false,
            supports_enable_thinking: true,
            supports_tools: true,
            is_embedding: false,
            tokenizer_model: None,
            tokenizer_pre: None,
            vocab_size: None,
            tensor_count: 500,
            tensor_summary: TensorMemorySummary {
                non_layer_hot_bytes: non_layer_hot,
                per_layer_hot_bytes: per_layer_hot,
                per_layer_routed_expert_bytes: per_layer_exp,
                per_layer_mtp_bytes: BTreeMap::new(),
                non_layer_mtp_bytes: 0,
                total_tensor_bytes: total,
                total_parameters: 30_000_000_000,
            },
        }
    }

    #[test]
    fn estimator_tracks_dense_mtp_vision_and_kv_separately() {
        let mut meta = sample_moe_meta();
        meta.has_mtp = true;
        meta.nextn_predict_layers = Some(1);
        meta.tensor_summary
            .per_layer_mtp_bytes
            .insert(48, 512 * 1024 * 1024);

        // Create a temporary dummy mmproj file (~200 MiB? No, 10 MiB is enough to test non-zero size)
        let mmproj = std::env::temp_dir().join(format!("test_mmproj_{}.gguf", uuid::Uuid::new_v4()));
        fs::write(&mmproj, vec![0u8; 4 * 1024 * 1024]).unwrap();

        let mut cfg = LoadConfig::new("qwen3-30b-a3b.gguf");
        cfg.mmproj_path = Some(mmproj.clone());
        cfg.enable_vision = true;
        cfg.offload_mmproj = true;
        cfg.mtp_enabled = true;

        let est_on = estimate_memory_usage(&meta, &cfg, 24000.0, 64000.0, GuardrailLevel::Off);
        assert!(est_on.breakdown.dense_vram_mib > 2000.0);
        assert!(est_on.breakdown.moe_experts_vram_mib > 14000.0);
        assert!(est_on.breakdown.mtp_vram_mib > 500.0);
        assert!(est_on.breakdown.vision_vram_mib > 4.0);
        assert_eq!(est_on.breakdown.vision_ram_mib, 0.0);

        // Disable vision and MTP -> both drop to 0 MiB while available sizes remain reported
        cfg.enable_vision = false;
        cfg.mtp_enabled = false;
        let est_off = estimate_memory_usage(&meta, &cfg, 24000.0, 64000.0, GuardrailLevel::Off);
        assert_eq!(est_off.breakdown.vision_vram_mib, 0.0);
        assert_eq!(est_off.breakdown.vision_ram_mib, 0.0);
        assert_eq!(est_off.breakdown.mtp_vram_mib, 0.0);
        assert!(est_off.vision_available_mib > 4.0);
        assert!(est_off.mtp_available_mib > 500.0);

        let _ = fs::remove_file(&mmproj);
    }

    #[test]
    fn auto_fit_moe_calculates_exact_n_cpu_moe_for_16gb_vram() {
        let meta = sample_moe_meta();
        let mut cfg = LoadConfig::new("qwen3-30b-a3b.gguf");
        cfg.ctx_size = 32768;
        cfg.cache_type_k = Some(KvQuantType::Q8_0);
        cfg.cache_type_v = Some(KvQuantType::Q8_0);

        let avail_vram_mib = 15.89 * 1024.0; // RTX 5070 Ti
        let mode = auto_fit_moe(&meta, &cfg, avail_vram_mib, 768.0);
        match mode {
            MoeOffloadMode::FirstNLayersExpertsCpu(n) => {
                assert!(n > 0 && n < 48, "expected partial MoE offload, got {n}");
                cfg.moe_offload = MoeOffloadMode::FirstNLayersExpertsCpu(n);
                let est = estimate_memory_usage(
                    &meta,
                    &cfg,
                    avail_vram_mib,
                    31.16 * 1024.0,
                    GuardrailLevel::Relaxed,
                );
                assert!(est.total_vram_mib <= avail_vram_mib);
                assert_eq!(est.fit_verdict, FitVerdict::MoeHybrid);
                assert!(est.weights_ram_mib > 0.0);
            }
            other => panic!("expected FirstNLayersExpertsCpu, got {other:?}"),
        }
    }

    #[test]
    fn looped_transformer_uses_effective_layers_for_ngl_and_kv() {
        let mut meta = sample_moe_meta();
        meta.architecture = Some("inkling2_loop".into());
        meta.block_count = Some(24);
        meta.effective_block_count = Some(48);
        meta.tensor_summary.per_layer_hot_bytes.retain(|&k, _| k < 24);
        meta.tensor_summary
            .per_layer_routed_expert_bytes
            .retain(|&k, _| k < 24);

        let mut cfg_full = LoadConfig::new("inkling2.gguf");
        cfg_full.gpu_layers = Some(999);
        let est_full = estimate_memory_usage(&meta, &cfg_full, 24000.0, 32000.0, GuardrailLevel::Off);
        assert_eq!(est_full.weights_ram_mib, 0.0);
        assert_eq!(est_full.kv_cache_ram_mib, 0.0);
        assert!(est_full.weights_vram_mib > 0.0);

        // Passing -ngl 25 on a 48-effective-layer looped model leaves physical blocks on CPU
        let mut cfg_partial = LoadConfig::new("inkling2.gguf");
        cfg_partial.gpu_layers = Some(25);
        let est_partial =
            estimate_memory_usage(&meta, &cfg_partial, 24000.0, 32000.0, GuardrailLevel::Off);
        assert!(est_partial.weights_ram_mib > 0.0);
        assert!(est_partial.kv_cache_ram_mib > 0.0);
    }
}
