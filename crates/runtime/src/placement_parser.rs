use bibulm_core::{BufferAllocation, MemoryPlacementReport};

/// Parses `llama-server` startup stdout/stderr logs into a structured [`MemoryPlacementReport`],
/// supporting classic `load_tensors`/`llama_kv_cache` lines, Qwen3.5/SSM `RS buffer size`,
/// `clip_model_load`/`mtmd` vision projector buffers, and modern `llama_memory_breakdown_print` /
/// `common_memory_breakdown_print` tables.
pub fn parse_placement_logs(log_text: &str) -> MemoryPlacementReport {
    let mut report = MemoryPlacementReport::default();

    for line in log_text.lines() {
        parse_placement_line(&mut report, line);
    }

    report
}

/// Incrementally updates a [`MemoryPlacementReport`] from a single stdout/stderr line.
/// Returns `true` if any field on `report` was updated.
pub fn parse_placement_line(report: &mut MemoryPlacementReport, raw_line: &str) -> bool {
    let cleaned = strip_ansi_codes(raw_line);
    let line = cleaned.trim();
    if line.is_empty() {
        return false;
    }

    let mut changed = false;

    // 1. Modern `llama_memory_breakdown_print` / `common_memory_breakdown_print` table rows:
    //    `| - CUDA0 (RTX 5070 Ti) | 15848 = 5847 + (7271 = 2789 + 1221 + 3261) + 2730 |`
    //    `| - Host                |   549 =  294 +    0 +  255 |`
    if (line.contains("memory_breakdown_print") || line.contains("| - "))
        && parse_memory_breakdown_row(report, line)
    {
        return true;
    }

    // 2. Layer offload progress:
    //    `load_tensors: offloaded 65/65 layers to GPU`
    if let Some(after) = line.split("offloaded ").nth(1)
        && let Some(fraction) = after.split_whitespace().next()
        && let Some((off, tot)) = fraction.split_once('/')
        && let (Ok(off_n), Ok(tot_n)) = (off.parse::<u32>(), tot.parse::<u32>())
    {
        if report.offloaded_layers != Some(off_n) || report.total_layers != Some(tot_n) {
            report.offloaded_layers = Some(off_n);
            report.total_layers = Some(tot_n);
            changed = true;
        }
        return changed;
    }

    // 3. Tensor override diagnostics
    if (line.contains("override") || (line.contains("tensor '") && line.contains("using CPU")))
        && report.tensor_overrides_applied.len() < 32
    {
        report.tensor_overrides_applied.push(line.to_string());
        changed = true;
    }

    let lower = line.to_ascii_lowercase();
    let is_vision_line = lower.contains("clip_")
        || lower.contains("mtmd")
        || lower.contains("mmproj")
        || lower.contains("alloc_compute_meta");
    let is_draft_or_mtp_line = lower.contains("draft") || lower.contains("mtp");

    // 4. Vision projector (`clip_model_load` / `mtmd` / `alloc_compute_meta`) buffers
    if is_vision_line {
        if let Some(alloc) = parse_buffer_line(line, " model buffer size =") {
            changed |= upsert_buffer(&mut report.vision_buffers, alloc);
            return changed;
        }
        if let Some(mut alloc) = parse_buffer_line(line, " compute buffer size =") {
            alloc.backend = format!("{}_compute", alloc.backend);
            changed |= upsert_buffer(&mut report.vision_buffers, alloc);
            return changed;
        }
        // e.g. `clip_model_load: model size: 1033.29 MiB`
        if let Some(size_mib) = parse_labelled_mib(line, "model size:")
            .or_else(|| parse_labelled_mib(line, "model size ="))
        {
            let has_backend_vision = report
                .vision_buffers
                .iter()
                .any(|b| !b.backend.ends_with("_compute") && b.backend != "VISION");
            if !has_backend_vision && size_mib > 0.0 {
                // Default to GPU unless line explicitly says CPU
                let is_gpu = !lower.contains("cpu");
                changed |= upsert_buffer(
                    &mut report.vision_buffers,
                    BufferAllocation {
                        backend: "VISION".to_string(),
                        is_gpu,
                        size_mib,
                    },
                );
            }
            return changed;
        }
    }

    // 5. Classic / detailed buffer allocation lines
    if let Some(alloc) = parse_buffer_line(line, " model buffer size =") {
        if is_draft_or_mtp_line {
            changed |= upsert_buffer(&mut report.mtp_buffers, alloc);
        } else {
            changed |= upsert_buffer(&mut report.model_buffers, alloc);
        }
    } else if let Some(alloc) = parse_buffer_line(line, " KV buffer size =") {
        changed |= upsert_buffer(&mut report.kv_buffers, alloc);
    } else if let Some(mut alloc) = parse_buffer_line(line, " RS buffer size =") {
        // Hybrid SSM / Gated DeltaNet recurrent state (e.g. Qwen3.5) — keep distinct from KV buffer
        alloc.backend = format!("{}_RS", alloc.backend);
        changed |= upsert_buffer(&mut report.kv_buffers, alloc);
    } else if let Some(alloc) = parse_buffer_line(line, " compute buffer size =") {
        changed |= upsert_buffer(&mut report.compute_buffers, alloc);
    } else if let Some(mut alloc) = parse_buffer_line(line, " output buffer size =") {
        alloc.backend = format!("{}_output", alloc.backend);
        changed |= upsert_buffer(&mut report.compute_buffers, alloc);
    }

    changed
}

fn parse_memory_breakdown_row(report: &mut MemoryPlacementReport, line: &str) -> bool {
    let parts: Vec<&str> = line.split('|').map(str::trim).collect();
    if parts.len() < 3 {
        return false;
    }

    // Find the column that starts with `- ` (device/host label)
    let Some(dev_idx) = parts.iter().position(|p| p.starts_with("- ")) else {
        return false;
    };
    let Some(expr) = parts.get(dev_idx + 1) else {
        return false;
    };

    let dev_label = parts[dev_idx].trim_start_matches("- ").trim();
    let backend = dev_label
        .split(|c: char| c.is_whitespace() || c == '(')
        .next()
        .unwrap_or("")
        .trim();
    if backend.is_empty() {
        return false;
    }

    // Extract `(self = model + context + compute)` if present, otherwise parse `self = model + context + compute`.
    let inner = if let (Some(l), Some(r)) = (expr.find('('), expr.rfind(')'))
        && r > l
    {
        &expr[l + 1..r]
    } else {
        expr
    };

    let rhs = inner.split_once('=').map(|(_, r)| r).unwrap_or(inner);
    let nums: Vec<f64> = rhs
        .split('+')
        .filter_map(|tok| tok.split_whitespace().next()?.parse::<f64>().ok())
        .collect();
    if nums.len() < 3 {
        return false;
    }

    let model_mib = nums[0];
    let context_mib = nums[1];
    let compute_mib = nums[2];
    let is_gpu = is_gpu_backend(backend);

    let mut changed = false;
    if model_mib > 0.0 {
        changed |= upsert_buffer(
            &mut report.model_buffers,
            BufferAllocation {
                backend: backend.to_string(),
                is_gpu,
                size_mib: model_mib,
            },
        );
    }
    if context_mib > 0.0 {
        changed |= upsert_buffer(
            &mut report.kv_buffers,
            BufferAllocation {
                backend: backend.to_string(),
                is_gpu,
                size_mib: context_mib,
            },
        );
    }
    if compute_mib > 0.0 {
        changed |= upsert_buffer(
            &mut report.compute_buffers,
            BufferAllocation {
                backend: backend.to_string(),
                is_gpu,
                size_mib: compute_mib,
            },
        );
    }

    changed
}

fn upsert_buffer(list: &mut Vec<BufferAllocation>, alloc: BufferAllocation) -> bool {
    if let Some(existing) = list.iter_mut().find(|b| b.backend == alloc.backend) {
        if (existing.size_mib - alloc.size_mib).abs() > 0.01 || existing.is_gpu != alloc.is_gpu {
            existing.size_mib = alloc.size_mib;
            existing.is_gpu = alloc.is_gpu;
            true
        } else {
            false
        }
    } else {
        list.push(alloc);
        true
    }
}

fn parse_buffer_line(line: &str, marker: &str) -> Option<BufferAllocation> {
    let (left, right) = line.split_once(marker)?;
    let size_token = right.split_whitespace().next()?;
    let size_mib: f64 = size_token.parse().ok()?;

    // Backend name is the last whitespace-delimited token before the marker (after optional `prefix:`).
    let backend = left.split_whitespace().last()?.trim_end_matches(':');
    if backend.is_empty() {
        return None;
    }

    Some(BufferAllocation {
        backend: backend.to_string(),
        is_gpu: is_gpu_backend(backend),
        size_mib,
    })
}

fn parse_labelled_mib(line: &str, label: &str) -> Option<f64> {
    let low = line.to_ascii_lowercase();
    let idx = low.find(label)?;
    let after = &line[idx + label.len()..];
    let mut parts = after.split_whitespace();
    let val: f64 = parts.next()?.parse().ok()?;
    let unit = parts.next().unwrap_or("").to_ascii_lowercase();
    if unit.starts_with("gib") || unit == "gb" {
        Some(val * 1024.0)
    } else if unit.starts_with("mib") || unit == "mb" || unit.is_empty() {
        Some(val)
    } else {
        None
    }
}

fn strip_ansi_codes(s: &str) -> String {
    if !s.contains('\x1b') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            if chars.peek() == Some(&'[') {
                chars.next();
                for ch in chars.by_ref() {
                    if ch.is_ascii_alphabetic() {
                        break;
                    }
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn is_gpu_backend(backend: &str) -> bool {
    let upper = backend.to_ascii_uppercase();
    if upper.starts_with("CPU") || upper == "HOST" || upper.contains("_HOST") {
        return false;
    }
    upper.starts_with("CUDA")
        || upper.starts_with("VULKAN")
        || upper.starts_with("ROCM")
        || upper.starts_with("HIP")
        || upper.starts_with("SYCL")
        || upper.starts_with("METAL")
        || upper.starts_with("MTL")
        || upper.starts_with("KOMPUTE")
        || upper.starts_with("RPC")
        || upper.starts_with("OPENCL")
        || upper.starts_with("WEBGPU")
}

#[cfg(test)]
mod tests {
    use super::*;
    use bibulm_core::PlacementStatus;

    #[test]
    fn detects_full_gpu_placement() {
        let log = r#"
load_tensors: offloaded 65/65 layers to GPU
load_tensors:        CUDA0 model buffer size = 11200.00 MiB
llama_kv_cache_init: CUDA0 KV buffer size =  2048.00 MiB
llama_context:       CUDA0 compute buffer size =   412.50 MiB
"#;
        let report = parse_placement_logs(log);
        assert_eq!(report.offloaded_layers, Some(65));
        assert_eq!(report.total_layers, Some(65));
        assert_eq!(report.gpu_model_mib(), 11200.0);
        assert_eq!(report.cpu_model_mib(), 0.0);
        assert_eq!(report.gpu_kv_mib(), 2048.0);
        assert_eq!(report.status(), PlacementStatus::GpuOnly);
    }

    #[test]
    fn detects_moe_hybrid_vram_plus_ram_placement() {
        let log = r#"
load_tensors: offloaded 49/49 layers to GPU
load_tensors:        CUDA0 model buffer size =  4850.25 MiB
load_tensors:   CPU_Mapped model buffer size = 12940.75 MiB
llama_kv_cache_init: CUDA0 KV buffer size =  1536.00 MiB
llama_context:       CUDA0 compute buffer size =   320.00 MiB
"#;
        let report = parse_placement_logs(log);
        assert_eq!(report.status(), PlacementStatus::HybridGpuCpu);
        assert_eq!(report.gpu_kv_mib(), 1536.0);
        assert_eq!(report.cpu_kv_mib(), 0.0);
        assert!(report.cpu_model_mib() > 12000.0);
        let ratio = report.gpu_weight_ratio().unwrap();
        assert!(ratio > 0.25 && ratio < 0.30);
    }

    #[test]
    fn detects_cpu_only_placement() {
        let log = r#"
load_tensors: offloaded 0/33 layers to GPU
load_tensors:   CPU_Mapped model buffer size = 4100.00 MiB
llama_kv_cache_init:   CPU KV buffer size =  512.00 MiB
"#;
        let report = parse_placement_logs(log);
        assert_eq!(report.status(), PlacementStatus::CpuOnly);
    }

    #[test]
    fn parses_modern_memory_breakdown_print_table() {
        let log = "\
llama_memory_breakdown_print: | memory breakdown [MiB] | total free self model context compute unaccounted |
llama_memory_breakdown_print: | - CUDA0 (RTX 5070 Ti) | 15848 = 1200 + (14200 = 12800 + 850 + 550) + 448 |
llama_memory_breakdown_print: | - Host                |   549 =  294 +    0 +  255 |
";
        let report = parse_placement_logs(log);
        assert_eq!(report.gpu_model_mib(), 12800.0);
        assert_eq!(report.gpu_kv_mib(), 850.0);
        assert_eq!(report.gpu_compute_mib(), 550.0);
        assert_eq!(report.cpu_model_mib(), 294.0);
        assert_eq!(report.cpu_compute_mib(), 255.0);
        assert_eq!(report.status(), PlacementStatus::HybridGpuCpu);
    }

    #[test]
    fn parses_qwen35_hybrid_rs_and_vision_without_overwriting_main_model() {
        let log = "\x1b[32mload_tensors: offloaded 65/65 layers to GPU\x1b[0m\n\
load_tensors:        CUDA0 model buffer size = 16400.00 MiB\n\
llama_kv_cache:      CUDA0 KV buffer size =   512.00 MiB (8192 cells, 16 layers, f16)\n\
llama_memory_recurrent: CUDA0 RS buffer size =  128.00 MiB\n\
llama_context:       CUDA0 compute buffer size =  320.00 MiB\n\
clip_model_load:     CUDA0 model buffer size =  820.50 MiB\n\
alloc_compute_meta:  CUDA0 compute buffer size =  110.25 MiB\n";

        let report = parse_placement_logs(log);
        assert_eq!(report.gpu_model_mib(), 16400.0);
        assert_eq!(report.gpu_kv_mib(), 640.0); // 512 KV + 128 RS
        assert_eq!(report.gpu_compute_mib(), 320.0);
        assert!((report.gpu_vision_mib() - 930.75).abs() < 0.01);
        assert_eq!(report.status(), PlacementStatus::GpuOnly);
    }
}
