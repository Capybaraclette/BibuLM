use bibulm_core::{
    CliFlagValue, FlashAttentionMode, LoadConfig, MoeOffloadMode, ReasoningEffort, ReasoningFormat,
};
use std::path::Path;

/// Deterministically builds `llama-server` CLI arguments from a [`LoadConfig`].
///
/// The returned argument list is suitable both for spawning `llama-server.exe` directly
/// and for rendering a live CLI preview in the UI.
pub fn build_llama_server_args(config: &LoadConfig, host: &str, port: u16) -> Vec<String> {
    let mut args = Vec::new();

    args.push("-m".to_string());
    args.push(config.model_path.to_string_lossy().into_owned());

    if let Some(alias) = &config.model_alias
        && !alias.trim().is_empty()
    {
        args.push("--alias".to_string());
        args.push(alias.clone());
    }

    args.push("--host".to_string());
    args.push(host.to_string());
    args.push("--port".to_string());
    args.push(port.to_string());

    if config.enable_vision
        && let Some(mmproj) = &config.mmproj_path
        && !mmproj.as_os_str().is_empty()
    {
        args.push("--mmproj".to_string());
        args.push(mmproj.to_string_lossy().into_owned());
        if !config.offload_mmproj {
            args.push("--no-mmproj-offload".to_string());
        }
    }

    // Context & batching
    args.push("-c".to_string());
    args.push(config.ctx_size.to_string());
    args.push("-b".to_string());
    args.push(config.batch_size.to_string());
    args.push("-ub".to_string());
    args.push(config.ubatch_size.to_string());
    args.push("-np".to_string());
    args.push(config.parallel_slots.max(1).to_string());

    if config.unified_kv_cache {
        args.push("-kvu".to_string());
    }
    if let Some(cache_ram) = config.cache_ram_mib {
        args.push("--cache-ram".to_string());
        args.push(cache_ram.to_string());
    }
    if let Some(ctx_shift) = config.context_shift {
        args.push(if ctx_shift {
            "--context-shift".to_string()
        } else {
            "--no-context-shift".to_string()
        });
    }

    // GPU offload & MoE selective placement
    if let Some(ngl) = config.gpu_layers {
        args.push("-ngl".to_string());
        args.push(ngl.to_string());
    }

    let has_manual_moe_override = !matches!(config.moe_offload, MoeOffloadMode::FullGpu);
    match &config.moe_offload {
        MoeOffloadMode::FullGpu => {}
        MoeOffloadMode::AllExpertsCpu => {
            args.push("--cpu-moe".to_string());
        }
        MoeOffloadMode::FirstNLayersExpertsCpu(n) => {
            args.push("--n-cpu-moe".to_string());
            args.push(n.to_string());
        }
        MoeOffloadMode::CustomOverrides(rules) => {
            for rule in rules {
                if !rule.pattern.trim().is_empty() && !rule.buffer.trim().is_empty() {
                    args.push("-ot".to_string());
                    args.push(format!("{}={}", rule.pattern.trim(), rule.buffer.trim()));
                }
            }
        }
    }

    // Avoid llama.cpp `--fit` overriding explicit MoE/tensor rules unless the user explicitly requested it.
    match (config.auto_fit, has_manual_moe_override) {
        (Some(true), _) => {
            args.push("--fit".to_string());
            args.push("on".to_string());
            if let Some(fit_ctx) = config.fit_ctx {
                args.push("--fit-ctx".to_string());
                args.push(fit_ctx.to_string());
            }
            if let Some(fit_target) = config.fit_target_mib {
                args.push("--fit-target".to_string());
                args.push(fit_target.to_string());
            }
        }
        (Some(false), _) | (None, true) => {
            args.push("--fit".to_string());
            args.push("off".to_string());
        }
        (None, false) => {}
    }

    if let Some(threads) = config.threads {
        args.push("-t".to_string());
        args.push(threads.to_string());
    }
    if let Some(threads_batch) = config.threads_batch {
        args.push("-tb".to_string());
        args.push(threads_batch.to_string());
    }
    if !config.devices.is_empty() {
        args.push("-dev".to_string());
        args.push(config.devices.join(","));
    }
    if let Some(split_mode) = config.split_mode {
        args.push("--split-mode".to_string());
        args.push(split_mode.as_cli_str().to_string());
    }
    if let Some(main_gpu) = config.main_gpu {
        args.push("-mg".to_string());
        args.push(main_gpu.to_string());
    }
    if let Some(ts) = &config.tensor_split
        && !ts.trim().is_empty()
    {
        args.push("-ts".to_string());
        args.push(ts.trim().to_string());
    }

    // KV Cache & Flash Attention
    if !config.offload_kv_cache {
        args.push("--no-kv-offload".to_string());
    }
    if config.flash_attention != FlashAttentionMode::Auto {
        args.push("-fa".to_string());
        args.push(config.flash_attention.as_cli_str().to_string());
    }
    if let Some(ctk) = config.cache_type_k {
        args.push("-ctk".to_string());
        args.push(ctk.as_cli_str().to_string());
    }
    if let Some(ctv) = config.cache_type_v {
        args.push("-ctv".to_string());
        args.push(ctv.as_cli_str().to_string());
    }

    // Memory / IO (skip legacy flags when explicit `--load-mode` is set in `cli_flags`)
    let has_explicit_load_mode = config.cli_flags.contains_key("--load-mode");
    if !has_explicit_load_mode {
        if !config.use_mmap {
            args.push("--no-mmap".to_string());
        }
        if config.use_mlock {
            args.push("--mlock".to_string());
        }
        if config.direct_io {
            args.push("--direct-io".to_string());
        }
    }

    // RoPE & YaRN
    if let Some(rope_scaling) = config.rope_scaling {
        args.push("--rope-scaling".to_string());
        args.push(rope_scaling.as_cli_str().to_string());
    }
    if let Some(base) = config.rope_freq_base {
        args.push("--rope-freq-base".to_string());
        args.push(base.to_string());
    }
    if let Some(scale) = config.rope_freq_scale {
        args.push("--rope-freq-scale".to_string());
        args.push(scale.to_string());
    }
    if let Some(v) = config.yarn_ext_factor {
        args.push("--yarn-ext-factor".to_string());
        args.push(v.to_string());
    }
    if let Some(v) = config.yarn_attn_factor {
        args.push("--yarn-attn-factor".to_string());
        args.push(v.to_string());
    }
    if let Some(v) = config.yarn_beta_fast {
        args.push("--yarn-beta-fast".to_string());
        args.push(v.to_string());
    }
    if let Some(v) = config.yarn_beta_slow {
        args.push("--yarn-beta-slow".to_string());
        args.push(v.to_string());
    }

    // Jinja, templates & reasoning controls
    if config.jinja {
        args.push("--jinja".to_string());
    }
    if let Some(template_file) = &config.chat_template_file
        && !template_file.as_os_str().is_empty()
    {
        args.push("--chat-template-file".to_string());
        args.push(template_file.to_string_lossy().into_owned());
    } else if let Some(template) = &config.chat_template_override
        && !template.trim().is_empty()
    {
        args.push("--chat-template".to_string());
        args.push(template.clone());
    }

    if config.reasoning.format != ReasoningFormat::Auto {
        args.push("--reasoning-format".to_string());
        args.push(config.reasoning.format.as_cli_str().to_string());
    }
    if let Some(budget) = config.reasoning.budget {
        args.push("--reasoning-budget".to_string());
        args.push(budget.to_string());
    }
    if let Some(msg) = &config.reasoning.budget_message
        && !msg.is_empty()
    {
        args.push("--reasoning-budget-message".to_string());
        args.push(msg.clone());
    }

    // Build `--chat-template-kwargs` JSON when reasoning effort or enable_thinking is set.
    let mut template_kwargs = Vec::new();
    if config.reasoning.effort != ReasoningEffort::Default {
        template_kwargs.push(format!(
            "\"reasoning_effort\":\"{}\"",
            config.reasoning.effort.as_cli_str()
        ));
    }
    if let Some(enable_thinking) = config.reasoning.enable_thinking {
        template_kwargs.push(format!("\"enable_thinking\":{enable_thinking}"));
    }
    if !template_kwargs.is_empty() {
        args.push("--chat-template-kwargs".to_string());
        args.push(format!("{{{}}}", template_kwargs.join(",")));
    }

    // Multi-Token Prediction (MTP) & Speculative Decoding
    if config.mtp_enabled {
        let spec_type = config
            .mtp_spec_type
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or("draft-mtp");
        args.push("--spec-type".to_string());
        args.push(spec_type.to_string());
        if let Some(max) = config.draft_max {
            args.push("--spec-draft-n-max".to_string());
            args.push(max.to_string());
        }
        if let Some(min) = config.draft_min {
            args.push("--spec-draft-n-min".to_string());
            args.push(min.to_string());
        }
        if let Some(p_min) = config.draft_p_min {
            args.push("--spec-draft-p-min".to_string());
            args.push(p_min.to_string());
        }
    } else if let Some(draft) = &config.draft_model_path
        && !draft.as_os_str().is_empty()
    {
        args.push("-md".to_string());
        args.push(draft.to_string_lossy().into_owned());
        if let Some(ngld) = config.draft_gpu_layers {
            args.push("-ngld".to_string());
            args.push(ngld.to_string());
        }
        if let Some(max) = config.draft_max {
            args.push("--spec-draft-n-max".to_string());
            args.push(max.to_string());
        }
        if let Some(min) = config.draft_min {
            args.push("--spec-draft-n-min".to_string());
            args.push(min.to_string());
        }
        if let Some(p_min) = config.draft_p_min {
            args.push("--spec-draft-p-min".to_string());
            args.push(p_min.to_string());
        }
    }

    // Misc
    if let Some(seed) = config.seed {
        args.push("-s".to_string());
        args.push(seed.to_string());
    }
    if config.embeddings {
        args.push("--embeddings".to_string());
    }

    // Always enable metrics/slots telemetry in llama-server for diagnostics.
    args.push("--metrics".to_string());

    // Structured llama.cpp flag overrides (full catalog coverage).
    for (flag, value) in &config.cli_flags {
        let key = flag.trim();
        if key.is_empty() {
            continue;
        }
        match value {
            CliFlagValue::Flag(true) => {
                args.push(key.to_string());
            }
            CliFlagValue::Flag(false) => {}
            CliFlagValue::Value(v) => {
                let trimmed = v.trim();
                if !trimmed.is_empty() {
                    args.push(key.to_string());
                    args.push(trimmed.to_string());
                }
            }
            CliFlagValue::MultiArg(values) => {
                let cleaned: Vec<String> = values
                    .iter()
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                if !cleaned.is_empty() {
                    args.push(key.to_string());
                    args.extend(cleaned);
                }
            }
            CliFlagValue::Repeatable(values) => {
                for raw in values {
                    let trimmed = raw.trim();
                    if !trimmed.is_empty() {
                        args.push(key.to_string());
                        args.push(trimmed.to_string());
                    }
                }
            }
        }
    }

    // Raw escape hatch (tokenized safely if user put multiple args in one string).
    for raw in &config.extra_args {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            args.extend(split_extra_args(trimmed));
        }
    }

    args
}

/// Safely splits a raw extra-args string respecting single and double quotes.
pub fn split_extra_args(input: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut in_single = false;
    let mut in_double = false;
    let mut chars = input.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            '\'' if !in_double => in_single = !in_single,
            '"' if !in_single => in_double = !in_double,
            '\\' if in_double => {
                if let Some(&next) = chars.peek() {
                    if next == '"' || next == '\\' {
                        current.push(chars.next().unwrap());
                    } else {
                        current.push(c);
                    }
                } else {
                    current.push(c);
                }
            }
            c if c.is_whitespace() && !in_single && !in_double => {
                if !current.is_empty() {
                    out.push(std::mem::take(&mut current));
                }
            }
            _ => current.push(c),
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

/// Formats a shell-copyable CLI preview string for display in the Load Config modal.
pub fn format_cli_preview(server_exe: Option<&Path>, args: &[String]) -> String {
    let exe = server_exe
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| "llama-server.exe".to_string());
    let mut parts = Vec::with_capacity(args.len() + 1);
    parts.push(quote_arg(&exe));
    for a in args {
        parts.push(quote_arg(a));
    }
    parts.join(" ")
}

fn quote_arg(a: &str) -> String {
    if a.is_empty()
        || a.chars()
            .any(|c| c.is_whitespace() || c == '"' || c == '{' || c == '}' || c == '\\')
    {
        format!("\"{}\"", a.replace('"', "\\\""))
    } else {
        a.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bibulm_core::{KvQuantType, TensorOverride};

    #[test]
    fn builds_moe_hybrid_and_reasoning_flags() {
        let mut cfg = LoadConfig::new("C:/models/qwen3-30b-a3b-q4_k_m.gguf");
        cfg.ctx_size = 32768;
        cfg.moe_offload = MoeOffloadMode::FirstNLayersExpertsCpu(20);
        cfg.flash_attention = FlashAttentionMode::On;
        cfg.cache_type_k = Some(KvQuantType::Q8_0);
        cfg.cache_type_v = Some(KvQuantType::Q8_0);
        cfg.reasoning.effort = ReasoningEffort::High;
        cfg.reasoning.budget = Some(8192);
        cfg.reasoning.format = ReasoningFormat::Deepseek;

        let args = build_llama_server_args(&cfg, "127.0.0.1", 51001);

        assert!(args.windows(2).any(|w| w == ["--n-cpu-moe", "20"]));
        assert!(args.windows(2).any(|w| w == ["--fit", "off"]));
        assert!(args.windows(2).any(|w| w == ["-fa", "on"]));
        assert!(args.windows(2).any(|w| w == ["-ctk", "q8_0"]));
        assert!(args.windows(2).any(|w| w == ["-ctv", "q8_0"]));
        assert!(args.windows(2).any(|w| w == ["--reasoning-budget", "8192"]));
        assert!(
            args.windows(2)
                .any(|w| w == ["--reasoning-format", "deepseek"])
        );
        assert!(
            args.windows(2)
                .any(|w| w == ["--chat-template-kwargs", "{\"reasoning_effort\":\"high\"}"])
        );
    }

    #[test]
    fn builds_custom_tensor_override_flags() {
        let mut cfg = LoadConfig::new("C:/models/deepseek.gguf");
        cfg.moe_offload = MoeOffloadMode::CustomOverrides(vec![TensorOverride {
            pattern: r"blk\.\d+\.ffn_.*_exps\.weight".to_string(),
            buffer: "CPU".to_string(),
        }]);

        let args = build_llama_server_args(&cfg, "127.0.0.1", 51002);
        assert!(
            args.windows(2)
                .any(|w| w == ["-ot", r"blk\.\d+\.ffn_.*_exps\.weight=CPU"])
        );
        assert!(args.windows(2).any(|w| w == ["--fit", "off"]));
    }

    #[test]
    fn splits_quoted_extra_args() {
        let parsed = split_extra_args("--chat-template-kwargs '{\"foo\":\"bar\"}' --no-warmup");
        assert_eq!(
            parsed,
            vec!["--chat-template-kwargs", "{\"foo\":\"bar\"}", "--no-warmup"]
        );
    }

    #[test]
    fn builds_vision_toggle_chat_template_file_and_mtp_flags() {
        let mut cfg = LoadConfig::new("C:/models/qwen3.5-27b-mtp.gguf");
        cfg.mmproj_path = Some("C:/models/mmproj-qwen3.5-f16.gguf".into());
        cfg.enable_vision = false;
        cfg.chat_template_file = Some("C:/templates/qwen35.jinja".into());
        cfg.chat_template_override = Some("chatml".into());
        cfg.mtp_enabled = true;
        cfg.draft_max = Some(3);

        let args = build_llama_server_args(&cfg, "127.0.0.1", 51003);

        assert!(!args.iter().any(|a| a == "--mmproj"));
        assert!(
            args.windows(2)
                .any(|w| w == ["--chat-template-file", "C:/templates/qwen35.jinja"])
        );
        assert!(!args.iter().any(|a| a == "--chat-template"));
        assert!(args.windows(2).any(|w| w == ["--spec-type", "draft-mtp"]));
        assert!(args.windows(2).any(|w| w == ["--spec-draft-n-max", "3"]));

        cfg.enable_vision = true;
        cfg.offload_mmproj = false;
        let args_vis = build_llama_server_args(&cfg, "127.0.0.1", 51003);
        assert!(
            args_vis
                .windows(2)
                .any(|w| w == ["--mmproj", "C:/models/mmproj-qwen3.5-f16.gguf"])
        );
        assert!(args_vis.iter().any(|a| a == "--no-mmproj-offload"));
    }

    #[test]
    fn builds_structured_cli_flags_and_modern_draft_flags() {
        let mut cfg = LoadConfig::new("C:/models/main.gguf");
        cfg.draft_model_path = Some("C:/models/draft.gguf".into());
        cfg.draft_gpu_layers = Some(24);
        cfg.draft_max = Some(8);
        cfg.draft_min = Some(2);
        cfg.draft_p_min = Some(0.5);
        cfg.cache_type_k = Some(KvQuantType::F8E4m3);
        cfg.split_mode = Some(bibulm_core::SplitMode::Tensor);
        cfg.use_mmap = false;
        cfg.cli_flags
            .insert("--load-mode".to_string(), CliFlagValue::Value("dio".to_string()));
        cfg.cli_flags
            .insert("--swa-full".to_string(), CliFlagValue::Flag(true));
        cfg.cli_flags
            .insert("--no-warmup".to_string(), CliFlagValue::Flag(true));
        cfg.cli_flags.insert(
            "--control-vector-layer-range".to_string(),
            CliFlagValue::MultiArg(vec!["5".to_string(), "18".to_string()]),
        );
        cfg.cli_flags.insert(
            "--logit-bias".to_string(),
            CliFlagValue::Repeatable(vec!["15043+1".to_string(), "1234-2".to_string()]),
        );

        let args = build_llama_server_args(&cfg, "127.0.0.1", 51004);

        assert!(args.windows(2).any(|w| w == ["-ngld", "24"]));
        assert!(args.windows(2).any(|w| w == ["--spec-draft-n-max", "8"]));
        assert!(args.windows(2).any(|w| w == ["--spec-draft-n-min", "2"]));
        assert!(args.windows(2).any(|w| w == ["--spec-draft-p-min", "0.5"]));
        assert!(!args.iter().any(|a| a == "--draft-max" || a == "--draft-min"));
        assert!(args.windows(2).any(|w| w == ["-ctk", "f8_e4m3"]));
        assert!(args.windows(2).any(|w| w == ["--split-mode", "tensor"]));
        assert!(args.windows(2).any(|w| w == ["--load-mode", "dio"]));
        assert!(!args.iter().any(|a| a == "--no-mmap"));
        assert!(args.iter().any(|a| a == "--swa-full"));
        assert!(args.iter().any(|a| a == "--no-warmup"));
        assert!(
            args.windows(3)
                .any(|w| w == ["--control-vector-layer-range", "5", "18"])
        );
        assert!(args.windows(2).any(|w| w == ["--logit-bias", "15043+1"]));
        assert!(args.windows(2).any(|w| w == ["--logit-bias", "1234-2"]));
    }
}
