use bibulm_core::{
    CliFlagValue, FlashAttentionMode, LoadConfig, MoeOffloadMode, ReasoningEffort, ReasoningFormat,
};
use std::path::Path;

/// Deterministically builds `llama-server` CLI arguments from a [`LoadConfig`].
///
/// The returned argument list is suitable both for spawning `llama-server.exe` directly
/// and for rendering a live CLI preview in the UI.
pub fn build_llama_server_args(config: &LoadConfig, host: &str, port: u16) -> Vec<String> {
    build_llama_server_args_inner(config, Some((host, port)))
}

/// Builds the editable `llama-server` CLI arguments for a [`LoadConfig`] without managed `--host`/`--port`.
pub fn build_editable_llama_args(config: &LoadConfig) -> Vec<String> {
    build_llama_server_args_inner(config, None)
}

fn build_custom_args(raw: &str, endpoint: Option<(&str, u16)>) -> Vec<String> {
    let parsed = split_extra_args(raw);
    let mut args = Vec::with_capacity(parsed.len() + 4);
    let mut iter = parsed.into_iter().peekable();

    // If the user pasted a full command preview starting with `llama-server(.exe)`, drop the exe token.
    if let Some(first) = iter.peek()
        && !first.starts_with('-')
        && first
            .rsplit(['/', '\\'])
            .next()
            .is_some_and(|n| {
                n.eq_ignore_ascii_case("llama-server") || n.eq_ignore_ascii_case("llama-server.exe")
            })
    {
        iter.next();
    }

    while let Some(tok) = iter.next() {
        if endpoint.is_some() && (tok == "--host" || tok == "--port") {
            let _ = iter.next();
            continue;
        }
        if endpoint.is_some() && (tok.starts_with("--host=") || tok.starts_with("--port=")) {
            continue;
        }
        args.push(tok);
    }

    if let Some((host, port)) = endpoint {
        args.push("--host".to_string());
        args.push(host.to_string());
        args.push("--port".to_string());
        args.push(port.to_string());
    }
    args
}

fn build_llama_server_args_inner(config: &LoadConfig, endpoint: Option<(&str, u16)>) -> Vec<String> {
    if let Some(raw) = &config.custom_cli_args {
        return build_custom_args(raw, endpoint);
    }

    let mut args = Vec::new();

    args.push("-m".to_string());
    args.push(config.model_path.to_string_lossy().into_owned());

    if let Some(alias) = &config.model_alias
        && !alias.trim().is_empty()
    {
        args.push("--alias".to_string());
        args.push(alias.clone());
    }

    if let Some((host, port)) = endpoint {
        args.push("--host".to_string());
        args.push(host.to_string());
        args.push("--port".to_string());
        args.push(port.to_string());
    }

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
    let spec_type_trimmed = config
        .mtp_spec_type
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let has_draft_path = config
        .draft_model_path
        .as_deref()
        .is_some_and(|p| !p.as_os_str().is_empty());
    let emit_spec_type = if config.mtp_enabled {
        Some(spec_type_trimmed.unwrap_or("draft-mtp"))
    } else if has_draft_path
        && spec_type_trimmed.is_some_and(bibulm_core::spec_type_uses_external_draft)
    {
        spec_type_trimmed
    } else {
        None
    };
    let emit_external_draft = config.uses_external_draft();

    if let Some(spec_type) = emit_spec_type {
        args.push("--spec-type".to_string());
        args.push(spec_type.to_string());
    }
    if emit_external_draft
        && let Some(draft) = &config.draft_model_path
    {
        args.push("-md".to_string());
        args.push(draft.to_string_lossy().into_owned());
        if let Some(ngld) = config.draft_gpu_layers {
            args.push("-ngld".to_string());
            args.push(ngld.to_string());
        }
    }
    if emit_spec_type.is_some() || emit_external_draft {
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

/// Formats `args` as a shell-escaped command-line argument string (without the executable).
pub fn format_cli_args(args: &[String]) -> String {
    args.iter()
        .map(|a| quote_arg(a))
        .collect::<Vec<_>>()
        .join(" ")
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
        // `split_extra_args` treats single-quoted segments literally, preserving UNC `\\` or trailing `\`.
        if (a.contains("\\\\") || a.ends_with('\\')) && !a.contains('\'') {
            format!("'{a}'")
        } else {
            format!("\"{}\"", a.replace('"', "\\\""))
        }
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

    #[test]
    fn custom_cli_args_override_generated_flags_and_manage_endpoint() {
        let mut cfg = LoadConfig::new(r"\\nas\models\my model.gguf");
        cfg.ctx_size = 32768;

        let editable = format_cli_args(&build_editable_llama_args(&cfg));
        assert!(!editable.contains("--host"));
        assert!(!editable.contains("--port"));
        assert!(editable.contains("-ctk q8_0"));
        assert!(editable.contains("-ctv q8_0"));
        assert_eq!(
            split_extra_args(&editable),
            build_editable_llama_args(&cfg),
            "editable args should round-trip losslessly"
        );

        cfg.custom_cli_args = Some(
            r#""C:\llama\llama-server.exe" -m "C:\m.gguf" --host 0.0.0.0 --port=9999 -c 4096 --no-warmup"#.into(),
        );
        let args = build_llama_server_args(&cfg, "127.0.0.1", 51005);
        assert_eq!(
            args,
            vec![
                "-m",
                r"C:\m.gguf",
                "-c",
                "4096",
                "--no-warmup",
                "--host",
                "127.0.0.1",
                "--port",
                "51005",
            ]
        );

        cfg.custom_cli_args = Some("/usr/local/bin/llama-server -m /models/m.gguf -c 2048".into());
        assert_eq!(
            build_llama_server_args(&cfg, "127.0.0.1", 51006),
            vec!["-m", "/models/m.gguf", "-c", "2048", "--host", "127.0.0.1", "--port", "51006"]
        );
    }

    #[test]
    fn emits_spec_type_and_external_draft_together_for_dspark_and_dflash() {
        // Case 1: mtp_enabled=true with draft-dspark and draft_model_path set -> both --spec-type and -md emitted
        let mut cfg = LoadConfig::new("C:/models/inkling2.gguf");
        cfg.mtp_enabled = true;
        cfg.mtp_spec_type = Some("draft-dspark".into());
        cfg.draft_model_path = Some("C:/models/dspark.gguf".into());
        cfg.draft_gpu_layers = Some(99);
        cfg.draft_max = Some(4);
        let args = build_llama_server_args(&cfg, "127.0.0.1", 51007);
        assert!(args.windows(2).any(|w| w == ["--spec-type", "draft-dspark"]));
        assert!(args.windows(2).any(|w| w == ["-md", "C:/models/dspark.gguf"]));
        assert!(args.windows(2).any(|w| w == ["-ngld", "99"]));
        assert!(args.windows(2).any(|w| w == ["--spec-draft-n-max", "4"]));

        // Case 2: mtp_enabled=false with draft-dflash and draft_model_path set -> both --spec-type and -md emitted
        let mut cfg2 = LoadConfig::new("C:/models/inkling2.gguf");
        cfg2.mtp_enabled = false;
        cfg2.mtp_spec_type = Some("draft-dflash".into());
        cfg2.draft_model_path = Some("C:/models/dflash.gguf".into());
        let args2 = build_llama_server_args(&cfg2, "127.0.0.1", 51008);
        assert!(args2.windows(2).any(|w| w == ["--spec-type", "draft-dflash"]));
        assert!(args2.windows(2).any(|w| w == ["-md", "C:/models/dflash.gguf"]));

        // Case 3: mtp_enabled=false with stale draft-mtp and draft_model_path -> only -md emitted
        let mut cfg3 = LoadConfig::new("C:/models/main.gguf");
        cfg3.mtp_enabled = false;
        cfg3.mtp_spec_type = Some("draft-mtp".into());
        cfg3.draft_model_path = Some("C:/models/draft.gguf".into());
        let args3 = build_llama_server_args(&cfg3, "127.0.0.1", 51009);
        assert!(!args3.iter().any(|a| a == "--spec-type"));
        assert!(args3.windows(2).any(|w| w == ["-md", "C:/models/draft.gguf"]));

        // Case 4: mtp_enabled=true with draft-mtp and stale draft_model_path -> only --spec-type emitted
        let mut cfg4 = LoadConfig::new("C:/models/main.gguf");
        cfg4.mtp_enabled = true;
        cfg4.mtp_spec_type = Some("draft-mtp".into());
        cfg4.draft_model_path = Some("C:/models/draft.gguf".into());
        let args4 = build_llama_server_args(&cfg4, "127.0.0.1", 51010);
        assert!(args4.windows(2).any(|w| w == ["--spec-type", "draft-mtp"]));
        assert!(!args4.iter().any(|a| a == "-md"));
    }
}
