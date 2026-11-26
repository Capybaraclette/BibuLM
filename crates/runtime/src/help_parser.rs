use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;
use tokio::process::Command;

/// CLI flag dynamically discovered from `llama-server --help`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiscoveredCliFlag {
    pub flag: String,
    pub aliases: Vec<String>,
    pub value_hint: Option<String>,
    pub description: String,
}

/// Runs `llama-server --help` on the specified runtime binary and parses supported CLI flags.
pub async fn discover_runtime_cli_flags(server_exe: &Path) -> Result<Vec<DiscoveredCliFlag>, String> {
    if !server_exe.is_file() {
        return Err(format!("Runtime binary not found: {}", server_exe.display()));
    }
    let mut cmd = Command::new(server_exe);
    cmd.arg("--help")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    crate::configure_runtime_command(&mut cmd, server_exe);

    let output = tokio::time::timeout(Duration::from_secs(6), cmd.output())
        .await
        .map_err(|_| "Timed out running llama-server --help".to_string())?
        .map_err(|e| format!("Failed to execute llama-server --help: {e}"))?;

    let mut combined = String::from_utf8_lossy(&output.stdout).into_owned();
    if combined.trim().is_empty() {
        combined = String::from_utf8_lossy(&output.stderr).into_owned();
    }
    Ok(parse_llama_help_output(&combined))
}

/// Parses `llama-server --help` text output into structured CLI flag specs.
pub fn parse_llama_help_output(help_text: &str) -> Vec<DiscoveredCliFlag> {
    let mut flags: Vec<DiscoveredCliFlag> = Vec::new();
    let mut seen = std::collections::BTreeSet::new();

    for raw_line in help_text.lines() {
        let trimmed = raw_line.trim_start();
        if !trimmed.starts_with('-') || trimmed.starts_with("---") {
            // Continuation line for the previous flag's description
            if !trimmed.is_empty()
                && raw_line.starts_with("      ")
                && let Some(last) = flags.last_mut()
                && !trimmed.starts_with("(env:")
            {
                if !last.description.is_empty() {
                    last.description.push(' ');
                }
                last.description.push_str(trimmed.trim());
            }
            continue;
        }

        // Split flag column from description (separated by >= 2 spaces or tab)
        let (flag_part, desc_part) = split_flag_and_desc(trimmed);
        let mut long_flag: Option<String> = None;
        let mut aliases: Vec<String> = Vec::new();
        let mut value_hint: Option<String> = None;

        for seg in flag_part.split(',') {
            let s = seg.trim();
            if s.is_empty() {
                continue;
            }
            let mut tokens = s.split_whitespace();
            let Some(name) = tokens.next() else {
                continue;
            };
            if !name.starts_with('-') {
                continue;
            }
            let rest: Vec<&str> = tokens.collect();
            if !rest.is_empty() && value_hint.is_none() {
                value_hint = Some(rest.join(" "));
            }
            if name.starts_with("--") {
                if long_flag.is_none() {
                    long_flag = Some(name.to_string());
                } else {
                    aliases.push(name.to_string());
                }
            } else {
                aliases.push(name.to_string());
            }
        }

        let Some(canonical) = long_flag.or_else(|| aliases.first().cloned()) else {
            continue;
        };
        if seen.insert(canonical.clone()) {
            flags.push(DiscoveredCliFlag {
                flag: canonical,
                aliases,
                value_hint,
                description: desc_part.trim().to_string(),
            });
        }
    }

    flags
}

fn split_flag_and_desc(trimmed: &str) -> (&str, &str) {
    if let Some(idx) = trimmed.find("  ") {
        let (l, r) = trimmed.split_at(idx);
        (l.trim(), r.trim())
    } else if let Some(idx) = trimmed.find('\t') {
        let (l, r) = trimmed.split_at(idx);
        (l.trim(), r.trim())
    } else {
        (trimmed, "")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_llama_server_help_flags() {
        let help = r#"
usage: llama-server [options]

----- common params -----
-h, --help                              print usage and exit
-c, --ctx-size N                        size of the prompt context (default: 4096)
                                        (env: LLAMA_ARG_CTX_SIZE)
-fa, --flash-attn [on|off|auto]         set Flash Attention use (default: auto)
--n-cpu-moe N                           keep the Mixture of Experts weights of the first N layers in the CPU
--jinja                                 use jinja template for chat
"#;
        let parsed = parse_llama_help_output(help);
        assert_eq!(parsed.len(), 5);
        assert_eq!(parsed[1].flag, "--ctx-size");
        assert_eq!(parsed[1].aliases, vec!["-c"]);
        assert_eq!(parsed[1].value_hint.as_deref(), Some("N"));
        assert!(parsed[1].description.contains("prompt context"));
        assert_eq!(parsed[3].flag, "--n-cpu-moe");
        assert_eq!(parsed[4].flag, "--jinja");
        assert_eq!(parsed[4].value_hint, None);
    }
}
