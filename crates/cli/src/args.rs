//! Zero-dependency command-line parser for the `bibulm` headless server CLI.

use bibulm_core::LinkAccessPolicy;
use std::path::PathBuf;

pub const MAIN_HELP: &str = "\
bibulm - headless Bibulm server (OpenAI-compatible inference + Bibulm Link)

USAGE:
    bibulm [GLOBAL OPTIONS] <COMMAND> [OPTIONS]

COMMANDS:
    serve       Run the inference gateway and connect to your Bibulm Link mesh
    login       Authenticate this machine on your Tailscale tailnet (one-time)
    logout      Remove this machine from your tailnet and forget its identity
    status      Show server, Link, runtime, model and download status
    hardware    Print detected CPU / RAM / GPU hardware
    runtime     Manage llama.cpp runtimes   (list | releases | install | use | remove)
    models      Manage GGUF models          (list | search | pull | remove)
    service     Generate service files      (service systemd)

GLOBAL OPTIONS:
        --data-dir <PATH>   Data directory (env: BIBULM_DATA_DIR)
        --json              Machine-readable JSON output where supported
    -h, --help              Print help
    -V, --version           Print version

Run `bibulm <COMMAND> --help` for command details.
";

pub const SERVE_HELP: &str = "\
bibulm serve - run the headless inference gateway

OPTIONS:
        --name <NAME>          Device name shown in Bibulm Desktop (env: BIBULM_DEVICE_NAME)
        --port <PORT>          OpenAI API port, default 1234 (env: BIBULM_PORT)
        --lan                  Listen on 0.0.0.0 instead of 127.0.0.1
        --mesh-port <PORT>     Tailnet mesh port, default 11435
        --policy <POLICY>      self_only | same_tailnet (env: BIBULM_ACCESS_POLICY)
        --control-url <URL>    Custom control server, e.g. Headscale (env: BIBULM_CONTROL_URL)
        --authkey <KEY>        Tailscale auth key for headless login (env: TS_AUTHKEY)
        --models-dir <PATH>    Primary models directory (env: BIBULM_MODELS_DIR)
        --runtimes-dir <PATH>  Runtimes directory (env: BIBULM_RUNTIMES_DIR)
        --hf-token <TOKEN>     HuggingFace token (env: HF_TOKEN)
        --github-token <TOKEN> GitHub token for release lookups (env: GITHUB_TOKEN)
        --no-link              Do not connect to Bibulm Link
        --no-remote-load       Refuse remote model load/unload requests
        --no-remote-manage     Refuse remote runtime install / model download requests
";

pub const LOGIN_HELP: &str = "\
bibulm login - authenticate this machine on your tailnet

OPTIONS:
        --name <NAME>          Device name (env: BIBULM_DEVICE_NAME)
        --authkey <KEY>        Tailscale auth key; otherwise a login URL is printed (env: TS_AUTHKEY)
        --control-url <URL>    Custom control server (env: BIBULM_CONTROL_URL)
        --policy <POLICY>      self_only | same_tailnet
";

pub const RUNTIME_HELP: &str = "\
bibulm runtime - manage llama.cpp runtimes

USAGE:
    bibulm runtime list
    bibulm runtime releases
    bibulm runtime install [--tag <TAG>] [--backend <BACKEND>]
    bibulm runtime use <ID>
    bibulm runtime remove <ID>
";

pub const MODELS_HELP: &str = "\
bibulm models - manage GGUF models

USAGE:
    bibulm models list
    bibulm models search <QUERY>
    bibulm models pull <REPO_ID> [--quant <QUANT>] [--mmproj]
    bibulm models remove <MODEL_ID>
";

pub const SERVICE_HELP: &str = "\
bibulm service systemd - generate a systemd unit

OPTIONS:
        --name <NAME>       Device name passed to `bibulm serve`
        --user <USER>       Run the service as this user (system unit)
        --port <PORT>       OpenAI API port
        --lan               Pass --lan to `bibulm serve`
        --user-unit         Generate a per-user unit (~/.config/systemd/user)
        --install           Write the unit file instead of printing it
";

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ServeArgs {
    pub name: Option<String>,
    pub port: Option<u16>,
    pub lan: bool,
    pub mesh_port: Option<u16>,
    pub policy: Option<LinkAccessPolicy>,
    pub control_url: Option<String>,
    pub authkey: Option<String>,
    pub models_dir: Option<PathBuf>,
    pub runtimes_dir: Option<PathBuf>,
    pub hf_token: Option<String>,
    pub github_token: Option<String>,
    pub no_link: bool,
    pub no_remote_load: bool,
    pub no_remote_manage: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LoginArgs {
    pub name: Option<String>,
    pub authkey: Option<String>,
    pub control_url: Option<String>,
    pub policy: Option<LinkAccessPolicy>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeCmd {
    List,
    Releases,
    Install {
        tag: Option<String>,
        backend: Option<String>,
    },
    Use(String),
    Remove(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelsCmd {
    List,
    Search(String),
    Pull {
        repo: String,
        quant: Option<String>,
        mmproj: bool,
    },
    Remove(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SystemdArgs {
    pub name: Option<String>,
    pub user: Option<String>,
    pub port: Option<u16>,
    pub lan: bool,
    pub user_unit: bool,
    pub install: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    Serve(ServeArgs),
    Login(LoginArgs),
    Logout,
    Status,
    Hardware,
    Runtime(RuntimeCmd),
    Models(ModelsCmd),
    ServiceSystemd(SystemdArgs),
    Help(&'static str),
    Version,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Parsed {
    pub data_dir: Option<PathBuf>,
    pub json: bool,
    pub command: Command,
}

struct Opts {
    rest: Vec<String>,
}

impl Opts {
    fn flag(&mut self, names: &[&str]) -> bool {
        let before = self.rest.len();
        self.rest.retain(|a| !names.contains(&a.as_str()));
        self.rest.len() != before
    }

    fn value(&mut self, names: &[&str]) -> Result<Option<String>, String> {
        let mut found: Option<String> = None;
        let mut i = 0;
        while i < self.rest.len() {
            let arg = self.rest[i].clone();
            if names.contains(&arg.as_str()) {
                if i + 1 >= self.rest.len() {
                    return Err(format!("option '{arg}' requires a value"));
                }
                found = Some(self.rest[i + 1].clone());
                self.rest.drain(i..=i + 1);
                continue;
            }
            if let Some((k, v)) = arg.split_once('=')
                && names.contains(&k)
            {
                found = Some(v.to_string());
                self.rest.remove(i);
                continue;
            }
            i += 1;
        }
        Ok(found)
    }

    fn port(&mut self, names: &[&str]) -> Result<Option<u16>, String> {
        match self.value(names)? {
            Some(v) => v
                .trim()
                .parse::<u16>()
                .map(Some)
                .map_err(|_| format!("invalid port '{v}'")),
            None => Ok(None),
        }
    }

    fn policy(&mut self, names: &[&str]) -> Result<Option<LinkAccessPolicy>, String> {
        match self.value(names)? {
            Some(v) => parse_policy(&v).map(Some),
            None => Ok(None),
        }
    }

    fn positionals(self) -> Result<Vec<String>, String> {
        if let Some(bad) = self.rest.iter().find(|a| a.starts_with('-') && a.len() > 1) {
            return Err(format!("unknown option '{bad}'"));
        }
        Ok(self.rest)
    }
}

pub fn parse_policy(raw: &str) -> Result<LinkAccessPolicy, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "self_only" | "self" => Ok(LinkAccessPolicy::SelfOnly),
        "same_tailnet" | "tailnet" => Ok(LinkAccessPolicy::SameTailnet),
        other => Err(format!(
            "invalid policy '{other}' (expected self_only or same_tailnet)"
        )),
    }
}

fn nonempty(v: Option<String>) -> Option<String> {
    v.map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

/// Parses `args` (excluding the program name). `env` resolves environment fallbacks.
pub fn parse(args: &[String], env: &dyn Fn(&str) -> Option<String>) -> Result<Parsed, String> {
    let mut g = Opts {
        rest: args.to_vec(),
    };
    let help = g.flag(&["-h", "--help"]);
    let version = g.flag(&["-V", "--version"]);
    let json = g.flag(&["--json"]);
    let data_dir = nonempty(g.value(&["--data-dir"])?)
        .or_else(|| nonempty(env("BIBULM_DATA_DIR")))
        .map(PathBuf::from);

    let mut rest = g.rest;
    let finish = |command: Command| Parsed {
        data_dir: data_dir.clone(),
        json,
        command,
    };

    if version {
        return Ok(finish(Command::Version));
    }
    if rest.is_empty() {
        return Ok(finish(Command::Help(MAIN_HELP)));
    }
    let sub = rest.remove(0);
    let mut o = Opts { rest };

    let command = match sub.as_str() {
        "serve" => {
            if help {
                return Ok(finish(Command::Help(SERVE_HELP)));
            }
            let a = ServeArgs {
                name: nonempty(o.value(&["--name"])?).or_else(|| nonempty(env("BIBULM_DEVICE_NAME"))),
                port: match o.port(&["--port"])? {
                    Some(p) => Some(p),
                    None => match nonempty(env("BIBULM_PORT")) {
                        Some(v) => Some(
                            v.parse::<u16>()
                                .map_err(|_| format!("invalid BIBULM_PORT '{v}'"))?,
                        ),
                        None => None,
                    },
                },
                lan: o.flag(&["--lan"]),
                mesh_port: o.port(&["--mesh-port"])?,
                policy: match o.policy(&["--policy"])? {
                    Some(p) => Some(p),
                    None => match nonempty(env("BIBULM_ACCESS_POLICY")) {
                        Some(v) => Some(parse_policy(&v)?),
                        None => None,
                    },
                },
                control_url: nonempty(o.value(&["--control-url"])?)
                    .or_else(|| nonempty(env("BIBULM_CONTROL_URL"))),
                authkey: nonempty(o.value(&["--authkey"])?).or_else(|| nonempty(env("TS_AUTHKEY"))),
                models_dir: nonempty(o.value(&["--models-dir"])?)
                    .or_else(|| nonempty(env("BIBULM_MODELS_DIR")))
                    .map(PathBuf::from),
                runtimes_dir: nonempty(o.value(&["--runtimes-dir"])?)
                    .or_else(|| nonempty(env("BIBULM_RUNTIMES_DIR")))
                    .map(PathBuf::from),
                hf_token: nonempty(o.value(&["--hf-token"])?).or_else(|| nonempty(env("HF_TOKEN"))),
                github_token: nonempty(o.value(&["--github-token"])?)
                    .or_else(|| nonempty(env("GITHUB_TOKEN"))),
                no_link: o.flag(&["--no-link"]),
                no_remote_load: o.flag(&["--no-remote-load"]),
                no_remote_manage: o.flag(&["--no-remote-manage"]),
            };
            no_extra(o)?;
            Command::Serve(a)
        }
        "login" => {
            if help {
                return Ok(finish(Command::Help(LOGIN_HELP)));
            }
            let a = LoginArgs {
                name: nonempty(o.value(&["--name"])?).or_else(|| nonempty(env("BIBULM_DEVICE_NAME"))),
                authkey: nonempty(o.value(&["--authkey"])?).or_else(|| nonempty(env("TS_AUTHKEY"))),
                control_url: nonempty(o.value(&["--control-url"])?)
                    .or_else(|| nonempty(env("BIBULM_CONTROL_URL"))),
                policy: o.policy(&["--policy"])?,
            };
            no_extra(o)?;
            Command::Login(a)
        }
        "logout" => {
            if help {
                return Ok(finish(Command::Help(MAIN_HELP)));
            }
            no_extra(o)?;
            Command::Logout
        }
        "status" => {
            if help {
                return Ok(finish(Command::Help(MAIN_HELP)));
            }
            no_extra(o)?;
            Command::Status
        }
        "hardware" => {
            if help {
                return Ok(finish(Command::Help(MAIN_HELP)));
            }
            no_extra(o)?;
            Command::Hardware
        }
        "runtime" => {
            if help {
                return Ok(finish(Command::Help(RUNTIME_HELP)));
            }
            let tag = nonempty(o.value(&["--tag"])?);
            let backend = nonempty(o.value(&["--backend"])?);
            let pos = o.positionals()?;
            let action = pos.first().map(String::as_str).unwrap_or("");
            let cmd = match action {
                "list" | "ls" => RuntimeCmd::List,
                "releases" => RuntimeCmd::Releases,
                "install" => RuntimeCmd::Install { tag, backend },
                "use" => RuntimeCmd::Use(
                    pos.get(1)
                        .cloned()
                        .ok_or("usage: bibulm runtime use <ID>")?,
                ),
                "remove" | "rm" => RuntimeCmd::Remove(
                    pos.get(1)
                        .cloned()
                        .ok_or("usage: bibulm runtime remove <ID>")?,
                ),
                _ => return Ok(finish(Command::Help(RUNTIME_HELP))),
            };
            Command::Runtime(cmd)
        }
        "models" => {
            if help {
                return Ok(finish(Command::Help(MODELS_HELP)));
            }
            let quant = nonempty(o.value(&["--quant"])?);
            let mmproj = o.flag(&["--mmproj"]);
            let pos = o.positionals()?;
            let action = pos.first().map(String::as_str).unwrap_or("");
            let cmd = match action {
                "list" | "ls" => ModelsCmd::List,
                "search" => {
                    if pos.len() < 2 {
                        return Err("usage: bibulm models search <QUERY>".into());
                    }
                    ModelsCmd::Search(pos[1..].join(" "))
                }
                "pull" | "download" => ModelsCmd::Pull {
                    repo: pos
                        .get(1)
                        .cloned()
                        .ok_or("usage: bibulm models pull <REPO_ID> [--quant <QUANT>] [--mmproj]")?,
                    quant,
                    mmproj,
                },
                "remove" | "rm" => ModelsCmd::Remove(
                    pos.get(1)
                        .cloned()
                        .ok_or("usage: bibulm models remove <MODEL_ID>")?,
                ),
                _ => return Ok(finish(Command::Help(MODELS_HELP))),
            };
            Command::Models(cmd)
        }
        "service" => {
            if help {
                return Ok(finish(Command::Help(SERVICE_HELP)));
            }
            let a = SystemdArgs {
                name: nonempty(o.value(&["--name"])?).or_else(|| nonempty(env("BIBULM_DEVICE_NAME"))),
                user: nonempty(o.value(&["--user"])?),
                port: o.port(&["--port"])?,
                lan: o.flag(&["--lan"]),
                user_unit: o.flag(&["--user-unit"]),
                install: o.flag(&["--install"]),
            };
            let pos = o.positionals()?;
            if pos.first().map(String::as_str) != Some("systemd") {
                return Ok(finish(Command::Help(SERVICE_HELP)));
            }
            Command::ServiceSystemd(a)
        }
        "help" => Command::Help(MAIN_HELP),
        other => return Err(format!("unknown command '{other}'. Run `bibulm --help`.")),
    };
    Ok(finish(command))
}

fn no_extra(o: Opts) -> Result<(), String> {
    let pos = o.positionals()?;
    if let Some(extra) = pos.first() {
        return Err(format!("unexpected argument '{extra}'"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(args: &[&str]) -> Result<Parsed, String> {
        let v: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        parse(&v, &|_| None)
    }

    #[test]
    fn parses_serve_flags_and_global_options_anywhere() {
        let parsed = p(&[
            "serve",
            "--name",
            "gpu-server",
            "--port=8080",
            "--lan",
            "--policy",
            "same_tailnet",
            "--no-remote-manage",
            "--data-dir",
            "/srv/bibulm",
        ])
        .unwrap();
        assert_eq!(parsed.data_dir, Some(PathBuf::from("/srv/bibulm")));
        let Command::Serve(a) = parsed.command else {
            panic!("expected serve");
        };
        assert_eq!(a.name.as_deref(), Some("gpu-server"));
        assert_eq!(a.port, Some(8080));
        assert!(a.lan && a.no_remote_manage && !a.no_link);
        assert_eq!(a.policy, Some(LinkAccessPolicy::SameTailnet));
    }

    #[test]
    fn env_fallbacks_apply_and_flags_win() {
        let args: Vec<String> = ["serve", "--name", "cli-name"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let env = |k: &str| match k {
            "BIBULM_DEVICE_NAME" => Some("env-name".to_string()),
            "TS_AUTHKEY" => Some("tskey-abc".to_string()),
            "BIBULM_PORT" => Some("4321".to_string()),
            _ => None,
        };
        let parsed = parse(&args, &env).unwrap();
        let Command::Serve(a) = parsed.command else {
            panic!("expected serve");
        };
        assert_eq!(a.name.as_deref(), Some("cli-name"));
        assert_eq!(a.authkey.as_deref(), Some("tskey-abc"));
        assert_eq!(a.port, Some(4321));
    }

    #[test]
    fn parses_runtime_and_models_subcommands() {
        assert_eq!(
            p(&["runtime", "install", "--backend", "cuda-12.4", "--tag", "b5200"])
                .unwrap()
                .command,
            Command::Runtime(RuntimeCmd::Install {
                tag: Some("b5200".into()),
                backend: Some("cuda-12.4".into())
            })
        );
        assert_eq!(
            p(&["runtime", "use", "b5200-cpu"]).unwrap().command,
            Command::Runtime(RuntimeCmd::Use("b5200-cpu".into()))
        );
        assert_eq!(
            p(&["models", "pull", "unsloth/Qwen3-8B-GGUF", "--quant", "Q4_K_M", "--mmproj"])
                .unwrap()
                .command,
            Command::Models(ModelsCmd::Pull {
                repo: "unsloth/Qwen3-8B-GGUF".into(),
                quant: Some("Q4_K_M".into()),
                mmproj: true
            })
        );
        assert_eq!(
            p(&["models", "search", "qwen", "coder"]).unwrap().command,
            Command::Models(ModelsCmd::Search("qwen coder".into()))
        );
    }

    #[test]
    fn rejects_bad_input_and_handles_help_version() {
        assert!(p(&["serve", "--port", "abc"]).is_err());
        assert!(p(&["serve", "--bogus"]).is_err());
        assert!(p(&["serve", "--policy", "nobody"]).is_err());
        assert!(p(&["frobnicate"]).is_err());
        assert!(p(&["runtime", "use"]).is_err());
        assert_eq!(p(&["--version"]).unwrap().command, Command::Version);
        assert_eq!(p(&[]).unwrap().command, Command::Help(MAIN_HELP));
        assert_eq!(
            p(&["serve", "--help"]).unwrap().command,
            Command::Help(SERVE_HELP)
        );
        assert!(p(&["status", "--json"]).unwrap().json);
    }

    #[test]
    fn parses_service_systemd() {
        let parsed = p(&["service", "systemd", "--name", "box", "--lan", "--install"]).unwrap();
        let Command::ServiceSystemd(a) = parsed.command else {
            panic!("expected systemd");
        };
        assert!(a.lan && a.install && !a.user_unit);
        assert_eq!(a.name.as_deref(), Some("box"));
    }
}
