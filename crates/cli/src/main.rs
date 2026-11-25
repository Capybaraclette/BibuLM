//! `bibulm` - headless server CLI entry point.

mod args;
mod commands;
mod node;

use args::Command;
use node::HeadlessNode;
use std::process::ExitCode;

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let parsed = match args::parse(&argv, &|k| std::env::var(k).ok()) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {e}\n\nRun `bibulm --help` for usage.");
            return ExitCode::from(2);
        }
    };

    match &parsed.command {
        Command::Help(text) => {
            print!("{text}");
            return ExitCode::SUCCESS;
        }
        Command::Version => {
            println!("bibulm {}", env!("CARGO_PKG_VERSION"));
            return ExitCode::SUCCESS;
        }
        _ => {}
    }

    if let Some(dir) = &parsed.data_dir {
        // SAFETY: executed at the very start of `main`, before the async runtime or any other
        // thread exists, so no concurrent environment access is possible.
        unsafe { std::env::set_var("BIBULM_DATA_DIR", dir) };
    }

    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "warn,bibulm=info".to_string()),
        )
        .try_init();

    let runtime = match tokio::runtime::Runtime::new() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: cannot start async runtime: {e}");
            return ExitCode::FAILURE;
        }
    };

    match runtime.block_on(run(parsed)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

async fn run(parsed: args::Parsed) -> anyhow::Result<()> {
    let node = HeadlessNode::open()?;
    let json = parsed.json;
    match parsed.command {
        Command::Serve(a) => commands::cmd_serve(&node, a).await,
        Command::Login(a) => commands::cmd_login(&node, a).await,
        Command::Logout => commands::cmd_logout(&node).await,
        Command::Status => commands::cmd_status(&node, json).await,
        Command::Hardware => commands::cmd_hardware(&node, json),
        Command::Runtime(c) => commands::cmd_runtime(&node, c, json).await,
        Command::Models(c) => commands::cmd_models(&node, c, json).await,
        Command::ServiceSystemd(a) => commands::cmd_service_systemd(&node.data_dir, a),
        Command::Help(_) | Command::Version => Ok(()),
    }
}
