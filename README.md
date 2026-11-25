# BibuLM

A lightweight, native **Rust + WebView** (Tauri v2) desktop workbench for running, benchmarking, and serving local LLMs powered directly by upstream **`llama.cpp`**.

## Core Features

- **Direct `ggml-org/llama.cpp` Runtime Manager** (`crates/runtime`): Fetch official GitHub releases (`bXXXX`) across **Windows (`x64`/`arm64`), Linux (`x64`/`arm64`), and macOS (`arm64`)**, automatically pair Windows CUDA binaries with companion `cudart` DLLs (matched against your NVIDIA driver via NVML), Vulkan, ROCm/HIP, SYCL, OpenVINO, or CPU builds, discover live `llama-server --help` CLI flags, keep multiple versions side-by-side (`llama-server` + `llama-bench`), or register your own compiled binary.
- **First-Class `llama.cpp` Controls** (`crates/core`, `crates/runtime`):
  - **Native Pure-Rust GGUF v2/v3 Inspector & MoE Estimator**: Inspects split and single `.gguf` headers in milliseconds (architecture, layers, heads, MoE experts, chat template, reasoning support, tokenizer family/vocab for **Speculative Decoding draft/main compatibility**) and computes exact VRAM vs. RAM breakdowns with **one-click Auto-Fit MoE to VRAM** (`--n-cpu-moe N`).
  - **MoE Selective VRAM/RAM Offloading**: Full GPU, All Experts on RAM (`--cpu-moe`), Smart Hybrid (`--n-cpu-moe N`), or Custom Tensor Override rules (`-ot <regex>=<buffer>`).
  - **Thinking / Reasoning Controls**: Effort levels (`off`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`), `enable_thinking` toggle, token budgets (`--reasoning-budget`), cutoff messages (`--reasoning-budget-message`), and output format (`--reasoning-format`).
  - Full parameter exposure (KV cache quantization `-ctk`/`-ctv`, Flash Attention `-fa`, RoPE/YaRN scaling, speculative decoding `-md`, multimodal `mmproj`, live CLI preview, full flags catalog, and raw CLI escape hatch) with load-time **VRAM Guardrails** (`Strict` / dedicated VRAM enforcement).
- **Model Discovery, Resumable Downloads & Local Library** (`crates/models`): Search HuggingFace GGUFs with capability badges and sanitized Markdown READMEs (`DOMPurify`), grouped split-GGUF quantizations with automatic VRAM/MoE fit verdicts, persistent pause/resume HTTP Range downloads (`files_json` + SHA-256 verification), and recursive local library scanning.
- **OpenAI & Anthropic-Compatible API Gateway** (`crates/server`): Local `axum` reverse-proxy gateway (`/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/responses`, plus **Anthropic `/v1/messages` & `/v1/messages/count_tokens`** with `x-api-key` auth for Claude Code) featuring configurable **Model Alias Routing** (exact & glob patterns + opt-in unknown-model fallback), SSE streaming, CORS, LAN bind toggle, Just-in-Time model loading, idle TTL auto-unload, hashed Bearer API keys (SHA-256) & OS-encrypted provider tokens (Windows DPAPI), OS process cleanup (`Job Objects` / `PR_SET_PDEATHSIG`), and non-blocking rotating logs.
- **BibuLM Link — BYO Tailscale Mesh** (`crates/link`, `sidecars/bibulm-linkd`): Connect your own devices over your personal Tailscale/Headscale network via an embedded `tsnet` userspace sidecar — discover remote GPUs, stream remote models over the API gateway, and trigger hardened remote model loads/unloads across devices with zero third-party relays or system VPN installation.
- **Interactive API Playground, `llama-bench` Matrix & Memory Placement Inspector** (`crates/bench`, `crates/hardware`): Multi-turn API Playground (Vision base64 upload, Tools JSON + `tool_calls`, Structured JSON Schema validation, KV prefix cache hit telemetry), multi-config `llama-bench -o json` sweep matrix, live `/metrics` & `/slots` server telemetry (including Linux `/sys/class/drm` AMD/Intel VRAM), SQLite benchmark history with CSV/JSON export, and GPU VRAM vs. RAM / WDDM spillover verification.

## Installing & Auto-Updates

- **Prebuilt Releases**: Download the installer for your platform (`BibuLM_<version>_x64-setup.exe` / `.msi` on Windows, `.dmg` on macOS, `.deb` / `.rpm` / `.AppImage` on Linux) from the [GitHub Releases](https://github.com/Capybaraclette/bibulm/releases) page.
- **Automatic Updates (Windows)**: BibuLM checks GitHub Releases on launch (configurable in **Settings → Application Updates**). Clicking **Update & Restart** downloads the installer, verifies its `SHA256SUMS.txt` digest, stops running models and the local API gateway cleanly, installs silently, and relaunches BibuLM.
- **Publishing a Release**: Bump the version in `Cargo.toml`, `src-tauri/tauri.conf.json`, and `ui/package.json`, then push a tag `vX.Y.Z` (e.g. `git tag v0.3.0 && git push origin v0.3.0`). `.github/workflows/release.yml` builds all platform installers, compiles `bibulm-linkd`, generates `SHA256SUMS.txt`, and publishes the GitHub Release.

## Building from Source

```powershell
# 1. Build the frontend bundle (Vite + React + TypeScript + Tailwind)
npm --prefix ui install
npm --prefix ui run build

# 2. (Optional) Build the BibuLM Link Go sidecar (requires Go 1.27.4+)
./scripts/build-linkd.ps1   # or ./scripts/build-linkd.sh on Linux/macOS

# 3. Run all Rust unit & integration tests across the workspace
cargo test --workspace

# 4. Launch BibuLM (uses embedded ui/dist)
cargo run -p bibulm-app

# 5. Build an optimized release executable
cargo build --release -p bibulm-app
```

> **Dev Hot-Reload**: Run `npm --prefix ui run dev` in one terminal and `cargo run -p bibulm-app --no-default-features` in another to connect the WebView directly to Vite HMR on `http://127.0.0.1:1420`.

## Keyboard Shortcuts

- `Ctrl + L` — Quick Load / Switch active model
- `Ctrl + R` — Reload last used model
- `Ctrl + Shift + R` — Open Runtime Manager
- `Ctrl + Shift + H` — Open Hardware & Live Telemetry
- `Ctrl + Enter` — Confirm load inside Load Config Modal

## License

BibuLM is licensed under the [MIT License](LICENSE).

---

*Bibu is my cat's surname, and the logo is her face.*
