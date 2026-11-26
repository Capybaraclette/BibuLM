# Contributing to BibuLM

Thanks for your interest in improving BibuLM! This guide covers how to set up the project locally, run tests, and submit changes.

By submitting a pull request, you agree that your contribution is licensed under the project's [MIT License](./LICENSE).

---

## Table of contents

1. [Code of conduct](#code-of-conduct)
2. [Ways to contribute](#ways-to-contribute)
3. [Prerequisites](#prerequisites)
4. [Local setup & running the app](#local-setup--running-the-app)
5. [Testing](#testing)
6. [Commits & pull requests](#commits--pull-requests)
7. [Branching and releases](#branching-and-releases)
8. [Reporting bugs and requesting features](#reporting-bugs-and-requesting-features)

---

## Code of conduct

Be respectful, constructive, and patient. Reviews focus on code, not on the contributor. Harassment, personal attacks, or discriminatory language are not tolerated in issues, PRs, discussions, or commit messages.

---

## Ways to contribute

- Fixing a bug from the [issues page](https://github.com/Capybaraclette/Bibulm/issues).
- Adding a feature that has been discussed in an issue.
- Improving documentation (`README.md`, `CONTRIBUTING.md`, CLI help text).
- Adding or improving unit and integration tests across the Rust workspace.
- Reviewing open pull requests.

For larger changes, **open an issue first** to align on scope before writing code.

---

## Prerequisites

| Tool | Minimum version | Purpose |
| --- | --- | --- |
| [Rust](https://rustup.rs/) | stable (edition 2024) | Core workspace, Tauri backend, and headless CLI |
| [Node.js](https://nodejs.org/) + npm | 22.x | Frontend build (`ui/` — Vite + React + TypeScript) |
| [Go](https://go.dev/) | 1.27.4+ *(optional)* | Only needed to build the `bibulm-linkd` Tailscale sidecar |
| `git` | recent | Version control |
| Docker + Compose | *(optional)* | Only needed if working on the headless server container |

On **Linux**, Tauri v2 also requires WebKitGTK and system libraries:

```bash
sudo apt-get install -y \
  libwebkit2gtk-4.1-dev libayatana-appindicator3-dev \
  librsvg2-dev libxdo-dev patchelf libssl-dev pkg-config
```

---

## Local setup & running the app

```bash
# 1. Fork the repo on GitHub, then clone your fork
git clone https://github.com/<your-username>/Bibulm.git
cd Bibulm

# 2. Add upstream so you can keep your fork up to date
git remote add upstream https://github.com/Capybaraclette/Bibulm.git

# 3. Install frontend dependencies and build the UI bundle
npm --prefix ui install
npm --prefix ui run build
```

### Desktop app

```bash
# Run the desktop app using the built ui/dist bundle
cargo run -p bibulm-app
```

For **hot-reload development**, run Vite in one terminal and Tauri in another:

```bash
# Terminal 1 — Vite dev server on http://127.0.0.1:1420
npm --prefix ui run dev

# Terminal 2 — Desktop shell connected to Vite HMR
cargo run -p bibulm-app --no-default-features
```

### BibuLM Link sidecar (optional)

If you are working on BibuLM Link, build the Go `tsnet` sidecar binary:

```powershell
# Windows
./scripts/build-linkd.ps1
```

```bash
# Linux / macOS
./scripts/build-linkd.sh
```

### Headless CLI server

```bash
cargo run -p bibulm-cli -- --help
```

### Docker (optional)

```bash
docker compose up -d --build
```

---

## Testing

Run the Rust test suite and make sure the frontend builds cleanly before opening a PR:

```bash
# Rust workspace tests
cargo test --workspace

# Frontend type-check & production build
npm --prefix ui run build
```

If you modified the Go sidecar (`sidecars/bibulm-linkd`):

```bash
cd sidecars/bibulm-linkd && go test ./...
```

If a change touches behavior that depends on a real GPU, runtime download, or live Tailscale connection and cannot be automated in unit tests, mention what you verified manually in the PR description.

---

## Commits & pull requests

### Commits

- Keep commits focused: one logical change per commit.
- Write commit messages in the imperative mood (e.g. `Add KV cache quantization preset`).
- Reference related issues with `Fixes #123` or `Closes #123` when applicable.

### Pull requests

Before opening a PR:

1. Rebase your branch on the latest `upstream/main`.
2. Ensure `cargo test --workspace` passes.
3. Ensure `npm --prefix ui run build` succeeds if you touched `ui/`.
4. Update `README.md` if your change is user-facing.

In your PR description, include:

- **What** changed and **why**.
- A link to any related issue.
- Screenshots for UI changes.
- How you tested the change.

---

## Branching and releases

- All pull requests target `main`.
- Releases follow [Semantic Versioning](https://semver.org/). Contributors should **not** bump version numbers in PRs.
- Maintainers bump versions in `Cargo.toml`, `src-tauri/tauri.conf.json`, and `ui/package.json`, then push a `vX.Y.Z` tag to trigger `.github/workflows/release.yml`.

---

## Reporting bugs and requesting features

Use the [issues page](https://github.com/Capybaraclette/Bibulm/issues):

- **Bug reports**: include your BibuLM version, OS, GPU / VRAM, installed `llama.cpp` runtime build, steps to reproduce, expected vs. actual behavior, and relevant logs from **Model Server → Server Logs**.
- **Feature requests**: describe the problem or workflow first, then your proposed solution.

---

Thanks for contributing!
