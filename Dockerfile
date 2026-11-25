# BibuLM headless server image (CLI + Tailscale userspace sidecar).
#   docker build -t bibulm-server .
#   docker run -d -p 1234:1234 -v bibulm_data:/data -e TS_AUTHKEY=tskey-... bibulm-server

FROM golang:1.27-bookworm AS go-builder
WORKDIR /src/sidecars/bibulm-linkd
COPY sidecars/bibulm-linkd/ ./
RUN go mod tidy && go mod download
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/bibulm-linkd .

FROM rust:1-bookworm AS rust-builder
WORKDIR /src
# The workspace lists src-tauri as a member, so its sources must exist for manifest resolution.
# `cargo build -p bibulm-cli` never compiles it, so no WebKit/GTK packages are needed.
COPY Cargo.toml Cargo.lock* ./
COPY crates ./crates
COPY src-tauri ./src-tauri
RUN cargo build --release -p bibulm-cli \
    && mkdir -p /out && cp target/release/bibulm /out/bibulm

FROM debian:bookworm-slim AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 libssl3 libvulkan1 curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=rust-builder /out/bibulm /usr/local/bin/bibulm
COPY --from=go-builder /out/bibulm-linkd /usr/local/bin/bibulm-linkd

ENV BIBULM_DATA_DIR=/data
RUN mkdir -p /data
VOLUME ["/data"]
EXPOSE 1234

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD sh -c 'curl -fsS "http://127.0.0.1:${BIBULM_PORT:-1234}/health" || exit 1'

ENTRYPOINT ["bibulm"]
CMD ["serve", "--lan"]
