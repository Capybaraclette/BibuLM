#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIDECAR_DIR="$ROOT_DIR/sidecars/bibulm-linkd"
OUT_DIR="$ROOT_DIR/src-tauri/binaries"
mkdir -p "$OUT_DIR"

HOST_TRIPLE="$(rustc -vV | awk '/host:/ { print $2 }')"
TARGET_TRIPLE="${1:-$HOST_TRIPLE}"

case "$TARGET_TRIPLE" in
  x86_64-unknown-linux-gnu)   GOOS=linux;   GOARCH=amd64; EXT="" ;;
  aarch64-unknown-linux-gnu)  GOOS=linux;   GOARCH=arm64; EXT="" ;;
  x86_64-apple-darwin)        GOOS=darwin;  GOARCH=amd64; EXT="" ;;
  aarch64-apple-darwin)       GOOS=darwin;  GOARCH=arm64; EXT="" ;;
  x86_64-pc-windows-msvc)     GOOS=windows; GOARCH=amd64; EXT=".exe" ;;
  aarch64-pc-windows-msvc)    GOOS=windows; GOARCH=arm64; EXT=".exe" ;;
  *) echo "Unsupported target triple: $TARGET_TRIPLE" >&2; exit 1 ;;
esac

OUT_FILE="$OUT_DIR/bibulm-linkd-${TARGET_TRIPLE}${EXT}"
echo "Building bibulm-linkd for $TARGET_TRIPLE (GOOS=$GOOS GOARCH=$GOARCH)..."
(
  cd "$SIDECAR_DIR"
  go mod tidy
  CGO_ENABLED=0 GOOS="$GOOS" GOARCH="$GOARCH" go build -trimpath -ldflags="-s -w" -o "$OUT_FILE" .
)
echo "Built sidecar: $OUT_FILE"
