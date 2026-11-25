param(
    [string]$TargetTriple = "x86_64-pc-windows-msvc",
    [string]$GoArch = "amd64"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$SidecarDir = Join-Path $Root "sidecars\bibulm-linkd"
$OutDir = Join-Path $Root "src-tauri\binaries"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$OutFile = Join-Path $OutDir "bibulm-linkd-$TargetTriple.exe"

Write-Host "Building bibulm-linkd for $TargetTriple (GOOS=windows GOARCH=$GoArch)..."
Push-Location $SidecarDir
try {
    $env:CGO_ENABLED = "0"
    $env:GOOS = "windows"
    $env:GOARCH = $GoArch
    go mod tidy
    go build -trimpath -ldflags="-s -w" -o $OutFile .
    Write-Host "Built sidecar: $OutFile"
} finally {
    Pop-Location
}
