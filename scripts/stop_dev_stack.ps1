# Stop the host-mode development stack started by scripts/dev_stack.ps1.
# Dependency containers (db/redis/minio) are left running unless -StopDeps is given.
param([switch]$StopDeps)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
$pidFile = Join-Path $root "var\dev_stack.pids.json"
if (-not (Test-Path $pidFile)) { Write-Host "[dev-stack] no pid file: $pidFile"; exit 0 }

$entries = Get-Content $pidFile -Raw | ConvertFrom-Json
foreach ($entry in @($entries)) {
  try {
    $proc = Get-Process -Id $entry.pid -ErrorAction Stop
    Stop-Process -Id $entry.pid -Force -ErrorAction Stop
    Write-Host ("[dev-stack] stopped {0} pid={1}" -f $entry.name, $entry.pid)
  } catch {
    Write-Host ("[dev-stack] {0} pid={1} already gone" -f $entry.name, $entry.pid)
  }
}
Remove-Item $pidFile -Force

if ($StopDeps) {
  Set-Location $root
  docker compose stop db redis minio | Out-Null
  Write-Host "[dev-stack] dependency containers stopped"
} else {
  Write-Host "[dev-stack] dependency containers left running (use -StopDeps to stop them)"
}
