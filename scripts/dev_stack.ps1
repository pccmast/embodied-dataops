# Windows host-mode development stack (ADR-0011).
#
# Starts the already-pulled dependency containers (PostgreSQL, Redis, PGSTY Silo),
# applies migrations, then runs the API, dispatcher and workers as host processes.
# No project container image is built on Windows.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/dev_stack.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/dev_stack.ps1 -WithUI
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/stop_dev_stack.ps1
param(
  [switch]$SkipDeps,
  [switch]$WithUI,
  [switch]$Force
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }
$logDir = Join-Path $root "var\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$pidFile = Join-Path $root "var\dev_stack.pids.json"

if ((Test-Path $pidFile) -and -not $Force) {
  Write-Host "[dev-stack] existing pid file found: $pidFile (use -Force to overwrite / stop first)"
  exit 1
}

# Container-internal endpoints are not usable from the host: point the app at
# localhost while keeping the same S3/MinIO-compatible API.
$env:STORAGE_BACKEND = "s3"
$env:S3_ENDPOINT_INTERNAL = "http://localhost:9000"
$env:S3_ENDPOINT_PUBLIC = "http://localhost:9000"
$env:DETECTOR = if ($env:DETECTOR) { $env:DETECTOR } else { "fake" }
$env:PYTHONPATH = "src"

if (-not $SkipDeps) {
  Write-Host "[dev-stack] starting dependency containers (db, redis, minio/Silo)"
  docker compose up -d db redis minio | Out-Null
  $deadline = (Get-Date).AddSeconds(120)
  while ((Get-Date) -lt $deadline) {
    $dbOk = (docker exec dataops-db pg_isready -U dataops -d dataops 2>$null) -match "accepting connections"
    $redisOk = ((docker exec dataops-redis redis-cli ping 2>$null) -match "PONG")
    $siloOk = $false
    try { $siloOk = (Invoke-WebRequest "http://127.0.0.1:9000/minio/health/ready" -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200 } catch { }
    if ($dbOk -and $redisOk -and $siloOk) { break }
    Start-Sleep -Seconds 2
  }
  if (-not ($dbOk -and $redisOk -and $siloOk)) { throw "dependency containers did not become healthy" }
}

Write-Host "[dev-stack] applying migrations"
& $python -m alembic upgrade head
if ($LASTEXITCODE -ne 0) { throw "alembic upgrade failed" }
& $python -c "from embodied_dataops.storage.base import get_storage; get_storage().ensure_bucket(); print('bucket ready')"
if ($LASTEXITCODE -ne 0) { throw "object-store bucket init failed" }

$processes = @()
function Start-Svc([string]$name, [string[]]$argsList) {
  $stdout = Join-Path $logDir "$name.log"
  $stderr = Join-Path $logDir "$name.err.log"
  $p = Start-Process -FilePath $python -ArgumentList $argsList -PassThru `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
        -WorkingDirectory $root
  Write-Host ("[dev-stack] {0} pid={1} log={2}" -f $name, $p.Id, $stdout)
  return [pscustomobject]@{ name = $name; pid = $p.Id; log = $stdout }
}

$processes += Start-Svc "api" @("-m", "uvicorn", "embodied_dataops.api.app:app", "--host", "127.0.0.1", "--port", "8000")
$processes += Start-Svc "dispatcher" @("-m", "embodied_dataops.workflow.dispatcher")
$cpuConcurrency = if ($env:WORKER_CPU_CONCURRENCY) { $env:WORKER_CPU_CONCURRENCY } else { "2" }
$processes += Start-Svc "worker-cpu" @("-m", "embodied_dataops.workflow.worker", "--pool", "cpu", "--concurrency", $cpuConcurrency)
$visionConcurrency = if ($env:WORKER_VISION_CONCURRENCY) { $env:WORKER_VISION_CONCURRENCY } else { "1" }
$processes += Start-Svc "worker-vision" @("-m", "embodied_dataops.workflow.worker", "--pool", "vision", "--concurrency", $visionConcurrency)

if ($WithUI) {
  $processes += Start-Svc "ui" @("-m", "embodied_dataops.cli.main", "ui", "--port", "8501")
}

$processes | ConvertTo-Json | Set-Content -Path $pidFile -Encoding utf8

$deadline = (Get-Date).AddSeconds(60)
$ready = $false
while ((Get-Date) -lt $deadline) {
  try {
    if ((Invoke-WebRequest "http://127.0.0.1:8000/health/ready" -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200) { $ready = $true; break }
  } catch { }
  Start-Sleep -Seconds 2
}
Write-Host ("[dev-stack] API: http://127.0.0.1:8000  ready={0}" -f $ready)
Write-Host ("[dev-stack] UI:  http://127.0.0.1:8501  (if -WithUI)" -f $ready)
Write-Host "[dev-stack] stop with: powershell -NoProfile -ExecutionPolicy Bypass -File scripts/stop_dev_stack.ps1"
if (-not $ready) { Write-Warning "API did not become ready; check var/logs/api.err.log" }
