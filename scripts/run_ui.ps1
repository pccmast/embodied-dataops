# Run the Streamlit UI on the host (no container image required).
# The UI image itself is built on Linux/CI (ADR-0010).
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
if (Test-Path ".venv\Scripts\python.exe") {
  & ".venv\Scripts\python.exe" -m embodied_dataops.cli.main ui @args
} else {
  & uv run python -m embodied_dataops.cli.main ui @args
}
