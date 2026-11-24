# Launch Jebrex AI Windows Desktop App using the project's local Python environment
$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$VenvPy = "D:\NourAI\.venv\Scripts\python.exe"
if (-not (Test-Path $VenvPy)) {
    $VenvPy = (Get-Command python).Source
}
$env:PYTHONPATH = $RepoRoot.Path
& $VenvPy (Join-Path $RepoRoot "apps\windows\jebrex_desktop.py") @args
