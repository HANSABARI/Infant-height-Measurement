param(
    [int]$Port = 8000,
    [switch]$Reload
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$PythonExe = $env:JARAM_HEIGHT_WEB_PYTHON
if (-not $PythonExe) {
    $PythonExe = "D:\Anaconda3\envs\jaram-height-web\python.exe"
}

if (-not $env:H_ALIGN_AI_API_KEY) {
    throw "Set H_ALIGN_AI_API_KEY before starting the AI server. Example: `$env:H_ALIGN_AI_API_KEY = '<shared-ai-api-key>'"
}

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python was not found at: $PythonExe. Set JARAM_HEIGHT_WEB_PYTHON if the Conda env lives elsewhere."
}

if (-not $env:NO_ALBUMENTATIONS_UPDATE) {
    $env:NO_ALBUMENTATIONS_UPDATE = "1"
}

Set-Location $ProjectRoot

$UvicornArgs = @(
    "-m", "uvicorn",
    "app.main:app",
    "--host", "127.0.0.1",
    "--port", "$Port"
)

if ($Reload) {
    $UvicornArgs += "--reload"
}

& $PythonExe @UvicornArgs
