# gateway-service.ps1 - Self-healing supervisor for the LiteLLM slim gateway.
#
# Run by the "LiteLLMGateway" scheduled task at logon (installed via
# .\start.ps1 --install-service). Launches the gateway in the foreground and
# relaunches it if it exits, so a crash or kill is recovered by the supervisor
# itself. This does NOT rely on Task Scheduler's RestartOnFailure, which is
# unreliable for Interactive / logon-trigger tasks (it does not fire when the
# task action exits with a non-zero code in a logged-on session).
#
# Gateway stdout/stderr go to litellm.stdout.log / litellm.stderr.log (overwritten
# per launch, same as start.ps1). Supervisor lifecycle events append to
# litellm.service.log so restart cycles survive across launches.

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$Port = 9374
$Root = $PSScriptRoot
$Config = Join-Path $Root "config.local.yaml"
$StdOutLog = Join-Path $Root "litellm.stdout.log"
$StdErrLog = Join-Path $Root "litellm.stderr.log"
$ServiceLog = Join-Path $Root "litellm.service.log"

function Write-ServiceLog([string]$Message) {
    Add-Content -Path $ServiceLog -Value "[$(Get-Date -Format 'o')] $Message" -Encoding utf8
}

$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) { $uv = Join-Path $env:USERPROFILE ".local\bin\uv.exe" }
if (-not (Test-Path $uv)) { Write-ServiceLog "uv not found at $uv"; exit 1 }
if (-not (Test-Path $Config)) { Write-ServiceLog "config not found: $Config"; exit 1 }

# --- Isolate the environment from any inherited session pollution ---
# When launched from an agent shell (Hermes, Claude Code, etc.) the child
# process may inherit PYTHONPATH / PYTHONHOME / VIRTUAL_ENV pointing at a
# DIFFERENT Python, so `uv run litellm` can resolve the wrong pydantic and
# crash with "ModuleNotFoundError: pydantic_core._pydantic_core". Pin the
# gateway's Python environment explicitly so it always uses this repo's
# .venv regardless of how the supervisor was started.
foreach ($envVar in @('PYTHONPATH', 'PYTHONHOME', 'PYTHONUSERBASE', 'VIRTUAL_ENV', 'VIRTUAL_ENV_PROMPT')) {
    [Environment]::SetEnvironmentVariable($envVar, $null, 'Process')
}
$RootVenvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $RootVenvPython)) { Write-ServiceLog ".venv not found: $RootVenvPython (run `uv sync` or `python -m venv .venv` first)"; exit 1 }
Write-ServiceLog "pinned gateway python: $RootVenvPython"

# Keys are embedded in config.local.yaml; the ARK_API_KEY_* env vars are no
# longer required. Keep a soft warning (not a hard exit) if they are absent.
$missing = @()
foreach ($i in 1..3) {
    if (-not [Environment]::GetEnvironmentVariable("ARK_API_KEY_$i")) { $missing += "ARK_API_KEY_$i" }
}
if ($missing.Count -gt 0) {
    Write-ServiceLog "warn: env vars not set (keys are in config.local.yaml): $($missing -join ', ')"
}

# Use the repo's own venv launcher directly. `uv run litellm` was failing in
# the scheduled-task context (uv can't resolve the project env without a
# parent shell), while .venv\Scripts\litellm.exe starts cleanly.
$GatewayExe = Join-Path $Root ".venv\Scripts\litellm.exe"
if (-not (Test-Path $GatewayExe)) { Write-ServiceLog "litellm.exe not found: $GatewayExe"; exit 1 }
$runArgs = @("--config", $Config, "--port", "$Port", "--slim")
# Relaunch delay: 5s after a run that lasted >= 10s (a healthy run, so recover
# fast); doubles up to 60s after a fast crash so a persistent failure can't
# hammer. Reset to 5s as soon as a run stays up.
$delay = 5

while ($true) {
    if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
        Write-ServiceLog "port $Port already in use; another instance is serving. Sleeping ${delay}s."
        Start-Sleep -Seconds $delay
        continue
    }

    Write-ServiceLog "starting slim gateway: $GatewayExe $($runArgs -join ' ')"
    $start = Get-Date
    $proc = Start-Process -FilePath $GatewayExe -ArgumentList $runArgs `
        -WorkingDirectory $Root -WindowStyle Hidden `
        -RedirectStandardOutput $StdOutLog -RedirectStandardError $StdErrLog `
        -PassThru -Wait
    $uptime = [int]((Get-Date) - $start).TotalSeconds
    if ($uptime -ge 10) { $delay = 5 } else { $delay = [Math]::Min($delay * 2, 60) }
    Write-ServiceLog "gateway exited with code $($proc.ExitCode) after ${uptime}s; relaunching in ${delay}s"
    Start-Sleep -Seconds $delay
}
