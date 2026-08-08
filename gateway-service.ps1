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

$missing = @()
foreach ($i in 1..3) {
    if (-not [Environment]::GetEnvironmentVariable("ARK_API_KEY_$i")) { $missing += "ARK_API_KEY_$i" }
}
if ($missing.Count -gt 0) {
    Write-ServiceLog "missing env vars: $($missing -join ', ')"
    exit 1
}

$runArgs = @("run", "litellm", "--config", $Config, "--port", "$Port", "--slim")
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

    Write-ServiceLog "starting slim gateway: uv $($runArgs -join ' ')"
    $start = Get-Date
    $proc = Start-Process -FilePath $uv -ArgumentList $runArgs `
        -WorkingDirectory $Root -WindowStyle Hidden `
        -RedirectStandardOutput $StdOutLog -RedirectStandardError $StdErrLog `
        -PassThru -Wait
    $uptime = [int]((Get-Date) - $start).TotalSeconds
    if ($uptime -ge 10) { $delay = 5 } else { $delay = [Math]::Min($delay * 2, 60) }
    Write-ServiceLog "gateway exited with code $($proc.ExitCode) after ${uptime}s; relaunching in ${delay}s"
    Start-Sleep -Seconds $delay
}
