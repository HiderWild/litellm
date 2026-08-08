# start.ps1 - LiteLLM Gateway for Claude Code -> Volcengine Ark Coding Plan
#
# Usage:
#   .\start.ps1            start the full proxy (detached; logs to litellm.*.log)
#   .\start.ps1 --lite     start in slim/lightweight mode (forwards litellm --slim)
#   .\start.ps1 --stop     stop the running gateway
#   .\start.ps1 --editable run the local litellm source (uv --with-editable .)
#   .\start.ps1 --no-proxy launch with HTTP(S)_PROXY / ALL_PROXY cleared
#
# --lite maps to litellm's --slim flag, which only exists on the
# litellm_single_model_multi_deploy_slimming branch. On any other checkout
# --lite is refused with a clear message instead of launching a doomed process.
#
# --editable overlays an editable install of the repo so edits to litellm/
# take effect on the next restart. It is not a hot reload; the running gateway
# still needs .\start.ps1 --stop and a rerun to load changed code.
#
# The gateway runs detached and hidden so a console / terminal crash cannot
# tear down the uv -> python process tree. A startup check waits for the port
# to come up and reports the result; tail the log files for live output.
#
# Requires: ARK_API_KEY_1, ARK_API_KEY_2, ARK_API_KEY_3 environment variables.

[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [object[]]$Rest
)

$ErrorActionPreference = "Stop"
$Port = 9374
$Root = $PSScriptRoot
$Config = Join-Path $Root "config.local.yaml"
$SlimServer = Join-Path $Root "litellm/proxy/slim_server.py"
$StdOutLog = Join-Path $Root "litellm.stdout.log"
$StdErrLog = Join-Path $Root "litellm.stderr.log"

function Find-GatewayProcess {
    $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if (-not $conns) { return $null }
    foreach ($p in ($conns | Select-Object -ExpandProperty OwningProcess -Unique)) {
        $pr = Get-Process -Id $p -ErrorAction SilentlyContinue
        if ($pr) { return $pr }
    }
    return $null
}

$slimMode = $false
$stopRequested = $false
$installService = $false
$uninstallService = $false
$editableMode = $false
$noProxy = $false
$unknown = New-Object System.Collections.Generic.List[string]
foreach ($t in $Rest) {
    $norm = "$t".TrimStart('-').ToLower()
    switch ($norm) {
        'lite' { $slimMode = $true }
        'slim' { $slimMode = $true }
        'stop' { $stopRequested = $true }
        'install-service' { $installService = $true }
        'uninstall-service' { $uninstallService = $true }
        'editable' { $editableMode = $true }
        'no-proxy' { $noProxy = $true }
        'noproxy' { $noProxy = $true }
        default { $unknown.Add("$t") }
    }
}

if ($stopRequested) {
    $task = Get-ScheduledTask -TaskName "LiteLLMGateway" -ErrorAction SilentlyContinue
    if ($task -and $task.State -eq "Running") {
        Stop-ScheduledTask -TaskName "LiteLLMGateway"
        Write-Host "Stopped supervisor task 'LiteLLMGateway' so it will not auto-relaunch." -ForegroundColor Cyan
    }
    $proc = Find-GatewayProcess
    if (-not $proc) {
        Write-Host "No gateway listening on port $Port." -ForegroundColor Yellow
        return
    }
    Stop-Process -Id $proc.Id -Force
    Write-Host "Stopped gateway (pid $($proc.Id))." -ForegroundColor Green
    return
}

if ($installService -or $uninstallService) {
    $TaskName = "LiteLLMGateway"
    if ($uninstallService) {
        try {
            Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction Stop
            Write-Host "Uninstalled scheduled task '$TaskName'." -ForegroundColor Green
        } catch {
            Write-Host "Task '$TaskName' not found or already removed." -ForegroundColor Yellow
        }
        return
    }

    $Supervisor = Join-Path $Root "gateway-service.ps1"
    if (-not (Test-Path $Supervisor)) {
        Write-Host "Supervisor script not found: $Supervisor" -ForegroundColor Red
        exit 1
    }

    $User = "$env:USERDOMAIN\$env:USERNAME"
    $Action = New-ScheduledTaskAction `
        -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$Supervisor`"" `
        -WorkingDirectory $Root
    $Trigger = New-ScheduledTaskTrigger -AtLogOn -User $User
    $Settings = New-ScheduledTaskSettingsSet `
        -MultipleInstances IgnoreNew `
        -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
        -ExecutionTimeLimit ([TimeSpan]::Zero) `
        -DontStopOnIdleEnd `
        -StartWhenAvailable
    $Principal = New-ScheduledTaskPrincipal -UserId $User -LogonType Interactive -RunLevel Limited

    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Set-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Principal $Principal | Out-Null
        Write-Host "Updated scheduled task '$TaskName'." -ForegroundColor Green
    } else {
        Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Principal $Principal `
            -Description "Auto-start the LiteLLM slim gateway (ark-code, glm-5.2 x 3 keys) at logon and restart it on failure." | Out-Null
        Write-Host "Installed scheduled task '$TaskName'." -ForegroundColor Green
    }

    Write-Host "Trigger: at logon for $User. Restart-on-failure: up to 999x, every 1 min." -ForegroundColor Cyan
    Write-Host "Start now without rebooting:  Start-ScheduledTask -TaskName '$TaskName'" -ForegroundColor Cyan
    Write-Host "Uninstall:  .\start.ps1 --uninstall-service" -ForegroundColor Cyan
    return
}

if ($unknown.Count -gt 0) {
    Write-Host "Ignoring unrecognized arguments: $($unknown -join ' ')" -ForegroundColor DarkGray
}

$missing = @()
foreach ($i in 1..3) {
    $varName = "ARK_API_KEY_$i"
    if (-not [Environment]::GetEnvironmentVariable($varName)) {
        $missing += $varName
    }
}
if ($missing.Count -gt 0) {
    Write-Host "Missing environment variables:" -ForegroundColor Red
    $missing | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    Write-Host ""
    Write-Host "Set them first:" -ForegroundColor Yellow
    Write-Host '  $env:ARK_API_KEY_1="your-key-1"' -ForegroundColor Yellow
    Write-Host '  $env:ARK_API_KEY_2="your-key-2"' -ForegroundColor Yellow
    Write-Host '  $env:ARK_API_KEY_3="your-key-3"' -ForegroundColor Yellow
    exit 1
}

if (Find-GatewayProcess) {
    Write-Host "Gateway already running on port $Port. Run .\start.ps1 --stop first." -ForegroundColor Yellow
    return
}

$uv = (Get-Command uv -ErrorAction Stop).Source
if (-not (Test-Path $Config)) {
    Write-Host "Config not found: $Config" -ForegroundColor Red
    exit 1
}

$modeLabel = "full"
$litellmArgs = @("run")
if ($editableMode) {
    $litellmArgs += @("--with-editable", $Root)
}
$litellmArgs += @("litellm", "--config", $Config, "--port", "$Port")
if ($slimMode) {
    if (-not (Test-Path $SlimServer)) {
        $branch = git -C $Root rev-parse --abbrev-ref HEAD 2>$null
        if (-not $branch) { $branch = "this checkout" }
        Write-Host "Lite (--slim) mode is not available on '$branch'." -ForegroundColor Red
        Write-Host "The slim server only exists on branch 'litellm_single_model_multi_deploy_slimming'." -ForegroundColor Red
        Write-Host "Switch to that branch, or run without --lite for the full proxy." -ForegroundColor Yellow
        exit 1
    }
    $litellmArgs += "--slim"
    $modeLabel = "slim"
}

Write-Host "Starting LiteLLM Gateway on port $Port ($modeLabel, detached)..." -ForegroundColor Green
Write-Host "  Claude Code: http://localhost:$Port/v1/messages" -ForegroundColor Cyan
Write-Host "  Model: ark-code (glm-5.2 x 3 keys)" -ForegroundColor Cyan
if ($editableMode) { Write-Host "  Source: local editable (uv --with-editable $Root)" -ForegroundColor Cyan }
if ($noProxy) { Write-Host "  Proxy env: HTTP(S)_PROXY / ALL_PROXY cleared for this process" -ForegroundColor Cyan }
Write-Host ""

$proxyVars = @('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY')
$savedProxy = @{}
if ($noProxy) {
    foreach ($v in $proxyVars) {
        $savedProxy[$v] = [Environment]::GetEnvironmentVariable($v, 'Process')
        [Environment]::SetEnvironmentVariable($v, $null, 'Process')
    }
}
try {
    $proc = Start-Process -FilePath $uv `
        -ArgumentList $litellmArgs `
        -WorkingDirectory $Root `
        -WindowStyle Hidden `
        -RedirectStandardOutput $StdOutLog `
        -RedirectStandardError $StdErrLog `
        -PassThru
} finally {
    if ($noProxy) {
        foreach ($v in $proxyVars) {
            [Environment]::SetEnvironmentVariable($v, $savedProxy[$v], 'Process')
        }
    }
}

Write-Host "Launched uv pid $($proc.Id). Waiting for port $Port..." -ForegroundColor Cyan

$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 2
    if ($proc.HasExited) { break }
    if (Find-GatewayProcess) { $ready = $true; break }
}

if (-not $ready) {
    Write-Host "Gateway did not come up on port $Port." -ForegroundColor Red
    if ($proc.HasExited) {
        Write-Host "Process exited (code $($proc.ExitCode))." -ForegroundColor Red
    }
    Write-Host "Last lines of stderr:" -ForegroundColor Yellow
    Get-Content $StdErrLog -Tail 20 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray }
    Write-Host "Full logs: $StdOutLog , $StdErrLog" -ForegroundColor Yellow
    return
}

Write-Host "Gateway is up on port $Port (mode: $modeLabel)." -ForegroundColor Green
Write-Host "Logs:" -ForegroundColor Cyan
Write-Host "  $StdOutLog" -ForegroundColor Cyan
Write-Host "  $StdErrLog" -ForegroundColor Cyan
Write-Host ""
Write-Host "Tail logs:" -ForegroundColor Yellow
Write-Host "  Get-Content '$StdOutLog' -Wait" -ForegroundColor Yellow
Write-Host "  Get-Content '$StdErrLog' -Wait" -ForegroundColor Yellow
Write-Host ""
Write-Host "Stop:" -ForegroundColor Yellow
Write-Host "  .\start.ps1 --stop" -ForegroundColor Yellow
