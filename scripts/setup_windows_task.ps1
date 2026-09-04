<#
=============================================================================
 setup_windows_task.ps1 — Register the weekly knowledge-base refresh as a
 Windows Scheduled Task (fires every Monday 18:00 local time, survives reboots).

 This is the recommended automation on Windows: unlike the always-on
 scripts/scheduler.py process, a Scheduled Task keeps working after you close
 the terminal or restart the machine, and it wakes the computer if needed.

 USAGE (run from a normal PowerShell window, inside the project folder):

     powershell -ExecutionPolicy Bypass -File scripts\setup_windows_task.ps1
     powershell -ExecutionPolicy Bypass -File scripts\setup_windows_task.ps1 -At 18:00 -Day Monday
     powershell -ExecutionPolicy Bypass -File scripts\setup_windows_task.ps1 -Remove   # delete the task

 After registering, verify it in Task Scheduler (taskschd.msc) under the name
 "TakshashilaKB-WeeklyUpdate", or run it on demand with:
     Start-ScheduledTask -TaskName "TakshashilaKB-WeeklyUpdate"
=============================================================================
#>

param(
    [string]$Day    = "Monday",               # day of week
    [string]$At     = "18:00",                # HH:mm, 24h local time
    [string]$TaskName = "TakshashilaKB-WeeklyUpdate",
    [switch]$Remove                            # remove the task instead of creating it
)

$ErrorActionPreference = "Stop"

# Project root = parent of this script's folder.
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$RunBat      = Join-Path $ScriptDir "run_update.bat"

if ($Remove) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Removed scheduled task '$TaskName'." -ForegroundColor Yellow
    } else {
        Write-Host "No scheduled task named '$TaskName' found." -ForegroundColor Yellow
    }
    return
}

if (-not (Test-Path $RunBat)) {
    throw "Could not find $RunBat. Run this from inside the project."
}

Write-Host "Registering weekly knowledge-base update…" -ForegroundColor Cyan
Write-Host "  Project : $ProjectRoot"
Write-Host "  Runner  : $RunBat"
Write-Host "  When    : every $Day at $At (local time)"

# Action: run the batch file from the project root.
$action = New-ScheduledTaskAction -Execute "cmd.exe" `
    -Argument "/c `"$RunBat`"" -WorkingDirectory $ProjectRoot

# Trigger: weekly on the chosen day/time.
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $Day -At $At

# Settings: allow on battery, wake to run, retry a few times if it misfires.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -WakeToRun `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 10) `
    -ExecutionTimeLimit (New-TimeSpan -Hours 3)

# Run as the current user, only when logged on (no stored password needed).
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERNAME" -LogonType Interactive -RunLevel Limited

# Replace any existing task with the same name.
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Principal $principal `
    -Description "Weekly incremental refresh of the Takshashila RAG knowledge base (website + Commit KB)." | Out-Null

Write-Host ""
Write-Host "Done. Task '$TaskName' registered." -ForegroundColor Green
Write-Host "  Verify : taskschd.msc  (or)  Get-ScheduledTask -TaskName '$TaskName'"
Write-Host "  Run now: Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "  Remove : powershell -ExecutionPolicy Bypass -File scripts\setup_windows_task.ps1 -Remove"