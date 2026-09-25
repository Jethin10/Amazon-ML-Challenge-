$ProgressPath = Join-Path $PSScriptRoot 'output_v5\progress.txt'
$ErrorActionPreference = 'Continue'

while ($true) {
    Clear-Host
    Write-Host 'ASTRALIS V5 - LIVE PROGRESS' -ForegroundColor Cyan
    Write-Host 'Training and evaluation are complete; test inference is live.'
    Write-Host 'This window refreshes every 3 seconds. Close it whenever you like.'
    Write-Host ''
    if (Test-Path -LiteralPath $ProgressPath) {
        Get-Content -LiteralPath $ProgressPath
    } else {
        Write-Host 'Waiting for the progress file...'
    }
    Write-Host ''
    $Inference = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object { $_.CommandLine -match 'predict_v5.py' } |
        ForEach-Object { Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue } |
        Sort-Object WorkingSet64 -Descending)[0]
    if ($Inference) {
        $MemoryGB = [math]::Round($Inference.WorkingSet64 / 1GB, 1)
        Write-Host "Inference: RUNNING (PID $($Inference.Id), RAM $MemoryGB GB)" -ForegroundColor Green
    } else {
        Write-Host 'Inference batch process: restarting or finished'
    }
    $RunnerLog = Join-Path $PSScriptRoot 'work\v5_finish.log'
    if (Test-Path -LiteralPath $RunnerLog) {
        Write-Host ''
        Write-Host 'LATEST RUNNER OUTPUT' -ForegroundColor Cyan
        Get-Content -LiteralPath $RunnerLog -Tail 4
    }
    $Gpu = & nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>$null
    if ($Gpu) {
        Write-Host "RTX 2050: GPU usage $Gpu (percent, MB). GPU is idle during LightGBM prediction."
    }
    $StatusPath = Join-Path $PSScriptRoot 'output_v5\finalization_status.json'
    if (Test-Path -LiteralPath $StatusPath) {
        $Status = Get-Content -LiteralPath $StatusPath -Raw | ConvertFrom-Json
        if ($Status.phase -eq 'ready') {
            Write-Host 'VALIDATED AND READY TO UPLOAD' -ForegroundColor Green
            break
        }
        if ($Status.phase -eq 'failed') {
            Write-Host 'Finalization failed; check this task for details.' -ForegroundColor Red
            break
        }
    }
    Start-Sleep -Seconds 3
}

Write-Host 'Press Enter to close this window.'
[void][Console]::ReadLine()
