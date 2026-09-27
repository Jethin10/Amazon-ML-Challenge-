param(
    [string]$Project = "C:\AMAZON ML CHALLENGE\Amazon-ML-Challenge-",
    [int]$PostV7Pid = 0
)

$py = Join-Path $Project ".venv\Scripts\python.exe"
$src = Join-Path $Project "src"

if ($PostV7Pid -gt 0) {
    Write-Host "waiting for post-V7 pid $PostV7Pid"
    while (Get-Process -Id $PostV7Pid -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 20
    }
}

Write-Host "running bounded anchor propagation audit"
& $py (Join-Path $src "probe_anchor_propagation_routes.py")
if ($LASTEXITCODE -ne 0) {
    throw "anchor propagation audit failed with exit code $LASTEXITCODE"
}

Write-Host "running France suffix sensitivity audit"
& $py (Join-Path $src "french_suffix_sensitivity.py")
if ($LASTEXITCODE -ne 0) {
    throw "France suffix sensitivity audit failed with exit code $LASTEXITCODE"
}

Write-Host "post-V7 research chain complete"
