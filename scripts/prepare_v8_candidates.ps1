param(
    [string]$Project = "C:\AMAZON ML CHALLENGE\Amazon-ML-Challenge-",
    [int]$ResearchPid = 0
)

$py = Join-Path $Project ".venv\Scripts\python.exe"
$src = Join-Path $Project "src"

if ($ResearchPid -gt 0) {
    Write-Host "waiting for research pid $ResearchPid"
    while (Get-Process -Id $ResearchPid -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 20
    }
}

Write-Host "building France suffix candidates"
& $py (Join-Path $src "build_france_suffix_pairs.py")
if ($LASTEXITCODE -ne 0) {
    throw "France suffix candidate build failed with exit code $LASTEXITCODE"
}

Write-Host "building V7 target collision pairs"
& $py (Join-Path $src "build_v7_collision_pairs.py")
if ($LASTEXITCODE -ne 0) {
    throw "V7 collision build failed with exit code $LASTEXITCODE"
}

Write-Host "V8 candidate preparation complete"
