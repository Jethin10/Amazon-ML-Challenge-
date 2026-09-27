param(
    [string]$Project = "C:\AMAZON ML CHALLENGE\Amazon-ML-Challenge-\",
    [int]$ExtractPid = 0
)

# Waits for the V7 uncertain-band extraction to exit, then runs the
# full-band reranker and the structural validator. Safe to run unattended:
# predict_v7_full.py refuses to start if any of the 20 batches is incomplete.

$py = Join-Path $Project ".venv\Scripts\python.exe"
$src = Join-Path $Project "src"
$log = Join-Path $Project "v7_chain.log"

function Say($m) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m
    Write-Host $line
    Add-Content -LiteralPath $log -Value $line
}

if ($ExtractPid -gt 0) {
    Say "waiting for extraction pid $ExtractPid"
    while (Get-Process -Id $ExtractPid -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 20
    }
    Say "extraction process exited"
}

Say "running predict_v7_full.py"
& $py (Join-Path $src "predict_v7_full.py") 2>&1 |
    Tee-Object -FilePath (Join-Path $Project "v7_predict.log")
$code = $LASTEXITCODE
Say "predict_v7_full.py exit code $code"
if ($code -ne 0) { exit $code }

Say "running validate_large.py"
& $py (Join-Path $src "validate_large.py") `
    --output-dir (Join-Path $Project "output_v7_full_rerank") 2>&1 |
    Tee-Object -FilePath (Join-Path $Project "v7_validate.log")
Say "validate exit code $LASTEXITCODE"
