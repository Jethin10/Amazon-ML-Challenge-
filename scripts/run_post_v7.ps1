param(
    [string]$Project = "C:\AMAZON ML CHALLENGE\Amazon-ML-Challenge-",
    [int]$V7Pid = 0,
    [int]$IndiaRetrievalPid = 0
)

$py = Join-Path $Project ".venv\Scripts\python.exe"
$src = Join-Path $Project "src"
$work = Join-Path $Project "work"

function Wait-ForProcess([int]$ProcessId, [string]$Label) {
    if ($ProcessId -le 0) { return }
    Write-Host "waiting for $Label pid $ProcessId"
    while (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 20
    }
    Write-Host "$Label process exited"
}

function Run-Step([string]$Label, [string[]]$Arguments) {
    Write-Host "running $Label"
    & $py @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

Wait-ForProcess $V7Pid "V7"
Wait-ForProcess $IndiaRetrievalPid "India retrieval"

Run-Step "V7 structural validation" @(
    (Join-Path $src "validate_large.py"),
    "--output-dir", (Join-Path $Project "output_v7_full_rerank")
)
Run-Step "India retrieval pair build" @(
    (Join-Path $src "build_v6_retrieval_pairs.py"),
    "--country", "india"
)
Run-Step "India retrieval cross-encoder scoring" @(
    (Join-Path $src "train_v6_reranker.py"),
    "--mode", "score",
    "--input-parquet", (Join-Path $work "v6_retrieval_india_audit3.parquet"),
    "--batch-size", "16"
)
Run-Step "joint V8 retrieval evaluation" @(
    (Join-Path $src "evaluate_v8_retrieval.py")
)
Run-Step "V7 entity-list decoding probe" @(
    (Join-Path $src "probe_v7_list_decoding.py")
)

Write-Host "post-V7 chain complete"
