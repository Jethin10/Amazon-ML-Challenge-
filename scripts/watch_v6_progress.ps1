$ErrorActionPreference = 'SilentlyContinue'
$repo = Split-Path -Parent $PSScriptRoot
$shards = Join-Path $repo 'work\v6_selected_shards'
$progressFile = Join-Path $shards 'progress.json'

while ($true) {
    Clear-Host
    Write-Host 'Amazon ML Challenge | V6 GPU reranker' -ForegroundColor Cyan
    Write-Host ('Updated: ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
    Write-Host 'Validated local audit3 F0.5: V5 0.93675 | reranked selected pairs 0.94728'
    Write-Host 'Full-pair research score: 0.95956 (requires an additional inference pass)'
    Write-Host ''
    $done = @(Get-ChildItem $shards -Filter 'selected_???.parquet').Count
    Write-Host "Completed output shards: $done / 20"
    if (Test-Path $progressFile) {
        try {
            $state = Get-Content $progressFile -Raw | ConvertFrom-Json
            Write-Host "Phase: $($state.phase)"
            if ($state.phase -eq 'scoring') {
                $number = [int]$state.shard + 1
                Write-Host "Current shard: $number / 20"
                Write-Host "Pairs scored in shard: $($state.scored_in_shard) / $($state.total_in_shard)"
                if ($state.scored_in_shard -gt 0) {
                    $rate = $state.scored_in_shard / [math]::Max($state.seconds_in_shard, 1)
                    $remainingShard = ($state.total_in_shard - $state.scored_in_shard) / $rate
                    $eta = $remainingShard + (19 - [int]$state.shard) * 110
                    Write-Host ('Estimated remaining: {0:N0} minutes' -f ($eta / 60))
                }
            }
            if ($state.phase -eq 'assembled') {
                Write-Host "Output ready: $($state.output_file)" -ForegroundColor Green
                break
            }
        } catch {
            Write-Host 'Reading the latest update...'
        }
    }
    Write-Host ''
    Write-Host 'Refreshes every 3 seconds. Press Ctrl+C to close this monitor.'
    Start-Sleep -Seconds 3
}
