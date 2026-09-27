param(
    [string]$Project = "C:\AMAZON ML CHALLENGE\Amazon-ML-Challenge-"
)

$progressFile = Join-Path $Project "work\v7_uncertain\progress.json"
while ($true) {
    Clear-Host
    Write-Host "Astralis V7 candidate expansion" -ForegroundColor Cyan
    Write-Host "Updated $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    if (Test-Path -LiteralPath $progressFile) {
        try {
            $state = Get-Content -LiteralPath $progressFile -Raw | ConvertFrom-Json
            Write-Host "Completed shards: $($state.leaves_done) / $($state.leaves_total)"
            Write-Host "Current shard: $($state.tag)"
            if ($null -ne $state.scored_in_leaf) {
                $fraction = if ($state.total_in_leaf -gt 0) {
                    [math]::Round(100 * $state.scored_in_leaf / $state.total_in_leaf, 1)
                } else { 0 }
                Write-Host "V5 candidate scoring: $($state.scored_in_leaf) / $($state.total_in_leaf) ($fraction%)"
                Write-Host "New uncertain pairs in this shard: $($state.extra_in_leaf)"
                Write-Host "Time in shard: $($state.seconds_in_leaf) seconds"
            }
            Write-Host "Saved uncertain pairs: $($state.extra_pairs)"
            Write-Host "Last progress: $([DateTimeOffset]::FromUnixTimeSeconds([long]$state.updated_unix).LocalDateTime)"
        } catch {
            Write-Host "Reading progress update..."
        }
    } else {
        Write-Host "Waiting for first progress update..."
    }
    Write-Host "Refreshes every 3 seconds. Close this window to stop watching."
    Start-Sleep -Seconds 3
}
