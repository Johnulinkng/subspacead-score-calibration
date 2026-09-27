param(
    [Parameter(Mandatory = $true)]
    [string]$Url,

    [Parameter(Mandatory = $true)]
    [string]$Destination,

    [Parameter(Mandatory = $true)]
    [long]$ExpectedBytes,

    [int]$MaxAttempts = 20,

    [int]$RetryDelaySeconds = 5
)

$ErrorActionPreference = "Stop"

$destinationPath = [System.IO.Path]::GetFullPath($Destination)
$destinationDirectory = Split-Path -Parent $destinationPath
New-Item -ItemType Directory -Path $destinationDirectory -Force | Out-Null

for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
    $currentBytes = if (Test-Path -LiteralPath $destinationPath) {
        (Get-Item -LiteralPath $destinationPath).Length
    } else {
        0L
    }

    if ($currentBytes -eq $ExpectedBytes) {
        Write-Host "Download already complete: $currentBytes bytes"
        exit 0
    }
    if ($currentBytes -gt $ExpectedBytes) {
        throw "Destination is larger than expected ($currentBytes > $ExpectedBytes): $destinationPath"
    }

    Write-Host "Attempt $attempt/$MaxAttempts; resuming at byte $currentBytes"
    & curl.exe `
        --fail `
        --location `
        --continue-at - `
        --connect-timeout 30 `
        --speed-time 120 `
        --speed-limit 1024 `
        --output $destinationPath `
        $Url

    $curlExit = $LASTEXITCODE
    $newBytes = if (Test-Path -LiteralPath $destinationPath) {
        (Get-Item -LiteralPath $destinationPath).Length
    } else {
        0L
    }

    if ($curlExit -eq 0 -and $newBytes -eq $ExpectedBytes) {
        Write-Host "Download complete: $newBytes bytes"
        exit 0
    }

    Write-Warning "curl exit=$curlExit; current size=$newBytes; expected=$ExpectedBytes"
    if ($attempt -lt $MaxAttempts) {
        Start-Sleep -Seconds $RetryDelaySeconds
    }
}

throw "Download did not complete after $MaxAttempts attempts: $destinationPath"
