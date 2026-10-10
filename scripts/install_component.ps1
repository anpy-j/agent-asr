param(
    [Parameter(Mandatory=$true)][string]$Wheel,
    [Parameter(Mandatory=$true)][string]$Sha256,
    [string]$Destination = "$env:LOCALAPPDATA/agent-asr/runtime/0.1.0",
    [ValidateSet('whisper','r2t2','mcp','core')][string]$Backend = 'whisper',
    [string]$Python = 'python'
)
$ErrorActionPreference = 'Stop'
$wheelPath = (Resolve-Path -LiteralPath $Wheel).Path
$actual = (Get-FileHash -LiteralPath $wheelPath -Algorithm SHA256).Hash
if ($actual -ine $Sha256) { throw 'Wheel SHA-256 mismatch; installation stopped' }
if (Test-Path -LiteralPath $Destination) { throw 'Choose a new version directory; active runtime is never overwritten' }
& $Python -m venv $Destination
if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed' }
$runtimePython = Join-Path $Destination 'Scripts/python.exe'
$requirement = if ($Backend -eq 'core') { $wheelPath } else { "$wheelPath[$Backend]" }
& $runtimePython -m pip install $requirement
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed; do not switch the application to this runtime' }
& $runtimePython -m agent_asr --version
if ($LASTEXITCODE -ne 0) { throw 'Runtime smoke check failed' }
Write-Output "Runtime ready: $runtimePython"
