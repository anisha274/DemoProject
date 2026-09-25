$ErrorActionPreference = "Stop"

Set-Location "D:\TrafficProjectFresh"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Running pilot..."
.\.venv\Scripts\python.exe -X utf8 -u -m traffic_pipeline run --config configs\pilot.yaml --video data\videos\2011_09_26_drive_0001_sync.mp4 --run-id pilot_0001 --resume

if ($LASTEXITCODE -ne 0) {
    Write-Error "Pilot failed."
    exit $LASTEXITCODE
}

Write-Host "Output directory: D:\TrafficProjectFresh\outputs\pilot_0001"
