param([string]$Config = 'configs/laptop.json', [string]$Run = 'runs/laptop', [string]$Device = 'auto')
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
foreach ($stage in @('prepare', 'basic', 'mt5', 'llama', 'gemma', 'report')) {
    & $python -m project $stage --config $Config --run $Run --device $Device
    if ($LASTEXITCODE -ne 0) {
        if (Test-Path "$Run/data/manifest.json") { & $python -m project report --run $Run }
        throw "Stage $stage failed. Resolve the error and rerun this stage; completed outputs are retained."
    }
}
& $python scripts/verify_results.py $Run
if ($LASTEXITCODE -ne 0) { throw 'Result verification failed' }
& $python scripts/summarize_results.py $Run
if ($LASTEXITCODE -ne 0) { throw 'Analysis generation failed' }
