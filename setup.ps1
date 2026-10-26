param([switch]$Cpu)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (!(Test-Path '.venv/Scripts/python.exe')) { python -m venv .venv }
if ($LASTEXITCODE -ne 0) { throw 'Could not create virtual environment' }
$python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed' }
if ($Cpu) { $index = 'https://download.pytorch.org/whl/cpu' }
else { $index = 'https://download.pytorch.org/whl/cu124' }
& $python -m pip install torch==2.6.0 --index-url $index
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
& $python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& $python -m project doctor
