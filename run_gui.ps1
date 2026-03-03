Param(
    [string]$Symbol,
    [string]$Interval,
    [int]$Bars
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=== Temporal Market Analysis Bot — Run GUI (Windows) ==="

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host ""
    Write-Host "Virtual environment not found at .venv."
    Write-Host "Run setup.ps1 first:"
    Write-Host "    .\setup.ps1"
    exit 1
}

Write-Host ""
Write-Host "Starting GUI using virtual environment ..."

$argsList = @("main_gui.py")
if ($Symbol)   { $argsList += $Symbol }
if ($Interval) { $argsList += $Interval }
if ($Bars)     { $argsList += $Bars }

& $VenvPython @argsList

