Param(
    [switch]$SkipPackages
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=== Temporal Market Analysis Bot — Setup (Windows) ==="

# Go to project root (folder where this script lives)
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$VenvDir     = Join-Path $ProjectRoot ".venv"
$VenvPython  = Join-Path $VenvDir "Scripts\python.exe"

Write-Host ""
Write-Host "Project folder:" (Get-Location)

# 1) Create virtual environment if missing
if (-not (Test-Path $VenvPython)) {
    Write-Host ""
    Write-Host "Creating virtual environment in .venv ..."

    $created = $false
    $pythonCandidates = @("python", "py -3")

    foreach ($cmd in $pythonCandidates) {
        try {
            Write-Host "Trying:" $cmd "-m venv .venv"
            & $cmd -m venv ".venv"
            $created = $true
            break
        } catch {
            Write-Host "  Failed with" $cmd
        }
    }

    if (-not $created) {
        Write-Error "Could not create a virtual environment. Make sure Python 3.11+ is installed and on PATH."
        exit 1
    }
} else {
    Write-Host ""
    Write-Host "Virtual environment already exists at .venv"
}

if (-not (Test-Path $VenvPython)) {
    Write-Error "Virtual environment python not found at $VenvPython"
    exit 1
}

# 2) Install / upgrade packages
if (-not $SkipPackages) {
    if (-not (Test-Path "requirements.txt")) {
        Write-Error "requirements.txt not found in project root."
        exit 1
    }

    Write-Host ""
    Write-Host "Upgrading pip inside the virtual environment ..."
    & $VenvPython -m pip install --upgrade pip

    Write-Host ""
    Write-Host "Installing dependencies from requirements.txt ..."
    & $VenvPython -m pip install -r "requirements.txt"

    if (Test-Path "requirements-dev.txt") {
        Write-Host ""
        Write-Host "Installing development/test dependencies from requirements-dev.txt ..."
        & $VenvPython -m pip install -r "requirements-dev.txt"
    }
}
else {
    Write-Host ""
    Write-Host "Skipping package installation because -SkipPackages was provided."
}

Write-Host ""
Write-Host "=== Setup complete ==="
Write-Host ""
Write-Host "Next steps for NOOBS:"
Write-Host ""
Write-Host "1) Open PowerShell in this folder:"
Write-Host "     $ProjectRoot"
Write-Host ""
Write-Host "2) To run the desktop GUI, use the helper script:"
Write-Host "     .\run_gui.ps1"
Write-Host ""
Write-Host "3) To run the automated test suite:"
Write-Host "     python run_all_tests.py"
Write-Host ""
Write-Host "You can still activate the virtualenv manually if you want to tinker:"
Write-Host "     .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "If activation fails because of execution policy, run this once:"
Write-Host "     Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned"
Write-Host "and then try activating again."

