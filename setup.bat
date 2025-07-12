# setup.bat - Windows Batch Setup
cat > setup.bat << 'EOF'
@echo off
echo 🏥 VHF-LLM Training Setup (Windows)
echo ===================================

echo 🖥️  Betriebssystem: Windows

REM Python Version prüfen
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python nicht gefunden. Bitte installieren Sie Python 3.8+
    echo 💡 Windows: https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set python_version=%%i
echo 🐍 Python Version: %python_version%

REM Virtual Environment erstellen
echo 📦 Erstelle Virtual Environment...
python -m venv vhf_training_env

REM Environment aktivieren
echo 🔄 Aktiviere Environment...
call vhf_training_env\Scripts\activate.bat

REM Requirements installieren
echo ⬇️  Installiere Requirements...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM PyTorch GPU Support prüfen
echo 🎮 Prüfe GPU Support...
python -c "import torch; print('CUDA verfügbar:', torch.cuda.is_available())" 2>nul
if %errorlevel% neq 0 (
    echo ⚠️  PyTorch Installation überprüfen
)

echo.
echo 🚀 Setup abgeschlossen!
echo.
echo Verwendung (Windows):
echo 1. Aktiviere Environment: vhf_training_env\Scripts\activate.bat
echo 2. Einfaches Training:    python vhf_training_simple.py
echo 3. Erweitert mit LoRA:    python vhf_training_advanced.py --use_lora
echo.
pause
EOF

# setup.ps1 - Windows PowerShell Setup
cat > setup.ps1 << 'EOF'
# VHF-LLM Training Setup (Windows PowerShell)
Write-Host "🏥 VHF-LLM Training Setup (Windows PowerShell)" -ForegroundColor Green
Write-Host "=============================================="

Write-Host "🖥️  Betriebssystem: Windows (PowerShell)" -ForegroundColor Cyan

# Execution Policy prüfen
$policy = Get-ExecutionPolicy
if ($policy -eq "Restricted") {
    Write-Host "⚠️  PowerShell Execution Policy ist 'Restricted'" -ForegroundColor Yellow
    Write-Host "💡 Führen Sie aus: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Yellow
    Write-Host "   Oder verwenden Sie setup.bat stattdessen"
    Read-Host "Drücken Sie Enter zum Fortfahren"
}

# Python Version prüfen
try {
    $pythonVersion = python --version 2>&1
    Write-Host "🐍 Python Version: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python nicht gefunden. Bitte installieren Sie Python 3.8+" -ForegroundColor Red
    Write-Host "💡 Windows: https://www.python.org/downloads/" -ForegroundColor Yellow
    Read-Host "Drücken Sie Enter zum Beenden"
    exit 1
}

# Virtual Environment erstellen
Write-Host "📦 Erstelle Virtual Environment..." -ForegroundColor Yellow
python -m venv vhf_training_env

# Environment aktivieren
Write-Host "🔄 Aktiviere Environment..." -ForegroundColor Yellow
& "vhf_training_env\Scripts\Activate.ps1"

# Requirements installieren
Write-Host "⬇️  Installiere Requirements..." -ForegroundColor Yellow
python -m pip install --upgrade pip
pip install -r requirements.txt

# PyTorch GPU Support prüfen
Write-Host "🎮 Prüfe GPU Support..." -ForegroundColor Yellow
try {
    $cudaAvailable = python -c "import torch; print(torch.cuda.is_available())" 2>$null
    Write-Host "CUDA verfügbar: $cudaAvailable" -ForegroundColor Green
} catch {
    Write-Host "⚠️  PyTorch Installation überprüfen" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🚀 Setup abgeschlossen!" -ForegroundColor Green
Write-Host ""
Write-Host "Verwendung (Windows PowerShell):" -ForegroundColor Cyan
Write-Host "1. Aktiviere Environment: vhf_training_env\Scripts\Activate.ps1"
Write-Host "2. Einfaches Training:    python vhf_training_simple.py"
Write-Host "3. Erweitert mit LoRA:    python vhf_training_advanced.py --use_lora"
Write-Host ""
Read-Host "Drücken Sie Enter zum Beenden"
EOF

echo "📋 Windows Setup-Dateien erstellt:"
echo "   - setup.bat (Windows Command Prompt)"
echo "   - setup.ps1 (Windows PowerShell)"