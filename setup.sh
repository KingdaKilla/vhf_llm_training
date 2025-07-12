# setup.sh - Setup Script für das VHF-LLM Training (Linux/Mac)
#!/bin/bash

echo "🏥 VHF-LLM Training Setup (Linux/Mac)"
echo "===================================="

# OS Detection
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
    PYTHON_CMD="python3"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="Linux"
    PYTHON_CMD="python3"
else
    OS="Unknown"
    PYTHON_CMD="python3"
fi

echo "🖥️  Betriebssystem: $OS"

# Python Version prüfen
if command -v $PYTHON_CMD &> /dev/null; then
    python_version=$($PYTHON_CMD --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
    echo "🐍 Python Version: $python_version"
    
    # Version vergleichen (vereinfacht)
    major=$(echo $python_version | cut -d. -f1)
    minor=$(echo $python_version | cut -d. -f2)
    
    if [[ $major -lt 3 ]] || [[ $major -eq 3 && $minor -lt 8 ]]; then
        echo "❌ Python 3.8+ erforderlich. Aktuelle Version: $python_version"
        exit 1
    fi
else
    echo "❌ Python nicht gefunden. Bitte installieren Sie Python 3.8+"
    if [[ "$OS" == "macOS" ]]; then
        echo "💡 Mac: brew install python3"
    else
        echo "💡 Linux: sudo apt-get install python3 python3-pip"
    fi
    exit 1
fi

# Virtual Environment erstellen
echo "📦 Erstelle Virtual Environment..."
$PYTHON_CMD -m venv vhf_training_env

# Environment aktivieren (OS-spezifisch)
if [[ "$OS" == "macOS" ]]; then
    source vhf_training_env/bin/activate
else
    source vhf_training_env/bin/activate
fi

# Requirements installieren
echo "⬇️  Installiere Requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# PyTorch GPU Support prüfen
echo "🎮 Prüfe GPU Support..."
$PYTHON_CMD -c "import torch; print(f'CUDA verfügbar: {torch.cuda.is_available()}')" 2>/dev/null
if $PYTHON_CMD -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "✅ GPU Support verfügbar"
else
    echo "⚠️  Nur CPU Training möglich"
    if [[ "$OS" == "macOS" ]]; then
        echo "💡 Mac: GPU Support über Metal Performance Shaders (MPS) möglich"
    fi
fi

echo ""
echo "🚀 Setup abgeschlossen!"
echo ""
echo "Verwendung ($OS):"
echo "1. Aktiviere Environment: source vhf_training_env/bin/activate"
echo "2. Einfaches Training:    python vhf_training_simple.py"
echo "3. Erweitert mit LoRA:    python vhf_training_advanced.py --use_lora"
echo ""
echo "Optionen für erweiterte Version:"
echo "--model_name microsoft/DialoGPT-medium  # Anderes Basis-Modell"
echo "--epochs 5                              # Mehr Epochen"
echo "--batch_size 2                          # Kleinere Batches für weniger RAM"
echo "--use_wandb                            # Logging mit Weights & Biases"
echo ""