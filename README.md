# VHF-LLM Training Guide 🏥

Komplettes Setup für das Training eines LLMs mit dem VHF-Überwachungssystem Datensatz.

## 📋 Inhaltsverzeichnis

- [Übersicht](#übersicht)
- [Installation](#installation)
- [Verwendung](#verwendung)
- [Scripts](#scripts)
- [Konfiguration](#konfiguration)
- [Troubleshooting](#troubleshooting)

## 🎯 Übersicht

Dieses Repository enthält Scripts zum Training eines Large Language Models (LLM) für ein VHF-Überwachungssystem. Der Datensatz umfasst 300+ Einträge mit medizinischen Dialogen und Anweisungen.

**Verfügbare Ansätze:**
- **Einfach**: GPT-2 basiertes Training für schnelle Experimente
- **Erweitert**: LoRA-basiertes Fine-tuning für effiziente Anpassung größerer Modelle

## 🚀 Installation

### Automatisches Setup

#### 🐧 Linux / 🍎 macOS
```bash
# Repository clonen oder Dateien herunterladen
git clone <repository-url>
cd vhf-llm-training

# Setup Script ausführen
chmod +x setup.sh
./setup.sh
```

#### 🪟 Windows

**Option 1: Command Prompt (CMD)**
```cmd
# Repository clonen oder Dateien herunterladen
git clone <repository-url>
cd vhf-llm-training

# Setup ausführen
setup.bat
```

**Option 2: PowerShell**
```powershell
# Repository clonen oder Dateien herunterladen
git clone <repository-url>
cd vhf-llm-training

# PowerShell Execution Policy setzen (falls nötig)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Setup ausführen
.\setup.ps1
```

**Option 3: Windows Subsystem for Linux (WSL)**
```bash
# Wie Linux verwenden
chmod +x setup.sh
./setup.sh
```

### Manuelle Installation

#### 🐧 Linux / 🍎 macOS
```bash
# Virtual Environment erstellen
python3 -m venv vhf_training_env
source vhf_training_env/bin/activate

# Requirements installieren
pip install -r requirements.txt
```

#### 🪟 Windows
```cmd
# Virtual Environment erstellen
python -m venv vhf_training_env

# Environment aktivieren (CMD)
vhf_training_env\Scripts\activate.bat

# ODER Environment aktivieren (PowerShell)
vhf_training_env\Scripts\Activate.ps1

# Requirements installieren
pip install -r requirements.txt
```

### GPU Setup (optional)

#### CUDA (NVIDIA GPUs)
```bash
# CUDA Version prüfen
nvidia-smi

# PyTorch mit CUDA Support (alle Plattformen)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 🍎 macOS (Apple Silicon)
```bash
# Metal Performance Shaders (MPS) Support
pip install torch torchvision torchaudio

# Prüfen ob MPS verfügbar
python -c "import torch; print('MPS verfügbar:', torch.backends.mps.is_available())"
```

## 📊 Datensatz

Platzieren Sie Ihre `Vollstaendiger_Datensatz.json` im Hauptverzeichnis. Der Datensatz sollte folgende Struktur haben:

```json
[
  {
    "id": "A_001",
    "instruction": "Beschreibung der Aufgabe",
    "input": "Eingabe des Patienten",
    "output": "Erwartete Systemantwort",
    "metadata": {
      "module": "A",
      "step": "system_init",
      "...": "..."
    }
  }
]
```

## 🛠️ Verwendung

### Virtual Environment aktivieren

#### 🐧 Linux / 🍎 macOS
```bash
source vhf_training_env/bin/activate
```

#### 🪟 Windows (CMD)
```cmd
vhf_training_env\Scripts\activate.bat
```

#### 🪟 Windows (PowerShell)
```powershell
vhf_training_env\Scripts\Activate.ps1
```

### Einfaches Training (GPT-2)

#### 🐧 Linux / 🍎 macOS
```bash
# Environment aktivieren
source vhf_training_env/bin/activate

# Training starten
python vhf_training_simple.py
```

#### 🪟 Windows
```cmd
# Environment aktivieren
vhf_training_env\Scripts\activate.bat

# Training starten
python vhf_training_simple.py
```

**Vorteile:**
- Schnell und einfach
- Geringer Memory-Verbrauch
- CPU-kompatibel

### Erweiteres Training (LoRA)

#### Alle Plattformen
```bash
# Standard LoRA Training
python vhf_training_advanced.py --use_lora

# Mit Wandb Logging
python vhf_training_advanced.py --use_lora --use_wandb

# Größeres Modell
python vhf_training_advanced.py --model_name microsoft/DialoGPT-large --epochs 5
```

**Vorteile:**
- Effiziente Anpassung größerer Modelle
- Weniger GPU Memory benötigt
- Bessere Performance

### Parameter-Übersicht

| Parameter | Beschreibung | Standard | Einfach | Erweitert |
|-----------|--------------|----------|---------|-----------|
| `--data_path` | Pfad zum JSON Datensatz | `Vollstaendiger_Datensatz.json` | ✅ | ✅ |
| `--model_name` | Basis-Modell | `gpt2` / `microsoft/DialoGPT-medium` | ✅ | ✅ |
| `--output_dir` | Output Verzeichnis | `./vhf-model-*` | ✅ | ✅ |
| `--epochs` | Anzahl Trainingsepochen | `3` | ✅ | ✅ |
| `--batch_size` | Batch Größe | `4` | ✅ | ✅ |
| `--learning_rate` | Lernrate | `5e-5` / `2e-4` | ✅ | ✅ |
| `--max_length` | Max Token Länge | `512` | ✅ | ✅ |
| `--use_lora` | LoRA verwenden | `False` / `True` | ❌ | ✅ |
| `--use_wandb` | Wandb Logging | `False` | ❌ | ✅ |

## 🧪 Model Testing

### Virtual Environment aktivieren (falls nicht aktiv)

#### 🐧 Linux / 🍎 macOS
```bash
source vhf_training_env/bin/activate
```

#### 🪟 Windows (CMD)
```cmd
vhf_training_env\Scripts\activate.bat
```

#### 🪟 Windows (PowerShell)
```powershell
vhf_training_env\Scripts\Activate.ps1
```

### Interaktiver Chat

#### Alle Plattformen
```bash
# Einfaches Modell testen
python vhf_inference.py --model_path ./vhf-model-gpt2 --interactive

# LoRA Modell testen
python vhf_inference.py --model_path ./vhf-model-lora --use_lora --interactive
```

### Batch Testing

#### Alle Plattformen
```bash
# Test Cases erstellen
python vhf_inference.py --create_test_cases

# Batch Test durchführen
python vhf_inference.py --model_path ./vhf-model-lora --use_lora --batch_test test_cases.json
```

### Beispiel-Chat

```
📋 Instruction: Patient meldet sich mit Symptomen, die auf Vorhofflimmern hindeuten könnten
📝 Input (optional): Ich habe Herzstolpern und fühle mich unwohl

🤖 Generiere Antwort...

💬 Antwort:
Ich verstehe, dass Sie Symptome verspüren. Um Ihre Beschwerden zu bewerten, führen wir zunächst eine EKG-Aufzeichnung durch. Dies hilft uns dabei festzustellen, ob Vorhofflimmern vorliegt. Sind Sie bereit, jetzt ein EKG durchzuführen?
```

## ⚙️ Konfiguration

### Standard-Konfigurationen

**CPU Training (wenig RAM) - Alle Plattformen:**
```bash
python vhf_training_simple.py --batch_size 1 --max_length 256
```

**GPU Training (optimiert) - Linux/Windows CUDA:**
```bash
python vhf_training_advanced.py --use_lora --batch_size 8 --max_length 512
```

**🍎 macOS (Apple Silicon MPS):**
```bash
# Automatische Erkennung von MPS
python vhf_training_advanced.py --use_lora --batch_size 4 --max_length 512
```

**Lange Training Session:**
```bash
python vhf_training_advanced.py --use_lora --epochs 10 --learning_rate 1e-4
```

### Plattform-spezifische Optimierungen

#### 🐧 Linux (CUDA)
- Beste GPU-Performance
- Unterstützt alle CUDA-Features
- Empfohlen für intensive Training-Sessions

```bash
# Optimiert für Linux + CUDA
python vhf_training_advanced.py --use_lora --batch_size 8 --gradient_accumulation_steps 1
```

#### 🍎 macOS (Metal Performance Shaders)
- Nutzt Apple Silicon GPU via MPS
- Automatische Erkennung in PyTorch 2.0+
- Begrenzt auf kleinere Batch-Sizes

```bash
# Optimiert für Apple Silicon
python vhf_training_advanced.py --use_lora --batch_size 4 --max_length 384
```

#### 🪟 Windows (CUDA/CPU)
- CUDA support mit NVIDIA GPUs
- Fallback auf CPU bei Problemen
- PowerShell vs CMD beachten

```cmd
# Windows optimiert
python vhf_training_advanced.py --use_lora --batch_size 6 --max_length 512
```

### Modell-Empfehlungen

| Anwendungsfall | Modell | Vorteile |
|----------------|--------|----------|
| Schnelle Experimente | `gpt2` | Klein, schnell, CPU-kompatibel |
| Ausgewogene Performance | `microsoft/DialoGPT-medium` | Gute Balance zwischen Größe und Qualität |
| Beste Qualität | `microsoft/DialoGPT-large` | Höchste Qualität, benötigt mehr GPU Memory |
| Deutsche Texte | `dbmdz/german-gpt2` | Deutsch-spezifisch |

## 📈 Monitoring & Logging

### Wandb Integration

```bash
# Wandb Account erstellen und Login
pip install wandb
wandb login

# Training mit Logging
python vhf_training_advanced.py --use_lora --use_wandb
```

### Metriken

Das Training protokolliert:
- **Training Loss**: Wie gut das Modell lernt
- **Evaluation Loss**: Wie gut das Modell auf ungesehenen Daten abschneidet
- **Learning Rate**: Aktuelle Lernrate
- **GPU Memory**: Speicherverbrauch

## 🔧 Troubleshooting

## 🔧 Troubleshooting

### Häufige Probleme

**1. CUDA Out of Memory**
```bash
# Lösung: Kleinere Batch Size
python vhf_training_advanced.py --batch_size 2 --gradient_accumulation_steps 2
```

**2. Slow Training**
```bash
# Lösung: Kleinere Sequence Length
python vhf_training_simple.py --max_length 256
```

**3. Import Errors**
```bash
# Lösung: Requirements neu installieren
pip install -r requirements.txt --force-reinstall
```

**4. Model Not Loading**
```bash
# Linux/macOS: Prüfe ob alle Dateien vorhanden sind
ls -la ./vhf-model-*/

# Windows (CMD): Prüfe Dateien
dir vhf-model-*

# Windows (PowerShell): Prüfe Dateien
Get-ChildItem -Path "vhf-model-*" -Recurse
```

### Plattform-spezifische Probleme

#### 🐧 Linux
**CUDA Driver Issues:**
```bash
# CUDA Installation prüfen
nvidia-smi
nvcc --version

# PyTorch CUDA Version prüfen
python -c "import torch; print(torch.version.cuda)"
```

**Permission Errors:**
```bash
# Ausführungsrechte setzen
chmod +x setup.sh

# Virtual Environment Rechte
sudo chown -R $USER:$USER vhf_training_env/
```

#### 🍎 macOS
**Apple Silicon (M1/M2) Probleme:**
```bash
# MPS Support prüfen
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"

# Rosetta fallback (wenn nötig)
arch -x86_64 python vhf_training_simple.py
```

**Homebrew Python Conflicts:**
```bash
# Python Version explizit verwenden
/opt/homebrew/bin/python3 -m venv vhf_training_env

# Oder pyenv verwenden
brew install pyenv
pyenv install 3.10.11
pyenv local 3.10.11
```

**Xcode Command Line Tools:**
```bash
# Falls Kompilierungsfehler auftreten
xcode-select --install
```

#### 🪟 Windows
**PowerShell Execution Policy:**
```powershell
# Policy prüfen
Get-ExecutionPolicy

# Policy ändern (als Administrator)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope LocalMachine

# Oder nur für aktuellen User
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Python nicht gefunden:**
```cmd
# Python zu PATH hinzufügen oder vollständigen Pfad verwenden
C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311\python.exe -m venv vhf_training_env
```

**Virtual Environment Aktivierung Probleme:**
```cmd
# Falls activate.bat nicht funktioniert
vhf_training_env\Scripts\python.exe -m pip install -r requirements.txt

# PowerShell alternative
& "vhf_training_env\Scripts\python.exe" -m pip install -r requirements.txt
```

**CUDA Installation Windows:**
```cmd
# NVIDIA Driver prüfen
nvidia-smi

# CUDA Toolkit Installation prüfen
nvcc --version

# PyTorch CUDA Windows
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Long Path Support:**
```cmd
# Git clone mit long paths (als Administrator)
git config --system core.longpaths true

# Oder Windows Registry ändern (nicht empfohlen)
```

### Environment Debugging

#### Alle Plattformen
```bash
# Python Environment Info
python -c "import sys; print('Python:', sys.version)"
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import transformers; print('Transformers:', transformers.__version__)"

# GPU Info
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"

# Memory Info
python -c "import psutil; print(f'RAM: {psutil.virtual_memory().total // (1024**3)} GB')"
```

#### Modell-Loading Debug
```bash
# Schritt-für-Schritt Debugging
python -c "
import os
model_path = './vhf-model-lora'
print('Model dir exists:', os.path.exists(model_path))
print('Files in model dir:')
if os.path.exists(model_path):
    for f in os.listdir(model_path):
        print(f'  {f}')
"
```

### System Requirements

#### Minimum (alle Plattformen):
- Python 3.8+
- 8GB RAM
- 10GB freier Speicherplatz

#### Empfohlen:

**🐧 Linux:**
- Ubuntu 20.04+ / CentOS 8+
- Python 3.10+
- 16GB+ RAM
- NVIDIA GPU mit 8GB+ VRAM (CUDA 11.8+)
- 50GB freier Speicherplatz

**🍎 macOS:**
- macOS 12.0+ (Monterey)
- Apple Silicon (M1/M2) oder Intel
- Python 3.10+ (Homebrew empfohlen)
- 16GB+ unified Memory
- 50GB freier Speicherplatz

**🪟 Windows:**
- Windows 10/11 (64-bit)
- Python 3.10+ (Microsoft Store oder python.org)
- 16GB+ RAM
- NVIDIA GPU mit 8GB+ VRAM (CUDA 11.8+)
- 50GB freier Speicherplatz
- PowerShell 5.1+ oder Windows Terminal

### Performance-Tipps

#### Plattform-übergreifend:
1. **Für CPU Training**: Verwenden Sie kleine Modelle (`gpt2`) und kleine Batch Sizes
2. **Für GPU Training**: Nutzen Sie LoRA für effizienten Memory-Verbrauch
3. **Für lange Sessions**: Aktivieren Sie Checkpointing mit `--save_steps 100`

#### 🐧 Linux-spezifisch:
- Nutzen Sie `htop` für Memory-Monitoring
- Setzen Sie `ulimit -n 65536` für mehr File Handles
- Verwenden Sie `screen` oder `tmux` für lange Training Sessions

#### 🍎 macOS-spezifisch:
- Überwachen Sie Memory mit Activity Monitor
- Nutzen Sie MPS für GPU-Beschleunigung auf Apple Silicon
- Beachten Sie thermal throttling bei längeren Sessions

#### 🪟 Windows-spezifisch:
- Verwenden Sie Task Manager für Performance-Monitoring
- Aktivieren Sie Developer Mode für bessere Performance
- Nutzen Sie Windows Terminal für bessere Kompatibilität

## 📁 Datei-Übersicht

```
vhf-llm-training/
├── 📄 vhf_training_simple.py      # Einfaches GPT-2 Training
├── 📄 vhf_training_advanced.py    # Erweiteres LoRA Training
├── 📄 vhf_inference.py           # Model Testing & Chat
├── 📄 requirements.txt           # Python Dependencies
├── 🐧 setup.sh                  # Setup Script (Linux/macOS)
├── 🪟 setup.bat                 # Setup Script (Windows CMD)
├── 🪟 setup.ps1                 # Setup Script (Windows PowerShell)
├── ⚙️ config.json              # Beispiel-Konfiguration
├── 📊 Vollstaendiger_Datensatz.json  # Ihr VHF Datensatz
├── 📚 README.md                # Diese Anleitung
└── 📁 vhf_training_env/         # Virtual Environment (nach Setup)
    ├── 🐧 bin/activate          # Linux/macOS Aktivierung
    └── 🪟 Scripts/
        ├── activate.bat         # Windows CMD Aktivierung
        └── Activate.ps1         # Windows PowerShell Aktivierung
```

## 💻 Plattform-spezifische Quickstarts

### 🐧 Linux Quickstart
```bash
git clone <repository-url>
cd vhf-llm-training
chmod +x setup.sh && ./setup.sh
source vhf_training_env/bin/activate
python vhf_training_simple.py
```

### 🍎 macOS Quickstart
```bash
# Homebrew Python installieren (falls nötig)
brew install python3

git clone <repository-url>
cd vhf-llm-training
chmod +x setup.sh && ./setup.sh
source vhf_training_env/bin/activate
python vhf_training_simple.py
```

### 🪟 Windows Quickstart (CMD)
```cmd
git clone <repository-url>
cd vhf-llm-training
setup.bat
vhf_training_env\Scripts\activate.bat
python vhf_training_simple.py
```

### 🪟 Windows Quickstart (PowerShell)
```powershell
git clone <repository-url>
cd vhf-llm-training
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\setup.ps1
vhf_training_env\Scripts\Activate.ps1
python vhf_training_simple.py
```

## 🤝 Support

Bei Problemen:

1. Prüfen Sie die [Troubleshooting](#troubleshooting) Sektion
2. Stellen Sie sicher, dass alle Requirements installiert sind
3. Testen Sie zunächst mit kleinen Konfigurationen
4. Öffnen Sie ein Issue mit detaillierter Fehlerbeschreibung

## 📄 Lizenz

Dieses Projekt steht unter der MIT Lizenz - siehe die [LICENSE](LICENSE) Datei für Details.

---

**Viel Erfolg beim Training Ihres VHF-LLMs! 🏥🤖**