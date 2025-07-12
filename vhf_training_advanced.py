#!/usr/bin/env python3
"""
Erweiterte VHF-LLM Training Script mit LoRA (Low-Rank Adaptation)
Für effizientes Fine-tuning größerer Modelle mit weniger GPU-Memory
"""

import json
import torch
import pandas as pd
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    prepare_model_for_kbit_training
)
from torch.utils.data import Dataset
import os
import wandb
from typing import List, Dict, Optional
import argparse

class VHFInstructionDataset(Dataset):
    """Optimiertes Dataset für Instruction-Following Training"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Erstelle die formatierten Prompts
        self.formatted_data = self._format_instruction_data()
    
    def _format_instruction_data(self):
        """Formatiert die Daten für Instruction Following"""
        formatted = []
        
        for item in self.data:
            # Instruction-Following Format
            prompt = f"""### Instruction:
{item['instruction']}

### Input:
{item.get('input', '')}

### Response:
{item['output']}"""
            
            formatted.append({
                'text': prompt,
                'module': item['metadata'].get('module', 'unknown'),
                'step': item['metadata'].get('step', 'unknown')
            })
        
        return formatted
    
    def __len__(self):
        return len(self.formatted_data)
    
    def __getitem__(self, idx):
        item = self.formatted_data[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }

class VHFTrainer:
    """Haupt-Trainer-Klasse für VHF-LLM"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        
    def setup_model_and_tokenizer(self):
        """Initialisiert Modell und Tokenizer mit LoRA"""
        print(f"🤖 Lade Modell: {self.config['model_name']}")
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model_name'],
            trust_remote_code=True
        )
        
        # Pad token setzen
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Modell laden
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model_name'],
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # LoRA Konfiguration
        if self.config.get('use_lora', True):
            self._setup_lora()
        
        print("✅ Modell und Tokenizer geladen")
    
    def _setup_lora(self):
        """Konfiguriert LoRA für effizientes Fine-tuning"""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.get('lora_r', 8),
            lora_alpha=self.config.get('lora_alpha', 32),
            lora_dropout=self.config.get('lora_dropout', 0.1),
            target_modules=self.config.get('lora_target_modules', ["q_proj", "v_proj"])
        )
        
        # Model für k-bit Training vorbereiten (falls GPU verfügbar)
        if torch.cuda.is_available():
            self.model = prepare_model_for_kbit_training(self.model)
        
        # LoRA anwenden
        self.peft_model = get_peft_model(self.model, lora_config)
        self.peft_model.print_trainable_parameters()
        
        print("✅ LoRA konfiguriert")
    
    def load_and_prepare_data(self, data_path: str):
        """Lädt und bereitet die VHF-Daten vor"""
        print("📚 Lade und verarbeite VHF-Daten...")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Dataset statistiken
        print(f"📊 Dataset-Statistiken:")
        print(f"   - Gesamt Einträge: {len(data)}")
        
        # Module auswerten
        modules = {}
        for item in data:
            module = item['metadata'].get('module', 'unknown')
            modules[module] = modules.get(module, 0) + 1
        
        print(f"   - Module: {modules}")
        
        # Train/Validation Split (80/20)
        split_idx = int(len(data) * 0.8)
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        # Datasets erstellen
        train_dataset = VHFInstructionDataset(
            train_data, 
            self.tokenizer, 
            self.config['max_length']
        )
        
        val_dataset = VHFInstructionDataset(
            val_data, 
            self.tokenizer, 
            self.config['max_length']
        )
        
        print(f"✅ Training: {len(train_dataset)}, Validation: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def train(self, train_dataset, val_dataset):
        """Startet das Training"""
        print("🚀 Starte Training...")
        
        # Training Arguments
        training_args = TrainingArguments(
            output_dir=self.config['output_dir'],
            overwrite_output_dir=True,
            num_train_epochs=self.config['epochs'],
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'],
            gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 1),
            warmup_steps=self.config.get('warmup_steps', 100),
            learning_rate=self.config['learning_rate'],
            logging_steps=50,
            save_steps=500,
            eval_steps=500,
            evaluation_strategy="steps",
            save_total_limit=3,
            load_best_model_at_end=True,
            report_to=["wandb"] if self.config.get('use_wandb', False) else [],
            fp16=torch.cuda.is_available(),
            dataloader_drop_last=False,
            remove_unused_columns=False,
        )
        
        # Data Collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Trainer
        model_to_train = self.peft_model if self.config.get('use_lora', True) else self.model
        
        trainer = Trainer(
            model=model_to_train,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        # Training starten
        trainer.train()
        
        # Modell speichern
        print("💾 Speichere Modell...")
        trainer.save_model()
        
        if self.config.get('use_lora', True):
            # LoRA Adapter separat speichern
            self.peft_model.save_pretrained(f"{self.config['output_dir']}/lora_adapter")
        
        self.tokenizer.save_pretrained(self.config['output_dir'])
        
        print(f"✅ Training abgeschlossen! Modell in: {self.config['output_dir']}")
        
        return trainer
    
    def test_model(self, test_prompts: List[str]):
        """Testet das trainierte Modell"""
        print("\n🔬 Teste das trainierte Modell...")
        
        model_to_test = self.peft_model if self.config.get('use_lora', True) else self.model
        
        for i, prompt in enumerate(test_prompts):
            print(f"\n--- Test {i+1} ---")
            print(f"Prompt: {prompt}")
            
            # Formatiere als Instruction
            formatted_prompt = f"""### Instruction:
{prompt}

### Input:
Ich habe Herzstolpern und Atemnot

### Response:
"""
            
            # Tokenize
            inputs = self.tokenizer.encode(formatted_prompt, return_tensors='pt')
            if torch.cuda.is_available():
                inputs = inputs.to(self.model.device)
            
            # Generate
            with torch.no_grad():
                outputs = model_to_test.generate(
                    inputs,
                    max_length=inputs.shape[1] + 150,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode nur die generierte Antwort
            generated_text = self.tokenizer.decode(
                outputs[0][inputs.shape[1]:], 
                skip_special_tokens=True
            )
            
            print(f"Antwort: {generated_text}")
            print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description='VHF-LLM Training')
    parser.add_argument('--data_path', default='Vollstaendiger_Datensatz.json', help='Pfad zu VHF JSON Daten')
    parser.add_argument('--model_name', default='microsoft/DialoGPT-medium', help='Basis-Modell')
    parser.add_argument('--output_dir', default='./vhf-model-lora', help='Output Directory')
    parser.add_argument('--epochs', type=int, default=3, help='Anzahl Epochen')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch Size')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning Rate')
    parser.add_argument('--max_length', type=int, default=512, help='Max Token Length')
    parser.add_argument('--use_lora', action='store_true', default=True, help='LoRA verwenden')
    parser.add_argument('--use_wandb', action='store_true', help='Wandb Logging')
    
    args = parser.parse_args()
    
    # Konfiguration
    config = {
        'model_name': args.model_name,
        'output_dir': args.output_dir,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'max_length': args.max_length,
        'use_lora': args.use_lora,
        'use_wandb': args.use_wandb,
        'lora_r': 8,
        'lora_alpha': 32,
        'lora_dropout': 0.1,
    }
    
    # Wandb Setup
    if config['use_wandb']:
        wandb.init(project="vhf-llm-training", config=config)
    
    # Device Info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
    
    try:
        # Trainer initialisieren
        trainer = VHFTrainer(config)
        
        # Modell setup
        trainer.setup_model_and_tokenizer()
        
        # Daten laden
        train_dataset, val_dataset = trainer.load_and_prepare_data(args.data_path)
        
        # Training
        trainer.train(train_dataset, val_dataset)
        
        # Tests
        test_prompts = [
            "Patient meldet sich mit Symptomen, die auf Vorhofflimmern hindeuten könnten",
            "EKG-Analyse erkennt Vorhofflimmern",
            "Bewertung der Herzfrequenz im gültigen Bereich"
        ]
        
        trainer.test_model(test_prompts)
        
    except Exception as e:
        print(f"❌ Fehler: {e}")
        raise

if __name__ == "__main__":
    main()