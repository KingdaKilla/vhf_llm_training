#!/usr/bin/env python3
"""
BiomistralAI VHF-LLM Training Script
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
    """Optimiertes Dataset für BiomistralAI Instruction-Following Training"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Erstelle die formatierten Prompts
        self.formatted_data = self._format_instruction_data()
    
    def _format_instruction_data(self):
        """Formatiert die Daten für BiomistralAI Instruction Following"""
        formatted = []
        
        for item in self.data:
            # BiomistralAI-optimiertes Format
            prompt = f"""<s>[INST] {item['instruction']}

Input: {item.get('input', '')} [/INST]

{item['output']}</s>"""
            
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
        
        # Tokenize mit korrekter Attention Mask
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            add_special_tokens=True
        )
        
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        
        # Labels für Causal LM - Input IDs als Labels
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padding tokens
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

class BiomistralVHFTrainer:
    """Spezialisierter Trainer für BiomistralAI VHF-Training"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        
    def setup_model_and_tokenizer(self):
        """Initialisiert BiomistralAI Modell und Tokenizer mit optimierter LoRA"""
        print(f"🧬 Lade BiomistralAI Modell: {self.config['model_name']}")
        
        # Tokenizer mit korrekter Konfiguration
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model_name'],
            trust_remote_code=True,
            use_fast=True
        )
        
        # Pad token explizit setzen
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Modell laden mit optimierten Einstellungen
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model_name'],
            torch_dtype=torch.float32,  # CPU benötigt float32
            trust_remote_code=True,
            device_map=None,  # Explizit kein device_map für CPU
            low_cpu_mem_usage=True
        )
        
        # Modell für Training vorbereiten
        self.model.train()
        
        # LoRA Konfiguration
        if self.config.get('use_lora', True):
            self._setup_lora()
        
        print("✅ BiomistralAI Modell und Tokenizer geladen")
    
    def _setup_lora(self):
        """Konfiguriert LoRA speziell für BiomistralAI"""
        # BiomistralAI-spezifische LoRA-Konfiguration
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.get('lora_r', 4),  # Kleinerer Rang für Stabilität
            lora_alpha=self.config.get('lora_alpha', 16),
            lora_dropout=self.config.get('lora_dropout', 0.05),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Nur Attention
            bias="none",
            use_rslora=False  # Für Stabilität
        )
        
        # Modell für LoRA vorbereiten
        self.model.requires_grad_(False)  # Basis-Modell einfrieren
        
        # LoRA anwenden
        self.peft_model = get_peft_model(self.model, lora_config)
        
        # Trainable Parameter aktivieren
        for name, param in self.peft_model.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
        
        print(f"📊 Trainierbare Parameter: {self.peft_model.get_nb_trainable_parameters()}")
        self.peft_model.print_trainable_parameters()
        
        print("✅ BiomistralAI LoRA konfiguriert")
    
    def load_and_prepare_data(self, data_path: str):
        """Lädt und bereitet die VHF-Daten für BiomistralAI vor"""
        print("📚 Lade VHF-Daten für BiomistralAI...")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Dataset statistiken
        print(f"🏥 Medizinische VHF-Daten:")
        print(f"   - Gesamt Einträge: {len(data)}")
        
        # Module auswerten
        modules = {}
        risk_levels = {}
        for item in data:
            module = item['metadata'].get('module', 'unknown')
            modules[module] = modules.get(module, 0) + 1
            
            risk_level = item['metadata'].get('risk_level', 'unknown')
            risk_levels[risk_level] = risk_levels.get(risk_level, 0) + 1
        
        print(f"   - Module: {modules}")
        print(f"   - Risiko-Level: {risk_levels}")
        
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
        
        print(f"✅ Medizinische Datasets: Training={len(train_dataset)}, Validation={len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def train(self, train_dataset, val_dataset):
        """Startet das BiomistralAI Training"""
        print("🚀 Starte BiomistralAI VHF Training...")
        
        # Optimierte Training Arguments für BiomistralAI
        training_args = TrainingArguments(
            output_dir=self.config['output_dir'],
            overwrite_output_dir=True,
            num_train_epochs=self.config['epochs'],
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'],
            gradient_accumulation_steps=max(1, 4 // self.config['batch_size']),  # Kompensiert kleine Batch Size
            warmup_steps=20,  # Weniger Warmup für kleinere Datasets
            learning_rate=self.config['learning_rate'],
            weight_decay=0.01,
            logging_steps=10,
            save_steps=50,
            eval_steps=50,
            eval_strategy="steps",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=["wandb"] if self.config.get('use_wandb', False) else [],
            fp16=False,  # CPU Training
            dataloader_drop_last=False,
            remove_unused_columns=False,
            gradient_checkpointing=False,  # Deaktiviert für Stabilität
            dataloader_pin_memory=False,  # CPU Training
            dataloader_num_workers=0,  # Single-threaded für Stabilität
        )
        
        # Data Collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=None,
            return_tensors="pt"
        )
        
        # Trainer mit LoRA Model
        model_to_train = self.peft_model if self.config.get('use_lora', True) else self.model
        
        trainer = Trainer(
            model=model_to_train,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        print("🏥 BiomistralAI medizinisches Training läuft...")
        
        # Training starten
        trainer.train()
        
        # Modell speichern
        print("💾 Speichere BiomistralAI Modell...")
        trainer.save_model()
        
        if self.config.get('use_lora', True):
            # LoRA Adapter separat speichern
            self.peft_model.save_pretrained(f"{self.config['output_dir']}/lora_adapter")
        
        self.tokenizer.save_pretrained(self.config['output_dir'])
        
        print(f"✅ BiomistralAI Training abgeschlossen! Modell in: {self.config['output_dir']}")
        
        return trainer

def main():
    parser = argparse.ArgumentParser(description='BiomistralAI VHF-LLM Training')
    parser.add_argument('--data_path', default='Vollstaendiger_Datensatz.json', help='Pfad zu VHF JSON Daten')
    parser.add_argument('--model_name', default='BioMistral/BioMistral-7B-DARE', help='BiomistralAI Modell')
    parser.add_argument('--output_dir', default='./vhf-biomistral-real', help='Output Directory')
    parser.add_argument('--epochs', type=int, default=3, help='Anzahl Epochen')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch Size (empfohlen: 1 für CPU)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--max_length', type=int, default=256, help='Max Token Length')
    parser.add_argument('--use_lora', action='store_true', default=True, help='LoRA verwenden')
    parser.add_argument('--use_wandb', action='store_true', help='Wandb Logging')
    parser.add_argument('--lora_r', type=int, default=4, help='LoRA Rang')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA Alpha')
    
    args = parser.parse_args()
    
    # Konfiguration für BiomistralAI
    config = {
        'model_name': args.model_name,
        'output_dir': args.output_dir,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'max_length': args.max_length,
        'use_lora': args.use_lora,
        'use_wandb': args.use_wandb,
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'lora_dropout': 0.05,
    }
    
    # Wandb Setup
    if config['use_wandb']:
        wandb.init(project="biomistral-vhf-training", config=config)
    
    # Device Info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")
    print(f"💾 RAM-optimiert für CPU-Training")
    
    try:
        # BiomistralAI Trainer initialisieren
        trainer = BiomistralVHFTrainer(config)
        
        # Modell setup
        trainer.setup_model_and_tokenizer()
        
        # Daten laden
        train_dataset, val_dataset = trainer.load_and_prepare_data(args.data_path)
        
        # Training
        trainer.train(train_dataset, val_dataset)
        
        print("🎉 BiomistralAI VHF-Training erfolgreich abgeschlossen!")
        print(f"🧪 Testen Sie das Modell mit:")
        print(f"   python vhf_inference.py --model_path {args.output_dir} --use_lora --interactive")
        
    except Exception as e:
        print(f"❌ Fehler: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()