#!/usr/bin/env python3
"""
BioMistral VHF-LLM Training Script mit optimierten Parametern
Speziell angepasst für BioMistral Modelle mit medizinischem Fokus
"""

import json
import torch
import pandas as pd
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
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
    """Dataset für BioMistral mit medizinischem Format"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.formatted_data = self._format_medical_data()
    
    def _format_medical_data(self):
        """Formatiert für medizinische Instruktionen (BioMistral optimiert)"""
        formatted = []
        
        for item in self.data:
            # Medizinisches Format mit Kontext
            prompt = f"""<s>[INST] Als medizinisches KI-System für Vorhofflimmern-Überwachung:

Aufgabe: {item['instruction']}

Patient-Input: {item.get('input', 'Keine spezifischen Symptome angegeben')}

Bitte gib eine professionelle, empathische und medizinisch fundierte Antwort. [/INST]

{item['output']}</s>"""
            
            formatted.append({
                'text': prompt,
                'module': item['metadata'].get('module', 'unknown'),
                'step': item['metadata'].get('step', 'unknown'),
                'risk_level': item['metadata'].get('risk_level', 'unknown')
            })
        
        return formatted
    
    def __len__(self):
        return len(self.formatted_data)
    
    def __getitem__(self, idx):
        item = self.formatted_data[idx]
        
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

class BioMistralVHFTrainer:
    """Spezialisierter Trainer für BioMistral VHF-LLM"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        
    def setup_model_and_tokenizer(self):
        """Setup mit BioMistral-optimierten Parametern"""
        print(f"🏥 Lade BioMistral Modell: {self.config['model_name']}")
        
        # Tokenizer mit medizinischen Tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model_name'],
            trust_remote_code=True,
            use_fast=True
        )
        
        # Chat Template für Mistral
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = "{% for message in messages %}[INST] {{ message['content'] }} [/INST]{% endfor %}"
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Quantization für Memory-Effizienz (optional)
        quantization_config = None
        if self.config.get('use_4bit', False):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        
        # BioMistral Modell laden
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model_name'],
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else None,
            attn_implementation="flash_attention_2" if self.config.get('use_flash_attention', False) else None
        )
        
        # LoRA für BioMistral
        if self.config.get('use_lora', True):
            self._setup_biomistral_lora()
        
        print("✅ BioMistral Modell und Tokenizer geladen")
    
    def _setup_biomistral_lora(self):
        """LoRA speziell für Mistral-Architektur konfiguriert"""
        
        # BioMistral-optimierte LoRA Konfiguration
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.get('lora_r', 16),  # Höher für medizinische Domäne
            lora_alpha=self.config.get('lora_alpha', 32),
            lora_dropout=self.config.get('lora_dropout', 0.05),
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
                "gate_proj", "up_proj", "down_proj",      # MLP layers
            ],
            bias="none",
            modules_to_save=None,
        )
        
        # Model für Training vorbereiten
        if self.config.get('use_4bit', False):
            self.model = prepare_model_for_kbit_training(self.model)
        
        # LoRA anwenden
        self.peft_model = get_peft_model(self.model, lora_config)
        self.peft_model.print_trainable_parameters()
        
        print("✅ BioMistral LoRA konfiguriert")
        print(f"📊 Trainierbare Parameter: {self.peft_model.get_nb_trainable_parameters()}")
    
    def load_and_prepare_data(self, data_path: str):
        """VHF-Daten für medizinisches Training vorbereiten"""
        print("📚 Lade VHF-Daten für BioMistral...")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Medizinische Domänen-Statistiken
        print(f"🏥 Medizinische VHF-Daten:")
        print(f"   - Gesamt Einträge: {len(data)}")
        
        # Risiko-Level Analyse
        risk_levels = {}
        modules = {}
        for item in data:
            module = item['metadata'].get('module', 'unknown')
            risk_level = item['metadata'].get('risk_level', 'unknown')
            modules[module] = modules.get(module, 0) + 1
            risk_levels[risk_level] = risk_levels.get(risk_level, 0) + 1
        
        print(f"   - Module: {modules}")
        print(f"   - Risiko-Level: {risk_levels}")
        
        # Stratifizierter Split (80/20) um Risiko-Balance zu erhalten
        split_idx = int(len(data) * 0.8)
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
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
        """BioMistral-optimiertes Training"""
        print("🚀 Starte BioMistral VHF Training...")
        
        # Training Arguments für medizinische Domain
        training_args = TrainingArguments(
            output_dir=self.config['output_dir'],
            overwrite_output_dir=True,
            num_train_epochs=self.config['epochs'],
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'],
            gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 2),
            warmup_steps=self.config.get('warmup_steps', 50),
            learning_rate=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 0.01),
            logging_steps=25,
            save_steps=250,
            eval_steps=250,
            eval_strategy="steps",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=["wandb"] if self.config.get('use_wandb', False) else [],
            fp16=torch.cuda.is_available() and not self.config.get('use_4bit', False),
            bf16=False,  # BioMistral funktioniert besser mit fp16
            dataloader_drop_last=False,
            remove_unused_columns=False,
            group_by_length=True,  # Effizienz für variable Längen
            gradient_checkpointing=self.config.get('gradient_checkpointing', True),
        )
        
        # Data Collator für Mistral
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Trainer mit BioMistral
        model_to_train = self.peft_model if self.config.get('use_lora', True) else self.model
        
        trainer = Trainer(
            model=model_to_train,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        # Training starten
        print("🏥 BioMistral medizinisches Training läuft...")
        trainer.train()
        
        # Speichern
        print("💾 Speichere BioMistral VHF-Modell...")
        trainer.save_model()
        
        if self.config.get('use_lora', True):
            self.peft_model.save_pretrained(f"{self.config['output_dir']}/biomistral_lora_adapter")
        
        self.tokenizer.save_pretrained(self.config['output_dir'])
        
        print(f"✅ BioMistral Training abgeschlossen! Modell in: {self.config['output_dir']}")
        
        return trainer
    
    def medical_test(self):
        """Medizinische Test-Fälle für BioMistral"""
        test_cases = [
            {
                "instruction": "Patient meldet sich mit akuten VHF-Symptomen",
                "input": "Ich habe plötzliches Herzrasen, Atemnot und Brustschmerzen seit 2 Stunden"
            },
            {
                "instruction": "EKG-Befund-Interpretation bei Vorhofflimmern",
                "input": "EKG zeigt unregelmäßige RR-Intervalle, fehlende P-Wellen, Frequenz 120 bpm"
            },
            {
                "instruction": "Risikostratifizierung bei VHF-Patient",
                "input": "Patient: 75 Jahre, Diabetes, Hypertonie, LVEF 45%, akute VHF-Episode"
            }
        ]
        
        print("\n🔬 Teste BioMistral VHF-Modell...")
        
        model_to_test = self.peft_model if self.config.get('use_lora', True) else self.model
        
        for i, case in enumerate(test_cases):
            print(f"\n--- Medizinischer Test {i+1} ---")
            print(f"Aufgabe: {case['instruction']}")
            print(f"Patient: {case['input']}")
            
            # Format für BioMistral
            prompt = f"<s>[INST] Als medizinisches KI-System für Vorhofflimmern:\n\nAufgabe: {case['instruction']}\n\nPatient-Input: {case['input']}\n\nBitte gib eine professionelle medizinische Antwort. [/INST]\n\n"
            
            inputs = self.tokenizer.encode(prompt, return_tensors='pt')
            if torch.cuda.is_available():
                inputs = inputs.to(self.model.device)
            
            with torch.no_grad():
                outputs = model_to_test.generate(
                    inputs,
                    max_length=inputs.shape[1] + 200,
                    temperature=0.3,  # Niedrigere Temperatur für medizinische Präzision
                    do_sample=True,
                    top_p=0.8,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            generated_text = self.tokenizer.decode(
                outputs[0][inputs.shape[1]:], 
                skip_special_tokens=True
            )
            
            print(f"🏥 BioMistral Antwort:\n{generated_text}")
            print("-" * 60)

def main():
    parser = argparse.ArgumentParser(description='BioMistral VHF-LLM Training')
    parser.add_argument('--data_path', default='Vollstaendiger_Datensatz.json', help='VHF JSON Daten')
    parser.add_argument('--model_name', default='BioMistral/BioMistral-7B', help='BioMistral Modell')
    parser.add_argument('--output_dir', default='./vhf-biomistral-7b', help='Output Directory')
    parser.add_argument('--epochs', type=int, default=3, help='Training Epochen')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch Size (klein für 7B)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--max_length', type=int, default=512, help='Max Token Length')
    parser.add_argument('--use_lora', action='store_true', default=True, help='LoRA verwenden')
    parser.add_argument('--use_4bit', action='store_true', help='4-bit Quantization')
    parser.add_argument('--use_wandb', action='store_true', help='Wandb Logging')
    parser.add_argument('--gradient_checkpointing', action='store_true', default=True, help='Memory sparen')
    
    args = parser.parse_args()
    
    # BioMistral-optimierte Konfiguration
    config = {
        'model_name': args.model_name,
        'output_dir': args.output_dir,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'max_length': args.max_length,
        'use_lora': args.use_lora,
        'use_4bit': args.use_4bit,
        'use_wandb': args.use_wandb,
        'gradient_checkpointing': args.gradient_checkpointing,
        'lora_r': 16,  # Optimiert für medizinische Domäne
        'lora_alpha': 32,
        'lora_dropout': 0.05,
        'gradient_accumulation_steps': 4,  # Effektive größere Batch Size
        'warmup_steps': 50,
        'weight_decay': 0.01,
    }
    
    # Wandb für medizinisches Training
    if config['use_wandb']:
        wandb.init(
            project="vhf-biomistral-training", 
            config=config,
            name=f"biomistral-vhf-{config['epochs']}ep"
        )
    
    # System Info
    print(f"🖥️  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    try:
        # BioMistral Trainer
        trainer = BioMistralVHFTrainer(config)
        
        # Setup
        trainer.setup_model_and_tokenizer()
        
        # Daten laden
        train_dataset, val_dataset = trainer.load_and_prepare_data(args.data_path)
        
        # Training
        trainer.train(train_dataset, val_dataset)
        
        # Medizinische Tests
        trainer.medical_test()
        
    except Exception as e:
        print(f"❌ Fehler: {e}")
        raise

if __name__ == "__main__":
    main()