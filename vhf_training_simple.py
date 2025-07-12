#!/usr/bin/env python3
"""
Einfaches Training Script für VHF-LLM mit einem kleinen Modell
Verwendet transformers mit GPT-2 als Basis für schnelles lokales Training
"""

import json
import torch
import pandas as pd
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    TextDataset, 
    DataCollatorForLanguageModeling,
    Trainer, 
    TrainingArguments
)
from torch.utils.data import Dataset
import os
from typing import List, Dict

class VHFDataset(Dataset):
    """Custom Dataset für VHF Conversation Data"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Formatiere die Daten als Conversations
        self.conversations = self._format_conversations()
    
    def _format_conversations(self):
        """Formatiert die VHF-Daten als Conversation-Strings"""
        conversations = []
        
        for item in self.data:
            # Erstelle einen Conversation-String im Format:
            # "Instruction: ... Input: ... Output: ..."
            conversation = f"Instruction: {item['instruction']}\n"
            if item.get('input'):
                conversation += f"Input: {item['input']}\n"
            conversation += f"Output: {item['output']}\n<|endoftext|>"
            
            conversations.append(conversation)
        
        return conversations
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conversation = self.conversations[idx]
        
        # Tokenize mit Padding und Truncation
        encoding = self.tokenizer(
            conversation,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()  # Für Language Modeling
        }

def load_vhf_data(filepath: str) -> List[Dict]:
    """Lädt die VHF JSON-Daten"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def setup_model_and_tokenizer(model_name: str = "gpt2"):
    """Initialisiert Modell und Tokenizer"""
    print(f"Lade Modell: {model_name}")
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Füge spezielle Tokens hinzu falls nötig
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def train_vhf_model(
    data_path: str,
    output_dir: str = "./vhf-model",
    model_name: str = "gpt2",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    max_length: int = 512
):
    """Hauptfunktion für das Training"""
    
    print("🏥 VHF-LLM Training Script gestartet")
    print("=" * 50)
    
    # 1. Daten laden
    print("📁 Lade VHF-Daten...")
    data = load_vhf_data(data_path)
    print(f"✅ {len(data)} VHF-Einträge geladen")
    
    # 2. Modell und Tokenizer setup
    model, tokenizer = setup_model_and_tokenizer(model_name)
    
    # 3. Dataset erstellen
    print("📚 Erstelle Dataset...")
    train_dataset = VHFDataset(data, tokenizer, max_length)
    print(f"✅ Dataset mit {len(train_dataset)} Einträgen erstellt")
    
    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=100,
        save_total_limit=2,
        prediction_loss_only=True,
        learning_rate=learning_rate,
        warmup_steps=100,
        logging_dir=f'{output_dir}/logs',
        logging_steps=50,
        remove_unused_columns=False,
        dataloader_drop_last=False,
    )
    
    # 5. Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Kein Masked Language Modeling für GPT-2
    )
    
    # 6. Trainer erstellen
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    
    # 7. Training starten
    print("🚀 Starte Training...")
    print(f"📊 Konfiguration:")
    print(f"   - Modell: {model_name}")
    print(f"   - Epochen: {epochs}")
    print(f"   - Batch Size: {batch_size}")
    print(f"   - Learning Rate: {learning_rate}")
    print(f"   - Max Length: {max_length}")
    print("=" * 50)
    
    trainer.train()
    
    # 8. Modell speichern
    print("💾 Speichere trainiertes Modell...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ Training abgeschlossen! Modell gespeichert in: {output_dir}")
    
    return trainer

def test_model(model_path: str, test_instruction: str = "Patient meldet sich mit Symptomen"):
    """Testet das trainierte Modell"""
    print("\n🔬 Teste trainiertes Modell...")
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    
    # Test-Prompt
    test_prompt = f"Instruction: {test_instruction}\nInput: Ich habe Herzstolpern und fühle mich unwohl\nOutput:"
    
    # Tokenize
    inputs = tokenizer.encode(test_prompt, return_tensors='pt')
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=inputs.shape[1] + 100,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("🤖 Modell Antwort:")
    print("-" * 30)
    print(generated_text)
    print("-" * 30)

if __name__ == "__main__":
    # Konfiguration
    DATA_PATH = "Vollstaendiger_Datensatz.json"  # Pfad zu Ihrer JSON-Datei
    OUTPUT_DIR = "./vhf-model-gpt2"
    
    # Prüfe ob GPU verfügbar ist
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Verwende Device: {device}")
    
    # Training starten
    try:
        trainer = train_vhf_model(
            data_path=DATA_PATH,
            output_dir=OUTPUT_DIR,
            model_name="gpt2",  # Kleines Modell für schnelles Training
            epochs=3,
            batch_size=2 if device == "cpu" else 4,  # Kleinere Batch Size für CPU
            learning_rate=5e-5,
            max_length=256  # Kleinere Länge für schnelleres Training
        )
        
        # Test das Modell
        test_model(OUTPUT_DIR)
        
    except Exception as e:
        print(f"❌ Fehler beim Training: {e}")
        print("💡 Tipps:")
        print("   - Stelle sicher, dass die JSON-Datei existiert")
        print("   - Bei Memory-Problemen: Reduziere batch_size oder max_length")
        print("   - Bei GPU-Problemen: Verwende CPU (automatisch erkannt)")