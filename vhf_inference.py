#!/usr/bin/env python3
"""
VHF-LLM Inference Script
Interaktives Testing des trainierten VHF-Modells
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
import json
import os

class VHFInference:
    """Inference-Klasse für trainierte VHF-Modelle"""
    
    def __init__(self, model_path: str, use_lora: bool = False):
        self.model_path = model_path
        self.use_lora = use_lora
        self.tokenizer = None
        self.model = None
        
        self._load_model()
    
    def _load_model(self):
        """Lädt das trainierte Modell"""
        print(f"🤖 Lade Modell aus: {self.model_path}")
        
        # Tokenizer laden
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        if self.use_lora:
            # LoRA Modell laden
            base_model = AutoModelForCausalLM.from_pretrained(
                "microsoft/DialoGPT-medium",  # Base model - anpassen falls nötig
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            self.model = PeftModel.from_pretrained(
                base_model, 
                f"{self.model_path}/lora_adapter"
            )
            print("✅ LoRA Modell geladen")
        else:
            # Standard Modell laden
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            print("✅ Standard Modell geladen")
    
    def generate_response(
        self, 
        instruction: str, 
        input_text: str = "", 
        max_length: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """Generiert eine Antwort basierend auf Instruction und Input"""
        
        # Formatiere den Prompt
        prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
        
        # Tokenize
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        if torch.cuda.is_available():
            inputs = inputs.to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + max_length,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode nur die generierte Antwort
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.shape[1]:], 
            skip_special_tokens=True
        )
        
        # Bereinige die Antwort
        response = generated_text.strip()
        if "###" in response:
            response = response.split("###")[0].strip()
        
        return response
    
    def interactive_chat(self):
        """Startet einen interaktiven Chat"""
        print("\n🏥 VHF-LLM Interaktiver Chat")
        print("=" * 40)
        print("Geben Sie 'quit' ein zum Beenden")
        print("Geben Sie 'help' für Beispiele ein")
        print("=" * 40)
        
        while True:
            print("\n" + "-" * 40)
            instruction = input("📋 Instruction: ").strip()
            
            if instruction.lower() == 'quit':
                print("👋 Auf Wiedersehen!")
                break
            
            if instruction.lower() == 'help':
                self._show_examples()
                continue
            
            if not instruction:
                print("❌ Bitte geben Sie eine Instruction ein")
                continue
            
            input_text = input("📝 Input (optional): ").strip()
            
            print("🤖 Generiere Antwort...")
            try:
                response = self.generate_response(instruction, input_text)
                print(f"\n💬 Antwort:\n{response}")
            except Exception as e:
                print(f"❌ Fehler bei der Generierung: {e}")
    
    def _show_examples(self):
        """Zeigt Beispiel-Instructions"""
        examples = [
            {
                "instruction": "Patient meldet sich mit Symptomen, die auf Vorhofflimmern hindeuten könnten",
                "input": "Ich habe Herzstolpern und fühle mich unwohl"
            },
            {
                "instruction": "EKG-Analyse erkennt Vorhofflimmern",
                "input": "Das EKG zeigt unregelmäßige R-R-Intervalle ohne erkennbare P-Wellen"
            },
            {
                "instruction": "Bewertung der Herzfrequenz im gültigen Bereich (50-150 bpm)",
                "input": "Meine gemessene Herzfrequenz beträgt 75 Schläge pro Minute"
            },
            {
                "instruction": "Beta-Blocker Medikationscheck",
                "input": "Ja, ich nehme täglich Metoprolol 50mg morgens ein"
            }
        ]
        
        print("\n📚 Beispiel-Instructions:")
        for i, ex in enumerate(examples, 1):
            print(f"\n{i}. Instruction: {ex['instruction']}")
            print(f"   Input: {ex['input']}")
    
    def batch_test(self, test_cases_file: str):
        """Testet mehrere Fälle aus einer JSON-Datei"""
        print(f"🧪 Batch-Testing mit: {test_cases_file}")
        
        with open(test_cases_file, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
        
        results = []
        
        for i, case in enumerate(test_cases):
            print(f"\n--- Test Case {i+1}/{len(test_cases)} ---")
            print(f"Instruction: {case['instruction']}")
            print(f"Input: {case.get('input', '')}")
            
            response = self.generate_response(
                case['instruction'], 
                case.get('input', '')
            )
            
            print(f"Antwort: {response}")
            
            results.append({
                'instruction': case['instruction'],
                'input': case.get('input', ''),
                'generated_response': response,
                'expected_response': case.get('expected_output', '')
            })
        
        # Ergebnisse speichern
        output_file = f"test_results_{len(test_cases)}_cases.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Ergebnisse gespeichert in: {output_file}")

def create_test_cases():
    """Erstellt eine Beispiel-Testdatei"""
    test_cases = [
        {
            "instruction": "Patient meldet sich mit Symptomen, die auf Vorhofflimmern hindeuten könnten",
            "input": "Ich habe Herzstolpern und fühle mich unwohl",
            "expected_output": "Ich verstehe, dass Sie Symptome verspüren..."
        },
        {
            "instruction": "EKG-Analyse erkennt Vorhofflimmern",
            "input": "Das EKG zeigt unregelmäßige R-R-Intervalle",
            "expected_output": "Die EKG-Analyse hat Vorhofflimmern erkannt..."
        },
        {
            "instruction": "Herzfrequenz unter 50 Schläge pro Minute erkannt",
            "input": "Meine gemessene Herzfrequenz beträgt 45 Schläge pro Minute",
            "expected_output": "Ihre Herzfrequenz von 45 Schlägen pro Minute liegt unter dem normalen Bereich..."
        }
    ]
    
    with open('test_cases.json', 'w', encoding='utf-8') as f:
        json.dump(test_cases, f, ensure_ascii=False, indent=2)
    
    print("✅ Test Cases erstellt: test_cases.json")

def main():
    parser = argparse.ArgumentParser(description='VHF-LLM Inference')
    parser.add_argument('--model_path', required=True, help='Pfad zum trainierten Modell')
    parser.add_argument('--use_lora', action='store_true', help='LoRA Modell verwenden')
    parser.add_argument('--interactive', action='store_true', help='Interaktiver Chat-Modus')
    parser.add_argument('--batch_test', help='Batch-Test mit JSON-Datei')
    parser.add_argument('--create_test_cases', action='store_true', help='Beispiel Test Cases erstellen')
    
    args = parser.parse_args()
    
    if args.create_test_cases:
        create_test_cases()
        return
    
    if not os.path.exists(args.model_path):
        print(f"❌ Modell-Pfad nicht gefunden: {args.model_path}")
        return
    
    # Device Info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")
    
    try:
        # Inference Setup
        inference = VHFInference(args.model_path, args.use_lora)
        
        if args.interactive:
            inference.interactive_chat()
        elif args.batch_test:
            inference.batch_test(args.batch_test)
        else:
            # Einzelner Test
            response = inference.generate_response(
                "Patient meldet sich mit Symptomen, die auf Vorhofflimmern hindeuten könnten",
                "Ich habe Herzstolpern und fühle mich unwohl"
            )
            print(f"🤖 Test-Antwort:\n{response}")
            
    except Exception as e:
        print(f"❌ Fehler: {e}")

if __name__ == "__main__":
    main()