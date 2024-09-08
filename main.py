import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from adapter.advanced_adapter import AdvancedAdapter
from config import AdapterConfig

def load_local_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def browse_files(directory):
    for root, dirs, files in os.walk(directory):
        level = root.replace(directory, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f'{indent}{os.path.basename(root)}/')
        sub_indent = ' ' * 4 * (level + 1)
        for file in files:
            print(f'{sub_indent}{file}')

def main():
    config = AdapterConfig()
    adapter = AdvancedAdapter(config)
    print(f"Adapter using device: {adapter.device}")

    print("Welcome to the LLaMA 3.1 70B Advanced Adapter")
    while True:
        print("\n1. Load local model")
        print("2. Browse model files")
        print("3. Generate code")
        print("4. Exit")
        choice = input("Enter your choice (1-4): ")

        if choice == '1':
            model_path = input("Enter the path to your local LLaMA 3.1 70B model: ")
            try:
                model, tokenizer = load_local_model(model_path)
                print("Model loaded successfully!")
            except Exception as e:
                print(f"Error loading model: {e}")
        elif choice == '2':
            model_path = input("Enter the path to your local LLaMA 3.1 70B model directory: ")
            browse_files(model_path)
        elif choice == '3':
            if 'model' not in locals():
                print("Please load a model first (option 1)")
            else:
                language = input("Enter the programming language (python/cpp/javascript/java/ruby/go): ").lower()
                if language not in config.supported_languages:
                    print(f"Unsupported language: {language}")
                    continue
                prompt = input("Enter your code generation prompt: ")
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(adapter.device)
                attention_mask = torch.ones_like(input_ids)
                refactored_code, suggestions = adapter.generate(input_ids, attention_mask, language)
                print(f"\nGenerated and Refactored {language.capitalize()} Code:")
                print(refactored_code)
                print("\nOptimization Suggestions:")
                for i, suggestion in enumerate(suggestions, 1):
                    print(f"{i}. {suggestion}")
        elif choice == '4':
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
