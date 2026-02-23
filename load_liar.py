from datasets import load_dataset
import os

print("Loading LIAR dataset from Hugging Face Hub (ucsbnlp/liar)...")
try:
    # Try loading with full repo id
    dataset = load_dataset("ucsbnlp/liar", trust_remote_code=True)
    print("Dataset loaded successfully!")
    print(dataset)
except Exception as e:
    print(f"Failed to load dataset: {e}")
