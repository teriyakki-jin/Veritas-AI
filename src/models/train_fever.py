import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
import json
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.common import FactCheckingDataset

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1_macro': f1_score(labels, predictions, average='macro')
    }

class FEVERDataset(FactCheckingDataset):
    def __getitem__(self, idx):
        item = self.samples[idx]
        claim = str(item['text'])
        
        # Evidence handling (Top-k Concat)
        evidence_texts = item.get('evidence_texts', []) # Expecting list of strings
        if not evidence_texts:
             # Placeholder if no text
             input_text = claim
        else:
             # Concatenation of up to 3 evidence sentences
             top_k = evidence_texts[:3] 
             input_text = claim + " [SEP] " + " [SEP] ".join(top_k)
        
        # FEVER 3-class mapping
        label_map = {
            "SUPPORTS": 0,
            "REFUTES": 1,
            "NOT ENOUGH INFO": 2
        }
        
        # Handle label variation if any
        label_str = item.get('label_class', 'NOT ENOUGH INFO')
        label_id = label_map.get(label_str, 2)
        
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_id, dtype=torch.long)
        }

def train_fever():
    model_name = "distilbert-base-uncased"
    data_dir = "data/fever"
    output_dir = "models/fever_baseline"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=3
    )
    
    # Check data
    # Data Selection: Augmented (with evidence text) > Normalized (pointers only)
    if os.path.exists(os.path.join(data_dir, "train_augmented.jsonl")):
        print("Using Augmented FEVER data (with evidence text)...")
        train_path = os.path.join(data_dir, "train_augmented.jsonl")
    else:
        print("Using Normalized FEVER data (pointers only, placeholder text)...")
        train_path = os.path.join(data_dir, "train_normalized.jsonl")
        
    valid_path = os.path.join(data_dir, "validation_normalized.jsonl")
    
    
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found. Please run src/data/download_fever_final.py first.")
        return

    # Load full train dataset
    full_train_dataset = FEVERDataset(train_path, tokenizer)
    
    # Validation Splitting Strategy
    if os.path.exists(valid_path):
        train_dataset = full_train_dataset
        valid_dataset = FEVERDataset(valid_path, tokenizer)
        print(f"Loaded train ({len(train_dataset)}) and valid ({len(valid_dataset)}) from files.")
    else:
        print(f"Validation file {valid_path} not found. Splitting train set (90/10)...")
        # Simple split using random_split
        total_size = len(full_train_dataset)
        train_size = int(0.9 * total_size)
        valid_size = total_size - train_size
        
        train_dataset, valid_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, valid_size])
        print(f"Split into train ({train_size}) and valid ({valid_size}).")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available() # Use mixed precision if GPU available
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics
    )
    
    print("Starting training for FEVER...")
    trainer.train()
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    train_fever()
