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
    # labels are classification labels for LIAR (0-5)
    # But wait, UnifiedSample has float labels.
    # In common.py we returned 'label_score'.
    # We need to map labels for this task.
    
    # We will need to map `labels` back to int if they came as floats?
    # Trainer passes what the dataset returns.
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1_macro': f1_score(labels, predictions, average='macro')
    }

class LIARDataset(FactCheckingDataset):
    def __getitem__(self, idx):
        item = self.samples[idx]
        text = str(item['text'])
        # Map LIAR 6-class labels to int
        label_map = {
            "pants-fire": 0,
            "false": 1,
            "barely-true": 2,
            "half-true": 3,
            "mostly-true": 4,
            "true": 5
        }
        label_str = item.get('label_class', 'false') # Default to false
        label_id = label_map.get(label_str, 1)
        
        encoding = self.tokenizer(
            text,
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

def train_liar():
    model_name = "distilbert-base-uncased"
    data_dir = "data/liar"
    output_dir = "models/liar_baseline"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=6
    )
    
    # Check data
    train_path = os.path.join(data_dir, "train.jsonl")
    valid_path = os.path.join(data_dir, "valid.jsonl") # LIAR has 'valid.tsv' mapped to 'valid.jsonl'
    test_path = os.path.join(data_dir, "test.jsonl")
    
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found. Please run src/data/load_liar.py first.")
        return

    train_dataset = LIARDataset(train_path, tokenizer)
    valid_dataset = LIARDataset(valid_path, tokenizer) if os.path.exists(valid_path) else None
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",  # Updated from evaluation_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics
    )
    
    print("Starting training for LIAR...")
    trainer.train()
    
    print("Evaluate on test set...")
    if os.path.exists(test_path):
        test_dataset = LIARDataset(test_path, tokenizer)
        results = trainer.evaluate(test_dataset)
        print(results)
        
        # Save results
        with open("results_liar.json", "w") as f:
            json.dump(results, f, indent=2)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save label mapping for inference
    label_map = {
        "pants-fire": 0,
        "false": 1,
        "barely-true": 2,
        "half-true": 3,
        "mostly-true": 4,
        "true": 5
    }
    with open(os.path.join(output_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f)
        
    print(f"Model and label map saved to {output_dir}")

if __name__ == "__main__":
    train_liar()
