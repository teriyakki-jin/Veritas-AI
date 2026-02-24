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

class FNNDataset(FactCheckingDataset):
    def __getitem__(self, idx):
        item = self.samples[idx]
        text = str(item['text'])
        
        # FNN 2-class mapping: fake->0, real->1
        # UnifiedSample 'label_class' is 'fake' or 'real'
        label_map = {
            "fake": 0,
            "real": 1
        }
        
        # Check if title exists, if not try text, else skip/empty
        text = str(item.get('text', ''))
        if not text:
             # Fallback to title if text is empty (UnifiedSample maps title -> text)
             text = str(item.get('title', ''))
        
        label_str = item.get('label_class', 'fake')
        label_id = label_map.get(label_str, 0)
        
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

def train_fakenewsnet():
    model_name = "distilbert-base-uncased"
    data_dir = "data/fakenewsnet"
    output_dir = "models/fakenewsnet_baseline"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2
    )
    
    
    # Check data
    # Priority: WELFake (if enabled/fallback) -> FNN
    # For now, check both.
    welfake_train = os.path.join("data/welfake", "train.jsonl")
    fnn_train = os.path.join(data_dir, "train.jsonl")
    
    train_path = None
    if os.path.exists(welfake_train):
        print(f"Using WELFake data from {welfake_train}")
        train_path = welfake_train
        valid_path = os.path.join("data/welfake", "test.jsonl")
    elif os.path.exists(fnn_train):
        print(f"Using Original FakeNewsNet data from {fnn_train}")
        train_path = fnn_train
        valid_path = os.path.join(data_dir, "test.jsonl")
    else:
        print(f"Error: No training data found (checked {fnn_train} and {welfake_train}).")
        return

    train_dataset = FNNDataset(train_path, tokenizer)
    valid_dataset = FNNDataset(valid_path, tokenizer) if os.path.exists(valid_path) else None
    
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
        fp16=torch.cuda.is_available()
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics
    )
    
    print("Starting training for FakeNewsNet...")
    trainer.train()
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    train_fakenewsnet()
