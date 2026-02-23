import torch
from torch.utils.data import Dataset
import json
from transformers import PreTrainedTokenizer
from typing import List, Dict
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.normalize import UnifiedSample

class FactCheckingDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, max_len: int = 128):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Load unified samples (JSONL)

        # Load unified samples (JSONL)
        if os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.samples.append(json.loads(line))
        else:
            print(f"Warning: {data_path} not found.")

        # FNN Coverage Check
        self.check_coverage()
        
    def check_coverage(self):
        """Checks percentage of samples with full text vs just title."""
        total = len(self.samples)
        if total == 0: return
        
        has_text = sum(1 for s in self.samples if s.get('text') and len(s['text']) > len(s.get('title', '')))
        coverage = has_text / total
        print(f"FNN Coverage Report: {coverage:.2%} have full text ({has_text}/{total})")
        
        # Save report
        os.makedirs("logs", exist_ok=True)
        with open("logs/data_coverage.json", "w") as f:
            json.dump({"fnn_text_coverage": coverage, "total_samples": total}, f)
            
        if coverage < 0.2:
            print("WARNING: Low text coverage. Consider using ISOT/WELFake fallback.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        text = item['text']
        label = float(item['label_unified']) # Using unified float score for regression or threshold
        
        # For classification, we might need mapped integer labels.
        # But UnifiedSample uses float. We'll support regression for now or map on the fly.
        # Let's map back to integer class based on task?
        # Actually simplest is to treat "Truthfulness" as a regression 0-1 or classification.
        # For this portfolio, let's stick to Classification (CrossEntropy) by mapping inputs.
        
        # HACK: For baseline, we'll specific mapping in the subclass or here if simple.
        # Let's add a 'label_id' field to the logic if we want classification.
        
        labels_map = {
            'liar': 6,
            'fever': 3,
            'fakenewsnet': 2
        }
        num_labels = labels_map.get(item['source'], 2)
        
        # Mapping float score to int label is tricky if we lost the original mapping.
        # Better to rely on 'label_raw' or 'label_class' and a fixed mapper.
        
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
            # 'labels': torch.tensor(label, dtype=torch.float) # Regression
             # We will handle labels in the training loop or map them appropriately
             'label_score': torch.tensor(label, dtype=torch.float),
             'source': item['source']
        }

def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score
    logits, labels = eval_pred
    # TODO: Implement based on task
    return {}
