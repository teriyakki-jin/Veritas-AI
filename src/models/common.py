import json
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import List, Dict

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.normalize import UnifiedSample

logger = logging.getLogger(__name__)

# ── Shared label mappings (single source of truth) ─────────────────────────────

LIAR_LABEL2ID: Dict[str, int] = {
    "pants-fire": 0,
    "false":      1,
    "barely-true": 2,
    "half-true":  3,
    "mostly-true": 4,
    "true":       5,
}

FEVER_LABEL2ID: Dict[str, int] = {
    "SUPPORTS":       0,
    "REFUTES":        1,
    "NOT ENOUGH INFO": 2,
}

FNN_LABEL2ID: Dict[str, int] = {
    "fake": 0,
    "real": 1,
}


# ── Shared compute_metrics (used by all training scripts) ───────────────────────

def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
    }


# ── Dataset base class ──────────────────────────────────────────────────────────

class FactCheckingDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, max_len: int = 128):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        if os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.samples.append(json.loads(line))
        else:
            logger.warning("Data file not found: %s", data_path)

        self.check_coverage()

    def check_coverage(self):
        """Log percentage of samples with full text vs just title."""
        total = len(self.samples)
        if total == 0:
            return

        has_text = sum(
            1 for s in self.samples
            if s.get('text') and len(s['text']) > len(s.get('title', ''))
        )
        coverage = has_text / total
        logger.info("FNN Coverage: %.1f%% have full text (%d/%d)", coverage * 100, has_text, total)

        os.makedirs("logs", exist_ok=True)
        with open("logs/data_coverage.json", "w") as f:
            json.dump({"fnn_text_coverage": coverage, "total_samples": total}, f)

        if coverage < 0.2:
            logger.warning("Low text coverage (%.1f%%). Consider using ISOT/WELFake fallback.", coverage * 100)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        text = item['text']
        label = float(item['label_unified'])

        labels_map = {'liar': 6, 'fever': 3, 'fakenewsnet': 2}
        num_labels = labels_map.get(item['source'], 2)  # noqa: F841 (kept for subclass override)

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
            'label_score': torch.tensor(label, dtype=torch.float),
            'source': item['source'],
        }
