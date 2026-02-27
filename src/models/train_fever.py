import logging
import os
import sys

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.common import FactCheckingDataset, compute_metrics, FEVER_LABEL2ID

logger = logging.getLogger(__name__)


class FEVERDataset(FactCheckingDataset):
    def __getitem__(self, idx):
        item = self.samples[idx]
        claim = str(item['text'])

        evidence_texts = item.get('evidence_texts', [])
        if evidence_texts:
            input_text = claim + " [SEP] " + " [SEP] ".join(evidence_texts[:3])
        else:
            input_text = claim

        label_str = item.get('label_class', 'NOT ENOUGH INFO')
        label_id = FEVER_LABEL2ID.get(label_str, 2)

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
            'labels': torch.tensor(label_id, dtype=torch.long),
        }


def train_fever():
    model_name = "distilbert-base-uncased"
    data_dir = "data/fever"
    output_dir = "models/fever_baseline"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    if os.path.exists(os.path.join(data_dir, "train_augmented.jsonl")):
        logger.info("Using augmented FEVER data (with evidence text)...")
        train_path = os.path.join(data_dir, "train_augmented.jsonl")
    else:
        logger.info("Using normalized FEVER data (pointers only)...")
        train_path = os.path.join(data_dir, "train_normalized.jsonl")

    valid_path = os.path.join(data_dir, "validation_normalized.jsonl")

    if not os.path.exists(train_path):
        logger.error("Training data not found: %s. Run src/data/download_fever_final.py first.", train_path)
        return

    full_train_dataset = FEVERDataset(train_path, tokenizer)

    if os.path.exists(valid_path):
        train_dataset = full_train_dataset
        valid_dataset = FEVERDataset(valid_path, tokenizer)
        logger.info("Loaded train (%d) and valid (%d) from files.", len(train_dataset), len(valid_dataset))
    else:
        logger.info("Validation file not found. Splitting train set (90/10)...")
        total_size = len(full_train_dataset)
        train_size = int(0.9 * total_size)
        valid_size = total_size - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(
            full_train_dataset, [train_size, valid_size]
        )
        logger.info("Split into train (%d) and valid (%d).", train_size, valid_size)

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
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training for FEVER...")
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Model saved to %s", output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_fever()
