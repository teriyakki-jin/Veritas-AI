import logging
import os
import sys

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.common import FactCheckingDataset, compute_metrics, FNN_LABEL2ID

logger = logging.getLogger(__name__)


class FNNDataset(FactCheckingDataset):
    def __getitem__(self, idx):
        item = self.samples[idx]
        text = str(item.get('text', '') or item.get('title', ''))

        label_str = item.get('label_class', 'fake')
        label_id = FNN_LABEL2ID.get(label_str, 0)

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
            'labels': torch.tensor(label_id, dtype=torch.long),
        }


def train_fakenewsnet():
    model_name = "distilbert-base-uncased"
    data_dir = "data/fakenewsnet"
    output_dir = "models/fakenewsnet_baseline"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    welfake_train = os.path.join("data/welfake", "train.jsonl")
    fnn_train = os.path.join(data_dir, "train.jsonl")

    if os.path.exists(welfake_train):
        logger.info("Using WELFake data from %s", welfake_train)
        train_path = welfake_train
        valid_path = os.path.join("data/welfake", "test.jsonl")
    elif os.path.exists(fnn_train):
        logger.info("Using FakeNewsNet data from %s", fnn_train)
        train_path = fnn_train
        valid_path = os.path.join(data_dir, "test.jsonl")
    else:
        logger.error("No training data found (checked %s and %s).", fnn_train, welfake_train)
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
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training for FakeNewsNet...")
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Model saved to %s", output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_fakenewsnet()
