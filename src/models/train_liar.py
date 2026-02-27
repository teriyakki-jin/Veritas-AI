import json
import logging
import os
import sys

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.common import FactCheckingDataset, compute_metrics, LIAR_LABEL2ID

logger = logging.getLogger(__name__)


class LIARDataset(FactCheckingDataset):
    def __getitem__(self, idx):
        item = self.samples[idx]
        text = str(item['text'])
        label_str = item.get('label_class', 'false')
        label_id = LIAR_LABEL2ID.get(label_str, 1)

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


def train_liar():
    model_name = "distilbert-base-uncased"
    data_dir = "data/liar"
    output_dir = "models/liar_baseline"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)

    train_path = os.path.join(data_dir, "train.jsonl")
    valid_path = os.path.join(data_dir, "valid.jsonl")
    test_path = os.path.join(data_dir, "test.jsonl")

    if not os.path.exists(train_path):
        logger.error("Training data not found: %s. Run src/data/load_liar.py first.", train_path)
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
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training for LIAR...")
    trainer.train()

    if os.path.exists(test_path):
        logger.info("Evaluating on test set...")
        test_dataset = LIARDataset(test_path, tokenizer)
        results = trainer.evaluate(test_dataset)
        logger.info("Test results: %s", results)

        with open("results_liar.json", "w") as f:
            json.dump(results, f, indent=2)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    with open(os.path.join(output_dir, "label_map.json"), "w") as f:
        json.dump(LIAR_LABEL2ID, f)

    logger.info("Model and label map saved to %s", output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_liar()
