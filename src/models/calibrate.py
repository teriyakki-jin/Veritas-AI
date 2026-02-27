import argparse
import json
import logging
import os
import sys

import torch
from torch import nn, optim
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.common import FactCheckingDataset
from models.train_fakenewsnet import FNNDataset
from models.train_fever import FEVERDataset
from models.train_liar import LIARDataset

logger = logging.getLogger(__name__)


class ModelWithTemperature(nn.Module):
    """Wraps a model with Temperature Scaling."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input_ids, attention_mask):
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature


def _select_dataset(task_name: str, data_path: str, tokenizer, max_samples: int = 0, max_len: int = 128):
    if task_name == "liar":
        dataset = LIARDataset(data_path, tokenizer, max_len=max_len)
    elif task_name == "fever":
        dataset = FEVERDataset(data_path, tokenizer, max_len=max_len)
    elif task_name == "fnn":
        dataset = FNNDataset(data_path, tokenizer, max_len=max_len)
    else:
        dataset = FactCheckingDataset(data_path, tokenizer)

    if max_samples and max_samples > 0:
        max_samples = min(max_samples, len(dataset))
        dataset = torch.utils.data.Subset(dataset, list(range(max_samples)))

    return dataset


def calibrate_model(task_name, model_path, data_path, num_labels, batch_size=16, max_samples=0, max_len=128):
    logger.info("Calibrating %s from %s...", task_name, model_path)
    logger.info("Validation data: %s", data_path)

    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset = _select_dataset(task_name, data_path, tokenizer, max_samples=max_samples, max_len=max_len)
    if len(dataset) == 0:
        logger.warning("Dataset is empty; skipping calibration")
        return None

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    logits_list = []
    labels_list = []

    logger.info("Collecting validation logits (samples=%s)...", len(dataset))
    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            output_logits = model(input_ids=input_ids, attention_mask=mask).logits
            logits_list.append(output_logits)
            labels_list.append(labels)

    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)

    temperature = nn.Parameter(torch.ones(1).to(device) * 1.5)
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)
    cross_entropy = nn.CrossEntropyLoss()

    def eval_func():
        optimizer.zero_grad()
        # Clamp temperature to positive values (>0.1) for valid scaling
        with torch.no_grad():
            temperature.clamp_(min=0.1)
        loss = cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(eval_func)

    t_val = max(float(temperature.item()), 0.1)
    logger.info("Optimal Temperature for %s: %.4f", task_name, t_val)

    save_path = os.path.join(model_path, "temperature.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "temperature": t_val,
                "task": task_name,
                "data_path": data_path,
                "num_samples": len(dataset),
                "batch_size": batch_size,
            },
            f,
            indent=2,
        )

    logger.info("Saved to %s", save_path)
    return t_val


def main():
    parser = argparse.ArgumentParser(description="Temperature scaling calibration")
    parser.add_argument("--task", choices=["liar", "fever", "fnn"], required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--num-labels", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-len", type=int, default=128)
    args = parser.parse_args()

    calibrate_model(
        task_name=args.task,
        model_path=args.model_path,
        data_path=args.data_path,
        num_labels=args.num_labels,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        max_len=args.max_len,
    )


if __name__ == "__main__":
    main()
