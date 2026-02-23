"""
Batch Evaluation Script for Fake News Detection Models.

Evaluates three individual models (LIAR, FEVER, FNN) on their respective
test sets, then performs ensemble evaluation via the FusionEngine.

Outputs:
  - results/liar_report.json, results/fever_report.json, results/fnn_report.json
  - results/liar_confusion.png, results/fever_confusion.png, results/fnn_confusion.png
  - results/comparison_table.json

Usage:
  python src/evaluate.py
"""

import argparse
import ast
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for headless / CI environments
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
except Exception:
    PLOT_AVAILABLE = False

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------------------------
# Path setup -- ensure src/ is on sys.path so relative imports resolve
# regardless of the working directory the script is launched from.
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from models.fusion import (
    FusionEngine,
    logits_to_credibility,
    LIAR_CREDIBILITY,
    FEVER_CREDIBILITY,
    FNN_CREDIBILITY,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LIAR_LABELS = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
FEVER_LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
FNN_LABELS = ["fake", "real"]

LABEL_MAPS: Dict[str, Dict[str, int]] = {
    "liar": {label: i for i, label in enumerate(LIAR_LABELS)},
    "fever": {label: i for i, label in enumerate(FEVER_LABELS)},
    "fnn": {label: i for i, label in enumerate(FNN_LABELS)},
}

LABEL_NAMES: Dict[str, List[str]] = {
    "liar": LIAR_LABELS,
    "fever": FEVER_LABELS,
    "fnn": FNN_LABELS,
}

MODEL_DIRS: Dict[str, str] = {
    "liar": os.path.join(PROJECT_ROOT, "models", "liar_baseline"),
    "fever": os.path.join(PROJECT_ROOT, "models", "fever_baseline"),
    "fnn": os.path.join(PROJECT_ROOT, "models", "fakenewsnet_baseline"),
}

DATA_PATHS: Dict[str, str] = {
    "liar": os.path.join(PROJECT_ROOT, "data", "liar", "test.jsonl"),
    "fever": os.path.join(PROJECT_ROOT, "data", "fever", "train_augmented.jsonl"),
    "fnn": os.path.join(PROJECT_ROOT, "data", "welfake", "test.jsonl"),
}

MAX_LENGTHS: Dict[str, int] = {
    "liar": 192,
    "fever": 256,
    "fnn": 256,
}

CREDIBILITY_MAPS: Dict[str, Dict[int, float]] = {
    "liar": LIAR_CREDIBILITY,
    "fever": FEVER_CREDIBILITY,
    "fnn": FNN_CREDIBILITY,
}

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
BATCH_SIZE = 32


# ===================================================================
# Data loading
# ===================================================================


def read_jsonl(path: str) -> List[Dict]:
    """Read every line of a JSONL file and return a list of dicts."""
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_fever_test_split(path: str, seed: int = 42) -> List[Dict]:
    """Return the last 10 % of FEVER data as a test split.

    Uses ``torch.Generator`` with a fixed seed so the split is
    identical to the one produced by ``torch.utils.data.random_split``
    during training (see ``train_fever.py``).
    """
    all_samples = read_jsonl(path)
    total = len(all_samples)
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(total, generator=g).tolist()
    train_size = int(0.9 * total)
    test_indices = indices[train_size:]
    return [all_samples[i] for i in test_indices]


# ===================================================================
# Text preparation (must mirror training-time preprocessing)
# ===================================================================


def prepare_text_liar(sample: Dict) -> str:
    """Construct LIAR input with metadata (speaker, subject, context).

    This matches the v2 training pipeline where metadata fields are
    appended to the claim separated by ``[SEP]``.
    """
    claim = str(sample.get("text", ""))

    meta = sample.get("metadata", {})
    if isinstance(meta, str):
        try:
            meta = ast.literal_eval(meta)
        except (ValueError, SyntaxError):
            meta = {}

    speaker = meta.get("speaker", "") if isinstance(meta, dict) else ""
    subject = meta.get("subject", "") if isinstance(meta, dict) else ""
    context = meta.get("context", "") if isinstance(meta, dict) else ""

    parts = [claim]
    if speaker:
        parts.append(f"Speaker: {speaker}")
    if subject:
        parts.append(f"Subject: {subject}")
    if context:
        parts.append(f"Context: {context}")
    return " [SEP] ".join(parts)


def prepare_text_fever(sample: Dict) -> str:
    """Construct FEVER input: claim [SEP] evidence_text(s)."""
    claim = str(sample.get("text", ""))
    evidence_texts = sample.get("evidence_texts", [])
    if isinstance(evidence_texts, list) and evidence_texts:
        return claim + " [SEP] " + " [SEP] ".join(evidence_texts[:3])
    return claim


def prepare_text_fnn(sample: Dict) -> str:
    """Return plain text for FNN (falls back to title if text is empty)."""
    text = str(sample.get("text", ""))
    if not text:
        text = str(sample.get("title", ""))
    return text


PREPARE_FN = {
    "liar": prepare_text_liar,
    "fever": prepare_text_fever,
    "fnn": prepare_text_fnn,
}


def gold_label(sample: Dict, dataset: str) -> int:
    """Map the ``label_class`` field to an integer id."""
    label_class = str(sample.get("label_class", "")).strip()
    return LABEL_MAPS[dataset].get(label_class, 0)


# ===================================================================
# Batch inference
# ===================================================================


def predict_batch_logits(
    model: torch.nn.Module,
    tokenizer,
    texts: List[str],
    max_len: int,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Run the model on *texts* in mini-batches and return a (N, C) logits array."""
    model.eval()
    all_logits: List[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits.cpu().numpy()
        all_logits.append(logits)

    return np.concatenate(all_logits, axis=0)


# ===================================================================
# Confusion-matrix heatmap (seaborn)
# ===================================================================


def save_confusion_png(
    cm: np.ndarray,
    labels: List[str],
    out_path: str,
    title: str,
) -> None:
    """Render confusion matrix image when plotting deps are available."""
    if not PLOT_AVAILABLE:
        print("  Plot libraries not installed; skipping confusion-matrix PNG")
        return

    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(7, n * 1.3), max(5, n * 1.1)))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved -> {out_path}")


# ===================================================================
# Per-model evaluation
# ===================================================================


def evaluate_single_model(
    dataset_name: str,
    samples: List[Dict],
    model: torch.nn.Module,
    tokenizer,
    device: str,
) -> Dict:
    """Evaluate a single model on its test set.

    Returns a dict containing metrics **and** the raw logits array
    (kept in memory for the subsequent ensemble step).
    """
    labels = LABEL_NAMES[dataset_name]
    max_len = MAX_LENGTHS[dataset_name]
    prep_fn = PREPARE_FN[dataset_name]

    texts = [prep_fn(s) for s in samples]
    y_true = [gold_label(s, dataset_name) for s in samples]

    print(f"\n{'=' * 60}")
    print(f"  Evaluating {dataset_name.upper()}  ({len(samples)} samples)")
    print(f"{'=' * 60}")

    logits = predict_batch_logits(model, tokenizer, texts, max_len, BATCH_SIZE, device)
    y_pred = np.argmax(logits, axis=1).tolist()

    acc = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)

    label_ids = list(range(len(labels)))
    report_dict = classification_report(
        y_true, y_pred,
        labels=label_ids,
        target_names=labels,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    report_str = classification_report(
        y_true, y_pred,
        labels=label_ids,
        target_names=labels,
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))

    # ---- console output ----
    print(f"\n  Accuracy : {acc:.4f}")
    print(f"  F1 Macro : {f1_mac:.4f}")
    print(f"\n  Classification Report:\n{report_str}")
    print(f"  Confusion Matrix (rows=true, cols=pred):")
    print(cm)

    # ---- save report JSON ----
    report_path = os.path.join(RESULTS_DIR, f"{dataset_name}_report.json")
    report_data = {
        "dataset": dataset_name,
        "num_samples": len(samples),
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_mac, 4),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"  Report saved -> {report_path}")

    # ---- save confusion-matrix PNG ----
    cm_path = os.path.join(RESULTS_DIR, f"{dataset_name}_confusion.png")
    save_confusion_png(cm, labels, cm_path, f"{dataset_name.upper()} Confusion Matrix")

    return {
        "dataset": dataset_name,
        "num_samples": len(samples),
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_mac, 4),
        "logits": logits,
    }


# ===================================================================
# Ensemble (FusionEngine) evaluation
# ===================================================================


def evaluate_ensemble(
    all_results: Dict[str, Dict],
    all_samples: Dict[str, List[Dict]],
) -> Dict:
    """Convert each model's logits into a credibility score, apply a 0.5
    threshold to obtain a binary (True / False) prediction, and evaluate
    the resulting binary classification across all test sets combined.
    """
    print(f"\n{'=' * 60}")
    print(f"  Evaluating ENSEMBLE (FusionEngine, threshold=0.5)")
    print(f"{'=' * 60}")

    engine = FusionEngine(model_dirs=MODEL_DIRS)

    # Ground-truth credibility anchors (same as fusion.py)
    gt_credibility: Dict[str, Dict[int, float]] = {
        "liar": dict(LIAR_CREDIBILITY),
        "fever": dict(FEVER_CREDIBILITY),
        "fnn": dict(FNN_CREDIBILITY),
    }

    all_y_true: List[int] = []
    all_y_pred: List[int] = []

    for ds_name in ["liar", "fever", "fnn"]:
        if ds_name not in all_results:
            continue

        logits_array: np.ndarray = all_results[ds_name]["logits"]
        samples = all_samples[ds_name]
        cmap = CREDIBILITY_MAPS[ds_name]
        temp = engine.temperatures.get(ds_name, 1.0)
        gt_cred = gt_credibility[ds_name]

        for i, sample in enumerate(samples):
            gt_id = gold_label(sample, ds_name)
            gt_binary = 1 if gt_cred[gt_id] >= 0.5 else 0

            pred_cred = logits_to_credibility(logits_array[i], cmap, temp)
            pred_binary = 1 if pred_cred >= 0.5 else 0

            all_y_true.append(gt_binary)
            all_y_pred.append(pred_binary)

    binary_labels = ["False", "True"]
    acc = accuracy_score(all_y_true, all_y_pred)
    f1_mac = f1_score(all_y_true, all_y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1])

    report_str = classification_report(
        all_y_true, all_y_pred,
        target_names=binary_labels,
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        all_y_true, all_y_pred,
        target_names=binary_labels,
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    print(f"\n  Ensemble Binary Accuracy : {acc:.4f}")
    print(f"  Ensemble Binary F1 Macro : {f1_mac:.4f}")
    print(f"\n{report_str}")
    print(f"  Confusion Matrix (rows=true, cols=pred):")
    print(cm)

    # Save ensemble confusion-matrix heatmap
    cm_path = os.path.join(RESULTS_DIR, "ensemble_confusion.png")
    save_confusion_png(cm, binary_labels, cm_path, "Ensemble (Fusion) Confusion Matrix")

    return {
        "ensemble_accuracy": round(acc, 4),
        "ensemble_f1_macro": round(f1_mac, 4),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
    }


# ===================================================================
# Main entry-point
# ===================================================================


def main() -> None:
    global RESULTS_DIR
    parser = argparse.ArgumentParser(description="Batch evaluate individual models and ensemble")
    parser.add_argument("--liar-max", type=int, default=0, help="max LIAR samples (0=all)")
    parser.add_argument("--fever-max", type=int, default=0, help="max FEVER samples (0=all)")
    parser.add_argument("--fnn-max", type=int, default=0, help="max FNN samples (0=all)")
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR)
    args = parser.parse_args()

    RESULTS_DIR = args.results_dir

    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = "cpu"  # Windows, CPU-only
    print(f"Device: {device}")
    print(f"Results directory: {RESULTS_DIR}")

    # ------------------------------------------------------------------
    # 1. Load test data
    # ------------------------------------------------------------------
    test_data: Dict[str, List[Dict]] = {}

    # LIAR -- dedicated test split
    liar_path = DATA_PATHS["liar"]
    if os.path.exists(liar_path):
        test_data["liar"] = read_jsonl(liar_path)
        if args.liar_max and args.liar_max > 0:
            test_data["liar"] = test_data["liar"][:args.liar_max]
        print(f"LIAR  test loaded : {len(test_data['liar']):>6} samples  ({liar_path})")
    else:
        print(f"WARNING: LIAR test data not found at {liar_path}")

    # FEVER -- last 10 % of train_augmented.jsonl (seed=42)
    fever_path = DATA_PATHS["fever"]
    if os.path.exists(fever_path):
        test_data["fever"] = load_fever_test_split(fever_path, seed=42)
        if args.fever_max and args.fever_max > 0:
            test_data["fever"] = test_data["fever"][:args.fever_max]
        print(f"FEVER test loaded : {len(test_data['fever']):>6} samples  (10 %% split of {fever_path})")
    else:
        print(f"WARNING: FEVER data not found at {fever_path}")

    # FNN -- dedicated test split
    fnn_path = DATA_PATHS["fnn"]
    if os.path.exists(fnn_path):
        test_data["fnn"] = read_jsonl(fnn_path)
        if args.fnn_max and args.fnn_max > 0:
            test_data["fnn"] = test_data["fnn"][:args.fnn_max]
        print(f"FNN   test loaded : {len(test_data['fnn']):>6} samples  ({fnn_path})")
    else:
        print(f"WARNING: FNN test data not found at {fnn_path}")

    # ------------------------------------------------------------------
    # 2. Evaluate each model individually
    # ------------------------------------------------------------------
    all_results: Dict[str, Dict] = {}

    for ds_name in ["liar", "fever", "fnn"]:
        if ds_name not in test_data:
            print(f"\nSkipping {ds_name} -- no test data available.")
            continue

        model_dir = MODEL_DIRS[ds_name]
        config_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_path):
            print(f"\nSkipping {ds_name} -- model not found at {model_dir}")
            continue

        print(f"\nLoading model: {model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.to(device)
        model.eval()

        result = evaluate_single_model(
            dataset_name=ds_name,
            samples=test_data[ds_name],
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        all_results[ds_name] = result

        # Free memory eagerly
        del model, tokenizer

    # ------------------------------------------------------------------
    # 3. Ensemble evaluation (FusionEngine credibility -> binary)
    # ------------------------------------------------------------------
    ensemble_result: Dict = {}
    if all_results:
        ensemble_result = evaluate_ensemble(all_results, test_data)

    # ------------------------------------------------------------------
    # 4. Comparison table
    # ------------------------------------------------------------------
    comparison: List[Dict] = []
    for ds_name in ["liar", "fever", "fnn"]:
        if ds_name in all_results:
            comparison.append({
                "dataset": ds_name,
                "accuracy": all_results[ds_name]["accuracy"],
                "f1_macro": all_results[ds_name]["f1_macro"],
            })
    if ensemble_result:
        comparison.append({
            "dataset": "ensemble",
            "accuracy": ensemble_result["ensemble_accuracy"],
            "f1_macro": ensemble_result["ensemble_f1_macro"],
        })

    comparison_path = os.path.join(RESULTS_DIR, "comparison_table.json")
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"\nComparison table saved -> {comparison_path}")

    # ------------------------------------------------------------------
    # 5. Console summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Dataset':<12} {'Accuracy':>10} {'F1 Macro':>10}")
    print(f"  {'-' * 34}")
    for row in comparison:
        print(f"  {row['dataset']:<12} {row['accuracy']:>10.4f} {row['f1_macro']:>10.4f}")
    print()


if __name__ == "__main__":
    main()
