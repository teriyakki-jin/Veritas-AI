import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models.fusion import FusionEngine
from models.inference import FactCheckPipeline

DATA_PATHS = {
    "liar": "data/liar/valid.jsonl",
    "fever": "data/fever/train_normalized.jsonl",
    "fnn": "data/welfake/test.jsonl",
}


def _read_jsonl(path: str, max_samples: int) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples > 0 and i >= max_samples:
                break
            rows.append(json.loads(line))
    return rows


def _target_score(sample: Dict) -> float:
    if "label_unified" in sample:
        return float(sample["label_unified"])

    # Fallback from label_class
    lc = str(sample.get("label_class", "")).lower()
    mapping = {
        "pants-fire": 0.0,
        "false": 0.1,
        "barely-true": 0.3,
        "half-true": 0.5,
        "mostly-true": 0.7,
        "true": 1.0,
        "supports": 1.0,
        "refutes": 0.0,
        "not enough info": 0.5,
        "fake": 0.0,
        "real": 1.0,
    }
    return float(mapping.get(lc, 0.5))


def _collect_outputs(pipeline: FactCheckPipeline, sample: Dict) -> Tuple[Dict[str, np.ndarray], float]:
    claim = str(sample.get("text", ""))
    target = _target_score(sample)

    evidence = pipeline.retrieve_evidence(claim, top_k=3)
    evidence_texts = [e["text"] for e in evidence]

    outputs: Dict[str, np.ndarray] = {}

    if "liar" in pipeline.models:
        logits = pipeline.predict_single("liar", claim)
        if logits.size > 0:
            outputs["liar"] = logits

    if "fever" in pipeline.models:
        fever_input = claim + " [SEP] " + " [SEP] ".join(evidence_texts[:3]) if evidence_texts else claim
        logits = pipeline.predict_single("fever", fever_input)
        if logits.size > 0:
            outputs["fever"] = logits

    if "fnn" in pipeline.models:
        logits = pipeline.predict_single("fnn", claim)
        if logits.size > 0:
            outputs["fnn"] = logits

    return outputs, target


def _build_dataset(pipeline: FactCheckPipeline, limits: Dict[str, int]) -> List[Tuple[Dict[str, np.ndarray], float]]:
    rows = []
    for source, path in DATA_PATHS.items():
        if not os.path.exists(path):
            continue
        samples = _read_jsonl(path, limits[source])
        for sample in samples:
            outputs, target = _collect_outputs(pipeline, sample)
            if outputs:
                rows.append((outputs, target))
    return rows


def _iter_weight_grid(step: float):
    grid = np.arange(0.0, 1.0 + 1e-9, step)
    for w_liar in grid:
        for w_fever in grid:
            w_fnn = 1.0 - w_liar - w_fever
            if w_fnn < -1e-9:
                continue
            if w_fnn < 0:
                w_fnn = 0.0
            yield {
                "liar": round(float(w_liar), 6),
                "fever": round(float(w_fever), 6),
                "fnn": round(float(w_fnn), 6),
            }


def _evaluate_weights(rows, weights, model_dirs) -> Dict[str, float]:
    engine = FusionEngine(weights=weights, model_dirs=model_dirs)
    preds = []
    trues = []
    for outputs, target in rows:
        fused = engine.fuse(outputs)
        preds.append(float(fused["credibility_score"]))
        trues.append(float(target))

    preds_arr = np.array(preds)
    trues_arr = np.array(trues)
    mse = float(np.mean((preds_arr - trues_arr) ** 2))
    mae = float(np.mean(np.abs(preds_arr - trues_arr)))
    return {"mse": mse, "mae": mae}


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid-search optimize fusion weights")
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--liar-max", type=int, default=300)
    parser.add_argument("--fever-max", type=int, default=300)
    parser.add_argument("--fnn-max", type=int, default=300)
    parser.add_argument("--out", default="models/optimal_weights.json")
    args = parser.parse_args()

    pipeline = FactCheckPipeline()
    pipeline.load()

    limits = {"liar": args.liar_max, "fever": args.fever_max, "fnn": args.fnn_max}
    rows = _build_dataset(pipeline, limits)
    if not rows:
        raise SystemExit("No samples available for optimization")

    best = None
    for weights in _iter_weight_grid(args.step):
        metrics = _evaluate_weights(rows, weights=weights, model_dirs=pipeline.model_dirs)
        cand = {"weights": weights, **metrics}
        if best is None or cand["mse"] < best["mse"]:
            best = cand

    output = {
        "num_samples": len(rows),
        "step": args.step,
        "best": best,
        "default": {"weights": {"liar": 0.35, "fever": 0.40, "fnn": 0.25}},
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
