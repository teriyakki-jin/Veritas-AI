"""
Fusion Module: Combines outputs from multiple fact-checking models.

Supported strategies:
  - Weighted Average of normalized credibility scores
  - Temperature-scaled calibration before fusion

Each model outputs probabilities over its own label space:
  - LIAR:  6 classes (pants-fire, false, barely-true, half-true, mostly-true, true)
  - FEVER: 3 classes (SUPPORTS, REFUTES, NOT ENOUGH INFO)
  - FNN:   2 classes (fake, real)

All are mapped to a unified credibility score in [0, 1].
"""

import json
import os
import numpy as np
from typing import Dict, Optional


# Credibility score for each class (higher = more truthful)
LIAR_CREDIBILITY = {0: 0.0, 1: 0.1, 2: 0.3, 3: 0.5, 4: 0.7, 5: 1.0}
FEVER_CREDIBILITY = {0: 1.0, 1: 0.0, 2: 0.5}  # SUPPORTS=1, REFUTES=0, NEI=0.5
FNN_CREDIBILITY = {0: 0.0, 1: 1.0}  # fake=0, real=1

# Default weights for each model (sum to 1.0)
# Tuned via grid search (models/optimal_weights_quick.json, 60 samples):
# LIAR accuracy ~30% (near-random on 6-class) → excluded from fusion score.
DEFAULT_WEIGHTS = {
    "liar": 0.0,
    "fever": 0.8,
    "fnn": 0.2,
}


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to logits and return softmax probabilities."""
    if temperature <= 0:
        temperature = 1.0
    scaled = logits / temperature
    exp_scaled = np.exp(scaled - np.max(scaled))
    return exp_scaled / exp_scaled.sum()


def logits_to_credibility(logits: np.ndarray, credibility_map: Dict[int, float],
                          temperature: float = 1.0) -> float:
    """Convert raw logits to a single credibility score in [0, 1]."""
    probs = apply_temperature(logits, temperature)
    score = sum(probs[i] * credibility_map[i] for i in range(len(probs)))
    return float(np.clip(score, 0.0, 1.0))


def load_temperature(model_dir: str) -> float:
    """Load calibrated temperature from model directory. Defaults to 1.0."""
    temp_path = os.path.join(model_dir, "temperature.json")
    if os.path.exists(temp_path):
        with open(temp_path, "r") as f:
            return json.load(f).get("temperature", 1.0)
    return 1.0


class FusionEngine:
    """Fuses credibility scores from multiple models into a final verdict."""

    def __init__(self, weights: Optional[Dict[str, float]] = None,
                 model_dirs: Optional[Dict[str, str]] = None):
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.temperatures = {"liar": 1.0, "fever": 1.0, "fnn": 1.0}

        # Load calibrated temperatures if model directories are provided
        if model_dirs:
            for key, path in model_dirs.items():
                self.temperatures[key] = load_temperature(path)

    def fuse(self, model_outputs: Dict[str, np.ndarray]) -> Dict:
        """
        Fuse model outputs into a final verdict.

        Args:
            model_outputs: dict mapping model name to raw logits array.
                e.g. {"liar": np.array([...]), "fever": np.array([...]), "fnn": np.array([...])}

        Returns:
            dict with 'credibility_score', 'verdict', and per-model details.
        """
        credibility_maps = {
            "liar": LIAR_CREDIBILITY,
            "fever": FEVER_CREDIBILITY,
            "fnn": FNN_CREDIBILITY,
        }

        details = {}
        weighted_score = 0.0
        total_weight = 0.0

        for model_name, logits in model_outputs.items():
            if model_name not in credibility_maps:
                continue

            cmap = credibility_maps[model_name]
            temp = self.temperatures.get(model_name, 1.0)
            score = logits_to_credibility(logits, cmap, temp)
            probs = apply_temperature(logits, temp).tolist()
            predicted_class = int(np.argmax(logits))

            w = self.weights.get(model_name, 0.0)
            weighted_score += w * score
            total_weight += w

            details[model_name] = {
                "credibility_score": round(score, 4),
                "predicted_class": predicted_class,
                "probabilities": [round(p, 4) for p in probs],
                "temperature": round(temp, 4),
                "weight": w,
            }

        if total_weight > 0:
            final_score = weighted_score / total_weight
        else:
            final_score = 0.5

        verdict = score_to_verdict(final_score)

        return {
            "credibility_score": round(final_score, 4),
            "verdict": verdict,
            "model_details": details,
        }


def score_to_verdict(score: float) -> str:
    """Map credibility score to human-readable verdict."""
    if score >= 0.75:
        return "TRUE"
    elif score >= 0.55:
        return "MOSTLY TRUE"
    elif score >= 0.45:
        return "HALF TRUE"
    elif score >= 0.25:
        return "MOSTLY FALSE"
    else:
        return "FALSE"


if __name__ == "__main__":
    # Demo with synthetic logits
    engine = FusionEngine()

    sample_outputs = {
        "liar": np.array([0.1, 0.1, 0.2, 0.3, 0.2, 0.1]),  # 6 classes
        "fever": np.array([1.5, -0.5, 0.2]),      # 3 classes
        "fnn": np.array([-0.3, 1.2]),             # 2 classes
    }

    result = engine.fuse(sample_outputs)
    print(json.dumps(result, indent=2))
