"""
End-to-End Inference Pipeline for the Combined Fact-Checking System.

Flow:
  Input Claim
    -> Evidence Retrieval (BM25 from FEVER wiki cache)
    -> Parallel Verification (LIAR / FEVER / FNN models)
    -> Fusion (weighted ensemble with temperature scaling)
    -> Output Verdict + Probabilities
"""

import os
import sys
import json
import logging
import torch
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from models.retrieval import RetrievalSystem
from models.fusion import FusionEngine, FEVER_CREDIBILITY, logits_to_credibility

logger = logging.getLogger(__name__)


class LRUCache:
    """Simple LRU cache for inference results."""

    def __init__(self, maxsize: int = 256):
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize

    def _make_key(self, claim: str, top_k: int) -> str:
        return f"{claim.strip().lower()}|{top_k}"

    def get(self, claim: str, top_k: int):
        key = self._make_key(claim, top_k)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, claim: str, top_k: int, result: Dict):
        key = self._make_key(claim, top_k)
        self._cache[key] = result
        self._cache.move_to_end(key)
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

# Default model directories
MODEL_DIRS = {
    "liar": "models/liar_baseline",
    "fever": "models/fever_baseline",
    "fnn": "models/fakenewsnet_baseline",
}

# Label maps for human-readable output
LIAR_LABELS = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
FEVER_LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
FNN_LABELS = ["fake", "real"]

LABEL_MAPS = {
    "liar": LIAR_LABELS,
    "fever": FEVER_LABELS,
    "fnn": FNN_LABELS,
}


class FactCheckPipeline:
    """End-to-end fact-checking inference pipeline."""

    def __init__(self, model_dirs: Optional[Dict[str, str]] = None,
                 retrieval_index: str = "models/retrieval_index.json",
                 device: Optional[str] = None):
        self.model_dirs = model_dirs or MODEL_DIRS.copy()
        self.retrieval_index = retrieval_index
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.models = {}
        self.tokenizers = {}
        self.retriever = None
        self.fusion = None
        self._cache = LRUCache(maxsize=256)

    def load(self):
        """Load all models, retrieval index, and fusion engine."""
        logger.info("Loading models on device: %s", self.device)

        # Load classification models
        for name, path in self.model_dirs.items():
            model_path = self._resolve_model_path(path)
            if model_path and os.path.exists(os.path.join(model_path, "config.json")):
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    model = AutoModelForSequenceClassification.from_pretrained(model_path)
                    model.to(self.device)
                    model.eval()
                    # Apply dynamic quantization on CPU for faster inference
                    if self.device == "cpu":
                        model = torch.quantization.quantize_dynamic(
                            model, {torch.nn.Linear}, dtype=torch.qint8
                        )
                    self.models[name] = model
                    self.tokenizers[name] = tokenizer
                    logger.info("  [%s] Loaded from %s", name, model_path)
                except Exception as e:
                    logger.warning("  [%s] Failed to load: %s", name, e)
            else:
                logger.warning("  [%s] Model not found at %s (skipped)", name, path)

        # Load retrieval system
        self.retriever = RetrievalSystem(index_path=self.retrieval_index)
        if os.path.exists(self.retrieval_index):
            self.retriever.load_index()
            logger.info("  [retrieval] Index loaded (%s docs)", len(self.retriever.corpus))
        else:
            # Try to build index if wiki cache exists
            cache_dir = "data/fever/wiki-cache"
            if os.path.exists(cache_dir) and os.listdir(cache_dir):
                logger.info("  [retrieval] Building index from wiki cache...")
                self.retriever.build_index(cache_dir)
            else:
                logger.warning("  [retrieval] No index or cache found (retrieval disabled)")

        # Initialize fusion engine
        self.fusion = FusionEngine(model_dirs=self.model_dirs)

        loaded = list(self.models.keys())
        logger.info("Pipeline ready. Active models: %s", loaded)
        return self

    def _resolve_model_path(self, path: str) -> Optional[str]:
        """Resolve model path, checking for checkpoints if main dir is empty."""
        if os.path.exists(os.path.join(path, "config.json")):
            return path
        # Check for latest checkpoint
        if os.path.isdir(path):
            checkpoints = sorted(
                [d for d in os.listdir(path) if d.startswith("checkpoint-")],
                key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0
            )
            if checkpoints:
                cp_path = os.path.join(path, checkpoints[-1])
                if os.path.exists(os.path.join(cp_path, "config.json")):
                    return cp_path
        return path

    def retrieve_evidence(self, claim: str, top_k: int = 3) -> List[Dict]:
        """Retrieve relevant evidence documents for a claim."""
        if self.retriever and self.retriever.bm25:
            return self.retriever.retrieve(claim, k=top_k)
        return []

    def predict_single(self, model_name: str, text: str, max_len: int = 512) -> np.ndarray:
        """Run a single model and return raw logits."""
        if model_name not in self.models:
            return np.array([])

        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]

        encoding = tokenizer(
            text,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output = model(**encoding)
            logits = output.logits.cpu().numpy()[0]

        return logits

    def verify(self, claim: str, top_k_evidence: int = 3) -> Dict:
        # Check cache first
        cached = self._cache.get(claim, top_k_evidence)
        if cached is not None:
            return cached

        evidence = self.retrieve_evidence(claim, top_k=top_k_evidence)
        evidence_texts = [e["text"] for e in evidence]
        model_outputs = self._build_model_outputs(claim, evidence_texts)
        if model_outputs and self.fusion:
            fusion_result = self.fusion.fuse(model_outputs)
        else:
            fusion_result = {
                "credibility_score": 0.5,
                "verdict": "UNKNOWN (no models loaded)",
                "model_details": {},
            }
        fusion_result = self._enrich_model_details(fusion_result)
        result = {
            "claim": claim,
            "evidence": [
                {"doc_id": e.get("doc_id", ""), "score": e.get("score", 0), "snippet": e.get("text", "")}
                for e in evidence
            ],
            "credibility_score": fusion_result["credibility_score"],
            "verdict": fusion_result["verdict"],
            "model_details": fusion_result["model_details"],
        }
        self._cache.put(claim, top_k_evidence, result)
        return result

    def _build_model_outputs(self, claim: str, evidence_texts: List[str]) -> Dict:
        """Run each loaded model and collect raw logits."""
        model_outputs = {}

        if "liar" in self.models:
            logits = self.predict_single("liar", claim)
            if logits.size > 0:
                model_outputs["liar"] = logits

        if "fever" in self.models:
            fever_input = claim + " [SEP] " + " [SEP] ".join(evidence_texts[:3]) if evidence_texts else claim
            logits = self.predict_single("fever", fever_input)
            if logits.size > 0:
                model_outputs["fever"] = logits

        if "fnn" in self.models:
            logits = self.predict_single("fnn", claim)
            if logits.size > 0:
                model_outputs["fnn"] = logits

        return model_outputs

    def explain_evidence(self, claim: str, evidence: List[Dict], top_k: int = 3) -> List[Dict]:
        """Leave-One-Out contribution of each evidence doc via the FEVER model.

        Returns a list parallel to `evidence`, each item containing:
          - contribution: float  (positive = doc supports claim, negative = refutes)
          - contribution_label: "supports" | "refutes" | "neutral"
        """
        if "fever" not in self.models or not evidence:
            return [{"contribution": 0.0, "contribution_label": "neutral"} for _ in evidence]

        temp = self.fusion.temperatures.get("fever", 1.0) if self.fusion else 1.0
        evidence_texts = [e.get("snippet", e.get("text", "")) for e in evidence]
        active = evidence_texts[:top_k]

        def _fever_score(texts: List[str]) -> float:
            inp = claim + " [SEP] " + " [SEP] ".join(texts) if texts else claim
            logits = self.predict_single("fever", inp)
            return float(logits_to_credibility(logits, FEVER_CREDIBILITY, temp))

        baseline = _fever_score(active)

        results = []
        for i in range(len(evidence)):
            if i >= top_k:
                results.append({"contribution": 0.0, "contribution_label": "neutral"})
                continue

            loo = [t for j, t in enumerate(active) if j != i]
            delta = round(baseline - _fever_score(loo), 4)

            if delta > 0.05:
                label = "supports"
            elif delta < -0.05:
                label = "refutes"
            else:
                label = "neutral"

            results.append({"contribution": delta, "contribution_label": label})

        return results

    def _enrich_model_details(self, fusion_result: Dict) -> Dict:
        """Add human-readable predicted_label to each model's detail dict (immutable)."""
        enriched = {}
        for name, detail in fusion_result.get("model_details", {}).items():
            labels = LABEL_MAPS.get(name, [])
            pred_class = detail.get("predicted_class", 0)
            label = labels[pred_class] if pred_class < len(labels) else None
            enriched[name] = {**detail, "predicted_label": label}
        return {**fusion_result, "model_details": enriched}


def main():
    """Interactive CLI for fact-checking."""
    pipeline = FactCheckPipeline()
    pipeline.load()

    logger.info("=== Fact-Checking System ===")
    logger.info("Enter a claim to verify (or 'quit' to exit):")

    while True:
        claim = input("Claim> ").strip()
        if not claim or claim.lower() in ("quit", "exit", "q"):
            break

        result = pipeline.verify(claim)
        logger.info("--- Result ---")
        logger.info("Verdict: %s", result["verdict"])
        logger.info("Credibility Score: %s", result["credibility_score"])

        if result["evidence"]:
            logger.info("Top Evidence:")
            for i, ev in enumerate(result["evidence"][:3], 1):
                doc_id = ev['doc_id'].encode('ascii', errors='replace').decode('ascii')
                snippet = ev['snippet'][:200].encode('ascii', errors='replace').decode('ascii')
                logger.info("  %s. [%s] (score: %.2f)", i, doc_id, ev["score"])
                logger.info("     %s", snippet)

        if result["model_details"]:
            logger.info("Model Breakdown:")
            for name, detail in result["model_details"].items():
                label = detail.get("predicted_label", "?")
                score = detail.get("credibility_score", 0)
                logger.info("  %s: %s (credibility: %.3f)", name, label, score)


if __name__ == "__main__":
    main()
