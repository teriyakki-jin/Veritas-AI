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
import hashlib
import torch
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from models.retrieval import RetrievalSystem
from models.fusion import FusionEngine


class LRUCache:
    """Simple LRU cache for inference results."""

    def __init__(self, maxsize: int = 256):
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize

    def _make_key(self, claim: str, top_k: int) -> str:
        return hashlib.md5(f"{claim.strip().lower()}|{top_k}".encode()).hexdigest()

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
                 retrieval_index: str = "models/retrieval_index.pkl",
                 device: Optional[str] = None):
        self.model_dirs = model_dirs or MODEL_DIRS.copy()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.models = {}
        self.tokenizers = {}
        self.retriever = None
        self.fusion = None
        self._cache = LRUCache(maxsize=256)

    def load(self):
        """Load all models, retrieval index, and fusion engine."""
        print(f"Loading models on device: {self.device}")

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
                    print(f"  [{name}] Loaded from {model_path}")
                except Exception as e:
                    print(f"  [{name}] Failed to load: {e}")
            else:
                print(f"  [{name}] Model not found at {path} (skipped)")

        # Load retrieval system
        retrieval_index = "models/retrieval_index.pkl"
        self.retriever = RetrievalSystem(index_path=retrieval_index)
        if os.path.exists(retrieval_index):
            self.retriever.load_index()
            print(f"  [retrieval] Index loaded ({len(self.retriever.corpus)} docs)")
        else:
            # Try to build index if wiki cache exists
            cache_dir = "data/fever/wiki-cache"
            if os.path.exists(cache_dir) and os.listdir(cache_dir):
                print("  [retrieval] Building index from wiki cache...")
                self.retriever.build_index(cache_dir)
            else:
                print("  [retrieval] No index or cache found (retrieval disabled)")

        # Initialize fusion engine
        self.fusion = FusionEngine(model_dirs=self.model_dirs)

        loaded = list(self.models.keys())
        print(f"Pipeline ready. Active models: {loaded}")
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
        """
        Full verification pipeline for a single claim.

        Returns dict with:
          - claim: original input
          - evidence: retrieved evidence snippets
          - credibility_score: 0.0 (fake) to 1.0 (true)
          - verdict: human-readable label
          - model_details: per-model breakdown
        """
        # Check cache first
        cached = self._cache.get(claim, top_k_evidence)
        if cached is not None:
            return cached

        # Step 1: Evidence Retrieval
        evidence = self.retrieve_evidence(claim, top_k=top_k_evidence)
        evidence_texts = [e["text"] for e in evidence]

        # Step 2: Build input for each model
        model_outputs = {}

        # LIAR: claim text only (political claim classification)
        if "liar" in self.models:
            logits = self.predict_single("liar", claim)
            if logits.size > 0:
                model_outputs["liar"] = logits

        # FEVER: claim + evidence (claim verification with evidence)
        if "fever" in self.models:
            if evidence_texts:
                fever_input = claim + " [SEP] " + " [SEP] ".join(evidence_texts[:3])
            else:
                fever_input = claim
            logits = self.predict_single("fever", fever_input)
            if logits.size > 0:
                model_outputs["fever"] = logits

        # FNN: claim text only (fake news detection)
        if "fnn" in self.models:
            logits = self.predict_single("fnn", claim)
            if logits.size > 0:
                model_outputs["fnn"] = logits

        # Step 3: Fusion
        if model_outputs and self.fusion:
            fusion_result = self.fusion.fuse(model_outputs)
        else:
            fusion_result = {
                "credibility_score": 0.5,
                "verdict": "UNKNOWN (no models loaded)",
                "model_details": {},
            }

        # Add human-readable labels to model details
        for name, detail in fusion_result.get("model_details", {}).items():
            labels = LABEL_MAPS.get(name, [])
            pred_class = detail.get("predicted_class", 0)
            if pred_class < len(labels):
                detail["predicted_label"] = labels[pred_class]

        # Step 4: Assemble response & cache
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


def main():
    """Interactive CLI for fact-checking."""
    pipeline = FactCheckPipeline()
    pipeline.load()

    print("\n=== Fact-Checking System ===")
    print("Enter a claim to verify (or 'quit' to exit):\n")

    while True:
        claim = input("Claim> ").strip()
        if not claim or claim.lower() in ("quit", "exit", "q"):
            break

        result = pipeline.verify(claim)
        print(f"\n--- Result ---")
        print(f"Verdict: {result['verdict']}")
        print(f"Credibility Score: {result['credibility_score']}")

        if result["evidence"]:
            print(f"\nTop Evidence:")
            for i, ev in enumerate(result["evidence"][:3], 1):
                doc_id = ev['doc_id'].encode('ascii', errors='replace').decode('ascii')
                snippet = ev['snippet'][:200].encode('ascii', errors='replace').decode('ascii')
                print(f"  {i}. [{doc_id}] (score: {ev['score']:.2f})")
                print(f"     {snippet}")

        if result["model_details"]:
            print(f"\nModel Breakdown:")
            for name, detail in result["model_details"].items():
                label = detail.get("predicted_label", "?")
                score = detail.get("credibility_score", 0)
                print(f"  {name}: {label} (credibility: {score:.3f})")

        print()


if __name__ == "__main__":
    main()
