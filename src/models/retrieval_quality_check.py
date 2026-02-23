"""Quick retrieval quality check for manual probes and FEVER subset metrics."""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models.retrieval import RetrievalSystem


PROBE_QUERIES: List[Dict] = [
    {"query": "The earth is flat", "expect_any": ["flat earth", "earth"]},
    {"query": "The Eiffel Tower is in Paris", "expect_any": ["eiffel tower", "paris"]},
    {"query": "Barack Obama was born in Hawaii", "expect_any": ["barack obama", "hawaii"]},
    {"query": "The capital of France is Paris", "expect_any": ["france", "paris"]},
]


def manual_probe(retriever: RetrievalSystem, top_k: int) -> Dict:
    rows = []
    hit_count = 0

    for item in PROBE_QUERIES:
        query = item["query"]
        expects = [e.lower() for e in item.get("expect_any", [])]
        results = retriever.retrieve(query, k=top_k)

        top_ids = [r["doc_id"] for r in results]
        top_ids_lower = [d.lower() for d in top_ids]
        hit = any(any(exp in doc_id for exp in expects) for doc_id in top_ids_lower)
        if hit:
            hit_count += 1

        rows.append(
            {
                "query": query,
                "hit": hit,
                "top_doc_ids": top_ids,
            }
        )

    return {
        "hit_rate": hit_count / len(PROBE_QUERIES),
        "rows": rows,
    }


def fever_subset_eval(retriever: RetrievalSystem, fever_path: str, sample_size: int) -> Dict:
    if not os.path.exists(fever_path):
        return {"error": f"Missing file: {fever_path}"}

    samples = []
    with open(fever_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not samples:
        return {"error": "No valid FEVER samples loaded"}

    corpus_titles = set(retriever.doc_ids)
    verifiable = 0
    with_gold = 0
    with_any_gold_in_corpus = 0
    for sample in samples:
        if str(sample.get("metadata", {}).get("verifiable", "")).upper() != "VERIFIABLE":
            continue
        verifiable += 1
        gold_pages = retriever._extract_gold_pages(sample.get("evidence", []))
        if not gold_pages:
            continue
        with_gold += 1
        if any(page in corpus_titles for page in gold_pages):
            with_any_gold_in_corpus += 1

    metrics = retriever.evaluate(samples, k_values=[1, 5, 10])
    return {
        "sample_size": len(samples),
        "verifiable": verifiable,
        "with_gold_pages": with_gold,
        "gold_page_coverage_in_corpus": (with_any_gold_in_corpus / with_gold) if with_gold else 0.0,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval quality check")
    parser.add_argument("--index", default="models/retrieval_index.pkl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--fever-path", default="data/fever/train_normalized.jsonl")
    parser.add_argument("--fever-sample", type=int, default=300)
    parser.add_argument("--output", default="results/retrieval_quality_report.json")
    args = parser.parse_args()

    retriever = RetrievalSystem(index_path=args.index)
    if not retriever.load_index():
        raise SystemExit("Failed to load retrieval index")

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "index": args.index,
        "manual_probe": manual_probe(retriever, top_k=args.top_k),
        "fever_subset": fever_subset_eval(retriever, fever_path=args.fever_path, sample_size=args.fever_sample),
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Saved: {args.output}")
    preview = json.dumps(report["manual_probe"], ensure_ascii=False, indent=2)
    print(preview.encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    main()
