import json
import os
import pickle
import re
from typing import Dict, List, Optional, Set

import nltk
import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

# Ensure NLTK data (simple tokenizer)
nltk_data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

try:
    nltk.data.find("tokenizers/punkt_tab")
except (LookupError, OSError):
    print(f"Downloading punkt_tab to {nltk_data_dir}...")
    nltk.download("punkt_tab", download_dir=nltk_data_dir, quiet=True)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    nltk.download("punkt", download_dir=nltk_data_dir, quiet=True)

# Domain-specific stopwords to reduce noisy retrieval on short factual claims.
STOPWORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is", "it",
    "its", "of", "on", "that", "the", "to", "was", "were", "will", "with", "this", "these", "those",
    "or", "not", "into", "over", "under", "about", "after", "before", "than", "then", "there", "their",
    "have", "had", "do", "does", "did", "which", "who", "whom", "what", "when", "where", "why", "how",
}


class RetrievalSystem:
    def __init__(self, index_path: str = "models/retrieval_index.pkl"):
        # Double check inside class as well just in case
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except (LookupError, OSError):
            nltk.download("punkt_tab", download_dir=nltk_data_dir, quiet=True)

        self.index_path = index_path
        self.bm25: Optional[BM25Okapi] = None
        self.corpus: List[str] = []
        self.doc_ids: List[str] = []
        self.doc_title_tokens: List[Set[str]] = []

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _tokenize(self, text: str, remove_stopwords: bool = True) -> List[str]:
        normalized = self._normalize_text(text)
        if not normalized:
            return []

        try:
            tokens = nltk.word_tokenize(normalized)
        except (LookupError, OSError):
            tokens = normalized.split()

        if remove_stopwords:
            tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
        return tokens

    def _build_title_tokens(self) -> None:
        self.doc_title_tokens = [set(self._tokenize(doc_id, remove_stopwords=True)) for doc_id in self.doc_ids]

    def save_index(self) -> None:
        """Save BM25 index and metadata to disk."""
        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(
                {
                    "bm25": self.bm25,
                    "corpus": self.corpus,
                    "doc_ids": self.doc_ids,
                },
                f,
            )
        print(f"Index saved to {self.index_path}")

    def load_index(self) -> bool:
        """Load BM25 index from disk. Returns True if successful."""
        if not os.path.exists(self.index_path):
            print(f"Index file {self.index_path} not found.")
            return False

        with open(self.index_path, "rb") as f:
            data = pickle.load(f)

        self.bm25 = data["bm25"]
        self.corpus = data["corpus"]
        self.doc_ids = data["doc_ids"]
        self._build_title_tokens()
        print(f"Index loaded: {len(self.corpus)} documents.")
        return True

    def build_index(self, cache_dir: str = "data/fever/wiki-cache") -> None:
        """Build BM25 index from cached Wiki pages."""
        print(f"Building index from {cache_dir}...")
        self.corpus = []
        self.doc_ids = []
        self.doc_title_tokens = []

        if not os.path.exists(cache_dir):
            print("Cache directory not found. Please run build_fever_cache.py first.")
            return

        files = os.listdir(cache_dir)
        for fname in tqdm(files):
            if not fname.endswith(".txt"):
                continue

            doc_id = fname.replace(".txt", "").replace("_", " ")
            fpath = os.path.join(cache_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()

            self.corpus.append(content)
            self.doc_ids.append(doc_id)

        if not self.corpus:
            print("No documents found in cache.")
            return

        print("Tokenizing corpus...")
        tokenized_corpus = [self._tokenize(doc, remove_stopwords=True) for doc in self.corpus]

        print("Fitting BM25...")
        self.bm25 = BM25Okapi(tokenized_corpus)
        self._build_title_tokens()

        self.save_index()
        print(f"Index built with {len(self.corpus)} documents.")

    def _rerank_score(self, query_tokens: Set[str], query_norm: str, idx: int, bm25_score: float) -> float:
        title_tokens = self.doc_title_tokens[idx] if idx < len(self.doc_title_tokens) else set()
        overlap = len(query_tokens & title_tokens)
        overlap_ratio = overlap / max(1, len(query_tokens))

        doc_title_norm = self._normalize_text(self.doc_ids[idx])
        phrase_bonus = 0.0
        if query_norm and query_norm in doc_title_norm:
            phrase_bonus = 1.5

        return float(bm25_score) + (2.0 * overlap_ratio) + phrase_bonus

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Return top-k evidence docs: {'doc_id', 'score', 'text'}"""
        if not self.bm25 and not self.load_index():
            raise ValueError("Index not loaded.")

        query_tokens_list = self._tokenize(query, remove_stopwords=True)
        if not query_tokens_list:
            query_tokens_list = self._tokenize(query, remove_stopwords=False)

        if not query_tokens_list:
            return []

        doc_scores = self.bm25.get_scores(query_tokens_list)

        candidate_k = min(len(doc_scores), max(k * 10, 50))
        candidate_idxs = np.argsort(doc_scores)[::-1][:candidate_k]

        query_tokens = set(query_tokens_list)
        query_norm = self._normalize_text(query)

        reranked = [
            (idx, self._rerank_score(query_tokens, query_norm, int(idx), float(doc_scores[idx])))
            for idx in candidate_idxs
        ]
        reranked.sort(key=lambda x: x[1], reverse=True)

        top_n = reranked[:k]
        results = []
        for idx, score in top_n:
            results.append(
                {
                    "doc_id": self.doc_ids[idx],
                    "score": float(score),
                    "text": self.corpus[idx][:500] + "...",
                }
            )
        return results

    @staticmethod
    def _extract_gold_pages(evidence_raw: List) -> Set[str]:
        gold_pages: Set[str] = set()

        for ev in evidence_raw:
            try:
                parsed = eval(ev) if isinstance(ev, str) else ev
            except Exception:
                continue

            if not isinstance(parsed, list):
                continue

            # FEVER normalized format often looks like:
            # "[[ann_id, evidence_id, 'Page_Title', line], ...]"
            # so parsed is a list of evidence entries.
            candidates = parsed
            if parsed and isinstance(parsed[0], (str, int, type(None))):
                candidates = [parsed]

            for item in candidates:
                if isinstance(item, list) and len(item) >= 3:
                    page_title = item[2]
                    if page_title:
                        gold_pages.add(str(page_title).replace("_", " "))

        return gold_pages

    def evaluate(self, test_samples: List[Dict], k_values: List[int] = [1, 5, 10]):
        """Evaluate Hit@k and Recall@k for FEVER page retrieval."""
        print("Evaluating Retrieval Performance...")
        if not self.bm25 and not self.load_index():
            print("Index not loaded. Skipping evaluation.")
            return {}

        metrics = {k: {"page_hit": 0, "page_recall": 0} for k in k_values}
        total = 0

        for sample in tqdm(test_samples):
            if str(sample.get("metadata", {}).get("verifiable", "")).upper() != "VERIFIABLE":
                continue

            claim = str(sample.get("text", ""))
            gold_pages = self._extract_gold_pages(sample.get("evidence", []))

            if not gold_pages:
                continue

            total += 1
            retrieved = self.retrieve(claim, k=max(k_values))
            retrieved_ids = [r["doc_id"] for r in retrieved]

            for k in k_values:
                top_k = retrieved_ids[:k]
                hits = sum(1 for p in top_k if p in gold_pages)
                if hits > 0:
                    metrics[k]["page_hit"] += 1
                metrics[k]["page_recall"] += hits / len(gold_pages)

        results = {}
        if total > 0:
            for k in k_values:
                results[f"Hit@{k} (Page)"] = metrics[k]["page_hit"] / total
                results[f"Recall@{k} (Page)"] = metrics[k]["page_recall"] / total

        print(f"Evaluated on {total} verifiable samples.")
        print("Retrieval Metrics:", json.dumps(results, indent=2))
        return results


if __name__ == "__main__":
    retriever = RetrievalSystem()
    cache_dir = "data/fever/wiki-cache"

    if os.path.exists(cache_dir):
        if not os.path.exists(retriever.index_path):
            retriever.build_index(cache_dir)
        else:
            retriever.load_index()

        test_query = "The Roman Empire was verified."
        docs = retriever.retrieve(test_query)
        print(f"Query: {test_query}")
        print("Retrieved:", docs)

        train_file = "data/fever/train_normalized.jsonl"
        if os.path.exists(train_file):
            with open(train_file, "r", encoding="utf-8") as f:
                samples = [json.loads(next(f)) for _ in range(10)]
            retriever.evaluate(samples, k_values=[1, 5])
