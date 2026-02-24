"""Heuristic claim extraction from article text for MVP usage."""

from __future__ import annotations

import re
from typing import List


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|(?<=다\.)\s+")


def _normalize_sentence(sentence: str) -> str:
    sentence = sentence.strip()
    sentence = re.sub(r"\s+", " ", sentence)
    return sentence


def _looks_like_claim(sentence: str) -> bool:
    s = sentence.lower()

    if len(sentence) < 20 or len(sentence) > 320:
        return False
    if sentence.endswith("?"):
        return False

    # Exclude high-opinion patterns.
    opinion_markers = ["i think", "we believe", "opinion", "editorial", "칼럼", "사설"]
    if any(marker in s for marker in opinion_markers):
        return False

    factual_markers = [
        " is ", " are ", " was ", " were ", " has ", " have ", " had ", " will ",
        "can", "said", "according", "reported", "announced", "발표", "확인", "밝혔",
    ]

    has_number = bool(re.search(r"\d", sentence))
    has_factual_marker = any(marker in s for marker in factual_markers)

    return has_number or has_factual_marker


def extract_claims(article_text: str, max_claims: int = 5) -> List[str]:
    text = re.sub(r"\s+", " ", article_text).strip()
    if not text:
        return []

    candidates = []
    for raw_sentence in SENTENCE_SPLIT_RE.split(text):
        sentence = _normalize_sentence(raw_sentence)
        if sentence and _looks_like_claim(sentence):
            candidates.append(sentence)

    # Deduplicate while preserving order.
    seen = set()
    deduped = []
    for sentence in candidates:
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(sentence)
        if len(deduped) >= max_claims:
            break

    if deduped:
        return deduped

    # Fallback: take the first long sentence if no heuristic match.
    for raw_sentence in SENTENCE_SPLIT_RE.split(text):
        sentence = _normalize_sentence(raw_sentence)
        if len(sentence) >= 20:
            return [sentence[:320]]

    return []
