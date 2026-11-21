from __future__ import annotations

import re
import string
from collections import Counter


def normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> bool:
    return normalize(prediction) == normalize(gold)


def f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize(prediction).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(pred_tokens)
    recall = same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def support_recall(evidence_keys: list[str], support_keys: list[str]) -> float:
    if not support_keys:
        return 0.0
    found = len(set(evidence_keys) & set(support_keys))
    return found / len(set(support_keys))
