"""Evaluation metrics: token F1, BLEU-1, ShardHit@B_probe, latency summary and mean/std over seeds."""
from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np

_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(s: str) -> str:
    """SQuAD normalisation: lowercase, no punctuation, no articles, single spaces."""
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    prec, rec = ns / len(p), ns / len(g)
    return 2 * prec * rec / (prec + rec)


def bleu1(pred: str, gold: str) -> float:
    """Unigram BLEU with brevity penalty (single reference)."""
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(bool(p) == bool(g))
    common = Counter(p) & Counter(g)
    prec = sum(common.values()) / len(p)
    bp = 1.0 if len(p) >= len(g) else float(np.exp(1 - len(g) / len(p)))
    return bp * prec


def shard_hit(probed: Iterable[int], gold: Iterable[int]) -> float:
    """1 if a probed shard is a gold shard."""
    return float(bool(set(probed) & set(gold)))


def latency_summary(ms: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(list(ms), dtype=np.float64)
    if arr.size == 0:
        return {"mean_ms": 0.0, "p95_ms": 0.0, "p50_ms": 0.0}
    return {"mean_ms": float(arr.mean()), "p95_ms": float(np.percentile(arr, 95)),
            "p50_ms": float(np.percentile(arr, 50))}


def mean_std_over_seeds(per_seed: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """{metric: {mean, std}} over seeds (population standard deviation)."""
    keys = sorted({k for d in per_seed for k in d})
    out = {}
    for k in keys:
        vals = np.array([d[k] for d in per_seed if k in d and d[k] == d[k]], dtype=np.float64)
        out[k] = {"mean": float(vals.mean()) if vals.size else float("nan"),
                  "std": float(vals.std()) if vals.size > 1 else 0.0}
    return out
