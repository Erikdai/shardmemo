"""BM25 scoring and the fused evidence score (BM25, dense similarity, routing probability)."""
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

import numpy as np

from .config import RankingConfig

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = set("a an the of to in on at for and or is are was were be been it its this that these those "
            "with by from as i you he she they we me him her them my your his their our do does did "
            "what when where who whom which why how".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP]


class MiniBM25:
    """BM25-Okapi that scores an arbitrary subset of its documents."""

    def __init__(self, docs: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf: list[Counter] = [Counter(d) for d in docs]
        self.dl = np.array([len(d) for d in docs], dtype=np.float32)
        self.avgdl = float(self.dl.mean()) if len(docs) else 1.0
        df: Counter = Counter()
        for d in self.tf:
            df.update(d.keys())
        N = max(len(docs), 1)
        self.idf: dict[str, float] = {t: math.log(1 + (N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def score(self, query_tokens: Sequence[str], doc_ids: Sequence[int]) -> np.ndarray:
        out = np.zeros(len(doc_ids), dtype=np.float32)
        q = [t for t in query_tokens if t in self.idf]
        if not q:
            return out
        for i, d in enumerate(doc_ids):
            tf, dl = self.tf[d], self.dl[d]
            s = 0.0
            for t in q:
                f = tf.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out[i] = s
        return out


def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.ones_like(x) if hi > 0 else np.zeros_like(x)
    return (x - lo) / (hi - lo)


def fuse_scores(bm25: np.ndarray, dense: np.ndarray, route: np.ndarray, cfg: RankingConfig) -> np.ndarray:
    """Weighted sum of min-max normalised BM25, dense and routing scores."""
    return cfg.w_bm25 * _minmax(bm25) + cfg.w_dense * _minmax(dense) + cfg.w_route * _minmax(route)
