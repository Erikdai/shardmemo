"""Shard-local ANN index that reports the number of vectors scanned per search.

HNSW counts the distance computations reported by faiss (hnsw_stats.ndis); flat search scans all n_j vectors.
"""
from __future__ import annotations

import faiss
import numpy as np

from .config import IndexConfig

faiss.omp_set_num_threads(1)   # multi-threaded HNSW construction is not deterministic


class ShardIndex:
    def __init__(self, dim: int, cfg: IndexConfig | None = None):
        self.cfg = cfg or IndexConfig()
        self.dim = dim
        self._vecs = np.zeros((0, dim), dtype=np.float32)
        self._index = None
        self._kind = "flat"

    @property
    def n(self) -> int:
        return int(self._vecs.shape[0])

    @property
    def kind(self) -> str:
        return self._kind

    def add(self, vecs: np.ndarray) -> None:
        vecs = np.ascontiguousarray(vecs, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs[None, :]
        self._vecs = np.vstack([self._vecs, vecs]) if self.n else vecs.copy()
        self._rebuild_if_needed(appended=vecs)

    def _rebuild_if_needed(self, appended: np.ndarray) -> None:
        want_hnsw = self.cfg.kind == "hnsw" and self.n >= self.cfg.flat_below
        if want_hnsw:
            if self._kind != "hnsw" or self._index is None:
                idx = faiss.IndexHNSWFlat(self.dim, self.cfg.hnsw_M, faiss.METRIC_INNER_PRODUCT)
                idx.hnsw.efConstruction = self.cfg.ef_construction
                idx.hnsw.efSearch = self.cfg.ef_search
                idx.add(self._vecs)
                self._index, self._kind = idx, "hnsw"
            else:
                self._index.add(np.ascontiguousarray(appended, dtype=np.float32))
        else:
            self._index, self._kind = None, "flat"

    def search(self, z: np.ndarray, L: int) -> tuple[np.ndarray, np.ndarray, int]:
        """(local ids, inner-product scores, vectors scanned) of the top-L items of this shard."""
        if self.n == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.float32), 0
        L = int(min(L, self.n))
        z = np.ascontiguousarray(z, dtype=np.float32).reshape(1, -1)
        if self._kind == "hnsw":
            stats = faiss.cvar.hnsw_stats
            stats.reset()
            self._index.hnsw.efSearch = max(self.cfg.ef_search, L)
            scores, ids = self._index.search(z, L)
            scanned = int(stats.ndis)
            keep = ids[0] >= 0
            return ids[0][keep].astype(np.int64), scores[0][keep].astype(np.float32), scanned
        scores = self._vecs @ z[0]
        if self.n > L:
            part = np.argpartition(-scores, L - 1)[:L]
            order = part[np.argsort(-scores[part])]
        else:
            order = np.argsort(-scores)
        return order.astype(np.int64), scores[order].astype(np.float32), self.n

    @property
    def vectors(self) -> np.ndarray:
        return self._vecs
