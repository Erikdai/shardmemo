"""Sentence embedder (all-MiniLM-L6-v2, L2-normalised) with an optional on-disk cache."""
from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np


class Embedder:
    """Sentence-transformers model with an in-memory cache, persisted to `cache_path` (.npz) on save."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu",
                 cache_path: str | None = None):
        self.model_name = model_name
        self._cache = {}
        self.cache_path = cache_path
        self._dirty = False
        if cache_path and os.path.exists(cache_path):
            with np.load(cache_path, allow_pickle=False) as f:
                texts = f["texts_blob"].tobytes().decode("utf-8").split("\x00")
                for t, v in zip(texts, f["vecs"]):
                    self._cache[t] = v.astype(np.float32)
        from sentence_transformers import SentenceTransformer
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self._model = SentenceTransformer(model_name, device=device)
        self.dim = int(self._model.get_sentence_embedding_dimension() or 384)

    def encode(self, texts: Sequence[str], batch_size: int = 256, progress: bool = False) -> np.ndarray:
        texts = list(texts)
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if todo:
            vecs = self._model.encode(todo, batch_size=batch_size, normalize_embeddings=True,
                                      show_progress_bar=progress, convert_to_numpy=True)
            for t, v in zip(todo, vecs):
                self._cache[t] = np.asarray(v, dtype=np.float32)
            self._dirty = True
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        return np.stack([self._cache[t] for t in texts])

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def save_cache(self) -> None:
        if not self.cache_path or not self._dirty or not self._cache:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.cache_path)), exist_ok=True)
        texts = [t for t in self._cache if "\x00" not in t]   # NUL separates texts in the blob
        blob = np.frombuffer("\x00".join(texts).encode("utf-8"), dtype=np.uint8)
        tmp = self.cache_path + ".tmp.npz"
        np.savez(tmp, texts_blob=blob, vecs=np.stack([self._cache[t] for t in texts]))
        os.replace(tmp, self.cache_path)
        self._dirty = False
