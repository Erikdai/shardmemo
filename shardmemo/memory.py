"""Tier-B memory: evidence items, shards (ANN index, summary sigma_j, size n_j) and the shard-map
builder that organises items into shard families."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from .config import IndexConfig
from .embedding import Embedder
from .index import ShardIndex
from .ranking import MiniBM25, tokenize
from .scope import Metadata

FAMILIES = ("profile", "obs", "session", "doc", "spec")
FAMILY_ID = {f: i for i, f in enumerate(FAMILIES)}


@dataclass
class EvidenceItem:
    """A Tier-B memory unit x with metadata m(x)."""
    uid: str
    text: str                         # embedded and BM25-indexed text
    meta: Metadata
    display: str | None = None        # text shown to the generator
    source_ids: tuple[str, ...] = ()  # provenance ids, used for evidence-to-shard supervision
    time: datetime | None = None
    unit_key: tuple = ()              # grouping key of the shard-map builder, e.g. (conversation, session)

    def shown(self) -> str:
        return self.display or self.text


@dataclass
class Shard:
    sid: int
    family: str
    meta: Metadata
    items: list[EvidenceItem] = field(default_factory=list)
    index: ShardIndex | None = None
    bm25: MiniBM25 | None = None
    summary: np.ndarray | None = None   # sigma_j

    @property
    def n(self) -> int:
        return len(self.items)

    @property
    def family_id(self) -> int:
        return FAMILY_ID.get(self.family, 0)


class ShardMap:
    """S shards with shard-local ANN indexes, summaries and BM25 statistics."""

    def __init__(self, shards: list[Shard], embedder: Embedder, index_cfg: IndexConfig | None = None):
        self.shards = shards
        self.embedder = embedder
        self.index_cfg = index_cfg or IndexConfig()
        self._uid2shard: dict[str, int] = {}
        self._src2shards: dict[str, set] = defaultdict(set)
        for sh in shards:
            for it in sh.items:
                self._uid2shard[it.uid] = sh.sid
                for s in it.source_ids:
                    self._src2shards[s].add(sh.sid)

    @classmethod
    def build(cls, items: Sequence[EvidenceItem], embedder: Embedder, S: int,
              index_cfg: IndexConfig | None = None, family_min_shards: int = 1,
              conv_locality: bool = True, progress: bool = False) -> ShardMap:
        """Partition items into S shards.

        Shard quotas are proportional to item counts. Units (items sharing a `unit_key`) are split
        into contiguous, size-balanced groups; with conv_locality and enough shards, no shard spans
        two conversations, so scope predicates mask whole shards.
        """
        index_cfg = index_cfg or IndexConfig()
        by_family: dict[str, list[EvidenceItem]] = defaultdict(list)
        for it in items:
            by_family[it.meta.family].append(it)
        fams = [f for f in FAMILIES if f in by_family] + [f for f in by_family if f not in FAMILIES]
        S = max(S, len(fams) * family_min_shards)
        convs = sorted({it.unit_key[0] for it in items if it.unit_key}, key=str)

        groups_out: list[tuple[str, list[EvidenceItem]]] = []
        if conv_locality and len(convs) > 1 and len(convs) * len(fams) <= S:
            per_conv: dict[object, dict[str, list[EvidenceItem]]] = defaultdict(lambda: defaultdict(list))
            for it in items:
                per_conv[it.unit_key[0] if it.unit_key else None][it.meta.family].append(it)
            conv_sizes = {c: sum(len(v) for v in per_conv[c].values()) for c in convs}
            conv_quota = _proportional_quotas(conv_sizes, S, len(fams))
            for c in convs:
                fam_counts = {f: len(per_conv[c][f]) for f in fams if per_conv[c][f]}
                fam_units = {f: len({it.unit_key for it in per_conv[c][f]}) for f in fam_counts}
                fq = _proportional_quotas(fam_counts, conv_quota[c], family_min_shards, caps=fam_units)
                for f in fam_counts:
                    for g in _partition_units(per_conv[c][f], fq[f], False):
                        groups_out.append((f, g))
        else:
            quotas = _proportional_quotas({f: len(by_family[f]) for f in fams}, S, family_min_shards)
            for f in fams:
                for g in _partition_units(by_family[f], quotas[f], conv_locality):
                    groups_out.append((f, g))

        shards: list[Shard] = []
        for f, g in groups_out:
            meta = Metadata.union([it.meta for it in g], family=f)
            shards.append(Shard(sid=len(shards), family=f, meta=meta, items=g))
        sm = cls(shards, embedder, index_cfg)
        sm._index_all(progress=progress)
        return sm

    @classmethod
    def build_per_unit(cls, items: Sequence[EvidenceItem], embedder: Embedder,
                       index_cfg: IndexConfig | None = None, progress: bool = False) -> ShardMap:
        """One shard per unit_key, e.g. one shard per document."""
        units: dict[tuple, list[EvidenceItem]] = {}
        for it in items:
            units.setdefault(it.unit_key, []).append(it)
        shards = [Shard(sid=i, family=g[0].meta.family, items=g,
                        meta=Metadata.union([it.meta for it in g], family=g[0].meta.family))
                  for i, g in enumerate(units.values())]
        sm = cls(shards, embedder, index_cfg or IndexConfig())
        sm._index_all(progress=progress)
        return sm

    def _index_all(self, progress: bool = False) -> None:
        all_items = [it for sh in self.shards for it in sh.items]
        vecs = self.embedder.encode([it.text for it in all_items], progress=progress)
        pos = 0
        for sh in self.shards:
            v = vecs[pos:pos + sh.n]
            pos += sh.n
            sh.index = ShardIndex(self.embedder.dim, self.index_cfg)
            if sh.n:
                sh.index.add(v)
                mu = v.mean(axis=0)
                sh.summary = (mu / (np.linalg.norm(mu) + 1e-8)).astype(np.float32)
            else:
                sh.summary = np.zeros(self.embedder.dim, np.float32)
            sh.bm25 = MiniBM25([tokenize(it.text) for it in sh.items])

    def add_items(self, items: Iterable[EvidenceItem], sid: int) -> None:
        """Append evidence items to an existing shard and refresh its index, summary and BM25 statistics."""
        items = list(items)
        if not items:
            return
        sh = self.shards[sid]
        vecs = self.embedder.encode([it.text for it in items])
        sh.items.extend(items)
        sh.index.add(vecs)
        allv = sh.index.vectors
        mu = allv.mean(axis=0)
        sh.summary = (mu / (np.linalg.norm(mu) + 1e-8)).astype(np.float32)
        sh.meta = Metadata.union([sh.meta] + [it.meta for it in items], family=sh.family)
        sh.bm25 = MiniBM25([tokenize(it.text) for it in sh.items])
        for it in items:
            self._uid2shard[it.uid] = sid
            for s in it.source_ids:
                self._src2shards[s].add(sid)

    @property
    def S(self) -> int:
        return len(self.shards)

    @property
    def mean_size(self) -> float:
        return float(np.mean([sh.n for sh in self.shards])) if self.shards else 1.0

    @property
    def summaries(self) -> np.ndarray:
        return np.stack([sh.summary for sh in self.shards])

    @property
    def family_ids(self) -> np.ndarray:
        return np.array([sh.family_id for sh in self.shards], dtype=np.int64)

    def shard_of_uid(self, uid: str) -> int | None:
        return self._uid2shard.get(uid)

    def shards_of_source(self, source_id: str) -> set:
        return set(self._src2shards.get(source_id, ()))

    def doc_prefilter(self):
        """Prefilter for document-scoped requests: shards whose doc_ids intersect q.allowed_doc_ids."""
        doc2shards: dict[str, list[int]] = defaultdict(list)
        for sh in self.shards:
            for d in sh.meta.doc_ids:
                doc2shards[d].append(sh.sid)
        unscoped = [sh.sid for sh in self.shards if not sh.meta.doc_ids]

        def _pre(q):
            if q.allowed_doc_ids is None:
                return None
            out = list(unscoped)
            for d in q.allowed_doc_ids:
                out.extend(doc2shards.get(d, ()))
            return out
        return _pre

    def gold_shards(self, source_ids: Iterable[str]) -> set:
        """Shards that store any of the given provenance ids."""
        g = set()
        for s in source_ids:
            g |= self.shards_of_source(s)
        return g

    def describe(self) -> dict:
        fam = defaultdict(int)
        for sh in self.shards:
            fam[sh.family] += 1
        return {"S": self.S, "items": sum(sh.n for sh in self.shards), "mean_shard_size": self.mean_size,
                "families": dict(fam), "index": self.index_cfg.kind}


def _proportional_quotas(counts: dict, S: int, min_each: int, caps: dict | None = None) -> dict:
    """Integer quotas proportional to `counts` that sum to S where possible, each in [min_each, caps[f]]."""
    fams = list(counts)
    caps = caps or {}
    total = sum(counts.values()) or 1

    def cap(f):
        return max(min_each, caps.get(f, 10 ** 9))
    quotas = {f: min(cap(f), max(min_each, int(round(S * counts[f] / total)))) for f in fams}
    diff = S - sum(quotas.values())
    order = sorted(fams, key=lambda f: -counts[f])
    guard = 0
    while diff != 0 and fams and guard < 100 * len(order) + abs(diff):
        moved = False
        for f in order:
            if diff > 0 and quotas[f] < cap(f):
                quotas[f] += 1
                diff -= 1
                moved = True
            elif diff < 0 and quotas[f] > min_each:
                quotas[f] -= 1
                diff += 1
                moved = True
            if diff == 0:
                break
        guard += 1
        if not moved:
            break
    return quotas


def _balanced_contiguous_split(units: list[list[EvidenceItem]], k: int) -> list[list[EvidenceItem]]:
    """Split ordered units into min(k, len(units)) contiguous, non-empty groups of balanced size."""
    if not units:
        return []
    k = max(1, min(k, len(units)))
    cum = np.cumsum([len(u) for u in units])
    total = float(cum[-1])
    cuts, prev = [], 0
    for i in range(1, k):
        lo = prev + 1
        hi = len(units) - (k - i)
        want = total * i / k
        b = int(np.searchsorted(cum, want))
        if b + 1 <= hi and b >= 0 and abs(cum[b] - want) < abs((cum[b - 1] if b > 0 else 0) - want):
            b = b + 1
        b = min(max(b, lo), hi)
        cuts.append(b)
        prev = b
    bounds = [0] + cuts + [len(units)]
    return [[it for u in units[bounds[i]:bounds[i + 1]] for it in u] for i in range(k)]


def _partition_units(items: list[EvidenceItem], k: int, conv_locality: bool) -> list[list[EvidenceItem]]:
    units: dict[tuple, list[EvidenceItem]] = defaultdict(list)
    for it in items:
        units[it.unit_key].append(it)
    keys = sorted(units, key=lambda t: tuple(str(x) if not isinstance(x, (int, float)) else x for x in t))
    convs: dict[object, list[list[EvidenceItem]]] = {}
    for key in keys:
        convs.setdefault(key[0] if key else None, []).append(units[key])
    if conv_locality and k >= len(convs) and len(convs) > 1:
        quotas = _proportional_quotas({c: sum(len(u) for u in us) for c, us in convs.items()}, k, 1)
        groups: list[list[EvidenceItem]] = []
        for c, us in convs.items():
            groups.extend(_balanced_contiguous_split(us, quotas[c]))
        return groups
    return _balanced_contiguous_split([units[key] for key in keys], k)
