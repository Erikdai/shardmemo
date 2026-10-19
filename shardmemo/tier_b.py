"""Tier B read path (Algorithm 1): eligibility masking, routing inside the admissible set,
shard-local ANN, item-level re-check and global Top-K."""
from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np

from .config import Budgets, ShardMemoConfig
from .memory import EvidenceItem, ShardMap
from .ranking import fuse_scores, tokenize
from .router import Router, request_features
from .scope import Request, ScopePredicate


@dataclass
class TierBResult:
    evidence: list[EvidenceItem] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    probed: list[int] = field(default_factory=list)          # P_t
    admissible: list[int] = field(default_factory=list)      # S_t
    probs: dict[int, float] = field(default_factory=dict)    # p_{t,j}
    vecscan: int = 0                                         # vectors scanned over probed shards
    latency_ms: float = 0.0                                  # scope filtering -> candidate ranking
    route_ms: float = 0.0
    n_candidates: int = 0
    n_admissible_candidates: int = 0
    tau: float | None = None
    empty_scope: bool = False

    @property
    def evidence_texts(self) -> list[str]:
        return [e.shown() for e in self.evidence]


class TierB:
    def __init__(self, shard_map: ShardMap, router: Router | None, psi_b: ScopePredicate, cfg: ShardMemoConfig,
                 prefilter: Callable[[Request], Iterable[int] | None] | None = None):
        """`prefilter(q)` may return a superset of the admissible shard ids (e.g. from a document
        index) so that psi^B is evaluated on fewer shards; the admissible set is unchanged."""
        self.sm = shard_map
        self.router = router
        self.psi_b = psi_b
        self.cfg = cfg
        self.prefilter = prefilter

    def admissible(self, q: Request) -> np.ndarray:
        """S_t = {j : psi^B(m_j, q_t) = 1}."""
        pool = self.prefilter(q) if self.prefilter is not None else None
        shards = self.sm.shards if pool is None else [self.sm.shards[j] for j in sorted(set(pool))]
        return np.array([sh.sid for sh in shards if self.psi_b(sh.meta, q)], dtype=np.int64)

    def admissible_items(self, q: Request, items: list[EvidenceItem]) -> list[bool]:
        return [self.psi_b(it.meta, q) for it in items]

    def read(self, q: Request, z: np.ndarray, budgets: Budgets | None = None,
             cues: tuple[bool, bool] = (False, False), query_text: str | None = None) -> TierBResult:
        b = budgets or self.cfg.budgets
        res = TierBResult()
        t0 = time.perf_counter()

        adm = self.admissible(q)
        res.admissible = adm.tolist()
        if adm.size == 0:
            res.empty_scope = True
            res.latency_ms = (time.perf_counter() - t0) * 1e3
            return res
        cand = adm if self.cfg.scope_before_routing else np.arange(self.sm.S, dtype=np.int64)

        tr = time.perf_counter()
        phi = request_features(q, int(adm.size), self.sm.S, b.B_probe, *cues)
        logits = self.router.logits(z, phi, cand)
        pos, probs, info = self.router.select(logits, b.B_probe)
        probed = cand[pos]
        res.probs = {int(j): float(p) for j, p in zip(cand, probs)}
        res.tau = info.get("tau")
        res.route_ms = (time.perf_counter() - tr) * 1e3
        res.probed = [int(j) for j in probed]

        cand_items: list[tuple[int, int, float]] = []   # (shard id, local id, dense score)
        for j in probed:
            sh = self.sm.shards[int(j)]
            ids, sc, scanned = sh.index.search(z, b.L)
            res.vecscan += int(scanned)
            cand_items.extend((int(j), int(i), float(s)) for i, s in zip(ids, sc))
        res.n_candidates = len(cand_items)

        kept = [(j, i, s) for (j, i, s) in cand_items if self.psi_b(self.sm.shards[j].items[i].meta, q)]
        res.n_admissible_candidates = len(kept)
        if not kept:
            res.latency_ms = (time.perf_counter() - t0) * 1e3
            return res

        qtok = tokenize(query_text or q.text)
        dense = np.array([s for (_, _, s) in kept], dtype=np.float32)
        bm25 = np.zeros(len(kept), dtype=np.float32)
        by_shard: dict[int, list[int]] = {}
        for k, (j, _, _) in enumerate(kept):
            by_shard.setdefault(j, []).append(k)
        for j, ks in by_shard.items():
            bm25[ks] = self.sm.shards[j].bm25.score(qtok, [kept[k][1] for k in ks])
        route = np.array([res.probs[j] for (j, _, _) in kept], dtype=np.float32)
        fused = fuse_scores(bm25, dense, route, self.cfg.ranking)
        order = np.argsort(-fused, kind="stable")[: b.K]
        res.evidence = [self.sm.shards[kept[k][0]].items[kept[k][1]] for k in order]
        res.scores = [float(fused[k]) for k in order]
        res.latency_ms = (time.perf_counter() - t0) * 1e3
        return res

    def write(self, items: list[EvidenceItem]) -> list[int]:
        """Append each item to the shard of the same family with the largest scope-key overlap."""
        sids = []
        for it in items:
            best, best_s = None, -1
            for sh in self.sm.shards:
                if sh.family != it.meta.family:
                    continue
                s = (4 * len(sh.meta.conv_ids & it.meta.conv_ids) + len(sh.meta.speakers & it.meta.speakers)
                     + 2 * len(sh.meta.session_ids & it.meta.session_ids) + 4 * len(sh.meta.doc_ids & it.meta.doc_ids)
                     + 4 * len(sh.meta.namespaces & it.meta.namespaces))
                if s > best_s:
                    best, best_s = sh.sid, s
            if best is None:
                best = max(self.sm.shards, key=lambda sh: sh.family == it.meta.family).sid
            self.sm.add_items([it], best)
            sids.append(best)
        if self.router is not None:
            self.router.refresh()
        return sids
