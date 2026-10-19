"""ShardMemo controller: Tier A read, tier gate, Tier C with C->B fallback, Tier B, generator."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from .config import Budgets, ShardMemoConfig
from .embedding import Embedder
from .generator import PROMPTS, Generator
from .memory import ShardMap
from .router import Router, ShardScorer
from .scope import PSI_A_DEFAULT, PSI_B_LOCOMO, Request, ScopePredicate
from .tier_a import ActiveContext, ActiveContextStore
from .tier_b import TierB, TierBResult
from .tier_c import TierC, TierCResult

_PROCEDURAL = re.compile(r"\b(get|fetch|look ?up|list|find|search|call|retrieve|check|convert|translate|"
                         r"book|send|calculate|compute|download|query)\b", re.I)


@dataclass
class ReadResult:
    request: Request
    tier_path: list[str]
    active: ActiveContext | None = None
    tier_b: TierBResult | None = None
    tier_c: TierCResult | None = None
    answer: str | None = None
    fallback: bool = False
    total_ms: float = 0.0

    def evidence_texts(self) -> list[str]:
        return self.tier_b.evidence_texts if self.tier_b else []


class ShardMemo:
    """Scope-constrained, budgeted agentic memory with three tiers."""

    def __init__(self, cfg: ShardMemoConfig, embedder: Embedder,
                 shard_map: ShardMap | None = None, scorer: ShardScorer | None = None,
                 psi_a: ScopePredicate = PSI_A_DEFAULT, psi_b: ScopePredicate = PSI_B_LOCOMO,
                 tier_c: TierC | None = None, generator: Generator | None = None,
                 prompt: str = PROMPTS["locomo"], active: ActiveContextStore | None = None, prefilter=None):
        self.cfg = cfg
        self.embedder = embedder
        self.active = active or ActiveContextStore(psi_a)
        self.tier_b: TierB | None = None
        self.router: Router | None = None
        if shard_map is not None:
            self.attach_shard_map(shard_map, scorer, psi_b, prefilter)
        self.tier_c = tier_c
        self.generator = generator
        self.prompt = prompt

    def attach_shard_map(self, shard_map: ShardMap, scorer: ShardScorer | None, psi_b: ScopePredicate,
                         prefilter=None) -> None:
        self.router = Router(self.cfg.router, self.cfg.routing_mode, shard_map, scorer, device=self.cfg.train.device)
        self.tier_b = TierB(shard_map, self.router, psi_b, self.cfg, prefilter=prefilter)

    def tier_gate(self, q: Request) -> list[str]:
        """Tiers to activate; ['C', 'B'] means the procedural path with Tier B as fallback."""
        if q.tier_hint in ("B", "C"):
            return [q.tier_hint] + (["B"] if q.tier_hint == "C" else [])
        procedural = q.allowed_namespaces is not None or _PROCEDURAL.search(q.text)
        if self.tier_c is not None and len(self.tier_c.store) and procedural:
            return ["C", "B"]
        return ["B"]

    def read(self, q: Request, budgets: Budgets | None = None, E_t: dict | None = None,
             generate: bool = True) -> ReadResult:
        b = budgets or self.cfg.budgets
        t0 = time.perf_counter()
        gate = self.tier_gate(q)
        out = ReadResult(request=q, tier_path=[])
        qtext = self.ranking_query(q)
        z = self.embedder.encode_one(qtext)

        cues = (False, False)
        if self.cfg.use_tier_a:
            out.active = self.active.read(q, b.M)
            cues = (out.active.temporal_cue, out.active.deictic_cue)
            out.tier_path.append("A")

        if gate[0] == "C" and self.tier_c is not None:
            out.tier_c = self.tier_c.read(q, E_t=E_t, R=b.R)
            out.tier_path.append("C")
            if out.tier_c.ok:
                out.answer = self._format_tool_result(out.tier_c.result)
                out.total_ms = (time.perf_counter() - t0) * 1e3
                return out
            out.fallback = "B" in gate and self.tier_b is not None

        if self.tier_b is not None and "B" in gate:
            out.tier_b = self.tier_b.read(q, z, b, cues, query_text=qtext)
            out.tier_path.append("B")
            if out.active is not None and out.tier_b.evidence:
                sessions = list(dict.fromkeys(s for e in out.tier_b.evidence for s in sorted(e.meta.session_ids)))
                out.active = self.active.read(q, b.M, focus_sessions=sessions)
        if generate and self.generator is not None:
            context = out.active.render() if out.active is not None else ""
            out.answer = self.generator.generate(q.text, out.evidence_texts(), context, self.prompt)
        out.total_ms = (time.perf_counter() - t0) * 1e3
        return out

    def ranking_query(self, q: Request) -> str:
        """Query text for routing, ANN and BM25; with strip_scope_terms the speaker names already
        enforced by psi^B are removed."""
        if not self.cfg.strip_scope_terms or not q.speakers_referenced:
            return q.text
        t = q.text
        for name in q.speakers_referenced:
            t = re.sub(r"(?i)\b" + re.escape(name) + r"('s)?\b", " ", t)
        t = " ".join(t.split())
        return t if len(t.split()) >= 2 else q.text

    @staticmethod
    def _format_tool_result(y: Any) -> str:
        if isinstance(y, dict):
            return "; ".join(f"{k}: {v}" for k, v in y.items())
        if isinstance(y, list):
            return "; ".join(str(x) for x in y)
        return str(y)
