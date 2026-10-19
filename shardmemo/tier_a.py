"""Tier A: bounded active context.

Holds session date anchors, plans and tool notes per conversation or agent and answers
ReadA(q_t, psi^A_t) with at most M items. The anchors let temporal and deictic references in the
evidence be resolved; the cue flags feed the router features.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from .scope import PSI_A_DEFAULT, Metadata, Request, ScopePredicate

_TEMPORAL_CUES = re.compile(
    r"\b(when|what (date|day|year|month|time)|how long|last (week|month|year|night|time)|yesterday|"
    r"ago|recently|earlier|later|before|after|since|until|during|next (week|month|year)|"
    r"this (week|month|year)|today|tomorrow|date|year|month|week)\b", re.I)
_DEICTIC_CUES = re.compile(r"\b(that|this|those|these|it|the (plan|idea|trip|event|one))\b", re.I)


def fmt_date(d: datetime) -> str:
    return f"{d.day} {d.strftime('%B')} {d.year}"


@dataclass
class ContextItem:
    kind: str            # session_anchor | turn | plan | tool_output | note
    text: str
    meta: Metadata
    time: datetime | None = None
    priority: float = 0.0


@dataclass
class ActiveContext:
    """A_t: ordered, bounded active context returned for a request."""
    items: list[ContextItem] = field(default_factory=list)
    temporal_cue: bool = False
    deictic_cue: bool = False
    anchor_time: datetime | None = None

    def render(self) -> str:
        return "\n".join(f"[{it.kind}] {it.text}" for it in self.items)


class ActiveContextStore:
    """Short-lived per-agent state with frequent updates."""

    def __init__(self, psi_a: ScopePredicate = PSI_A_DEFAULT):
        self.psi_a = psi_a
        self._items: dict[str, list[ContextItem]] = {}

    def add(self, key: str, item: ContextItem) -> None:
        self._items.setdefault(key, []).append(item)

    def add_session_anchor(self, conv_id: str, session_id: int, when: datetime, owner: str | None = None,
                           label: str | None = None) -> None:
        txt = label or f"Session {session_id} took place on {fmt_date(when)}."
        meta = Metadata.make(conv_id=conv_id, session_id=session_id, family="session", t=when, owner=owner)
        self.add(conv_id, ContextItem("session_anchor", txt, meta, time=when, priority=1.0))

    def add_note(self, key: str, kind: str, text: str, meta: Metadata | None = None, priority: float = 0.7) -> None:
        self.add(key, ContextItem(kind, text, meta or Metadata(), priority=priority))

    @staticmethod
    def cues(q: Request):
        return bool(_TEMPORAL_CUES.search(q.text)), bool(_DEICTIC_CUES.search(q.text))

    def read(self, q: Request, M: int, focus_sessions: list[int] | None = None) -> ActiveContext:
        """ReadA(q_t, psi^A_t): scope-filtered, priority-ordered, |A_t| <= M. Anchors of
        `focus_sessions` (sessions of the retrieved evidence, best first) are ranked first."""
        temporal, deictic = self.cues(q)
        pool = [it for it in self._items.get(q.conv_id or "", []) if self.psi_a(it.meta, q)]
        rank = {}
        for i, s in enumerate(focus_sessions or []):
            rank.setdefault(s, i)
        n_focus = max(len(rank), 1)

        def _prio(it: ContextItem) -> float:
            p = it.priority
            if it.kind == "session_anchor":
                p += 1.0 if temporal else 0.0
                hits = [rank[s] for s in it.meta.session_ids if s in rank]
                if hits:
                    p += 2.0 + (1.0 - min(hits) / n_focus)
            if it.kind == "turn" and deictic:
                p += 0.5
            if it.time is not None:
                p += 1e-10 * it.time.timestamp()
            return p

        pool.sort(key=_prio, reverse=True)
        chosen = pool[:M]
        anchors = [it.time for it in chosen if it.kind == "session_anchor" and it.time]
        return ActiveContext(items=chosen, temporal_cue=temporal, deictic_cue=deictic,
                             anchor_time=max(anchors) if anchors else (q.time or None))

    def anchor_for(self, conv_id: str, session_id: int) -> datetime | None:
        for it in self._items.get(conv_id, []):
            if it.kind == "session_anchor" and session_id in it.meta.session_ids:
                return it.time
        return None
