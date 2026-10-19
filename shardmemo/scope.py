"""Scope predicates psi(m, q) in {0, 1}.

`Metadata` describes a single memory unit or a shard (the union of its units' metadata), so a
predicate evaluated on a shard admits it whenever it may contain admissible units; the item-level
re-check in Tier B removes the inadmissible ones.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol


def _fs(x: Iterable | None) -> frozenset:
    if x is None:
        return frozenset()
    if isinstance(x, (str, bytes)):
        return frozenset([x])
    return frozenset(x)


@dataclass(frozen=True)
class Metadata:
    """Scope keys, permissions and provenance of a memory unit or shard."""
    conv_ids: frozenset[str] = frozenset()
    speakers: frozenset[str] = frozenset()
    session_ids: frozenset[int] = frozenset()
    family: str = "obs"                        # profile | obs | session | doc | spec ...
    namespaces: frozenset[str] = frozenset()   # tool namespaces (Tier C / spec shards)
    doc_ids: frozenset[str] = frozenset()      # document ids (HotpotQA shards)
    permissions: frozenset[str] = frozenset()  # permission tags required to read
    t_min: datetime | None = None
    t_max: datetime | None = None
    owner: str | None = None                   # user / agent id owning the memory

    @staticmethod
    def make(conv_id=None, speaker=None, session_id=None, family="obs", namespace=None,
             doc_id=None, permissions=None, t=None, owner=None) -> Metadata:
        return Metadata(conv_ids=_fs([conv_id] if conv_id is not None else None),
                        speakers=_fs([speaker] if speaker is not None else None),
                        session_ids=_fs([session_id] if session_id is not None else None),
                        family=family,
                        namespaces=_fs([namespace] if namespace is not None else None),
                        doc_ids=_fs([doc_id] if doc_id is not None else None),
                        permissions=_fs(permissions), t_min=t, t_max=t, owner=owner)

    @staticmethod
    def union(ms: Sequence[Metadata], family: str | None = None) -> Metadata:
        ts = [m.t_min for m in ms if m.t_min is not None]
        te = [m.t_max for m in ms if m.t_max is not None]
        fams = {m.family for m in ms}
        return Metadata(
            conv_ids=frozenset().union(*[m.conv_ids for m in ms]),
            speakers=frozenset().union(*[m.speakers for m in ms]),
            session_ids=frozenset().union(*[m.session_ids for m in ms]),
            family=family or (next(iter(fams)) if len(fams) == 1 else "mixed"),
            namespaces=frozenset().union(*[m.namespaces for m in ms]),
            doc_ids=frozenset().union(*[m.doc_ids for m in ms]),
            permissions=frozenset().union(*[m.permissions for m in ms]),
            t_min=min(ts) if ts else None, t_max=max(te) if te else None,
            owner=ms[0].owner if ms and len({m.owner for m in ms}) == 1 else None,
        )


@dataclass
class Request:
    """A memory read request q with the request-time metadata used by psi and by the router features."""
    text: str
    conv_id: str | None = None
    speakers_referenced: frozenset[str] = frozenset()
    allowed_doc_ids: frozenset[str] | None = None       # HotpotQA: docs in the input context
    allowed_namespaces: frozenset[str] | None = None    # ToolBench: authorised tool namespaces
    permissions: frozenset[str] = frozenset()           # permissions held by the requester
    owner: str | None = None
    time: datetime | None = None
    time_window: tuple | None = None                    # (t_from, t_to) admissible window
    tier_hint: str | None = None                        # "B", "C", or None (gate decides)
    meta: dict = field(default_factory=dict)            # benchmark annotations (gold labels, category)


class ScopePredicate(Protocol):
    def __call__(self, m: Metadata, q: Request) -> bool: ...


def conversation_scope(m: Metadata, q: Request) -> bool:
    """Memory from the request's conversation."""
    if q.conv_id is None:
        return True
    return (not m.conv_ids) or (q.conv_id in m.conv_ids)


def speaker_scope(m: Metadata, q: Request) -> bool:
    """Memory produced by a speaker named in the request (all speakers if none is named)."""
    if not q.speakers_referenced or not m.speakers:
        return True
    return bool(m.speakers & q.speakers_referenced)


def doc_set_scope(m: Metadata, q: Request) -> bool:
    """Documents that appear in the request's input context."""
    if q.allowed_doc_ids is None:
        return True
    return (not m.doc_ids) or bool(m.doc_ids & q.allowed_doc_ids)


def namespace_scope(m: Metadata, q: Request) -> bool:
    """Skills and API specs whose tool namespace is authorised for the request."""
    if q.allowed_namespaces is None:
        return True
    return (not m.namespaces) or bool(m.namespaces & q.allowed_namespaces)


def permission_scope(m: Metadata, q: Request) -> bool:
    """All permissions required by the memory must be held by the requester."""
    return m.permissions <= q.permissions if m.permissions else True


def owner_scope(m: Metadata, q: Request) -> bool:
    if q.owner is None or m.owner is None:
        return True
    return m.owner == q.owner


def time_window_scope(m: Metadata, q: Request) -> bool:
    if q.time_window is None or (m.t_min is None and m.t_max is None):
        return True
    t_from, t_to = q.time_window
    lo = m.t_min or m.t_max
    hi = m.t_max or m.t_min
    return not (hi < t_from or lo > t_to)


def all_of(*preds: Callable[[Metadata, Request], bool]) -> ScopePredicate:
    def _pred(m: Metadata, q: Request) -> bool:
        return all(p(m, q) for p in preds)
    _pred.__name__ = "all_of(" + ",".join(getattr(p, "__name__", "p") for p in preds) + ")"
    return _pred


PSI_B_LOCOMO = all_of(conversation_scope, speaker_scope, permission_scope, owner_scope, time_window_scope)
PSI_B_HOTPOT = all_of(doc_set_scope, permission_scope)
PSI_C_TOOLBENCH = all_of(namespace_scope, permission_scope, owner_scope)
PSI_A_DEFAULT = all_of(conversation_scope, owner_scope)
