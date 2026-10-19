"""Tier C: procedural skill memory.

Read (Algorithm 2): scope filter psi^C, latest in-scope version per skill id, Top-R by similarity,
order by (reliability, similarity), slot filling, preconditions, execution, validation.
Write (Algorithm 3): canonicalise a successful trace, abstract constants into typed slots, reuse or
mint a skill id, synthesise a deterministic test and store a new version only if it passes.
"""
from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .embedding import Embedder
from .scope import Metadata, Request, ScopePredicate


@dataclass
class ToolCall:
    tool: str
    args: dict[str, Any]


@dataclass
class Trace:
    """A tool-use trajectory; successful if `final` is not None."""
    task_id: str
    query: str
    calls: list[ToolCall]
    outputs: list[Any]
    final: Any
    meta: Metadata
    errors: list[bool] | None = None

    @property
    def n_steps(self) -> int:
        return len(self.calls)


@dataclass
class SlotSpec:
    name: str
    type: str                                 # int | float | year | date | email | url | entity | str
    templates: tuple[str, ...] = ()           # words preceding the constant in induction queries ("in", "for")
    example: Any = None


@dataclass
class SkillTest:
    slots: dict[str, Any]
    expected: Any


@dataclass
class SkillRecord:
    skill_id: str
    version: int
    meta: Metadata
    name: str
    description: str
    procedure: list[ToolCall]              # args may be {"$slot": ...}, {"$ref": ...} or {"$lookup": ...}
    slots: dict[str, SlotSpec]
    tests: list[SkillTest]
    reliability: float = 0.5
    n_success: int = 0
    n_fail: int = 0
    created_at: float = field(default_factory=time.time)
    result_keys: tuple[str, ...] = ()
    source_task: str = ""
    query_templates: tuple[str, ...] = ()  # induction queries with slot values abstracted to {slot}

    @property
    def key(self) -> str:
        return f"{self.skill_id}_v{self.version}"

    def text(self) -> str:
        """Retrieval text: the abstracted query templates the skill was induced from."""
        if self.query_templates:
            return " | ".join(self.query_templates[-8:])
        return f"{self.name}: {self.description}"


class ToolExecutor(Protocol):
    def call(self, tool: str, args: dict[str, Any]) -> Any: ...


class ToolError(Exception):
    pass


_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_URL = re.compile(r"https?://\S+")
_CAP_SPAN = re.compile(r"\b([A-Z][\w'&.-]*(?:[ -](?:[A-Z][\w'&.-]*|and|of|de|du|la|le))*)\b")


def infer_type(v: Any) -> str:
    if isinstance(v, bool):
        return "str"
    if isinstance(v, int):
        return "year" if 1900 <= v <= 2100 else "int"
    if isinstance(v, float):
        return "float"
    s = str(v)
    if _DATE.fullmatch(s):
        return "date"
    if _EMAIL.fullmatch(s):
        return "email"
    if _URL.fullmatch(s):
        return "url"
    if _NUM.fullmatch(s):
        return "year" if _YEAR.fullmatch(s) else ("float" if "." in s else "int")
    if s[:1].isupper():
        return "entity"
    return "str"


def _find_in_outputs(value: Any, outputs: list[Any]) -> tuple[int, str] | None:
    """Step and field of an earlier output that produced `value`."""
    for si, out in enumerate(outputs):
        if isinstance(out, dict):
            for k, v in out.items():
                if v == value or (isinstance(v, (int, float, str)) and str(v) == str(value)):
                    return si, k
        elif isinstance(out, list):
            for el in out:
                if isinstance(el, dict):
                    for k, v in el.items():
                        if v == value or str(v) == str(value):
                            return si, f"[*].{k}"
        elif out == value:
            return si, ""
    return None


def canonicalize(trace: Trace) -> tuple[list[ToolCall], list[Any]]:
    """Drop failed and repeated steps, normalise tool names and argument order."""
    calls, outs = [], []
    errs = trace.errors or [False] * len(trace.calls)
    prev = None
    for c, o, e in zip(trace.calls, trace.outputs, errs):
        if e or (isinstance(o, dict) and o.get("error")):
            continue
        norm = ToolCall(c.tool.strip().lower(), {k: c.args[k] for k in sorted(c.args)})
        sig = (norm.tool, tuple(sorted((k, str(v)) for k, v in norm.args.items())))
        if sig == prev:
            continue
        prev = sig
        calls.append(norm)
        outs.append(o)
    return calls, outs


_ARTICLES = {"the", "a", "an"}
_MASK_SKIP = {"i", "what", "which", "who", "whom", "where", "when", "how", "please", "list", "get", "find", "check",
              "look", "fetch", "give", "show", "tell", "retrieve", "search", "is", "are", "do", "does", "can", "could"}


def _template_before(query: str, pos: int) -> str:
    """Last non-article word before position `pos` (e.g. 'details of the X' -> 'of')."""
    before = [w.lower().strip(",.;:") for w in query[:pos].split()]
    while before and before[-1] in _ARTICLES:
        before.pop()
    return before[-1] if before else ""


def _find_mention(query: str, value: str) -> int:
    """Position of `value` in `query` as a whole word / phrase (case-insensitive), else -1."""
    m = re.search(r"(?<![\w])" + re.escape(value) + r"(?![\w])", query, flags=re.I)
    return m.start() if m else -1


def mask_entities(text: str) -> str:
    """Replace slot placeholders, quoted strings, numbers and non-initial capitalised spans by one
    placeholder token, so requests and templates match on their structure."""
    t = re.sub(r"\{[^}]*\}", " XSLOT ", text)
    t = re.sub(r'"[^"]+"', " XSLOT ", t)
    t = re.sub(r"(?<![\w])-?\d+(?:\.\d+)?(?![\w])", " XSLOT ", t)

    def _cap(m):
        span = m.group(0)
        if m.start() == 0 or span.lower() in _MASK_SKIP or span == "XSLOT":
            return span
        return "XSLOT"
    t = _CAP_SPAN.sub(_cap, t)
    t = re.sub(r"(XSLOT\s*)+", "XSLOT ", t)
    return " ".join(t.split()).replace("XSLOT", "something")


def _find_lookup(value: Any, outputs: list[Any], query: str) -> tuple[int, str, str, Any] | None:
    """(step, key, match_field, mention) when `value` is field `key` of an element of an earlier list
    output whose field `match_field` is mentioned in the query."""
    q_low = query.lower()
    for si, out in enumerate(outputs):
        if not isinstance(out, list) or len(out) < 2:
            continue
        for el in out:
            if not isinstance(el, dict):
                continue
            for k, v in el.items():
                if v == value or str(v) == str(value):
                    for mk, mv in el.items():
                        if mk != k and isinstance(mv, str) and len(mv) >= 2 and mv.lower() in q_low:
                            return si, k, mk, mv
    return None


def abstract_constants(calls: list[ToolCall], outputs: list[Any],
                       query: str) -> tuple[list[ToolCall], dict[str, SlotSpec]]:
    """Constants mentioned in the query become typed slots, constants produced by earlier steps become
    references or lookups into an earlier list output, all others stay constant."""
    slots: dict[str, SlotSpec] = {}
    proc: list[ToolCall] = []
    for si, c in enumerate(calls):
        new_args: dict[str, Any] = {}
        for k, v in c.args.items():
            sv = str(v)
            pos = _find_mention(query, sv) if sv else -1
            if pos >= 0 and len(sv) >= 2:
                name = k if k not in slots else f"{k}_{si}"
                slots[name] = SlotSpec(name=name, type=infer_type(v), templates=(_template_before(query, pos),),
                                       example=v)
                new_args[k] = {"$slot": name}
                continue
            lk = _find_lookup(v, outputs[:si], query) if si > 0 else None
            if lk is not None:
                step, key, match_field, mention = lk
                name = match_field if match_field not in slots else f"{match_field}_{si}"
                mpos = _find_mention(query, mention)
                slots[name] = SlotSpec(name=name, type=infer_type(mention),
                                       templates=(_template_before(query, mpos),), example=mention)
                new_args[k] = {"$lookup": {"step": step, "key": key, "match": match_field, "slot": name}}
                continue
            ref = _find_in_outputs(v, outputs[:si]) if si > 0 else None
            if ref is not None:
                new_args[k] = {"$ref": [ref[0], ref[1]]}
            else:
                new_args[k] = v
        proc.append(ToolCall(c.tool, new_args))
    return proc, slots


def _arg_kind(v: Any) -> str:
    if isinstance(v, dict):
        for kind in ("$slot", "$ref", "$lookup"):
            if kind in v:
                return kind[1:]
    return "const"


def procedure_signature(proc: list[ToolCall]) -> str:
    parts = []
    for c in proc:
        kinds = [f"{k}:{_arg_kind(v)}" for k, v in sorted(c.args.items())]
        parts.append(c.tool + "(" + ",".join(kinds) + ")")
    return "->".join(parts)


def render_procedure(proc: list[ToolCall]) -> str:
    def _a(v):
        kind = _arg_kind(v)
        if kind == "slot":
            return "{" + v["$slot"] + "}"
        if kind == "ref":
            return f"<step{v['$ref'][0]}.{v['$ref'][1]}>"
        if kind == "lookup":
            lk = v["$lookup"]
            return f"<step{lk['step']}[{lk['match']}=={{{lk['slot']}}}].{lk['key']}>"
        return repr(v)
    return " -> ".join(f"{c.tool}(" + ", ".join(f"{k}={_a(v)}" for k, v in c.args.items()) + ")" for c in proc)


class SlotFiller:
    """Fills typed slots from the request text and the context E_t, which may hold
    values {slot: value} and resolvers {slot: callable(text) -> value or None}."""

    def fill(self, q: Request, skill: SkillRecord, E_t: dict | None = None) -> tuple[dict[str, Any], bool]:
        E_t = E_t or {}
        values, resolvers = E_t.get("values", {}), E_t.get("resolvers", {})
        filled: dict[str, Any] = {}
        for name, spec in skill.slots.items():
            v = values.get(name)
            if v is None and name in resolvers:
                v = resolvers[name](q.text)
            if v is None:
                v = self._extract(q.text, spec)
            if v is None:
                return filled, False
            filled[name] = v
        return filled, True

    _SKIP = _MASK_SKIP | {"the", "a", "an", "today", "now", "usd", "api"}
    _PREP = {"in", "for", "of", "from", "about", "at", "on", "to", "named", "called", "titled"}

    def _extract(self, text: str, spec: SlotSpec) -> Any:
        t = spec.type
        if t in ("int", "float", "year"):
            cands = list(_NUM.finditer(text))
            if t == "year":
                cands = [m for m in cands if _YEAR.fullmatch(m.group(0))]
            if not cands:
                return None
            m = self._pick(text, cands, spec.templates)
            s = m.group(0)
            return int(s) if t in ("int", "year") and "." not in s else float(s)
        if t == "date":
            m = _DATE.search(text)
            return m.group(0) if m else None
        if t == "email":
            m = _EMAIL.search(text)
            return m.group(0) if m else None
        if t == "url":
            m = _URL.search(text)
            return m.group(0) if m else None
        quoted = re.findall(r'"([^"]+)"', text)
        if quoted:
            return quoted[0]
        cands = list(_CAP_SPAN.finditer(text))
        cands = [m for m in cands if m.group(0).lower() not in self._SKIP
                 and not (m.group(0).isupper() and len(m.group(0)) <= 4)]
        if not cands:
            return None
        return self._pick(text, cands, spec.templates).group(0)

    def _pick(self, text: str, cands: list, templates: tuple[str, ...]):
        """Prefer the candidate preceded by one of the skill's templates, then one preceded by a
        preposition, then the first candidate that is not sentence-initial."""
        def _before(m) -> str:
            return _template_before(text, m.start())
        for tpl in templates:
            if not tpl:
                continue
            for m in cands:
                if _before(m) == tpl:
                    return m
        for m in cands:
            if _before(m) in self._PREP:
                return m
        for m in cands:
            if m.start() > 0:
                return m
        return cands[0]


@dataclass
class TierCResult:
    skills: list[SkillRecord] = field(default_factory=list)
    sims: list[float] = field(default_factory=list)
    ok: bool = False
    executed: SkillRecord | None = None
    result: Any = None
    slots: dict[str, Any] = field(default_factory=dict)
    n_admissible: int = 0
    latency_ms: float = 0.0
    reason: str = ""


class SkillStore:
    """Versioned skill records with a procedure-signature index for deduplication."""

    def __init__(self, embedder: Embedder, mask: bool = True):
        self.embedder = embedder
        self.mask = mask
        self.records: list[SkillRecord] = []
        self._by_sig: dict[str, str] = {}
        self._vecs: np.ndarray | None = None
        self._tpl_vecs: list[np.ndarray] = []

    def add(self, rec: SkillRecord, signature: str | None = None) -> None:
        self.records.append(rec)
        if signature:
            self._by_sig[signature] = rec.skill_id
        v = self.embedder.encode([rec.text()])
        self._vecs = v if self._vecs is None else np.vstack([self._vecs, v])
        tpls = list(rec.query_templates) or [rec.text()]
        self._tpl_vecs.append(self.embedder.encode([mask_entities(t) if self.mask else t for t in tpls]))

    def encode_query(self, text: str) -> np.ndarray:
        return self.embedder.encode_one(mask_entities(text) if self.mask else text)

    def similarity(self, idxs: Sequence[int], zq: np.ndarray) -> np.ndarray:
        """sim(q, s): maximum cosine between the masked request and the skill's masked templates."""
        return np.array([float((self._tpl_vecs[i] @ zq).max()) for i in idxs], dtype=np.float32)

    def id_for_signature(self, signature: str) -> str | None:
        return self._by_sig.get(signature)

    def copy_signatures(self, other: SkillStore) -> None:
        """Reuse the procedure-signature index of `other`, so re-induced procedures keep their skill ids."""
        self._by_sig = dict(other._by_sig)

    def next_version(self, skill_id: str) -> int:
        vs = [r.version for r in self.records if r.skill_id == skill_id]
        return (max(vs) + 1) if vs else 1

    @property
    def vectors(self) -> np.ndarray:
        return self._vecs if self._vecs is not None else np.zeros((0, self.embedder.dim), np.float32)

    def __len__(self) -> int:
        return len(self.records)


class TierC:
    def __init__(self, store: SkillStore, executor: ToolExecutor, psi_c: ScopePredicate,
                 R: int = 3, tau_c: float = 0.5, slot_filler: SlotFiller | None = None,
                 ema: float = 0.9):
        self.store = store
        self.executor = executor
        self.psi_c = psi_c
        self.R = R
        self.tau_c = tau_c
        self.filler = slot_filler or SlotFiller()
        self.ema = ema

    def execute(self, proc: list[ToolCall], slots: dict[str, Any]) -> tuple[bool, Any, int]:
        outputs: list[Any] = []
        for c in proc:
            args = {}
            for k, v in c.args.items():
                if isinstance(v, dict) and "$slot" in v:
                    args[k] = slots.get(v["$slot"])
                elif isinstance(v, dict) and "$lookup" in v:
                    lk = v["$lookup"]
                    src = outputs[lk["step"]] if lk["step"] < len(outputs) else None
                    want = str(slots.get(lk["slot"], "")).lower()
                    hit = next((el for el in (src or []) if isinstance(el, dict)
                                and str(el.get(lk["match"], "")).lower() == want), None)
                    if hit is None:
                        hit = next((el for el in (src or []) if isinstance(el, dict) and want
                                    and (want in str(el.get(lk["match"], "")).lower()
                                         or str(el.get(lk["match"], "")).lower() in want)), None)
                    args[k] = hit.get(lk["key"]) if hit is not None else None
                    if args[k] is None:
                        return False, {"error": f"lookup failed for {lk['slot']}={want!r}"}, len(outputs)
                elif isinstance(v, dict) and "$ref" in v:
                    si, fld = v["$ref"]
                    src = outputs[si] if si < len(outputs) else None
                    if fld.startswith("[*]."):
                        key = fld[4:]
                        args[k] = next((el.get(key) for el in (src or []) if isinstance(el, dict) and key in el), None)
                    elif fld == "":
                        args[k] = src
                    else:
                        args[k] = src.get(fld) if isinstance(src, dict) else None
                    if args[k] is None:
                        return False, {"error": f"unresolved reference {fld}"}, len(outputs)
                else:
                    args[k] = v
            try:
                out = self.executor.call(c.tool, args)
            except Exception as e:
                return False, {"error": str(e)}, len(outputs) + 1
            if isinstance(out, dict) and out.get("error"):
                return False, out, len(outputs) + 1
            outputs.append(out)
        return True, (outputs[-1] if outputs else None), len(outputs)

    def admissible(self, q: Request) -> list[int]:
        return [i for i, r in enumerate(self.store.records) if self.psi_c(r.meta, q)]

    def current_versions(self, idxs: list[int]) -> list[int]:
        """Highest admissible version of each skill id."""
        best: dict[str, int] = {}
        for i in idxs:
            r = self.store.records[i]
            if r.skill_id not in best or r.version > self.store.records[best[r.skill_id]].version:
                best[r.skill_id] = i
        return sorted(best.values())

    def read(self, q: Request, E_t: dict | None = None, R: int | None = None, execute: bool = True) -> TierCResult:
        R = R or self.R
        t0 = time.perf_counter()
        res = TierCResult()
        adm = self.admissible(q)
        res.n_admissible = len(adm)
        if not adm:
            res.reason = "no admissible skill"
            res.latency_ms = (time.perf_counter() - t0) * 1e3
            return res
        cur = self.current_versions(adm)
        sims = self.store.similarity(cur, self.store.encode_query(q.text))
        order = np.argsort(-sims, kind="stable")[:R]
        res.skills = [self.store.records[cur[i]] for i in order]
        res.sims = [float(sims[i]) for i in order]
        if not execute:
            res.latency_ms = (time.perf_counter() - t0) * 1e3
            return res
        V = sorted(zip(res.skills, res.sims), key=lambda p: (p[0].reliability, p[1]), reverse=True)
        for s, sim in V:
            if sim < self.tau_c:
                continue
            slots, fillable = self.filler.fill(q, s, E_t)
            if not fillable or not self.validate_preconditions(s, slots):
                continue
            ok, y, _ = self.execute(s.procedure, slots)
            if ok and self.validate_result(s, slots, y):
                self._update_reliability(s, True)
                res.ok, res.executed, res.result, res.slots = True, s, y, slots
                res.latency_ms = (time.perf_counter() - t0) * 1e3
                return res
            self._update_reliability(s, False)
        res.reason = "no skill cleared tau_C / preconditions / validation" if res.skills else "empty"
        res.latency_ms = (time.perf_counter() - t0) * 1e3
        return res

    def validate_preconditions(self, s: SkillRecord, slots: dict[str, Any]) -> bool:
        for name, spec in s.slots.items():
            v = slots.get(name)
            if v is None or (isinstance(v, str) and not v.strip()):
                return False
            if spec.type in ("int", "year") and not isinstance(v, int):
                return False
            if spec.type == "float" and not isinstance(v, (int, float)):
                return False
        return True

    def validate_result(self, s: SkillRecord, slots: dict[str, Any], y: Any) -> bool:
        if y is None or (isinstance(y, dict) and y.get("error")):
            return False
        if isinstance(y, (list, dict, str)) and len(y) == 0:
            return False
        if s.result_keys and isinstance(y, dict):
            return all(k in y for k in s.result_keys)
        return True

    def _update_reliability(self, s: SkillRecord, success: bool) -> None:
        s.reliability = self.ema * s.reliability + (1 - self.ema) * float(success)
        if success:
            s.n_success += 1
        else:
            s.n_fail += 1

    def induce(self, trace: Trace, name: str | None = None) -> SkillRecord | None:
        """Induce, validate and version a skill from a successful trace; None if it fails its test."""
        if trace.final is None:
            return None
        calls, outs = canonicalize(trace)
        if not calls:
            return None
        proc, slots = abstract_constants(calls, outs, trace.query)
        sig = procedure_signature(proc)
        skill_id = self.store.id_for_signature(sig) or ("skill_" + hashlib.md5(sig.encode()).hexdigest()[:8])
        template = trace.query
        for n, s in slots.items():
            if s.example is not None and str(s.example):
                template = re.sub(re.escape(str(s.example)), "{" + n + "}", template, flags=re.I)
        query_templates: tuple[str, ...] = (template,)
        prev = [r for r in self.store.records if r.skill_id == skill_id]
        if prev:
            latest = max(prev, key=lambda r: r.version)
            for n, s in slots.items():
                if n in latest.slots:
                    merged = tuple(dict.fromkeys(latest.slots[n].templates + s.templates))
                    slots[n] = SlotSpec(name=s.name, type=s.type, templates=merged, example=s.example)
            query_templates = tuple(dict.fromkeys(latest.query_templates + query_templates))
        tests = self.synthesize_tests(trace, proc, slots, outs)
        if not self.validate_induction_tests(proc, tests):
            return None
        ver = self.store.next_version(skill_id)
        result_keys = tuple(sorted(outs[-1].keys())) if isinstance(outs[-1], dict) else ()
        rec = SkillRecord(
            skill_id=skill_id, version=ver, meta=trace.meta,
            name=name or (calls[-1].tool + "_skill"),
            description=f"{trace.query} || procedure: {render_procedure(proc)}",
            procedure=proc, slots=slots, tests=tests, result_keys=result_keys, source_task=trace.task_id,
            query_templates=query_templates)
        self.store.add(rec, signature=sig)
        return rec

    @staticmethod
    def synthesize_tests(trace: Trace, proc: list[ToolCall], slots: dict[str, SlotSpec],
                         outs: list[Any]) -> list[SkillTest]:
        """The induction constants must reproduce the trace result."""
        return [SkillTest(slots={n: s.example for n, s in slots.items()}, expected=outs[-1] if outs else trace.final)]

    def validate_induction_tests(self, proc: list[ToolCall], tests: list[SkillTest]) -> bool:
        for t in tests:
            ok, y, _ = self.execute(proc, t.slots)
            if not ok:
                return False
            if isinstance(t.expected, dict) and isinstance(y, dict):
                if set(t.expected) != set(y):
                    return False
            elif y != t.expected:
                return False
        return True
