"""HotpotQA under the MemAgent long-context protocol.

The memory store holds every document of the HotpotQA dev (distractor) split, one Tier-B shard per
document with one item per sentence. A request's context for a target length (56K / 224K / 448K
tokens) contains its gold documents, the distractors of the example and documents sampled from the
pool until the length is reached; psi^B admits only the documents of that context.
"""
from __future__ import annotations

import json
import random
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..memory import EvidenceItem
from ..scope import Metadata, Request

HF_PARQUET_URL = "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet"
TOKENS_PER_WORD = 1.3


def _from_parquet(path: Path) -> list[dict]:
    import pandas as pd
    df = pd.read_parquet(path)
    out = []
    for r in df.itertuples(index=False):
        ctx = [[str(t), [str(s) for s in sents]] for t, sents in zip(r.context["title"], r.context["sentences"])]
        sf = [[str(t), int(i)] for t, i in zip(r.supporting_facts["title"], r.supporting_facts["sent_id"])]
        out.append({"_id": r.id, "question": r.question, "answer": r.answer, "type": r.type, "level": r.level,
                    "supporting_facts": sf, "context": ctx})
    return out


def load_hotpot_dev(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(HF_PARQUET_URL, p)
    if p.suffix == ".parquet":
        return _from_parquet(p)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def approx_tokens(text: str) -> int:
    return int(len(text.split()) * TOKENS_PER_WORD)


@dataclass
class HotpotPool:
    items: list[EvidenceItem]
    docs: dict[str, list[str]]
    doc_tokens: dict[str, int]
    titles: list[str]


@dataclass
class HotpotData:
    requests: list[Request]
    context_tokens: int
    stats: dict = field(default_factory=dict)


def build_pool(examples: Sequence[dict]) -> HotpotPool:
    docs: dict[str, list[str]] = {}
    for ex in examples:
        for title, sents in ex["context"]:
            docs.setdefault(title, [s.strip() for s in sents if s.strip()])
    titles = sorted(docs)
    items = [EvidenceItem(uid=f"{t}::{si}", text=f"{t}: {s}", display=f"[{t}] {s}",
                          meta=Metadata.make(doc_id=t, family="doc"), source_ids=(f"{t}::{si}", t), unit_key=(t,))
             for t in titles for si, s in enumerate(docs[t])]
    doc_tokens = {t: approx_tokens(" ".join(docs[t])) for t in titles}
    return HotpotPool(items=items, docs=docs, doc_tokens=doc_tokens, titles=titles)


def build_requests(examples: Sequence[dict], pool: HotpotPool, context_tokens: int, seed: int = 0) -> HotpotData:
    """Requests whose admissible documents form a context of about `context_tokens` tokens."""
    rng = random.Random(seed * 1_000_003 + context_tokens)
    reqs: list[Request] = []
    sizes, ndocs = [], []
    for ex in examples:
        gold_titles = sorted({t for t, _ in ex["supporting_facts"]})
        allowed = list(dict.fromkeys(gold_titles + [t for t, _ in ex["context"]]))
        seen = set(allowed)
        tok = sum(pool.doc_tokens.get(t, 0) for t in allowed)
        while tok < context_tokens:
            t = pool.titles[rng.randrange(len(pool.titles))]
            if t in seen:
                continue
            seen.add(t)
            allowed.append(t)
            tok += pool.doc_tokens[t]
        sizes.append(tok)
        ndocs.append(len(allowed))
        sf_ids = [f"{t}::{i}" for t, i in ex["supporting_facts"]]
        reqs.append(Request(text=ex["question"], allowed_doc_ids=frozenset(allowed),
                            meta={"qid": ex["_id"], "answer": ex["answer"], "gold_docs": gold_titles,
                                  "evidence": sf_ids, "type": ex.get("type"), "level": ex.get("level"),
                                  "distractors": [t for t, _ in ex["context"] if t not in gold_titles]}))
    stats = {"n_questions": len(reqs), "context_tokens": context_tokens,
             "mean_context_tokens": sum(sizes) / max(len(sizes), 1),
             "mean_context_docs": sum(ndocs) / max(len(ndocs), 1), "pool_docs": len(pool.titles)}
    return HotpotData(requests=reqs, context_tokens=context_tokens, stats=stats)


def split_questions(examples: Sequence[dict], n_eval: int = 500, n_train: int = 1500, seed: int = 0):
    """Disjoint evaluation and router-training questions."""
    rng = random.Random(seed)
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    ev = [examples[i] for i in idx[:n_eval]]
    tr = [examples[i] for i in idx[n_eval:n_eval + n_train]]
    return ev, tr
