"""Helpers shared by the experiment scripts: routing supervision, router training, evaluation,
aggregation over seeds and Markdown output."""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
load_dotenv(ROOT / ".env")

from shardmemo.config import GeneratorConfig, ShardMemoConfig  # noqa: E402
from shardmemo.controller import ShardMemo  # noqa: E402
from shardmemo.generator import Generator  # noqa: E402
from shardmemo.memory import ShardMap  # noqa: E402
from shardmemo.metrics import bleu1, latency_summary, mean_std_over_seeds, shard_hit, token_f1  # noqa: E402
from shardmemo.router import ShardScorer, request_features  # noqa: E402
from shardmemo.scope import Request  # noqa: E402
from shardmemo.tier_a import ActiveContextStore  # noqa: E402
from shardmemo.tier_b import TierB  # noqa: E402
from shardmemo.training import RouteExample, corrupt_labels, train_router  # noqa: E402

RESULTS = ROOT / "results"
CACHE = RESULTS / "cache"
CATEGORIES = ["single_hop", "multi_hop", "temporal", "open_domain"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def set_seed(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_generator(cfg: GeneratorConfig, benchmark: str) -> Generator:
    model = re.sub(r"[^\w.-]", "_", cfg.model)
    return Generator(cfg, cache_path=str(CACHE / f"llm_{model}_{benchmark}.jsonl"))


def make_route_examples(reqs: Sequence[Request], tb: TierB, sm: ShardMap, embedder, B_probe: int,
                        all_shards: bool = False, max_neg: int | None = None, hard_neg_fn=None,
                        seed: int = 0, query_fn=None) -> list[RouteExample]:
    """One example per request whose gold shards are among its candidates.

    all_shards: candidates are all shards instead of the admissible set.
    max_neg: subsample negatives, keeping the hard negatives returned by hard_neg_fn.
    query_fn: text the router sees for a request (must match inference).
    """
    rng = random.Random(seed)
    zs = embedder.encode([query_fn(r) if query_fn else r.text for r in reqs])
    out: list[RouteExample] = []
    for r, z in zip(reqs, zs):
        adm = tb.admissible(r)
        cand = np.arange(sm.S, dtype=np.int64) if all_shards else adm
        gold = sm.gold_shards(r.meta.get("evidence", [])) & set(cand.tolist())
        if not gold:
            continue
        if max_neg is not None and cand.size > len(gold) + max_neg:
            hard = set(hard_neg_fn(r)) & set(cand.tolist()) if hard_neg_fn else set()
            pool = [int(c) for c in cand if int(c) not in gold and int(c) not in hard]
            negs = list(hard) + rng.sample(pool, min(max_neg, len(pool)))
            cand = np.array(sorted(set(gold) | set(negs)), dtype=np.int64)
        t_cue, d_cue = ActiveContextStore.cues(r)
        phi = request_features(r, int(adm.size), sm.S, B_probe, t_cue, d_cue)
        out.append(RouteExample(z, phi, cand, np.array(sorted(gold), dtype=np.int64), r.meta.get("qid", "")))
    return out


def train_scorer(examples: list[RouteExample], sm: ShardMap, cfg: ShardMemoConfig, seed: int,
                 noise: float = 0.0) -> ShardScorer:
    set_seed(seed)
    ex = corrupt_labels(examples, noise, seed=seed) if noise > 0 else examples
    scorer = ShardScorer(sm.embedder.dim, cfg.router)
    train_router(scorer, ex, sm, cfg.router, cfg.train, seed)
    return scorer


def _with_date(text: str, when) -> str:
    """'(session 3) ...' -> '(session 3, 8 May 2023) ...'."""
    if when is None or not text.startswith("(session "):
        return text
    close = text.find(")")
    return text[:close] + f", {when.day} {when.strftime('%B')} {when.year}" + text[close:]


def _anchor_dated(text: str, item, memo: ShardMemo, conv_id: str | None) -> str:
    if not item.meta.session_ids or conv_id is None:
        return text
    return _with_date(text, memo.active.anchor_for(conv_id, min(item.meta.session_ids)))


def _chronological_key(item) -> tuple:
    session = min(item.meta.session_ids) if item.meta.session_ids else 10 ** 6
    m = re.search(r":D(\d+):(\d+)$", item.uid)
    return session, int(m.group(2)) if m else -1, item.uid


def evaluate(memo: ShardMemo, reqs: Sequence[Request], sm: ShardMap, generate: bool = True,
             evidence_order: str = "score") -> list[dict]:
    """Retrieval for every request (sequential, for latency), then concurrent answer generation.

    evidence_order: 'score' (retrieval rank) or 'chronological' (the same items in conversation order).
    """
    outs, jobs = [], []
    for r in reqs:
        out = memo.read(r, generate=False)
        items = out.tier_b.evidence if out.tier_b else []
        if evidence_order == "chronological":
            items = sorted(items, key=_chronological_key)
        if memo.cfg.use_tier_a and out.active is not None:
            texts = [_anchor_dated(e.shown(), e, memo, r.conv_id) for e in items]
        else:
            texts = [e.shown() for e in items]
        context = out.active.render() if out.active is not None else ""
        outs.append(out)
        jobs.append((r.text, texts, context, memo.prompt))
    answers: list[str | None] = memo.generator.generate_many(jobs) if generate else [None] * len(reqs)

    records = []
    for r, out, answer in zip(reqs, outs, answers):
        tb = out.tier_b
        items = tb.evidence if tb else []
        gold_ev = set(r.meta.get("evidence", []))
        got = {s for e in items for s in e.source_ids}
        gold_sh = sm.gold_shards(gold_ev) if gold_ev else set()
        gold_ans = str(r.meta.get("answer", ""))
        routed = tb is not None and bool(gold_sh)
        records.append({
            "qid": r.meta.get("qid"), "category": r.meta.get("category_name", r.meta.get("type")),
            "answer": answer, "gold": gold_ans,
            "f1": token_f1(answer, gold_ans) if answer is not None else None,
            "bleu1": bleu1(answer, gold_ans) if answer is not None else None,
            "shard_hit": shard_hit(tb.probed, gold_sh) if routed else None,
            "ev_recall": len(gold_ev & got) / len(gold_ev) if gold_ev else None,
            "vecscan": tb.vecscan if tb else 0, "latency_ms": tb.latency_ms if tb else 0.0,
            "route_ms": tb.route_ms if tb else 0.0, "n_probed": len(tb.probed) if tb else 0,
            "inadmissible_returned": sum(0 if memo.tier_b.psi_b(e.meta, r) else 1 for e in items),
            "probed": list(tb.probed) if tb else [], "gold_shards": sorted(gold_sh),
        })
    return records


def summarize(records: list[dict], categories: Sequence[str] | None = CATEGORIES) -> dict[str, float]:
    """Mean metrics; with categories, F1 and BLEU-1 are macro-averaged over them."""
    def mean(key, rs):
        v = [x[key] for x in rs if x.get(key) is not None]
        return float(np.mean(v)) if v else float("nan")
    out: dict[str, float] = {}
    if categories:
        for c in categories:
            rs = [x for x in records if x["category"] == c]
            out[f"f1_{c}"] = 100 * mean("f1", rs)
            out[f"bleu1_{c}"] = 100 * mean("bleu1", rs)
        for m in ("f1", "bleu1"):
            vals = [out[f"{m}_{c}"] for c in categories if not np.isnan(out[f"{m}_{c}"])]
            out[m] = float(np.mean(vals)) if vals else float("nan")
    else:
        out["f1"] = 100 * mean("f1", records)
        out["bleu1"] = 100 * mean("bleu1", records)
    out["shard_hit"] = mean("shard_hit", records)
    out["ev_recall"] = mean("ev_recall", records)
    out["vecscan"] = mean("vecscan", records)
    out.update(latency_summary([x["latency_ms"] for x in records]))
    out["route_ms"] = mean("route_ms", records)
    out["avg_probed"] = mean("n_probed", records)
    out["inadmissible_returned"] = float(sum(x["inadmissible_returned"] for x in records))
    return out


def over_seeds(per_seed: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return mean_std_over_seeds(per_seed)


def fmt(ms: dict[str, dict[str, float]], key: str, digits: int = 2) -> str:
    if key not in ms or np.isnan(ms[key]["mean"]):
        return "-"
    return f"{ms[key]['mean']:.{digits}f} ± {ms[key]['std']:.{digits}f}"


def md_table(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
