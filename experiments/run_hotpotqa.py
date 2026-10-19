"""HotpotQA under the MemAgent long-context protocol.

The memory store holds every document of the dev (distractor) split, one Tier-B shard per document.
Each evaluation question is asked at three context lengths; its request admits only the documents
of its context. The router is trained on disjoint dev questions.

Usage: python experiments/run_hotpotqa.py [--seeds 0 1 2]
"""
from __future__ import annotations

import argparse
import copy

import numpy as np

from common import (CACHE, RESULTS, ROOT, ShardMemo, TierB, evaluate, fmt, log, make_generator, make_route_examples,
                    md_table, over_seeds, save_json, summarize, train_scorer)
from shardmemo.config import HotpotQAConfig, ShardMemoConfig
from shardmemo.data.hotpotqa import build_pool, build_requests, load_hotpot_dev, split_questions
from shardmemo.embedding import Embedder
from shardmemo.generator import PROMPTS
from shardmemo.memory import ShardMap
from shardmemo.scope import PSI_B_HOTPOT

OUT = RESULTS / "hotpotqa"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/raw/hotpotqa/validation.parquet"))
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    args = ap.parse_args()
    cfg = HotpotQAConfig()
    if args.seeds:
        cfg.seeds = tuple(args.seeds)

    emb = Embedder(cfg.memory.embedding_model, cache_path=str(CACHE / "emb_hotpotqa.npz"))
    gen = make_generator(cfg.generator, "hotpotqa")
    examples = load_hotpot_dev(args.data)
    eval_q, train_q = split_questions(examples, cfg.n_eval, cfg.n_train, seed=0)
    pool = build_pool(examples)
    log(f"HotpotQA: {len(pool.titles)} documents, {len(pool.items)} sentences; "
        f"{len(eval_q)} eval / {len(train_q)} router-training questions")
    emb.encode([it.text for it in pool.items], progress=True)
    emb.save_cache()
    sm = ShardMap.build_per_unit(pool.items, emb, index_cfg=cfg.memory.index)
    prefilter = sm.doc_prefilter()
    title2sid = {next(iter(sh.meta.doc_ids)): sh.sid for sh in sm.shards if sh.meta.doc_ids}

    def hard_negatives(r):
        return [title2sid[t] for t in r.meta.get("distractors", []) if t in title2sid]

    lengths = [f"{L // 1000}K" for L in cfg.context_lengths]
    test = {name: build_requests(eval_q, pool, L, seed=0) for name, L in zip(lengths, cfg.context_lengths)}
    train_reqs = build_requests(train_q, pool, cfg.context_lengths[0], seed=1).requests
    stats = {name: {**d.stats, "admitted_pct": 100 * d.stats["mean_context_docs"] / sm.S} for name, d in test.items()}

    def run(name: str, c: ShardMemoConfig, scorer, seed: int, per_seed: dict[str, list[dict]]) -> None:
        memo = ShardMemo(c, emb, shard_map=sm, scorer=scorer, psi_b=PSI_B_HOTPOT, generator=gen,
                         prompt=PROMPTS[cfg.generator.prompt], prefilter=prefilter)
        for length in lengths:
            records = evaluate(memo, test[length].requests, sm, evidence_order=cfg.generator.evidence_order)
            s = summarize(records, categories=None)
            both = [float(set(r["gold_shards"]) <= set(r["probed"])) for r in records if r["gold_shards"]]
            s["both_gold"] = float(np.mean(both)) if both else float("nan")
            per_seed.setdefault(f"{name}|{length}", []).append(s)
            log(f"  seed {seed} {name:<24s} {length:>5s} F1 {s['f1']:6.2f}  BothGold {s['both_gold']:.3f}  "
                f"SF recall {s['ev_recall']:.3f}  VecScan {s['vecscan']:8.1f}  p95 {s['p95_ms']:7.2f}ms")

    per_seed: dict[str, list[dict]] = {}
    budgets = (cfg.memory.budgets.B_probe,) + tuple(cfg.extra_probe_budgets)
    for seed in cfg.seeds:
        c0 = copy.deepcopy(cfg.memory)
        c0.seed = seed
        tb = TierB(sm, None, PSI_B_HOTPOT, c0, prefilter=prefilter)
        examples_seed = make_route_examples(train_reqs, tb, sm, emb, c0.train.probe_budget, max_neg=cfg.max_negatives,
                                            hard_neg_fn=hard_negatives, seed=seed)
        scorer = train_scorer(examples_seed, sm, c0, seed)
        for B in budgets:
            c = copy.deepcopy(c0)
            c.budgets.B_probe = B
            run(method_name(B, budgets), c, scorer, seed, per_seed)

    agg = {k: over_seeds(v) for k, v in per_seed.items()}
    save_json({"config": {"memory": cfg.memory.to_dict(), "generator": vars(cfg.generator)}, "seeds": cfg.seeds,
               "stats": stats, "metrics": agg}, OUT / "summary.json")
    write_report(agg, stats, lengths, budgets, cfg)
    log(f"wrote {OUT / 'report.md'}; LLM calls {gen.n_calls}, tokens {gen.usage}")


def method_name(B: int, budgets: tuple) -> str:
    return "ShardMemo" if budgets[0] == B else f"ShardMemo (B_probe={B})"


def write_report(agg: dict, stats: dict, lengths: list[str], budgets: tuple, cfg: HotpotQAConfig) -> None:
    names = [method_name(B, budgets) for B in budgets]
    b = cfg.memory.budgets
    L = ["# HotpotQA\n", f"Generator `{cfg.generator.model}`, seeds {list(cfg.seeds)}, {cfg.n_eval} questions, "
                         f"B_probe={b.B_probe}, K={b.K}\n",
         "## Admissible-set statistics\n",
         md_table([[x, f"{stats[x]['mean_context_docs']:.0f}", f"{stats[x]['admitted_pct']:.2f}"] for x in lengths],
                  ["context", "admitted documents (mean)", "admitted % of the store"]), ""]
    for metric, title, digits in (("f1", "Answer F1", 2), ("both_gold", "BothGold@B_probe", 3),
                                  ("ev_recall", "Supporting-fact recall@K", 3), ("vecscan", "VecScan", 0),
                                  ("p95_ms", "p95 latency (ms)", 2)):
        rows = [[n] + [fmt(agg.get(f"{n}|{x}", {}), metric, digits) for x in lengths] for n in names]
        L += [f"## {title}\n", md_table(rows, ["Method"] + lengths), ""]
    (OUT / "report.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
