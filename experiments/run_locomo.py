"""LoCoMo experiments.

  stats        admissible-set statistics
  main         ShardMemo, per-category F1 / BLEU-1
  ablations    component ablations
  sweep        probe-budget sweep, B_probe in {1, 2, 4, 8}
  scaling      shard-count scaling
  noise        label-noise robustness
  sensitivity  adaptive Top-P parameters

Usage: python experiments/run_locomo.py [--groups main ablations ...] [--seeds 0 1 2]
"""
from __future__ import annotations

import argparse
import copy
from collections import defaultdict

import numpy as np

from common import (CACHE, CATEGORIES, RESULTS, ROOT, ShardMemo, ShardMap, TierB, evaluate, fmt, log, make_generator,
                    make_route_examples, md_table, over_seeds, save_json, summarize, train_scorer)
from shardmemo.config import LocomoConfig, ShardMemoConfig
from shardmemo.data.locomo import build_active_context, prepare_locomo
from shardmemo.embedding import Embedder
from shardmemo.generator import PROMPTS
from shardmemo.scope import PSI_B_LOCOMO
from shardmemo.tier_a import ActiveContextStore

OUT = RESULTS / "locomo"
GROUPS = ("stats", "main", "ablations", "sweep", "scaling", "noise", "sensitivity")
ABLATIONS = ["w/o cost-aware gating (alpha=0)", "w/o eligibility masking (post-filter only)",
             "w/o eligibility masking, router retrained on all shards", "w/o adaptive Top-P (fixed Top-B_probe)",
             "w/ adaptive Top-P (capped by B_probe)", "w/o heterogeneous shard families (session only)",
             "w/o Tier A active context", "w/o scope-term stripping (raw query)",
             "w/o budget-aware surrogate (L_route only)"]
SWEEP_BUDGETS = (1, 2, 4, 8)
SHARD_COUNTS = (5, 10, 20, 40, 80)


def variant(cfg: ShardMemoConfig, **changes) -> ShardMemoConfig:
    """Copy of cfg with top-level fields or router fields (prefix router_) replaced."""
    c = copy.deepcopy(cfg)
    for key, value in changes.items():
        if key.startswith("router_"):
            setattr(c.router, key[len("router_"):], value)
        else:
            setattr(c, key, value)
    return c


class LocomoRunner:
    def __init__(self, cfg: LocomoConfig, data_path: str):
        self.cfg = cfg
        self.data_path = data_path
        self.emb = Embedder(cfg.memory.embedding_model, cache_path=str(CACHE / "emb_locomo.npz"))
        self.gen = make_generator(cfg.generator, "locomo")
        self.data = prepare_locomo(data_path, show_previous_turn=cfg.show_previous_turn)
        self.train_reqs, self.test_reqs = self.data.split(cfg.test_conversations)
        self.active = ActiveContextStore()
        build_active_context(self.active, self.data.anchors)
        self.maps: dict[str, ShardMap] = {}
        self.per_seed: dict[str, list[dict]] = defaultdict(list)
        log(f"LoCoMo: {len(self.data.items)} evidence items, "
            f"{len(self.train_reqs)}/{len(self.test_reqs)} train/test questions")

    def shard_map(self, S: int, session_only: bool = False) -> ShardMap:
        key = f"S{S}" + ("_session_only" if session_only else "")
        if key not in self.maps:
            items = self.data.items
            if session_only:
                items = prepare_locomo(self.data_path, session_only=True,
                                       show_previous_turn=self.cfg.show_previous_turn).items
            self.maps[key] = ShardMap.build(items, self.emb, S=S, index_cfg=self.cfg.memory.index)
            self.emb.save_cache()
        return self.maps[key]

    def route_examples(self, sm: ShardMap, cfg: ShardMemoConfig, all_shards: bool = False):
        tb = TierB(sm, None, PSI_B_LOCOMO, cfg)
        query_fn = ShardMemo(cfg, self.emb, shard_map=sm, psi_b=PSI_B_LOCOMO, active=self.active).ranking_query
        return make_route_examples(self.train_reqs, tb, sm, self.emb, cfg.train.probe_budget,
                                   all_shards=all_shards, query_fn=query_fn)

    def train(self, sm: ShardMap, cfg: ShardMemoConfig, seed: int, noise: float = 0.0, all_shards: bool = False):
        return train_scorer(self.route_examples(sm, cfg, all_shards), sm, cfg, seed, noise=noise)

    def run(self, name: str, cfg: ShardMemoConfig, sm: ShardMap, scorer, seed: int) -> None:
        memo = ShardMemo(cfg, self.emb, shard_map=sm, scorer=scorer, psi_b=PSI_B_LOCOMO, generator=self.gen,
                         prompt=PROMPTS[self.cfg.generator.prompt], active=self.active)
        records = evaluate(memo, self.test_reqs, sm, evidence_order=self.cfg.generator.evidence_order)
        s = summarize(records)
        self.per_seed[name].append(s)
        log(f"  seed {seed} {name:<58s} F1 {s['f1']:6.2f}  ShardHit {s['shard_hit']:.3f}  "
            f"EvRecall {s['ev_recall']:.3f}  VecScan {s['vecscan']:7.1f}  p95 {s['p95_ms']:6.2f}ms")

    def stats(self) -> dict:
        sm = self.shard_map(self.cfg.num_shards)
        tb = TierB(sm, None, PSI_B_LOCOMO, self.cfg.memory)
        uid = {it.uid: it for it in self.data.items}
        admitted, gold_shard, gold_item = [], [], []
        for r in self.data.requests:
            adm = tb.admissible(r)
            admitted.append(adm.size / sm.S)
            gold = sm.gold_shards(r.meta["evidence"])
            if gold:
                gold_shard.append(bool(gold & set(adm.tolist())))
            ev = [uid[e] for e in r.meta["evidence"] if e in uid]
            if ev:
                gold_item.append(all(PSI_B_LOCOMO(it.meta, r) for it in ev))
        return {"admitted_pct_mean": 100 * float(np.mean(admitted)),
                "admitted_pct_p10": 100 * float(np.percentile(admitted, 10)),
                "admitted_pct_p90": 100 * float(np.percentile(admitted, 90)),
                "gold_in_scope_shard_pct": 100 * float(np.mean(gold_shard)),
                "gold_in_scope_item_pct": 100 * float(np.mean(gold_item))}

    def seed_runs(self, seed: int, groups: set) -> None:
        cfg = variant(self.cfg.memory, seed=seed)
        sm = self.shard_map(self.cfg.num_shards)
        full = self.train(sm, cfg, seed)
        if groups & {"main", "ablations", "noise"}:
            self.run("ShardMemo", cfg, sm, full, seed)
        if "ablations" in groups:
            self.ablations(cfg, sm, full, seed)
        if "sweep" in groups:
            self.sweep(cfg, sm, full, seed)
        if "noise" in groups:
            for eps in (0.2, 0.4):
                self.run(f"noise|{int(eps * 100)}%", cfg, sm, self.train(sm, cfg, seed, noise=eps), seed)
        if "scaling" in groups:
            self.scaling(cfg, full, seed)
        if "sensitivity" in groups:
            self.sensitivity(cfg, sm, full, seed)
        self.emb.save_cache()

    def ablations(self, cfg: ShardMemoConfig, sm: ShardMap, full, seed: int) -> None:
        c = variant(cfg, router_alpha=0.0)
        self.run(ABLATIONS[0], c, sm, self.train(sm, c, seed), seed)
        c = variant(cfg, scope_before_routing=False)
        self.run(ABLATIONS[1], c, sm, full, seed)
        self.run(ABLATIONS[2], c, sm, self.train(sm, c, seed, all_shards=True), seed)
        top_p = ABLATIONS[3] if cfg.router.adaptive_top_p else ABLATIONS[4]
        self.run(top_p, variant(cfg, router_adaptive_top_p=not cfg.router.adaptive_top_p), sm, full, seed)
        sm_session = self.shard_map(self.cfg.num_shards, session_only=True)
        self.run(ABLATIONS[5], cfg, sm_session, self.train(sm_session, cfg, seed), seed)
        self.run(ABLATIONS[6], variant(cfg, use_tier_a=False), sm, full, seed)
        c = variant(cfg, strip_scope_terms=False)
        self.run(ABLATIONS[7], c, sm, self.train(sm, c, seed), seed)
        c = variant(cfg, router_budget_aware_loss=False)
        self.run(ABLATIONS[8], c, sm, self.train(sm, c, seed), seed)

    def sweep(self, cfg: ShardMemoConfig, sm: ShardMap, full, seed: int) -> None:
        for B in SWEEP_BUDGETS:
            c = variant(cfg)
            c.budgets.B_probe = B
            self.run(f"sweep|B={B}", c, sm, full, seed)

    def scaling(self, cfg: ShardMemoConfig, full, seed: int) -> None:
        for S in SHARD_COUNTS:
            sm = self.shard_map(S)
            scorer = full if self.cfg.num_shards == S else self.train(sm, cfg, seed)
            self.run(f"scaling|S={S}", cfg, sm, scorer, seed)

    def sensitivity(self, cfg: ShardMemoConfig, sm: ShardMap, full, seed: int) -> None:
        for name, values in (("gamma", (0.2, 0.4, 0.6, 0.8, 1.0)), ("p_min", (0.3, 0.4, 0.5, 0.6, 0.7)),
                             ("p_max", (0.80, 0.85, 0.90, 0.95, 1.00))):
            for v in values:
                c = variant(cfg, router_adaptive_top_p=True, **{f"router_{name}": v})
                self.run(f"sens|{name}={v:.2f}", c, sm, full, seed)


def write_report(results: dict, agg: dict, cfg: LocomoConfig) -> None:
    b = cfg.memory.budgets
    L = ["# LoCoMo\n", f"Generator `{cfg.generator.model}`, seeds {list(results['seeds'])}, "
                        f"{results['n_test']} test questions, S={cfg.num_shards}, B_probe={b.B_probe}, K={b.K}\n"]
    if "stats" in results:
        s = results["stats"]
        L += ["## Admissible-set statistics\n",
              md_table([[f"{s['admitted_pct_mean']:.1f}", f"{s['admitted_pct_p10']:.1f}-{s['admitted_pct_p90']:.1f}",
                         f"{s['gold_in_scope_shard_pct']:.1f}", f"{s['gold_in_scope_item_pct']:.1f}"]],
                       ["admitted shards % (mean)", "p10-p90", "gold in scope % (shard)",
                        "gold in scope % (item)"]), ""]
    if "ShardMemo" in agg:
        a = agg["ShardMemo"]
        rows = [[c, fmt(a, f"f1_{c}"), fmt(a, f"bleu1_{c}")] for c in CATEGORIES]
        rows.append(["macro average", fmt(a, "f1"), fmt(a, "bleu1")])
        L += ["## Per-category F1 / BLEU-1\n", md_table(rows, ["Category", "F1", "BLEU-1"]), ""]
    names = [n for n in ["ShardMemo"] + ABLATIONS if n in agg]
    if names:
        rows = [[n, fmt(agg[n], "f1"), fmt(agg[n], "shard_hit", 3), fmt(agg[n], "ev_recall", 3),
                 fmt(agg[n], "vecscan", 0), fmt(agg[n], "p95_ms"), fmt(agg[n], "avg_probed"),
                 f"{agg[n]['inadmissible_returned']['mean']:.0f}"] for n in names]
        L += ["## Retrieval metrics and ablations\n",
              md_table(rows, ["Variant", "Macro F1", "ShardHit@B", "Evidence recall@K", "VecScan", "p95 (ms)",
                              "avg probed", "inadmissible items returned"]), ""]
    if any(k.startswith("sweep|") for k in agg):
        rows = [[B, fmt(a, "f1"), fmt(a, "shard_hit", 3), fmt(a, "vecscan", 0), fmt(a, "p95_ms")]
                for B in SWEEP_BUDGETS if (a := agg.get(f"sweep|B={B}"))]
        L += ["## Probe-budget sweep\n", md_table(rows, ["B_probe", "F1", "ShardHit", "VecScan", "p95 (ms)"]), ""]
    if any(k.startswith("scaling|") for k in agg):
        rows = [[S, fmt(a, "f1"), fmt(a, "shard_hit", 3), fmt(a, "vecscan", 0), fmt(a, "p95_ms"), fmt(a, "route_ms", 3)]
                for S in SHARD_COUNTS if (a := agg.get(f"scaling|S={S}"))]
        L += ["## Shard-count scaling\n",
              md_table(rows, ["S", "F1", "ShardHit", "VecScan", "p95 (ms)", "Route (ms)"]), ""]
    if "noise|20%" in agg:
        keys = (("f1", 2), ("shard_hit", 3), ("vecscan", 0), ("p95_ms", 2))
        rows = [[label] + [fmt(agg[name], k, d) for k, d in keys]
                for label, name in (("0%", "ShardMemo"), ("20%", "noise|20%"), ("40%", "noise|40%"))]
        L += ["## Label-noise robustness\n", md_table(rows, ["Noise", "F1", "ShardHit", "VecScan", "p95 (ms)"]), ""]
    for name in ("gamma", "p_min", "p_max"):
        keys = sorted(k for k in agg if k.startswith(f"sens|{name}="))
        if keys:
            rows = [[k.split("=")[1], fmt(agg[k], "f1"), fmt(agg[k], "vecscan", 0), fmt(agg[k], "p95_ms"),
                     fmt(agg[k], "avg_probed")] for k in keys]
            L += [f"## Adaptive Top-P sensitivity: {name}\n",
                  md_table(rows, [name, "F1", "VecScan", "p95 (ms)", "avg probed"]), ""]
    (OUT / "report.md").write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/raw/locomo/locomo10.json"))
    ap.add_argument("--groups", nargs="+", default=list(GROUPS), choices=GROUPS)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    args = ap.parse_args()
    cfg = LocomoConfig()
    if args.seeds:
        cfg.seeds = tuple(args.seeds)
    groups = set(args.groups)

    runner = LocomoRunner(cfg, args.data)
    results = {"config": {"memory": cfg.memory.to_dict(), "generator": vars(cfg.generator)}, "seeds": cfg.seeds,
               "n_train": len(runner.train_reqs), "n_test": len(runner.test_reqs)}
    if "stats" in groups:
        results["stats"] = runner.stats()
        log(f"admissible-set statistics: {results['stats']}")
    if groups - {"stats"}:
        for seed in cfg.seeds:
            log(f"===== seed {seed} =====")
            runner.seed_runs(seed, groups)
    agg = {k: over_seeds(v) for k, v in runner.per_seed.items()}
    results["metrics"] = agg
    save_json(results, OUT / "summary.json")
    write_report(results, agg, cfg)
    log(f"wrote {OUT / 'report.md'}; LLM calls {runner.gen.n_calls}, tokens {runner.gen.usage}")


if __name__ == "__main__":
    main()
