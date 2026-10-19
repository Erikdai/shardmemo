"""ToolBench-style procedural memory: Tier-C skill retrieval and ablations, family-level OOD with online
induction and a case study of direct execution and C->B fallback.

Usage: python experiments/run_toolbench.py [--seeds 0 1 2]
"""
from __future__ import annotations

import argparse
import copy
from typing import Any

import numpy as np

from common import CACHE, RESULTS, ShardMemo, fmt, log, md_table, over_seeds, save_json
from shardmemo.config import ShardMemoConfig, ToolBenchConfig
from shardmemo.data.toolbench import FAMILY_SPECS, ToolTask, make_toolbench, skill_valid_for
from shardmemo.embedding import Embedder
from shardmemo.memory import ShardMap
from shardmemo.scope import PSI_C_TOOLBENCH, Request, all_of, namespace_scope, permission_scope
from shardmemo.tier_c import SkillRecord, SkillStore, TierC, render_procedure

OUT = RESULTS / "toolbench"
PSI_B_SPEC = all_of(namespace_scope, permission_scope)
TIER_C = "Tier-C Skills"
POST_FILTER = "w/o scope-first (post-filter top-R)"
RAW_SIMILARITY = "w/o masked templates (raw similarity)"
VARIANTS = (TIER_C, POST_FILTER, RAW_SIMILARITY)


def execute_first(tc: TierC, ranked: list[SkillRecord], task: ToolTask) -> Any | None:
    """Result of the first skill whose slots fill, preconditions hold and result validates."""
    req = task.request()
    for s in ranked:
        slots, ok = tc.filler.fill(req, s)
        if not ok or not tc.validate_preconditions(s, slots):
            continue
        done, y, _ = tc.execute(s.procedure, slots)
        if done and tc.validate_result(s, slots, y):
            return y
    return None


def score_task(valid_flags: list[bool], R: int, executed: Any | None,
               task: ToolTask) -> tuple[float, float | None]:
    p_at_r = sum(valid_flags) / R
    if not task.gold_in_scope:
        return p_at_r, None                       # StepRed is averaged over in-scope targets only
    n_skill = 1 if (executed is not None and executed == task.gold) else None
    return p_at_r, (max(0, task.n_gold_steps - n_skill) if n_skill is not None else 0.0)


def run_seed(seed: int, emb: Embedder, cfg: ToolBenchConfig, single_provider: bool = False):
    R, tau_c = cfg.R, cfg.tau_c
    data = make_toolbench(seed=seed, n_in_domain=cfg.n_in_domain, n_ood=cfg.n_ood, single_provider=single_provider)
    env = data.env
    store = SkillStore(emb)
    tc = TierC(store, env, PSI_C_TOOLBENCH, R=R, tau_c=tau_c)
    for t in data.train:
        tc.induce(t.trace)
    for t in data.train:                                     # reliability estimates from the induction tasks
        tc.read(t.request())
    log(f"seed {seed}: induced {len(store)} versioned records / {len({r.skill_id for r in store.records})} skill ids "
        f"from {len(data.train)} traces; eval {len(data.test)}, OOD {len(data.ood)}")

    raw_store = SkillStore(emb, mask=False)                  # same records with raw-text vectors
    for r in store.records:
        raw_store.add(copy.copy(r))
    res: dict[str, dict[str, list]] = {m: {"p": [], "sr": []} for m in VARIANTS}

    def record(name: str, scores: tuple[float, float | None]) -> None:
        res[name]["p"].append(scores[0])
        res[name]["sr"].append(scores[1])

    all_idx = list(range(len(store.records)))
    for task in data.test:
        req = task.request()
        out = tc.read(req)
        valid = [skill_valid_for(tc, s, task) for s in out.skills]
        record(TIER_C, score_task(valid, R, out.result if out.ok else None, task))
        # Tier-C ranking over every namespace, scope applied after the top-R
        cur_all = tc.current_versions(all_idx)
        sims = store.similarity(cur_all, store.encode_query(req.text))
        top = [store.records[cur_all[i]] for i in np.argsort(-sims, kind="stable")[:R]]
        top = [s for s in top if PSI_C_TOOLBENCH(s.meta, req)]
        ranked = sorted(top, key=lambda s: s.reliability, reverse=True)
        valid = [skill_valid_for(tc, s, task) for s in top]
        record(POST_FILTER, score_task(valid, R, execute_first(tc, ranked, task), task))
        # scope-first and latest versions, raw (unmasked) text similarity
        cur = tc.current_versions(tc.admissible(req))
        if cur:
            sims = raw_store.vectors[cur] @ emb.encode_one(req.text)
            top = [store.records[cur[i]] for i in np.argsort(-sims, kind="stable")[:R]]
        else:
            top = []
        ranked = sorted(top, key=lambda s: s.reliability, reverse=True)
        valid = [skill_valid_for(tc, s, task) for s in top]
        record(RAW_SIMILARITY, score_task(valid, R, execute_first(tc, ranked, task), task))

    summary = {m: {"P@R": float(np.mean(v["p"])), "StepRed": float(np.mean([x for x in v["sr"] if x is not None]))}
               for m, v in res.items()}

    adm = [len(tc.admissible(t.request())) / max(len(store), 1) for t in data.test]
    summary["_stats"] = {"admitted_pct": 100 * float(np.mean(adm)),
                         "admitted_p10": 100 * float(np.percentile(adm, 10)),
                         "admitted_p90": 100 * float(np.percentile(adm, 90)),
                         "gold_in_scope_pct": 100 * float(np.mean([t.gold_in_scope for t in data.test])),
                         "fallback_rate_pct": 100 * float(np.mean([not tc.read(t.request()).ok for t in data.test]))}

    # family-level OOD, online: try Tier C, then induce a skill from the solved trace
    ood_store = SkillStore(emb)
    for r in store.records:
        ood_store.add(copy.copy(r))
    ood_store.copy_signatures(store)
    ood_tc = TierC(ood_store, env, PSI_C_TOOLBENCH, R=R, tau_c=tau_c)
    ok_flags, p_list = [], []
    for task in data.ood:
        out = ood_tc.read(task.request())
        ok_flags.append(bool(out.ok and out.result == task.gold))
        p_list.append(sum(skill_valid_for(ood_tc, s, task) for s in out.skills) / R)
        ood_tc.induce(task.trace)
    half = len(ok_flags) // 2
    summary["_ood"] = {"exec_rate_all": float(np.mean(ok_flags)),
                       "exec_rate_second_half": float(np.mean(ok_flags[half:])),
                       "P@R_all": float(np.mean(p_list)), "P@R_second_half": float(np.mean(p_list[half:])),
                       "first_success_index": int(next((i for i, f in enumerate(ok_flags) if f), -1))}
    return summary, (store, tc, data)


def case_study(emb: Embedder, store: SkillStore, tc: TierC, data, R: int) -> list[str]:
    cfg = ShardMemoConfig()
    cfg.routing_mode = "cosine"
    cfg.budgets.R = R
    spec_map = ShardMap.build_per_unit(data.spec_items, emb)
    memo = ShardMemo(cfg, emb, shard_map=spec_map, psi_b=PSI_B_SPEC, tier_c=tc)
    lines: list[str] = []
    directory = {e: e for e in FAMILY_SPECS["transitaires"]["entities"]}
    # Tier-A directory cache, e.g. "ACT" -> full name
    alias = {e.split()[0]: e for e in directory}

    def resolver(text: str) -> str | None:
        for a, full in alias.items():
            if a in text.split():
                return full
        return None
    memo.active.add_note("toolbench", "directory_cache",
                         "transitaires directory: " + "; ".join(f"{a} -> {f}" for a, f in alias.items()))

    q1 = Request(text="Get contact details of the ACT customs agency in New Caledonia.",
                 allowed_namespaces=frozenset(["transitaires"]), tier_hint="C")
    r1 = memo.read(q1, E_t={"resolvers": {"name": resolver}}, generate=False)
    c = r1.tier_c
    lines += ["### Case 1 - direct skill execution", f"Task: {q1.text}", f"Tier path: {' -> '.join(r1.tier_path)}",
              f"Tier C retrieval (R={R}, scope: transitaires):"]
    lines += [f"  [{s.key}] {render_procedure(s.procedure)}  (sim {x:.2f}, rho {s.reliability:.2f})"
              for s, x in zip(c.skills, c.sims)]
    lines += [f"Slot filling: {c.slots} (canonicalised via the Tier-A directory cache)",
              f"Execution: {c.result}", f"Success: {c.ok}"]

    q2 = Request(text="List freight forwarders serving Wallis-and-Futuna.",
                 allowed_namespaces=frozenset(["transitaires"]), tier_hint="C")
    r2 = memo.read(q2, generate=False)
    c = r2.tier_c
    max_sim = max(c.sims) if c.sims else float("nan")
    lines += ["", "### Case 2 - C->B fallback", f"Task: {q2.text}", f"Tier path: {' -> '.join(r2.tier_path)}",
              f"Tier C: max sim {max_sim:.2f} (tau_C={tc.tau_c}); executed: {c.ok}; reason: {c.reason}",
              f"C->B fallback: {r2.fallback}; Tier-B probed shards {r2.tier_b.probed} "
              f"under B_probe={cfg.budgets.B_probe}, K={cfg.budgets.K}"]
    lines += [f"  evidence: {e}" for e in r2.evidence_texts()[:3]]
    return lines


def run_setting(emb: Embedder, cfg: ToolBenchConfig, single_provider: bool):
    per_seed: dict[str, list[dict]] = {}
    stats, ood = [], []
    last = None
    for seed in cfg.seeds:
        summ, last = run_seed(seed, emb, cfg, single_provider=single_provider)
        for m, v in summ.items():
            if m == "_stats":
                stats.append(v)
            elif m == "_ood":
                ood.append(v)
            else:
                per_seed.setdefault(m, []).append(v)
        log(f"[{'single' if single_provider else 'multi'}-provider] seed {seed}: " +
            ", ".join(f"{m} P@R {v['P@R']:.3f} StepRed {v['StepRed']:.2f}" for m, v in summ.items()
                      if not m.startswith("_")))
    return {m: over_seeds(v) for m, v in per_seed.items()}, over_seeds(stats), over_seeds(ood), last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    args = ap.parse_args()
    cfg = ToolBenchConfig()
    if args.seeds:
        cfg.seeds = tuple(args.seeds)
    emb = Embedder(cache_path=str(CACHE / "emb_toolbench.npz"))
    agg, st, oo, last = run_setting(emb, cfg, single_provider=False)
    agg1, st1, oo1, _ = run_setting(emb, cfg, single_provider=True)
    emb.save_cache()
    lines = case_study(emb, *last, R=cfg.R)
    save_json({"multi_provider": {"metrics": agg, "stats": st, "ood": oo},
               "single_provider": {"metrics": agg1, "stats": st1, "ood": oo1},
               "case_study": lines, "seeds": cfg.seeds}, OUT / "summary.json")

    def rows_for(a):
        return [[m, fmt(a[m], "P@R"), fmt(a[m], "StepRed")] for m in VARIANTS]
    header = ["Variant", f"P@{cfg.R}", "StepRed"]
    rep = ["# ToolBench-style procedural memory\n",
           f"Seeds {list(cfg.seeds)}; R={cfg.R}; tau_C={cfg.tau_c}\n",
           "## Tier-C skill retrieval and ablations (several providers per tool family)\n",
           md_table(rows_for(agg), header), "",
           "## Admissible-set statistics\n",
           md_table([[fmt(st, "admitted_pct", 1), fmt(st, "gold_in_scope_pct", 1), fmt(st, "fallback_rate_pct", 1)]],
                    ["admitted skills %", "gold in scope %", "C->B fallback rate %"]), "",
           "## Family-level OOD with online induction\n",
           md_table([[fmt(oo, "exec_rate_all", 3), fmt(oo, "exec_rate_second_half", 3), fmt(oo, "P@R_all", 3),
                      fmt(oo, "P@R_second_half", 3), fmt(oo, "first_success_index", 1)]],
                    ["exec rate (all)", "exec rate (2nd half)", "P@R (all)", "P@R (2nd half)",
                     "first success at task #"]), "",
           "## Tier-C skill retrieval and ablations (one provider per tool family)\n",
           f"Admitted skills: {fmt(st1, 'admitted_pct', 1)} %\n",
           md_table(rows_for(agg1), header), "",
           "## Case study\n", "```", *lines, "```"]
    (OUT / "report.md").write_text("\n".join(rep), encoding="utf-8")
    log(f"wrote {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
