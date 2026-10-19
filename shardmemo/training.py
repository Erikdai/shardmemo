"""Router training: set-likelihood loss L_route, budget-aware surrogate L_route^BA, label-noise
corruption of the supervision, AdamW or Adam with cosine annealing."""
from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from .config import RouterConfig, TrainConfig
from .memory import ShardMap
from .router import ShardScorer


@dataclass
class RouteExample:
    z: np.ndarray
    phi: np.ndarray
    cand: np.ndarray     # candidate shard ids
    gold: np.ndarray     # gold shard ids, a subset of cand
    qid: str = ""


def corrupt_labels(examples: Sequence[RouteExample], eps: float, seed: int = 0) -> list[RouteExample]:
    """Replace each gold shard by a random non-gold candidate with probability eps."""
    if eps <= 0:
        return list(examples)
    rng = random.Random(seed)
    out = []
    for ex in examples:
        gold = {int(g) for g in ex.gold}
        wrong = [int(c) for c in ex.cand if int(c) not in gold]
        new_gold = set()
        for g in sorted(gold):
            if wrong and rng.random() < eps:
                new_gold.add(rng.choice(wrong))
            else:
                new_gold.add(g)
        out.append(RouteExample(ex.z, ex.phi, ex.cand, np.array(sorted(new_gold), dtype=np.int64), ex.qid))
    return out


def _collate(batch: Sequence[RouteExample], sm: ShardMap, device: str):
    B = len(batch)
    Smax = max(int(ex.cand.size) for ex in batch)
    d = batch[0].z.shape[-1]
    z = torch.zeros(B, d)
    phi = torch.zeros(B, batch[0].phi.shape[-1])
    sigma = torch.zeros(B, Smax, d)
    fam = torch.zeros(B, Smax, dtype=torch.long)
    mask = torch.zeros(B, Smax, dtype=torch.bool)
    gold = torch.zeros(B, Smax, dtype=torch.bool)
    summ, fams = sm.summaries, sm.family_ids
    for i, ex in enumerate(batch):
        n = int(ex.cand.size)
        z[i] = torch.from_numpy(ex.z)
        phi[i] = torch.from_numpy(ex.phi)
        sigma[i, :n] = torch.from_numpy(summ[ex.cand])
        fam[i, :n] = torch.from_numpy(fams[ex.cand])
        mask[i, :n] = True
        gset = {int(g) for g in ex.gold}
        gold[i, :n] = torch.tensor([int(c) in gset for c in ex.cand])
    return (t.to(device) for t in (z, phi, sigma, fam, mask, gold))


def route_loss(logp: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    """L_route = -log sum_{j in G_t} p_{t,j} per example."""
    neg_inf = torch.full_like(logp, -1e9)
    return -torch.logsumexp(torch.where(gold, logp, neg_inf), dim=-1)


def budget_aware_weights(logits: torch.Tensor, mask: torch.Tensor, B_probe: int, kappa: float,
                         floor: float) -> torch.Tensor:
    """w_{t,j}: 1 inside the current Top-B_probe window, decaying with the rank distance below it."""
    with torch.no_grad():
        masked = logits.detach().masked_fill(~mask, -1e9)
        order = torch.argsort(masked, dim=-1, descending=True)
        ranks = torch.empty_like(order)
        ar = torch.arange(masked.shape[-1], device=masked.device)[None, :].expand_as(order)
        ranks.scatter_(-1, order, ar)
        dist = (ranks - (B_probe - 1)).clamp(min=0).float()
        w = torch.exp(-dist / max(kappa, 1e-6)).clamp(min=floor)
        w = torch.where(dist > 0, w, torch.ones_like(w))
    return w


def budget_aware_loss(logp: torch.Tensor, gold: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """L_route^BA = -log sum_{j in G_t} w_{t,j} p_{t,j}."""
    neg_inf = torch.full_like(logp, -1e9)
    return -torch.logsumexp(torch.where(gold, logp + torch.log(w.clamp(min=1e-6)), neg_inf), dim=-1)


def train_router(scorer: ShardScorer, examples: Sequence[RouteExample], sm: ShardMap,
                 rcfg: RouterConfig, tcfg: TrainConfig, seed: int) -> list[float]:
    """Train f_theta with L_route (+ L_route^BA); returns the mean loss per epoch."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = tcfg.device
    B_probe = tcfg.probe_budget
    scorer.to(device)
    optimizer = torch.optim.AdamW if tcfg.optimizer == "adamw" else torch.optim.Adam
    opt = optimizer(scorer.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    steps_per_epoch = max(1, math.ceil(len(examples) / tcfg.batch_size))
    sched = None
    if tcfg.cosine_annealing:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tcfg.epochs * steps_per_epoch)
    cost = torch.from_numpy(np.array([sh.n for sh in sm.shards], np.float32) / max(sm.mean_size, 1e-6)).to(device)
    history: list[float] = []
    ex_list = list(examples)
    for _ in range(tcfg.epochs):
        scorer.train()
        random.shuffle(ex_list)
        tot, nb = 0.0, 0
        for s in range(0, len(ex_list), tcfg.batch_size):
            batch = ex_list[s: s + tcfg.batch_size]
            z, phi, sigma, fam, mask, gold = _collate(batch, sm, device)
            logits = scorer(z, phi, sigma, fam)
            if rcfg.alpha:
                cand_cost = torch.stack([torch.nn.functional.pad(cost[torch.from_numpy(ex.cand).to(device)],
                                                                 (0, mask.shape[1] - ex.cand.size)) for ex in batch])
                logits = logits - rcfg.alpha * cand_cost
            logits = logits.masked_fill(~mask, -1e9)
            logp = torch.log_softmax(logits, dim=-1)
            loss = route_loss(logp, gold)
            if rcfg.budget_aware_loss:
                w = budget_aware_weights(logits, mask, B_probe, rcfg.ba_kappa, rcfg.ba_floor)
                loss = loss + rcfg.ba_lambda * budget_aware_loss(logp, gold, w)
            loss = loss.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scorer.parameters(), 5.0)
            opt.step()
            if sched is not None:
                sched.step()
            tot += float(loss.item())
            nb += 1
        history.append(tot / max(nb, 1))
    scorer.eval()
    return history
