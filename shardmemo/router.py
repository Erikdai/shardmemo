"""Masked router: f_theta scores admissible shards, cost bias, masked softmax, Top-B_probe or
adaptive Top-P selection; cosine routing for shards without routing supervision."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .config import RouterConfig, RoutingMode
from .memory import FAMILIES, ShardMap
from .scope import Request

PHI_DIM = 6


def request_features(q: Request, n_admissible: int, S: int, B_probe: int,
                     temporal_cue: bool, deictic_cue: bool) -> np.ndarray:
    """phi_t: scope size, budget and cue features available at request time."""
    return np.array([
        np.log1p(n_admissible) / max(np.log1p(S), 1e-6),
        min(1.0, B_probe / max(n_admissible, 1)),
        float(temporal_cue),
        float(deictic_cue),
        min(len(q.speakers_referenced), 2) / 2.0,
        float(q.time_window is not None or q.allowed_namespaces is not None),
    ], dtype=np.float32)


class ShardScorer(nn.Module):
    """f_theta(r_t, sigma_j): two-layer MLP over [z_t; sigma_j; z_t * sigma_j; phi_t; family embedding]."""

    def __init__(self, dim: int, cfg: RouterConfig, n_families: int = len(FAMILIES), meta_dim: int = PHI_DIM):
        super().__init__()
        self.cfg = cfg
        in_dim = 2 * dim
        if cfg.use_product_feature:
            in_dim += dim
        if cfg.use_meta_features:
            in_dim += meta_dim
        if cfg.use_family_emb:
            self.fam = nn.Embedding(n_families, cfg.family_emb_dim)
            in_dim += cfg.family_emb_dim
        else:
            self.fam = None
        self.mlp = nn.Sequential(nn.Linear(in_dim, cfg.hidden), nn.ReLU(), nn.Dropout(cfg.dropout),
                                 nn.Linear(cfg.hidden, 1))
        self.beta = nn.Parameter(torch.tensor(float(cfg.beta_init))) if cfg.cosine_residual else None

    def forward(self, z: torch.Tensor, phi: torch.Tensor, sigma: torch.Tensor, fam: torch.Tensor) -> torch.Tensor:
        """z: [B,d], phi: [B,m], sigma: [B,S,d], fam: [B,S] -> logits [B,S]."""
        B, S, d = sigma.shape
        zz = z[:, None, :].expand(B, S, d)
        feats = [zz, sigma]
        if self.cfg.use_product_feature:
            feats.append(zz * sigma)
        if self.cfg.use_meta_features:
            feats.append(phi[:, None, :].expand(B, S, phi.shape[-1]))
        if self.fam is not None:
            feats.append(self.fam(fam))
        out = self.mlp(torch.cat(feats, dim=-1)).squeeze(-1)
        if self.beta is not None:
            out = out + self.beta * (zz * sigma).sum(-1)
        return out


def masked_softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax over the admissible shards only."""
    if logits.size == 0:
        return logits
    x = logits - logits.max()
    e = np.exp(x)
    return e / e.sum()


def adaptive_top_p_count(probs_sorted_desc: np.ndarray, cfg: RouterConfig, B_probe: int) -> tuple[int, float]:
    """tau_t = clip(P_min + gamma (1 - max p), P_min, P_max);
    b = min(min{b : sum_{l<=b} p_l >= tau_t}, B_probe, |S_t|)."""
    n = probs_sorted_desc.size
    if n == 0:
        return 0, 0.0
    tau = float(np.clip(cfg.p_min + cfg.gamma * (1.0 - probs_sorted_desc.max()), cfg.p_min, cfg.p_max))
    cum = np.cumsum(probs_sorted_desc)
    b_hat = int(np.searchsorted(cum, tau - 1e-9) + 1)
    return int(min(b_hat, B_probe, n)), tau


class Router:
    """Routing policy over the candidate shards (the admissible set S_t under scope-before-routing)."""

    def __init__(self, cfg: RouterConfig, mode: RoutingMode, shard_map: ShardMap,
                 scorer: ShardScorer | None = None, device: str = "cpu"):
        self.cfg = cfg
        self.mode = mode
        self.sm = shard_map
        self.device = device
        self.scorer = scorer
        if mode == "learned" and scorer is None:
            self.scorer = ShardScorer(shard_map.embedder.dim, cfg)
        if self.scorer is not None:
            self.scorer.to(device).eval()
        self._sigma = torch.from_numpy(shard_map.summaries).to(device)
        self._fam = torch.from_numpy(shard_map.family_ids).to(device)
        self._sizes = np.array([sh.n for sh in shard_map.shards], dtype=np.float32)
        self._cost = self._sizes / max(shard_map.mean_size, 1e-6)

    def refresh(self) -> None:
        """Re-read shard summaries and sizes after Tier-B writes."""
        self._sigma = torch.from_numpy(self.sm.summaries).to(self.device)
        self._fam = torch.from_numpy(self.sm.family_ids).to(self.device)
        self._sizes = np.array([sh.n for sh in self.sm.shards], dtype=np.float32)
        self._cost = self._sizes / max(self.sm.mean_size, 1e-6)

    def logits(self, z: np.ndarray, phi: np.ndarray, cand: np.ndarray) -> np.ndarray:
        """s_{t,j} = f_theta(r_t, sigma_j) - alpha * n_j / n_bar for the candidate shards."""
        if cand.size == 0:
            return np.zeros(0, np.float32)
        if self.mode == "learned":
            with torch.no_grad():
                zt = torch.from_numpy(np.asarray(z, np.float32))[None].to(self.device)
                pt = torch.from_numpy(np.asarray(phi, np.float32))[None].to(self.device)
                idx = torch.from_numpy(np.asarray(cand, np.int64)).to(self.device)
                s = self.scorer(zt, pt, self._sigma[idx][None], self._fam[idx][None])[0]
                s = s.cpu().numpy().astype(np.float32)
        else:
            s = 10.0 * (self.sm.summaries[cand] @ np.asarray(z, np.float32)).astype(np.float32)
        if self.cfg.alpha:
            s = s - self.cfg.alpha * self._cost[cand]
        return s

    def select(self, logits: np.ndarray, B_probe: int) -> tuple[np.ndarray, np.ndarray, dict]:
        """Masked softmax and Top-B_probe or adaptive Top-P; returns (positions into cand, probs, info)."""
        n = logits.size
        if n == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.float32), {"tau": None, "b": 0}
        probs = masked_softmax(logits.astype(np.float64) / max(self.cfg.temperature, 1e-6)).astype(np.float32)
        order = np.argsort(-logits, kind="stable")
        if self.cfg.adaptive_top_p:
            b, tau = adaptive_top_p_count(probs[order], self.cfg, B_probe)
        else:
            b, tau = int(min(B_probe, n)), None
        return order[:b], probs, {"tau": tau, "b": b}

    def state_dict(self):
        return self.scorer.state_dict() if self.scorer is not None else {}
