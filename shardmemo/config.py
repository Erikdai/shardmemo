"""Configuration of the ShardMemo memory system and of the three benchmark runs."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass
class Budgets:
    """Per-request budgets: |A| <= M, |P| <= B_probe, |R_B| <= K, |U| <= R; L caps ANN candidates per shard."""
    M: int = 10
    B_probe: int = 3
    K: int = 10
    R: int = 3
    L: int = 50


RoutingMode = Literal[
    "learned",        # trained shard scorer f_theta
    "cosine",         # cosine similarity to the shard prototype sigma_j, for shards without routing supervision
]


@dataclass
class RouterConfig:
    hidden: int = 128
    dropout: float = 0.1
    family_emb_dim: int = 8
    use_family_emb: bool = True
    use_meta_features: bool = True
    use_product_feature: bool = True
    cosine_residual: bool = True      # s = MLP(.) + beta * cos(z, sigma)
    beta_init: float = 1.0
    alpha: float = 0.1                # cost-bias coefficient
    temperature: float = 1.0
    adaptive_top_p: bool = True       # adaptive Top-P capped by B_probe, otherwise fixed Top-B_probe
    gamma: float = 0.6
    p_min: float = 0.5
    p_max: float = 0.95
    budget_aware_loss: bool = True
    ba_lambda: float = 1.0
    ba_kappa: float = 2.0
    ba_floor: float = 0.2


@dataclass
class TrainConfig:
    optimizer: Literal["adamw", "adam"] = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 50
    batch_size: int = 32
    probe_budget: int = 3             # B_probe seen by the router during training
    cosine_annealing: bool = True
    device: str = "cpu"


@dataclass
class RankingConfig:
    """Evidence score: weighted sum of min-max normalised BM25, dense and routing scores."""
    w_bm25: float = 0.4
    w_dense: float = 0.4
    w_route: float = 0.2


@dataclass
class IndexConfig:
    kind: Literal["hnsw", "flat"] = "hnsw"
    hnsw_M: int = 32
    ef_construction: int = 200
    ef_search: int = 64
    flat_below: int = 32


@dataclass
class ShardMemoConfig:
    budgets: Budgets = field(default_factory=Budgets)
    router: RouterConfig = field(default_factory=RouterConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    routing_mode: RoutingMode = "learned"
    scope_before_routing: bool = True
    strip_scope_terms: bool = False   # remove speaker names enforced by psi^B from the ranking query
    use_tier_a: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GeneratorConfig:
    """Answer generator behind an OpenAI-compatible API, configured by LLM_MODEL, LLM_BASE_URL and LLM_API_KEY."""
    model: str = field(default_factory=lambda: os.environ.get("LLM_MODEL", ""))
    base_url: str = field(default_factory=lambda: os.environ.get("LLM_BASE_URL", ""))
    api_key_env: str = "LLM_API_KEY"
    prompt: str = "locomo"                    # key of shardmemo.generator.PROMPTS
    evidence_order: Literal["score", "chronological"] = "score"
    max_tokens: int = 256
    n_samples: int = 1                        # > 1: majority vote over n sampled answers and the greedy answer
    sample_temperature: float = 0.7
    workers: int = 32


def _locomo_memory() -> ShardMemoConfig:
    return ShardMemoConfig(
        budgets=Budgets(B_probe=3, K=20),
        router=RouterConfig(adaptive_top_p=False),
        ranking=RankingConfig(w_bm25=0.5, w_dense=0.3, w_route=0.2),
        strip_scope_terms=True,
    )
    # return ShardMemoConfig(
    #     budgets=Budgets(B_probe=3, K=10),
    #     router=RouterConfig(adaptive_top_p=False),
    #     ranking=RankingConfig(w_bm25=0.5, w_dense=0.3, w_route=0.2),
    #     strip_scope_terms=True,
    # )
    # return ShardMemoConfig(
    #     budgets=Budgets(B_probe=3, K=10),
    #     router=RouterConfig(adaptive_top_p=True, gamma=0.6, p_min=0.5, p_max=0.95,
    #                         use_product_feature=False, cosine_residual=False),
    #     ranking=RankingConfig(w_bm25=0.4, w_dense=0.4, w_route=0.2),
    #     train=TrainConfig(optimizer="adam", epochs=50),
    #     strip_scope_terms=True,
    # )


def _locomo_generator() -> GeneratorConfig:
    return GeneratorConfig(prompt="locomo", evidence_order="chronological")
    # return GeneratorConfig(prompt="locomo", evidence_order="score")
    # return GeneratorConfig(prompt="locomo", evidence_order="chronological", n_samples=5)


@dataclass
class LocomoConfig:
    memory: ShardMemoConfig = field(default_factory=_locomo_memory)
    generator: GeneratorConfig = field(default_factory=_locomo_generator)
    num_shards: int = 40
    show_previous_turn: bool = True
    test_conversations: tuple[str, ...] = ("conv-49", "conv-50")
    seeds: tuple[int, ...] = (0, 1, 2)


def _hotpotqa_memory() -> ShardMemoConfig:
    return ShardMemoConfig(
        budgets=Budgets(B_probe=12, K=20),
        router=RouterConfig(beta_init=10.0, adaptive_top_p=False),
        train=TrainConfig(epochs=20, probe_budget=3),
        index=IndexConfig(ef_construction=100),
    )
    # return ShardMemoConfig(
    #     budgets=Budgets(B_probe=12, K=10),
    #     router=RouterConfig(beta_init=10.0, adaptive_top_p=False),
    #     train=TrainConfig(epochs=20, probe_budget=3),
    #     index=IndexConfig(ef_construction=100),
    # )
    # return ShardMemoConfig(
    #     budgets=Budgets(B_probe=3, K=10),
    #     router=RouterConfig(beta_init=10.0, adaptive_top_p=False),
    #     train=TrainConfig(epochs=20, probe_budget=3),
    #     index=IndexConfig(ef_construction=100),
    # )
    # return ShardMemoConfig(
    #     budgets=Budgets(B_probe=3, K=10),
    #     router=RouterConfig(adaptive_top_p=True, gamma=0.6, p_min=0.5, p_max=0.95,
    #                         use_product_feature=False, cosine_residual=False),
    #     train=TrainConfig(optimizer="adam", epochs=50, probe_budget=3),
    #     index=IndexConfig(ef_construction=200),
    # )


def _hotpotqa_generator() -> GeneratorConfig:
    return GeneratorConfig(prompt="hotpotqa", n_samples=5)
    # return GeneratorConfig(prompt="hotpotqa", n_samples=1)


@dataclass
class HotpotQAConfig:
    memory: ShardMemoConfig = field(default_factory=_hotpotqa_memory)
    generator: GeneratorConfig = field(default_factory=_hotpotqa_generator)
    context_lengths: tuple[int, ...] = (56_000, 224_000, 448_000)
    n_eval: int = 500
    n_train: int = 1500
    max_negatives: int = 64
    extra_probe_budgets: tuple[int, ...] = ()
    # extra_probe_budgets: tuple[int, ...] = (3, 5, 8)
    seeds: tuple[int, ...] = (0, 1, 2)


@dataclass
class ToolBenchConfig:
    R: int = 3
    tau_c: float = 0.5
    n_in_domain: int = 348
    n_ood: int = 181
    seeds: tuple[int, ...] = (0, 1, 2)
