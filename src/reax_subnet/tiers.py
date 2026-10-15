"""Tiered miner pools (feature REAX_FEATURE_TIERED_POOLS, development networks only; docs/DESIGN-POOLS.md).

Three hardware tiers, each with its own pinned model(s) and a fixed share of the miner part:

* ``gpu``   - large NVIDIA GPUs (>= 24 GB): s1-fast (and s1-pro when that pool is enabled),
* ``small`` - small GPUs incl. NVIDIA Jetson Orin (8-16 GB): s1-lite, a 4B GGUF model on llama.cpp CUDA,
* ``cpu``   - CPU-only machines incl. ARM: s1-cpu, a sub-1B GGUF model on llama.cpp CPU.

Inside a pool a miner's weight is proportional to ``ema * capacity_factor``: ``ema`` is the existing
correctness-gated round score (a failed verification is a strike; latency counts 20 %), the factor is the verified
throughput from capacity probes, capped at the pool's target. The probes hit every UID of a pool at the same moment,
so UIDs that share one backend split its throughput: running N hotkeys on one machine earns what one hotkey on it
earns (minus N registrations), while N real nodes earn N times. With ``weight = 1`` there is deliberately no
per-UID floor, because a floor would pay per registered hotkey rather than per machine.

Two rules keep the expensive tier ahead of the cheap ones whatever the miner counts are:

1. fixed tier budgets (release constants, announced before any change), and
2. the tier-order cap: the average weight per miner in a tier may not exceed ``order_ratio`` times the average
   weight per miner in the tier above it; the excess moves up to the tier above (never to a lower tier).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional

TIER_ORDER = ("gpu", "small", "cpu")


@dataclass(frozen=True)
class TierBudgets:
    # Shares of the miner part per tier (normalised over tiers that have an enabled pool).
    gpu: float = 0.80
    small: float = 0.15
    cpu: float = 0.05
    # Average weight per miner in a tier <= order_ratio * the average in the tier above.
    order_ratio: float = 0.5

    def share(self, tier: str) -> float:
        return float(getattr(self, tier))

    def __post_init__(self) -> None:
        if not (all(0 <= self.share(t) <= 1 for t in TIER_ORDER) and 0 < self.order_ratio <= 1
                and math.isclose(sum(self.share(t) for t in TIER_ORDER), 1.0, abs_tol=1e-9)):
            raise ValueError("invalid tier budgets")


@dataclass(frozen=True)
class CapacityParams:
    """Verified-throughput probe of one pool (decisions per second, all UIDs of the pool probed at once)."""
    target_dps: float  # throughput at which the capacity factor reaches 1
    weight: float = 1.0  # in-pool weight = ema * ((1 - weight) + weight * factor); 1 = no per-UID floor
    window_s: float = 10.0  # the probe keeps every UID busy for this long
    items_per_request: int = 4
    inflight: int = 2  # concurrent requests per UID (miner default per-caller concurrency is 2)
    verified: int = 3  # items per UID checked against the reference
    every_rounds: int = 10
    alpha: float = 0.3  # moving average of the measured throughput

    @property
    def max_requests(self) -> int:
        """Enough requests that a UID below the target is still busy when the window ends, so all UIDs of a
        probe are loaded for the whole window (a UID that finished early would free a shared machine for its
        siblings). A UID that runs out of requests early is above the target, where the factor is capped."""
        return math.ceil(self.target_dps * self.window_s / self.items_per_request) + self.inflight

    def factor(self, dps: float) -> float:
        if not math.isfinite(dps) or dps <= 0:
            return 0.0
        return min(1.0, dps / self.target_dps)

    def multiplier(self, dps: Optional[float]) -> float:
        """A miner that was never probed counts at the floor (1 - weight), i.e. 0 by default."""
        return (1.0 - self.weight) + self.weight * self.factor(dps or 0.0)


POOL_TIER = {"s1-fast": "gpu", "s1-pro": "gpu", "s1-lite": "small", "s1-cpu": "cpu"}
# Within the gpu tier the existing 1.1 split applies.
POOL_SHARE_IN_TIER = {"s1-fast": 0.35, "s1-pro": 0.65, "s1-lite": 1.0, "s1-cpu": 1.0}
CAPACITY = {
    # Targets from the measured speeds (DESIGN-POOLS.md section 6): about the throughput of a
    # well-run node of the tier, so the bigger node of a tier earns more, up to that point.
    # Requests per probe stay within the miner's default limit of 20 per minute per validator.
    "s1-fast": CapacityParams(target_dps=40.0, window_s=3.0, items_per_request=16),
    "s1-pro": CapacityParams(target_dps=10.0, window_s=5.0, items_per_request=8),
    "s1-lite": CapacityParams(target_dps=8.0, window_s=16.0, items_per_request=8),
    "s1-cpu": CapacityParams(target_dps=1.0, window_s=30.0, items_per_request=2),
}


def pool_shares(pools: list[str], budgets: TierBudgets) -> dict[str, float]:
    """Share of the miner part per enabled pool (tier budget x share inside the tier)."""
    tiers = {POOL_TIER[p] for p in pools}
    tier_total = math.fsum(budgets.share(t) for t in tiers)
    out: dict[str, float] = {}
    for p in sorted(pools):
        t = POOL_TIER[p]
        in_tier = math.fsum(POOL_SHARE_IN_TIER[q] for q in pools if POOL_TIER[q] == t)
        out[p] = budgets.share(t) / tier_total * POOL_SHARE_IN_TIER[p] / in_tier
    return out


def compute_tiered_weights(
    ema_by_pool: Mapping[str, Mapping[int, float]],
    capacity_by_pool: Mapping[str, Mapping[int, float]],
    budgets: TierBudgets,
    owner_uid: Optional[int],
    min_ema: float,
    burn_fraction: float = 0.0,
) -> dict[int, float]:
    """Weights over all UIDs. ``capacity_by_pool`` holds measured decisions/s (missing = never probed).

    An empty pool's share goes to the owner UID, as in compute_pool_weights; a tier whose own cap leaves
    budget over passes it to the next tier up that has eligible miners (or to the owner if none has).
    """
    if owner_uid is None and burn_fraction >= 1.0:
        return {}
    owner_part = burn_fraction if owner_uid is not None else 0.0
    miner_part = 1.0 - owner_part
    shares = pool_shares(sorted(ema_by_pool), budgets)

    # Eligible miners and their in-pool scores. A pool for which this validator has no capacity measurement at all
    # (probes not run yet, or failing on the validator's side) falls back to the correctness score alone, so a
    # validator problem never sends a whole pool's share to the owner.
    scores: dict[str, dict[int, float]] = {}
    for pool in sorted(ema_by_pool):
        caps = capacity_by_pool.get(pool, {})
        cap = CAPACITY[pool] if caps else CapacityParams(target_dps=1.0, weight=0.0)
        scores[pool] = {}
        for uid, s in sorted(ema_by_pool[pool].items()):
            if uid != owner_uid and math.isfinite(s) and s >= min_ema:
                v = s * cap.multiplier(caps.get(uid))
                if v > 0:
                    scores[pool][uid] = v

    # Budget per tier, then the tier-order cap from the top tier down.
    tier_budget = {t: 0.0 for t in TIER_ORDER}
    tier_miners = {t: 0 for t in TIER_ORDER}
    for pool, share in shares.items():
        t = POOL_TIER[pool]
        if scores[pool]:
            tier_budget[t] += share
            tier_miners[t] += len(scores[pool])
        else:
            owner_part += miner_part * share if owner_uid is not None else 0.0
    present = [t for t in TIER_ORDER if tier_miners[t] > 0]
    # Budget only ever moves up, so repeating the pass until nothing moves terminates (a few passes at most).
    for _ in range(4 * len(TIER_ORDER)):
        moved = False
        for upper, lower in zip(present, present[1:]):
            cap_lower = budgets.order_ratio * tier_budget[upper] / tier_miners[upper] * tier_miners[lower]
            if tier_budget[lower] > cap_lower * (1 + 1e-12):
                tier_budget[upper] += tier_budget[lower] - cap_lower
                tier_budget[lower] = cap_lower
                moved = True
        if not moved:
            break

    weights: dict[int, float] = {}
    for t in present:
        pools = [p for p in shares if POOL_TIER[p] == t and scores[p]]
        base = math.fsum(shares[p] for p in pools)
        for p in pools:
            part = miner_part * tier_budget[t] * shares[p] / base
            total = math.fsum(scores[p].values())
            if total <= 0:
                if owner_uid is not None:
                    owner_part += part
                continue
            for uid, s in scores[p].items():
                weights[uid] = weights.get(uid, 0.0) + part * s / total
    if owner_uid is not None and owner_part > 0:
        weights[owner_uid] = weights.get(owner_uid, 0.0) + owner_part
    if not weights:
        return {}
    norm = math.fsum(weights.values())
    return {uid: w / norm for uid, w in sorted(weights.items())}
