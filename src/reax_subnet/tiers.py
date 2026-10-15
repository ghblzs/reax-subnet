"""Tiered miner pools (release behaviour in 1.2; docs/DESIGN-POOLS.md). REAX_FEATURE_TIERED_POOLS=0 switches them off on
development networks only.

Three hardware tiers, each with its own pinned model(s) and a fixed share of the miner part:

* ``gpu``   - large NVIDIA GPUs (>= 24 GB): s1-fast (and s1-pro when that pool is enabled),
* ``small`` - small GPUs incl. NVIDIA Jetson Orin (8-16 GB): s1-lite, a 4B GGUF model on llama.cpp CUDA,
* ``cpu``   - CPU-only machines incl. ARM: s1-cpu, a sub-1B GGUF model on llama.cpp CPU.

Inside a pool a miner's weight is proportional to ``ema * capacity_factor``: ``ema`` is the existing
correctness-gated round score (a failed verification is a strike; latency counts 20 %), the factor is the verified
throughput from capacity probes in units of the pool's ``target_dps``, linear up to ``max_factor`` units (1.2.0rc3: in
the GPU pools far above one machine, in s1-lite / s1-cpu near the best honest node of the tier; 1.2.0rc2 capped it at
1 unit, which paid a machine faster than the target once per hotkey). The probes hit every UID of a pool at the same
moment (first sends within PROBE_SEND_SKEW_MAX_S) and count only answers inside each UID's window of the same length,
opened at the validator's first send to it, so UIDs that share one backend split its throughput: N hotkeys
on one machine together measure at most what the machine does, and because the factor is linear, they earn at most
what one hotkey on that machine earns (minus N registrations), while a hotkey with more GPUs earns more. With ``weight = 1`` there is deliberately no per-UID floor, because a floor would pay per
registered hotkey rather than per machine.

Two rules keep the expensive tier ahead of the cheap ones whatever the miner counts are:

1. fixed tier budgets (release constants, announced before any change), and
2. the tier-order cap: the average weight per miner (hotkey with weight) in a tier may not exceed ``order_ratio``
   times the average weight per miner in the tier above it; the excess moves up to the tier above (never to a lower
   tier). This is the published 1.2 rule and it stays per miner in 1.2.0rc3, so how fast GPU miners get never
   shrinks the cheap tiers' pay per miner. Splitting a cheap-tier machine into more hotkeys can raise its tier's cap
   (more miners), but never above the tier's fixed budget; inside the pool the split still earns nothing extra.
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
    target_dps: float  # one capacity unit: a UID measured at this throughput has factor 1
    weight: float = 1.0  # in-pool weight = ema * ((1 - weight) + weight * factor); 1 = no per-UID floor
    window_s: float = 10.0  # the probe keeps every UID busy for this long; only answers inside it count
    items_per_request: int = 4
    inflight: int = 2  # concurrent request lanes the validator offers each UID (the miner accepts up to its own limit)
    # Items per UID checked against the reference: a quarter of what it answered, at least 3, at most verified_max
    # (bounded by the validator's reference cost). Unverified probe answers cannot be recognised by the miner.
    verified_share: float = 0.25
    verified_min: int = 3
    verified_max: int = 40
    every_rounds: int = 10
    alpha: float = 0.3  # moving average of the measured throughput
    # The factor is linear up to max_factor units (1 = the 1.2.0rc2 cap at the target). Sybil-neutral only while one
    # machine stays below max_factor * target_dps (above it a split is paid in proportion to throughput, never more).
    max_factor: float = 1.0
    # Requests prepared per UID and probe: enough for headroom x its previous measurement (at least the target, at
    # most the cap), so the validator's probe cost follows the real capacity of the pool, not the cap.
    headroom: float = 4.0

    @property
    def cap_dps(self) -> float:
        return self.target_dps * self.max_factor

    def budget_dps(self, previous: Optional[float]) -> float:
        """Throughput a UID's prepared requests can show in one window. A UID measured at its budget gets
        ``headroom`` times as much next time, so a new big node reaches its real throughput within a few probes."""
        prev = previous if previous is not None and math.isfinite(previous) and previous > 0 else 0.0
        return min(self.cap_dps, max(self.target_dps, self.headroom * prev))

    def requests_for(self, budget_dps: float) -> int:
        """Enough requests that a UID at its budget is still busy when the window ends (so every UID of a probe is
        loaded for the whole window), plus one per lane for the requests in flight at the end."""
        return math.ceil(budget_dps * self.window_s / self.items_per_request) + self.inflight

    @property
    def max_requests(self) -> int:
        """Requests a UID at the cap can receive in one probe (bounds the miner's per-validator rate limit)."""
        return self.requests_for(self.cap_dps)

    def verified(self, answered: int) -> int:
        return min(answered, max(self.verified_min, min(self.verified_max, math.ceil(self.verified_share * answered))))

    def factor(self, dps: float) -> float:
        if not math.isfinite(dps) or dps <= 0:
            return 0.0
        return min(self.max_factor, dps / self.target_dps)

    def multiplier(self, dps: Optional[float]) -> float:
        """A miner that was never probed counts at the floor (1 - weight), i.e. 0 by default."""
        return (1.0 - self.weight) + self.weight * self.factor(dps or 0.0)


POOL_TIER = {"s1-fast": "gpu", "s1-pro": "gpu", "s1-lite": "small", "s1-cpu": "cpu"}
# Within the gpu tier the existing 1.1 split applies.
POOL_SHARE_IN_TIER = {"s1-fast": 0.35, "s1-pro": 0.65, "s1-lite": 1.0, "s1-cpu": 1.0}
CAPACITY = {
    # One unit = the throughput of a well-run node of the tier (DESIGN-POOLS.md section 6); pay is linear in verified
    # throughput up to max_factor units. In the GPU pools the cap is far above one machine (ANALYSIS: capacity v2), so
    # one hotkey per machine is never worse than several. inflight x items_per_request decisions in flight per UID cover the
    # round-trip time of a distant miner at the cap; the window keeps the counting error of the last request small.
    # Requests per probe at the cap stay within the miner's default limit per validator (config.py).
    "s1-fast": CapacityParams(target_dps=40.0, window_s=5.0, items_per_request=16, inflight=16, max_factor=16.0),
    "s1-pro": CapacityParams(target_dps=10.0, window_s=8.0, items_per_request=8, inflight=16, verified_max=10,
                             max_factor=16.0),
    # s1-lite / s1-cpu keep small requests: the reference runtimes answer one decision at a time in arrival order,
    # so the first request of a slow node (Orin ~0.7/s, a small CPU) must finish well inside the window.
    # Their cap sits near the best honest node of the tier (s1-lite 32/s = an RTX 5070 Ti at 33/s; s1-cpu 4/s), not
    # far above it: these models run 10-100x faster on a big GPU, and a high cap would make the cheap pools pay a GPU
    # more than s1-fast does (review H1). Above the cap a split is paid in proportion, bounded by the tier budget.
    "s1-lite": CapacityParams(target_dps=8.0, window_s=16.0, items_per_request=8, inflight=8, max_factor=4.0),
    "s1-cpu": CapacityParams(target_dps=1.0, window_s=30.0, items_per_request=4, inflight=4, verified_max=8,
                             max_factor=4.0),
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

    An empty pool's share goes pro rata to the other pools (as in compute_pool_weights, never burned) and the
    owner UID gets only the burn fraction, or the whole miner part when no pool has an eligible miner; a tier whose own cap
    leaves budget over passes it to the next tier up that has eligible miners.
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

    # Budget per tier. A pool without an eligible miner passes its share pro rata to the pools that have one (inside
    # its tier first, then across tiers): an unclaimed share is never burned (1.2). Only when no pool has an eligible
    # miner does the miner part go to the owner UID (rc3: no miners -> owner).
    tier_budget = {t: 0.0 for t in TIER_ORDER}
    tier_miners = {t: 0 for t in TIER_ORDER}
    for pool in shares:
        if scores[pool]:
            tier_miners[POOL_TIER[pool]] += len(scores[pool])
    for pool, share in shares.items():  # a tier with any live pool keeps its whole budget (empty pools: see below)
        if tier_miners[POOL_TIER[pool]] > 0:
            tier_budget[POOL_TIER[pool]] += share
    live_share = math.fsum(tier_budget.values())
    if live_share <= 0:
        owner_part += miner_part if owner_uid is not None else 0.0
    else:
        for t in TIER_ORDER:
            tier_budget[t] /= live_share
    # Then the tier-order cap from the top tier down.
    present = [t for t in TIER_ORDER if tier_miners[t] > 0]
    # Budget only ever moves up, so repeating the pass until nothing moves terminates (a few passes at most). The cap
    # is computed on the upper tier's budget at that moment and excess is not handed back when the upper average rises
    # later, so the result can be up to ~15 % stricter than the stated ratio; the ratio itself always holds.
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
