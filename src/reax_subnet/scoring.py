"""Deterministic scoring for the Minimal Production Mechanism.

Everything in this module is a pure function of its inputs. Two validators
that see the same reference distributions, the same miner responses and the
same latencies compute bit-identical scores and weights.

Per round and miner::

    item passes      <=> total_variation(p_miner, p_ref) <= tau
    fidelity          = passed items / items sent
    round_score       = 0                                if fidelity < min_round_fidelity
                      = fidelity * ((1 - lw) + lw * speed) otherwise
    speed             = clamp((slow - latency) / (slow - fast), 0, 1)

A missing, late, unauthenticated or malformed response scores 0 for the
whole round. Across rounds the validator keeps an exponential moving average
per UID. Weights are proportional to the EMA among miners whose EMA is at
least ``min_ema``; ``burn_fraction`` of the weight (and all of it when no
miner qualifies) goes to the subnet owner's UID.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence


@dataclass(frozen=True)
class ScoringParams:
    # Largest total-variation distance at which a miner answer still matches the
    # reference. Provisional; see docs/MPM.md "Calibrating tau".
    tau: float = 0.15
    # A round with fewer matching items than this share scores zero.
    min_round_fidelity: float = 0.8
    # Latency bounds for one batch, in seconds (measured by the validator).
    latency_fast_s: float = 2.0
    latency_slow_s: float = 20.0
    # Share of the round score that depends on speed. Kept small on purpose:
    # fidelity is what matters, speed only separates otherwise equal miners.
    latency_weight: float = 0.2
    ema_alpha: float = 0.1
    # Miners below this EMA get no weight (filters flapping/new miners).
    min_ema: float = 0.3
    # Share of every weight vector assigned to the subnet owner UID.
    burn_fraction: float = 0.0

    def __post_init__(self) -> None:
        checks = {
            "tau": 0.0 < self.tau < 1.0,
            "min_round_fidelity": 0.0 <= self.min_round_fidelity <= 1.0,
            "latency bounds": 0.0 <= self.latency_fast_s < self.latency_slow_s,
            "latency_weight": 0.0 <= self.latency_weight <= 1.0,
            "ema_alpha": 0.0 < self.ema_alpha <= 1.0,
            "min_ema": 0.0 <= self.min_ema < 1.0,
            "burn_fraction": 0.0 <= self.burn_fraction <= 1.0,
        }
        bad = [name for name, ok in checks.items() if not ok]
        if bad:
            raise ValueError(f"invalid scoring parameters: {', '.join(bad)}")


def total_variation(p: Sequence[float], q: Sequence[float]) -> float:
    if len(p) != len(q):
        raise ValueError("distributions have different lengths")
    return 0.5 * math.fsum(abs(a - b) for a, b in zip(p, q))


def item_passes(p_miner: Sequence[float], p_ref: Sequence[float], params: ScoringParams) -> bool:
    return total_variation(p_miner, p_ref) <= params.tau


def speed_factor(latency_s: float, params: ScoringParams) -> float:
    if not math.isfinite(latency_s):
        return 0.0
    span = params.latency_slow_s - params.latency_fast_s
    return min(1.0, max(0.0, (params.latency_slow_s - latency_s) / span))


def round_score(
    answers: Optional[Mapping[str, Sequence[float]]],
    references: Mapping[str, Sequence[float]],
    latency_s: float,
    params: ScoringParams,
) -> tuple[float, float]:
    """Score one miner for one round. Returns ``(round_score, fidelity)``.

    ``answers`` is None when the miner failed the round as a whole (no
    response, timeout, bad signature, protocol error). Items the miner did not
    answer count as failed.
    """
    if answers is None or not references:
        return 0.0, 0.0
    passed = sum(
        1
        for item_id in sorted(references)
        if item_id in answers
        and len(answers[item_id]) == len(references[item_id])
        and item_passes(answers[item_id], references[item_id], params)
    )
    fidelity = passed / len(references)
    if fidelity < params.min_round_fidelity:
        return 0.0, fidelity
    lw = params.latency_weight
    return fidelity * ((1.0 - lw) + lw * speed_factor(latency_s, params)), fidelity


def ema_update(previous: float, observation: float, alpha: float) -> float:
    return (1.0 - alpha) * previous + alpha * observation


def compute_weights(
    ema: Mapping[int, float], owner_uid: Optional[int], params: ScoringParams
) -> dict[int, float]:
    """Turn per-UID EMA scores into a normalised weight vector.

    The owner UID never earns miner weight from its own score; it only
    receives the burn share. Without an owner UID the burn share is dropped
    and the miner weights are renormalised. Returns ``{}`` only when there is
    neither a qualifying miner nor an owner UID (the caller then skips
    setting weights).
    """
    eligible = {
        uid: score
        for uid, score in sorted(ema.items())
        if uid != owner_uid and math.isfinite(score) and score >= params.min_ema
    }
    total = math.fsum(eligible.values())
    if total <= 0.0:
        return {owner_uid: 1.0} if owner_uid is not None else {}
    if owner_uid is None and params.burn_fraction >= 1.0:
        return {}  # full burn requested but no owner UID: set nothing rather than pay miners
    miner_share = 1.0 - params.burn_fraction if owner_uid is not None else 1.0
    weights = {uid: miner_share * score / total for uid, score in eligible.items() if miner_share > 0.0}
    if owner_uid is not None and params.burn_fraction > 0.0:
        weights[owner_uid] = params.burn_fraction
    return weights
