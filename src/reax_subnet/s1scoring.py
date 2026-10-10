"""Scoring for ``reax-s1/1`` (pure functions; see docs/DESIGN-v1.1.md).

Per item the validator compares the miner's probabilities ``p`` with its own
reference ``q`` in *logit space*::

    d(p, q) = T * max_i | (ln p_i - mean_j ln p_j) - (ln q_i - mean_j ln q_j) |

``T`` is the model's reference temperature, so ``d`` is the largest
disagreement between two option logits in the model's own units. It does not
depend on the temperature and does not saturate for confident answers, which is
why reference answers (T = 2.07 / 2.0) can be checked without rc3's raised
protocol temperature.

Per round and miner: n items are sent; the validator verifies a secret random
subset of k of them after the answers arrive. A malformed or failed response
scores 0. With D = mean distance over the verified items:

* D > tau_mean                      -> the round scores 0 (an honest outlier costs one round);
* D > tau_strike or any d > tau_item -> strike: the moving average is reset to 0;
* otherwise                         -> the round scores (1 - lw) + lw * speed.

The thresholds come from the GPU calibration (CALIBRATION.md): honest setups
(reference path on another GPU, other honest engines) stay far below tau_mean on
average, an FP8 build of the same weights sits around 4x higher, other models far
above tau_strike.
"""

from __future__ import annotations

import math
import secrets
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

# Probabilities below this are treated as equal. The reference engine returns float32 probabilities and builds
# p(false) = 1 - p(true) for yes/no questions, so on extremely confident items the small side is pure
# rounding; below 1e-6 an option carries no usable fingerprint anyway.
LOG_FLOOR = 1e-6


@dataclass(frozen=True)
class PoolParams:
    model: str
    temperature: float
    # Release constants from the GPU calibration (CALIBRATION.md), in logit units.
    tau_mean: float  # largest mean distance over the verified items for a scoring round
    tau_strike: float  # a mean above this is a strike
    tau_item: float  # any single verified item above this is a strike
    items_per_round: int
    verified_per_round: int
    latency_fast_s: float
    latency_slow_s: float
    latency_weight: float = 0.2

    def __post_init__(self) -> None:
        ok = (
            self.temperature > 0
            and 0 < self.tau_mean < self.tau_strike < 50
            and self.tau_mean < self.tau_item < 50
            and 1 <= self.verified_per_round <= self.items_per_round <= 16
            and 0 <= self.latency_fast_s < self.latency_slow_s
            and 0 <= self.latency_weight <= 1
        )
        if not ok:
            raise ValueError(f"invalid pool parameters for {self.model}")


def centred_logs(p: Sequence[float]) -> list[float]:
    logs = [math.log(max(float(x), LOG_FLOOR)) for x in p]
    mean = math.fsum(logs) / len(logs)
    return [x - mean for x in logs]


def logit_distance(p: Sequence[float], q: Sequence[float], temperature: float) -> float:
    if len(p) != len(q) or not p:
        raise ValueError("distributions have different lengths")
    a, b = centred_logs(p), centred_logs(q)
    return temperature * max(abs(x - y) for x, y in zip(a, b))


def total_variation(p: Sequence[float], q: Sequence[float]) -> float:
    return 0.5 * math.fsum(abs(a - b) for a, b in zip(p, q))


def choose_verified(item_ids: Sequence[str], k: int) -> list[str]:
    """Secret uniform choice of k items; drawn by the validator after the answers arrive."""
    pool = list(item_ids)
    chosen = []
    for _ in range(min(k, len(pool))):
        chosen.append(pool.pop(secrets.randbelow(len(pool))))
    return sorted(chosen)


def speed_factor(latency_s: float, params: PoolParams) -> float:
    if not math.isfinite(latency_s):
        return 0.0
    span = params.latency_slow_s - params.latency_fast_s
    return min(1.0, max(0.0, (params.latency_slow_s - latency_s) / span))


@dataclass(frozen=True)
class RoundOutcome:
    score: float
    verified: int
    passed: int  # verified items within tau_item
    strike: bool
    max_distance: float
    mean_distance: float


def score_round(
    answers: Optional[Mapping[str, Sequence[float]]],
    references: Mapping[str, Sequence[float]],
    latency_s: float,
    params: PoolParams,
) -> RoundOutcome:
    """``references`` holds the verified items only; ``answers`` all answered items (None = failed response)."""
    if answers is None or not references:
        return RoundOutcome(0.0, len(references), 0, False, math.inf, math.inf)
    ds = []
    for item_id in sorted(references):
        got = answers.get(item_id)
        ref = references[item_id]
        ds.append(math.inf if got is None or len(got) != len(ref) else logit_distance(got, ref, params.temperature))
    worst = max(ds)
    mean = math.fsum(ds) / len(ds) if math.isfinite(worst) else math.inf
    passed = sum(1 for d in ds if d <= params.tau_item)
    if worst > params.tau_item or mean > params.tau_strike:
        return RoundOutcome(0.0, len(ds), passed, True, worst, mean)
    if mean > params.tau_mean:
        return RoundOutcome(0.0, len(ds), passed, False, worst, mean)
    lw = params.latency_weight
    return RoundOutcome((1.0 - lw) + lw * speed_factor(latency_s, params), len(ds), passed, False, worst, mean)


def compute_pool_weights(
    ema_by_pool: Mapping[str, Mapping[int, float]],
    shares: Mapping[str, float],
    owner_uid: Optional[int],
    min_ema: float,
    burn_fraction: float = 0.0,
) -> dict[int, float]:
    """Combine per-pool EMAs into one weight vector.

    Each enabled pool gets its fixed share of the miner part; inside a pool weights
    are proportional to EMA among miners at or above ``min_ema``. A pool without an
    eligible miner passes its share to the owner UID (rc3: no miners -> owner).
    """
    if owner_uid is None and burn_fraction >= 1.0:
        return {}  # full burn requested but no owner UID: set nothing rather than pay miners (rc3 rule)
    weights: dict[int, float] = {}
    owner_part = burn_fraction if owner_uid is not None else 0.0
    miner_part = 1.0 - owner_part if owner_uid is not None else 1.0
    total_share = math.fsum(shares.values())
    if total_share <= 0:
        return {owner_uid: 1.0} if owner_uid is not None else {}
    for pool in sorted(shares):
        share = miner_part * shares[pool] / total_share
        eligible = {
            uid: s for uid, s in sorted(ema_by_pool.get(pool, {}).items())
            if uid != owner_uid and math.isfinite(s) and s >= min_ema
        }
        total = math.fsum(eligible.values())
        if total <= 0:
            if owner_uid is not None:
                owner_part += share
            continue
        for uid, s in eligible.items():
            weights[uid] = weights.get(uid, 0.0) + share * s / total
    if owner_uid is not None and owner_part > 0:
        weights[owner_uid] = weights.get(owner_uid, 0.0) + owner_part
    if not weights:
        return {}
    norm = math.fsum(weights.values())
    return {uid: w / norm for uid, w in sorted(weights.items())}
