"""Specialist score: ``S = 0.75 x competence + 0.25 x calibration`` (docs/SPECIALIST-TRACKS.md).

Competence is the JevBench O1S chance-corrected score per primitive (choice, noul, score), averaged over the
primitives present; calibration is ``100 x max(0, 1 - Brier / Brier_uniform)``. The per-item definitions are those of
the dev-set baselines used to calibrate the tracks, so the numbers are comparable.
The paired standard error of a score difference comes from a seeded bootstrap over the shared items.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

W_COMPETENCE = 0.75
W_CALIBRATION = 0.25
NOUL_NO, NOUL_YES = 0.20, 0.80   # p(yes) <= 0.20 -> no, >= 0.80 -> yes, in between = abstain (wrong)
PRIMITIVES = ("choice", "noul", "score")
BOOTSTRAP_RESAMPLES = 400


def combine(competence: float, calibration: float) -> float:
    return W_COMPETENCE * competence + W_CALIBRATION * calibration


def clean_probs(probs: Any, labels: Sequence[str]) -> dict[str, float] | None:
    """A probability dict that is exactly over ``labels``, finite, in [0,1], summing to ~1; else None."""
    if not isinstance(probs, Mapping) or set(map(str, probs)) != set(labels):
        return None
    out: dict[str, float] = {}
    for k, v in probs.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 or v > 1:
            return None
        out[str(k)] = float(v)
    s = sum(out.values())
    if abs(s - 1) > 2e-2 or s <= 0:
        return None
    return {k: v / s for k, v in out.items()}


@dataclass(frozen=True)
class ItemScore:
    item_id: str
    kind: str
    tier: str
    valid: bool
    correct: bool
    chance: float       # choice: 1/k; score: mean abs error of a uniform guess (normalised); noul: 0.5
    err: float          # score only: normalised abs error of the expected level
    brier: float
    brier_uniform: float


def score_item(row: Mapping[str, Any], probs: Any) -> ItemScore:
    kind = row["question"]["type"]
    labels = [str(x) for x in row["labels"]]
    k = len(labels)
    gold = str(int(row["expected"])) if kind == "score" else str(row["expected"])
    p = clean_probs(probs, labels)
    uniform = {lab: 1 / k for lab in labels}
    d = p or uniform
    brier = sum((d[lab] - (1.0 if lab == gold else 0.0)) ** 2 for lab in labels)
    brier_u = sum((uniform[lab] - (1.0 if lab == gold else 0.0)) ** 2 for lab in labels)
    tier = str(row.get("tier", ""))
    iid = str(row.get("id", ""))
    if kind == "choice":
        correct = bool(p) and sorted(p, key=lambda lab: (-p[lab], lab))[0] == gold
        return ItemScore(iid, kind, tier, p is not None, correct, 1 / k, 0.0, brier, brier_u)
    if kind == "noul":
        dec = None
        if p:
            py = p["yes"]
            dec = "no" if py <= NOUL_NO else "yes" if py >= NOUL_YES else None
        return ItemScore(iid, kind, tier, p is not None, dec == gold, 0.5, 0.0, brier, brier_u)
    gi = int(gold)
    chance = sum(abs(lv - gi) for lv in range(k)) / k / (k - 1)
    if p:
        pred = sum(int(lab) * pr for lab, pr in p.items())
        err = abs(pred - gi) / (k - 1)
        correct = round(pred) == gi
    else:
        err = max(abs(lv - gi) for lv in range(k)) / (k - 1)
        correct = False
    return ItemScore(iid, kind, tier, p is not None, correct, chance, err, brier, brier_u)


@dataclass
class Score:
    n: int
    failures: int
    competence: float
    calibration: float
    total: float
    primitives: dict[str, dict[str, float]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"n": self.n, "failures": self.failures, "competence": self.competence,
                "calibration": self.calibration, "score": self.total, "primitives": self.primitives}


def _clip(x: float) -> float:
    return max(0.0, min(100.0, x))


def _arrays(items: Sequence[ItemScore]) -> dict[str, dict[str, np.ndarray]]:
    by: dict[str, dict[str, list]] = {}
    for idx, it in enumerate(items):
        d = by.setdefault(it.kind, {"idx": [], "correct": [], "chance": [], "err": [], "brier": [], "bu": [], "valid": []})
        d["idx"].append(idx)
        d["correct"].append(1.0 if it.correct else 0.0)
        d["chance"].append(it.chance)
        d["err"].append(it.err)
        d["brier"].append(it.brier)
        d["bu"].append(it.brier_uniform)
        d["valid"].append(1.0 if it.valid else 0.0)
    return {kind: {k: np.asarray(v, dtype=float) for k, v in d.items()} for kind, d in by.items()}


def _score_weighted(arrs: dict[str, dict[str, np.ndarray]], w: np.ndarray | None) -> tuple[float, float, dict]:
    """Competence, calibration and per-primitive detail for item weights ``w`` (None = all ones)."""
    comp, cal, prim = [], [], {}
    for kind in PRIMITIVES:
        a = arrs.get(kind)
        if a is None:
            continue
        ww = np.ones(len(a["idx"])) if w is None else w[a["idx"].astype(int)]
        tot = ww.sum()
        if tot <= 0:
            continue
        acc = float((ww * a["correct"]).sum() / tot)
        if kind == "choice":
            c = float((ww * a["chance"]).sum() / tot)
            cc = 100 * (acc - c) / (1 - c) if c < 1 else 0.0
        elif kind == "noul":
            cc = 100 * (acc - 0.5) / 0.5
        else:
            chance = float((ww * a["chance"]).sum() / tot)
            mae = float((ww * a["err"]).sum() / tot)
            cc = 100 * (1 - mae / chance) if chance > 0 else 0.0
        b = float((ww * a["brier"]).sum() / tot)
        bu = float((ww * a["bu"]).sum() / tot)
        ca = 100 * max(0.0, 1 - b / bu) if bu > 0 else 0.0
        comp.append(_clip(cc))
        cal.append(ca)
        prim[kind] = {"n": float(tot), "accuracy": acc, "competence": _clip(cc), "calibration": ca}
    if not comp:
        return 0.0, 0.0, {}
    return sum(comp) / len(comp), sum(cal) / len(cal), prim


def score_items(items: Sequence[ItemScore]) -> Score:
    arrs = _arrays(items)
    comp, cal, prim = _score_weighted(arrs, None)
    return Score(n=len(items), failures=sum(not it.valid for it in items), competence=comp, calibration=cal,
                 total=combine(comp, cal), primitives=prim)


def score_run(rows: Sequence[Mapping[str, Any]], probs: Sequence[Any]) -> tuple[list[ItemScore], Score]:
    items = [score_item(r, p) for r, p in zip(rows, probs)]
    return items, score_items(items)


def paired_se(a: Sequence[ItemScore], b: Sequence[ItemScore], seed: int, resamples: int = BOOTSTRAP_RESAMPLES) -> float:
    """Bootstrap standard error of ``S(a) - S(b)`` over the shared items (same indices for both models)."""
    if len(a) != len(b) or any(x.item_id != y.item_id for x, y in zip(a, b)):
        raise ValueError("paired scores need the same items in the same order")
    n = len(a)
    if n < 2:
        return float("inf")
    aa, bb = _arrays(a), _arrays(b)
    rng = np.random.RandomState(seed % (2 ** 32))
    diffs = np.empty(resamples)
    for r in range(resamples):
        w = np.bincount(rng.randint(0, n, size=n), minlength=n).astype(float)
        ca, la, _ = _score_weighted(aa, w)
        cb, lb, _ = _score_weighted(bb, w)
        diffs[r] = combine(ca, la) - combine(cb, lb)
    return float(diffs.std(ddof=1))


def by_tier(items: Sequence[ItemScore]) -> dict[str, Score]:
    tiers = sorted({it.tier for it in items})
    return {t: score_items([it for it in items if it.tier == t]) for t in tiers}


# ----------------------------------------------------------------------------- cross-author gap (eval mechanism v3, stage 2)
# Cross-author tier (sampling.X_TIER): items written by authors outside the generator families. The gap of an entry is
# ``s_gen - s_x`` (score on the generator-family items minus score on the cross-author items). A large gap means the model
# learned the generator families' style rather than the task. In this release the gap is only REPORTED (history row
# "gap_shadow"); the penalty is never applied (binding use waits for the pilot and a notice to miners).
GAP_WEIGHT = 0.5
GAP_ALLOWANCE_FLOOR = 8.0
GAP_MIN_COLDKEYS = 5


def split_x(items: Sequence[ItemScore], x_tier: str = "cross_author") -> tuple[list[ItemScore], list[ItemScore]]:
    """(items outside the cross-author tier, items of it)."""
    return [it for it in items if it.tier != x_tier], [it for it in items if it.tier == x_tier]


def gap_se(gen: Sequence[ItemScore], x: Sequence[ItemScore], seed: int, resamples: int = BOOTSTRAP_RESAMPLES) -> float:
    """Bootstrap standard error of ``S(gen) - S(x)``: the two item groups are resampled independently."""
    if len(gen) < 2 or len(x) < 2:
        return float("inf")
    ag, ax = _arrays(gen), _arrays(x)
    rng = np.random.RandomState(seed % (2 ** 32))
    diffs = np.empty(resamples)
    for r in range(resamples):
        wg = np.bincount(rng.randint(0, len(gen), size=len(gen)), minlength=len(gen)).astype(float)
        wx = np.bincount(rng.randint(0, len(x), size=len(x)), minlength=len(x)).astype(float)
        cg, lg, _ = _score_weighted(ag, wg)
        cx, lx, _ = _score_weighted(ax, wx)
        diffs[r] = combine(cg, lg) - combine(cx, lx)
    return float(diffs.std(ddof=1))


def median_gap(gap_by_coldkey: Mapping[str, float], reference: Sequence[float] = (), min_coldkeys: int = GAP_MIN_COLDKEYS) -> float | None:
    """G_med: the median of the per-coldkey gaps (plus reference models' gaps if given). Fewer than ``min_coldkeys``
    coldkeys -> None: no penalty is computed, the gap is only reported."""
    if len(gap_by_coldkey) < min_coldkeys:
        return None
    vals = sorted([float(v) for v in gap_by_coldkey.values() if math.isfinite(v)] + [float(v) for v in reference if math.isfinite(v)])
    if not vals:
        return None
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2


def gap_penalty(gaps: Sequence[float], ses: "Sequence[float | None] | float | None", g_med: float | None,
                w: float = GAP_WEIGHT, allowance_floor: float = GAP_ALLOWANCE_FLOOR) -> tuple[float, dict[str, Any]]:
    """Penalty for an entry whose gap exceeds the median gap by more than the allowance. Pure.

    ``gaps``: the entry's gap (``s_gen - s_x``) on each draw. ``ses``: the bootstrap SE of the gap on each draw (a single
    number is used for every draw; None or a missing value counts as 0, i.e. the allowance floor decides).
    pooled gap = mean(gaps); SE_gap_pooled = rms(ses) / sqrt(k); excess = pooled gap - g_med;
    allowance A = max(allowance_floor, 2 x SE_gap_pooled); penalty = w x max(0, excess - A).
    ``g_med`` None (fewer than ``GAP_MIN_COLDKEYS`` coldkeys): penalty 0, details only. Returns (penalty, details)."""
    k = len(gaps)
    if k == 0:
        return 0.0, {"draws": 0, "reason": "no draws with cross-author items", "penalty": 0.0}
    if ses is None or isinstance(ses, (int, float)):
        per = [0.0 if ses is None else float(ses)] * k
    else:
        per = [0.0 if (v is None or not math.isfinite(v)) else float(v) for v in ses]
        per = (per + [0.0] * k)[:k]
    gap = sum(gaps) / k
    se = math.sqrt(sum(v * v for v in per) / k) / math.sqrt(k)
    allowance = max(allowance_floor, 2 * se)
    details: dict[str, Any] = {"draws": k, "gap": gap, "se_gap": se, "g_med": g_med, "allowance": allowance, "w": w}
    if g_med is None:
        details.update(excess=None, penalty=0.0, reason="fewer than %d coldkeys: reported only" % GAP_MIN_COLDKEYS)
        return 0.0, details
    excess = gap - g_med
    penalty = w * max(0.0, excess - allowance)
    details.update(excess=excess, penalty=penalty)
    return penalty, details
