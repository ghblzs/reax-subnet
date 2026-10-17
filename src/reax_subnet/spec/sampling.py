"""Fresh draws from the sealed pool (host directory ``REAX_SPEC_SEALED_DIR``, mounted read-only by the operator).

Layout: ``<dir>/<category>/sealed/sealed.clean.jsonl`` (``sealed.jsonl`` as fallback). Inside a tier, items that were
drawn before are used last (a fresh item of the tier is always preferred), an item used ``ITEM_MAX_USES`` times is
retired, and a confirmation draw is disjoint from the draws it confirms.
Sealed rows never leave this process: only item ids are persisted.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Mapping, Sequence

SAMPLE_SIZE = 300   # 1.2.0rc6 (was 600): with 85 % non-proc items, a first draw and its disjoint confirmation fit the pools
MIN_SAMPLE = 200
# Share of the draw per tier (docs/SPECIALIST-TRACKS.md/5). Since 1.2.0rc6 (owner decision 16 Oct 2026) 15 % programmatic
# gold from the published procedural families and 85 % LLM-written items, split 50/25/25 same-family, new-domain,
# new-family; 1.2.0rc5 drew 50 % programmatic gold, which let a model that only learned the public procedures clear the
# entry floor. A tier with too few unused items fills its quota with its own least-used items first (1.2.0rc6:
# before, the shortfall went to the other tiers' fresh items, so the public-family proc tier grew as the small tiers were
# used up); only a tier with too few eligible items at all hands the rest to the other non-proc tiers, then to proc.
TIER_MIX = {"proc_sealed": 0.15, "same_family": 0.425, "new_domain": 0.2125, "new_family": 0.2125}
# An item is reused by rule (the miner's model runs in our sandbox, so the miner never sees the item) but not without
# bound: an item drawn this many times is retired, never drawn again, and becomes eligible for later publication.
# 24: sized at the 17 Oct 2026 rate of about one 300-item draw per hour and track; at the rc8 maximum of two per hour a
# ~3,500-item pool lasts about 5 days, so pools are refilled on the "refill needed" warning (8 would retire the binding same_family tier of demand_forecasting after ~54 more draws, ~2 days).
ITEM_MAX_USES = 24
# A track whose LLM-written (non-proc) items have fewer uses left than this many draws' worth logs "refill needed".
# rc8 (review-rc8 F1): counted in remaining non-proc uses, not items: use counts stay even across items, so the item count
# drops off a cliff at the end, and the proc tier (at most 15 % of a draw) kept it above the old 3-draw bar until draws
# failed. 48 draws = about one day at the rc8 maximum of two draws per hour and track.
REFILL_DRAWS = 48
PRIMITIVES = ("choice", "noul", "score")


class SampleError(RuntimeError):
    pass


def load_pool(sealed_dir: str, category: str) -> list[dict[str, Any]]:
    base = os.path.join(sealed_dir, category, "sealed")
    for name in ("sealed.clean.jsonl", "sealed.jsonl"):
        path = os.path.join(base, name)
        if os.path.exists(path):
            rows = []
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        rows.append(json.loads(line))
            return rows
    raise SampleError(f"no sealed pool for {category} under {sealed_dir}")


def _spread(rows: list[dict], n: int, rng: random.Random, use: Mapping[str, int]) -> list[dict]:
    """n rows, about 1/3 per primitive, least-used first (random inside equal use counts)."""
    by: dict[str, list[dict]] = {p: [] for p in PRIMITIVES}
    for r in rows:
        by.setdefault(r["question"]["type"], []).append(r)
    for lst in by.values():
        rng.shuffle(lst)
        lst.sort(key=lambda r: use.get(r["id"], 0))
    kinds = [k for k in by if by[k]]
    picked: list[dict] = []
    while len(picked) < n and kinds:
        for k in list(kinds):
            if len(picked) >= n:
                break
            if by[k]:
                picked.append(by[k].pop(0))
            else:
                kinds.remove(k)
    return picked


def eligible_count(pool: Sequence[dict[str, Any]], use_counts: Mapping[str, int]) -> int:
    """Items that can still be drawn (used fewer than ``ITEM_MAX_USES`` times)."""
    return sum(1 for r in pool if use_counts.get(r["id"], 0) < ITEM_MAX_USES)


def nonproc_draws_left(pool: Sequence[dict[str, Any]], use_counts: Mapping[str, int], sample_size: int = SAMPLE_SIZE) -> float:
    """Draws of ``sample_size`` the LLM-written tiers can still fill: their remaining uses (``ITEM_MAX_USES`` minus the
    uses so far, per item) over the non-proc part of one draw."""
    left = sum(max(0, ITEM_MAX_USES - use_counts.get(r["id"], 0)) for r in pool if r.get("tier") != "proc_sealed")
    per_draw = (1.0 - TIER_MIX.get("proc_sealed", 0.0)) * sample_size
    return left / per_draw if per_draw > 0 else float("inf")


def draw(pool: Sequence[dict[str, Any]], *, n: int, seed: int, use_counts: Mapping[str, int] | None = None,
         exclude: set[str] | frozenset[str] = frozenset(), min_n: int = MIN_SAMPLE) -> tuple[list[dict], bool]:
    """Pick ``n`` items. Returns (rows, fresh) where fresh is False if any already-used item had to be reused.

    Each tier takes its ``TIER_MIX`` quota from its own items, fresh ones first, then its least-used ones. Only what a
    tier cannot fill from its own eligible items goes to the other non-proc tiers (fresh, then least used); the proc
    tier never takes more than its ``TIER_MIX`` share of the draw that results, so a draw without enough non-proc items
    is smaller. Items in ``exclude`` are never picked, nor are retired items (used ``ITEM_MAX_USES`` times or more). If fewer than ``min_n`` eligible items exist, or the draw came out
    smaller than both ``n`` and ``min_n``, SampleError."""
    use = use_counts or {}
    rng = random.Random(seed)
    eligible = [r for r in pool if r["id"] not in exclude and use.get(r["id"], 0) < ITEM_MAX_USES]
    if len(eligible) < min_n:
        raise SampleError(f"only {len(eligible)} eligible sealed items, need {min_n}")
    n = min(n, len(eligible))
    chosen: list[dict] = []
    left = n
    fresh = True
    quota = {t: int(round(n * w)) for t, w in TIER_MIX.items()}
    for t in TIER_MIX:
        rows = [r for r in eligible if r.get("tier") == t]
        take = _spread([r for r in rows if use.get(r["id"], 0) == 0], min(quota[t], left), rng, use)
        short = min(quota[t], left) - len(take)
        if short > 0:   # the tier's own least-used items before any other tier's fresh ones
            have = {r["id"] for r in take}
            again = _spread([r for r in rows if r["id"] not in have], short, rng, use)
            take += again
            fresh = fresh and not again
        chosen += take
        left -= len(take)
    if left > 0:   # a tier with too few eligible items: the other non-proc tiers, fresh before reused
        have = {r["id"] for r in chosen}
        rest = [r for r in eligible if r["id"] not in have and r.get("tier") != "proc_sealed"]
        for reused in (False, True):
            part = [r for r in rest if (use.get(r["id"], 0) > 0) == reused]
            take = _spread(part, left, rng, use)
            chosen += take
            left -= len(take)
            fresh = fresh and not (reused and take)
    # The proc tier never takes another tier's share (1.2.0rc6): without enough non-proc items the draw is smaller, and
    # proc keeps at most its TIER_MIX share of the draw that results.
    w = TIER_MIX.get("proc_sealed", 0.0)
    proc = [r for r in chosen if r.get("tier") == "proc_sealed"]
    others = len(chosen) - len(proc)
    cap = len(proc) if w >= 1.0 else int(w / (1.0 - w) * others)
    if len(proc) > cap:
        drop = {r["id"] for r in proc[cap:]}
        chosen = [r for r in chosen if r["id"] not in drop]
    if len(chosen) < min(n, min_n):   # a smaller draw asked for (REAX_SPEC_SAMPLE_SIZE) is fine when it is complete
        raise SampleError(f"only {len(chosen)} items fit the tier shares, need {min_n}")
    chosen.sort(key=lambda r: r["id"])   # a stable order, the same for every model of the draw
    return chosen, fresh
