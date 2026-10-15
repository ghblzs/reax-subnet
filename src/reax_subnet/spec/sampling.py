"""Fresh draws from the sealed pool (host directory ``REAX_SPEC_SEALED_DIR``, mounted read-only by the operator).

Layout: ``<dir>/<category>/sealed/sealed.clean.jsonl`` (``sealed.jsonl`` as fallback). Items that were drawn before are
used last (a fresh item is always preferred), and a confirmation draw is disjoint from the draws it confirms.
Sealed rows never leave this process: only item ids are persisted.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Mapping, Sequence

SAMPLE_SIZE = 600
MIN_SAMPLE = 200
# Share of the draw per tier (docs/SPECIALIST-TRACKS.md/5: >= 50 % programmatic gold; the rest 50/25/25 same-family,
# new-domain, new-family). A tier with too few unused items hands its quota to the others.
TIER_MIX = {"proc_sealed": 0.5, "same_family": 0.25, "new_domain": 0.125, "new_family": 0.125}
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


def draw(pool: Sequence[dict[str, Any]], *, n: int, seed: int, use_counts: Mapping[str, int] | None = None,
         exclude: set[str] | frozenset[str] = frozenset(), min_n: int = MIN_SAMPLE) -> tuple[list[dict], bool]:
    """Pick ``n`` items. Returns (rows, fresh) where fresh is False if any already-used item had to be reused.

    Items in ``exclude`` are never picked. If fewer than ``min_n`` eligible items exist, SampleError."""
    use = use_counts or {}
    rng = random.Random(seed)
    eligible = [r for r in pool if r["id"] not in exclude]
    if len(eligible) < min_n:
        raise SampleError(f"only {len(eligible)} eligible sealed items, need {min_n}")
    n = min(n, len(eligible))
    fresh_rows = [r for r in eligible if use.get(r["id"], 0) == 0]
    # tier quotas over the fresh items first
    chosen: list[dict] = []
    left = n
    tiers = {t: [r for r in fresh_rows if r.get("tier") == t] for t in TIER_MIX}
    quota = {t: int(round(n * w)) for t, w in TIER_MIX.items()}
    for t, rows in tiers.items():
        take = _spread(rows, min(quota[t], left), rng, use)
        chosen += take
        left -= len(take)
    if left > 0:   # shortfall: any other fresh item, then reused items (least used first)
        have = {r["id"] for r in chosen}
        rest = [r for r in fresh_rows if r["id"] not in have]
        take = _spread(rest, left, rng, use)
        chosen += take
        left -= len(take)
    fresh = True
    if left > 0:
        have = {r["id"] for r in chosen}
        rest = [r for r in eligible if r["id"] not in have]
        take = _spread(rest, left, rng, use)
        chosen += take
        fresh = not take
    chosen.sort(key=lambda r: r["id"])   # a stable order, the same for every model of the draw
    return chosen, fresh
