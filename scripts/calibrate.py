#!/usr/bin/env python3
"""Measure how far honest setups drift from the validator reference (launch blocker V3).

Run on each GPU type miners and validators will use, with the pinned reference model:

    python scripts/calibrate.py --model qwen3-4b-instruct-2507 --device cuda --dtype bfloat16 \
        --items 256 --out calib-<gpu>.json

The script computes reference distributions for a fixed, seeded item set and saves them. Give it a
reference file from another machine (--compare other.json) to get the cross-hardware distances. It also
reports how often simple cheats (uniform, first option, one-hot of the reference's top option) pass.

Pass criterion for launch: the largest honest distance across every pair of hardware setups is at most
0.10 (tau is 0.15); every cheat that does not need the model (uniform, cheaper models) has a round-pass
probability below 1 %; the oracle-top cheat (which needs the model's own top choice) stays below 10 %.
The strongest built-in cheat knows the reference's top choice exactly and uses the best top mass per option
count ("oracle top", optimistic for the cheater); `--cheat-model <hf id>@<revision>` adds a real cheaper model,
evaluated at the temperature that suits the cheater best. The weight threshold (moving average 0.3) needs
roughly 70 % of items per round; any cheat below that earns nothing even if it passes some rounds.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from math import comb
import statistics
import sys
import time

from reax_subnet.backends import HFBackend, ModelSpec, resolve_model
from reax_subnet.config import load_validator
from reax_subnet.items import generate_items
from reax_subnet.scoring import total_variation


PARAMS = load_validator({"REAX_NETUID": "1"}).scoring
N_ITEMS = load_validator({"REAX_NETUID": "1"}).items_per_round
NEED = -(-int(PARAMS.min_round_fidelity * N_ITEMS * 1000) // 1000)  # ceil


def p_round(p_item: float) -> float:
    return sum(comb(N_ITEMS, k) * p_item**k * (1 - p_item) ** (N_ITEMS - k) for k in range(NEED, N_ITEMS + 1))


def summary(name, answers, refs, taus=(0.05, 0.10, 0.15, 0.20)):
    tv = sorted(total_variation(a, r) for a, r in zip(answers, refs))
    out = {"name": name, "mean": round(statistics.mean(tv), 4), "p99": round(tv[int(0.99 * (len(tv) - 1))], 4),
           "max": round(tv[-1], 4)}
    for tau in taus:
        out[f"pass@{tau}"] = round(sum(x <= tau for x in tv) / len(tv), 3)
    rate = sum(x <= PARAMS.tau for x in tv) / len(tv)
    out[f"P(round passes, {NEED}/{N_ITEMS} at tau {PARAMS.tau})"] = round(p_round(rate), 5)
    return out


def oracle_top(refs):
    """Knows the reference's top choice exactly; top mass tuned per option count (in-sample, so optimistic)."""
    by_n: dict[int, list[float]] = {}
    for r in refs:
        by_n.setdefault(len(r), []).append(max(r))
    best: dict[int, float] = {}
    for n, tops in by_n.items():
        group = [r for r in refs if len(r) == n]
        candidates = [i / 100 for i in range(int(100 / n) + 1, 100)]
        best[n] = max(candidates, key=lambda t: sum(total_variation(_shape(r, t), r) <= PARAMS.tau for r in group))
    return [_shape(r, best[len(r)]) for r in refs]


def _shape(r, top_mass):
    top = max(range(len(r)), key=r.__getitem__)
    rest = (1 - top_mass) / (len(r) - 1)
    return tuple(top_mass if i == top else rest for i in range(len(r)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-4b-instruct-2507")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--items", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--compare", help="JSON written by this script on another machine")
    ap.add_argument("--allow-dev", action="store_true")
    ap.add_argument("--temperature", type=float, help="override the catalog T (to choose T for a new model)")
    ap.add_argument("--cheat-model", action="append", default=[], help="<hf id>@<revision> of a cheaper model")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    items = generate_items(args.items, seed=args.seed)
    spec = resolve_model(args.model, args.allow_dev)
    started = time.monotonic()
    if args.temperature:
        spec = dataclasses.replace(spec, temperature=args.temperature)
    backend = HFBackend(spec, device=args.device, dtype=args.dtype)
    refs = []
    for i in range(0, len(items), N_ITEMS):  # same requests as the validator sends
        refs.extend(backend.predict(items[i : i + N_ITEMS]))
    seconds = time.monotonic() - started
    single = [backend.predict([it])[0] for it in items[:64]]
    result = {
        "model": args.model, "temperature": spec.temperature, "device": args.device, "dtype": args.dtype, "items": args.items, "seed": args.seed,
        "seconds": round(seconds, 1),
        "honest": [summary("batch of 1 vs batch of 8 (same machine)", single, refs[:64])],
        "cheats": [
            summary("uniform", [tuple([1 / len(r)] * len(r)) for r in refs], refs),
            summary("first option", [tuple([1.0] + [0.0] * (len(r) - 1)) for r in refs], refs),
            summary("one-hot of reference top", [tuple(float(i == max(range(len(r)), key=r.__getitem__))
                                                        for i in range(len(r))) for r in refs], refs),
            summary("oracle top choice + shape tuned per option count", oracle_top(refs), refs),
        ],
        "references": [list(r) for r in refs],
    }
    for cheat in args.cheat_model:
        hf_id, _, rev = cheat.partition("@")
        best = None
        # A cheater chooses its own temperature; report the best one.
        for t_cheat in (0.5, 1.0, 2.0, 4.0, 8.0):
            other = HFBackend(ModelSpec(hf_id, hf_id, rev or "main", False, temperature=t_cheat),
                              device=args.device, dtype=args.dtype)
            answers = []
            for i in range(0, len(items), N_ITEMS):
                answers.extend(other.predict(items[i : i + N_ITEMS]))
            row = summary(f"model {hf_id} at T={t_cheat}", answers, refs)
            if best is None or row[f"pass@{PARAMS.tau}"] > best[f"pass@{PARAMS.tau}"]:
                best = row
            del other
        result["cheats"].append(best)
    if args.compare:
        other = json.load(open(args.compare))
        if (other["seed"], other["items"], other["model"], other.get("temperature")) != (
            args.seed, args.items, args.model, spec.temperature
        ):
            sys.exit("compare file uses a different seed, item count, model or temperature")
        result["honest"].append(summary(f"vs {other['device']}/{other['dtype']} reference", other["references"], refs))
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "references"}, indent=1))


if __name__ == "__main__":
    main()
