#!/usr/bin/env python3
"""Distances between two run.py outputs with the validator's metric (s1scoring.logit_distance).

usage: compare.py <a.jsonl> <b.jsonl> [temperature]   (temperature: the pool's PoolParams.temperature)
"""
import json, os, statistics, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from reax_subnet.s1scoring import logit_distance  # noqa: E402


def load(p):
    lines = [json.loads(x) for x in open(p)]
    return lines[0]["meta"], {r["id"]: r for r in lines[1:]}


ma, a = load(sys.argv[1]); mb, b = load(sys.argv[2])
T = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
ids = sorted(set(a) & set(b))
d = sorted(logit_distance(a[i]["probs"], b[i]["probs"], T) for i in ids)
flips = sum(1 for i in ids if max(range(len(a[i]["probs"])), key=a[i]["probs"].__getitem__)
            != max(range(len(b[i]["probs"])), key=b[i]["probs"].__getitem__))
q = lambda f: d[min(len(d) - 1, int(f * len(d)))]
lat = lambda r: statistics.median(x["seconds"] for x in r.values())
print(json.dumps({"a": ma, "b": mb, "items": len(ids), "mean": sum(d) / len(d), "p50": q(0.5), "p95": q(0.95),
                  "p99": q(0.99), "max": d[-1], "argmax_flips": flips,
                  "median_s_a": lat(a), "median_s_b": lat(b),
                  "median_tokens": statistics.median(x["tokens"] for x in a.values())}, indent=1))
