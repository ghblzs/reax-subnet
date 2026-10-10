#!/usr/bin/env python3
"""Round-level statistics from measured item distances (bootstrap). usage: rounds.py <ref_run> <run> <T> <k> [rounds]"""
import json, random, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from reax_subnet.s1scoring import logit_distance
a, b, T, k = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
R = int(sys.argv[5]) if len(sys.argv) > 5 else 20000
A, B = json.load(open(a))["items"], json.load(open(b))["items"]
ds = []
for i in sorted(set(A) & set(B)):
    pa, pb = A[i]["p"], B[i]["p"]
    if A[i]["keys"] != B[i]["keys"]:
        m = dict(zip(B[i]["keys"], pb)); pb = [m[x] for x in A[i]["keys"]]
    ds.append(logit_distance(pa, pb, T))
rng = random.Random(1)
means = sorted(sum(rng.sample(ds, k)) / k for _ in range(R))
q = lambda f: means[min(R - 1, int(f * R))]
print(f"{os.path.basename(b):22s} k={k} item_d50={sorted(ds)[len(ds)//2]:.3f} mean-of-{k}: q50={q(.5):.3f} q01={q(.01):.3f} q99={q(.99):.3f} q999={q(.999):.3f} max={means[-1]:.3f}",
      " ".join(f"P(mean<={t})={sum(m <= t for m in means)/R:.4f}" for t in (0.10, 0.125, 0.15, 0.2)))
