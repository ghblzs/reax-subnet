#!/usr/bin/env python3
"""Calibration analysis: distances between runs with the validator's own metric (s1scoring.logit_distance).

usage: analyze.py <results_dir> <out.json>
<results_dir>/<gpu>/<run>.json as written by calib/kit (p_* / r_* honest, a_* adversary).
"""
import itertools, json, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from reax_subnet.s1scoring import logit_distance, total_variation

root, out_path = sys.argv[1], sys.argv[2]
runs = {}
for gpu in sorted(os.listdir(root)):
    d = os.path.join(root, gpu)
    if not os.path.isdir(d):
        continue
    for f in sorted(os.listdir(d)):
        if f.endswith(".json") and f[:2] in ("p_", "r_", "a_"):
            try:
                runs[f"{gpu}/{f[:-5]}"] = json.load(open(os.path.join(d, f)))
            except Exception as e:  # noqa
                print("skip", gpu, f, e)
TAUS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]


def q(xs, f):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(f * len(xs)))] if xs else None


def compare(a, b, T, scale_search=False):
    A, B = runs[a]["items"], runs[b]["items"]
    ids = sorted(set(A) & set(B))
    ds, tvs, agree, errors = [], [], 0, 0
    best_scale = 1.0
    if scale_search:  # the cheater may rescale its logits to fit best: pick the scale minimising the median distance
        cands = [0.5 + 0.05 * i for i in range(31)]
        def med(s):
            vals = []
            for i in ids[:300]:
                if "p" not in A[i] or "p" not in B[i]:
                    continue
                zb = [s * T * math.log(max(x, 1e-30)) for x in B[i]["p"]]
                m = max(zb); e = [math.exp((z - m) / T) for z in zb]; t = sum(e)
                vals.append(logit_distance(A[i]["p"], [x / t for x in e], T))
            return q(vals, 0.5)
        best_scale = min(cands, key=med)
    for i in ids:
        if "p" not in A[i] or "p" not in B[i]:
            errors += 1
            continue
        pa, pb = A[i]["p"], B[i]["p"]
        if A[i]["keys"] != B[i]["keys"]:
            # align by key (Rune noul order is false,true; Plumb true,false)
            m = dict(zip(B[i]["keys"], pb)); pb = [m[k] for k in A[i]["keys"]]
        if best_scale != 1.0:
            zb = [best_scale * T * math.log(max(x, 1e-30)) for x in pb]
            mx = max(zb); e = [math.exp((z - mx) / T) for z in zb]; t = sum(e); pb = [x / t for x in e]
        ds.append(logit_distance(pa, pb, T))
        tvs.append(total_variation(pa, pb))
        agree += max(range(len(pa)), key=lambda k: pa[k]) == max(range(len(pb)), key=lambda k: pb[k])
    n = len(ds)
    return {"a": a, "b": b, "n": n, "errors": errors, "scale": round(best_scale, 2),
            "d_q50": q(ds, .5), "d_q90": q(ds, .9), "d_q99": q(ds, .99), "d_q999": q(ds, .999), "d_max": max(ds) if ds else None,
            "tv_max": max(tvs) if tvs else None, "argmax_agree": agree / n if n else None,
            "pass_rate": {str(t): sum(d <= t for d in ds) / n for t in TAUS} if n else {}}


res = {"plumb": [], "rune": [], "runs": sorted(runs)}
gpus = sorted({k.split("/")[0] for k in runs})
for model, T, honest_ref, prefix, advs in (("plumb", 2.07, "p_graph_a", "p_", ("a_qwen35base", "a_jevk5orig", "a_vllm_fp8")),
                                          ("rune", 2.0, "r_det_c1_a", "r_", ("a_rune_fp8",))):
    for g in gpus:
        ref = f"{g}/{honest_ref}"
        if ref not in runs:
            continue
        for k in sorted(runs):
            if k.startswith(g + "/" + prefix) and k != ref:
                res[model].append({"kind": "honest-same-gpu", **compare(ref, k, T)})
        for adv in advs:
            if f"{g}/{adv}" in runs:
                res[model].append({"kind": "adversary", **compare(ref, f"{g}/{adv}", T)})
                res[model].append({"kind": "adversary-best-scale", **compare(ref, f"{g}/{adv}", T, scale_search=True)})
    for g1, g2 in itertools.combinations(gpus, 2):
        for run in sorted({k.split("/")[1] for k in runs if k.split("/")[1].startswith(prefix)}):
            a, b = f"{g1}/{run}", f"{g2}/{run}"
            if a in runs and b in runs:
                res[model].append({"kind": "honest-cross-gpu", **compare(a, b, T)})
        # cross-GPU against the reference run of the other GPU (validator on g1, miner on g2 with any honest engine)
        for k in sorted(runs):
            if k.startswith(g2 + "/" + prefix) and f"{g1}/{honest_ref}" in runs:
                res[model].append({"kind": "honest-cross-gpu-vs-ref", **compare(f"{g1}/{honest_ref}", k, T)})
json.dump(res, open(out_path, "w"), indent=1)
for model in ("plumb", "rune"):
    print(f"== {model}")
    for r in res[model]:
        pr = r["pass_rate"]
        print(f"{r['kind']:24s} {r['a']:26s} vs {r['b']:26s} n={r['n']:4d} err={r['errors']} scale={r['scale']} "
              f"d50={r['d_q50']:.4f} d99={r['d_q99']:.4f} dmax={r['d_max']:.4f} argmax={r['argmax_agree']:.4f} "
              f"pass@0.5={pr['0.5']:.4f} pass@1={pr['1.0']:.4f} pass@1.5={pr['1.5']:.4f}" if r["n"] else f"{r['kind']} {r['a']} {r['b']} EMPTY")
