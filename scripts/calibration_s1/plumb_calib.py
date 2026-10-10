"""Plumb (s1-fast) calibration runner. Runs inside the reference fast runtime (pytorch 2.13 + jevk5 bf54965).

usage: plumb_calib.py <model_dir> <items.jsonl> <out.json> <mode> [temperature]
modes: graph   = reference path (JevK5 graphs=True, letter_logits batch-1 CUDA-graph replay), identical to
                 the reference PlumbBatchBackend, max_batch=1
       eager8  = engine micro-batch path (_slot_logits, padded to GRAPH_LENGTHS, 8 per batch)
       eager1  = exact-length eager forward, one at a time (no graphs)
Output: {id: {keys, z (raw letter logits, float32), p (reference probabilities), tokens}} + timing.
"""
import json, sys, time
import numpy as np
import torch
from jevk5.runtime import JevK5, decision_options, GRAPH_LENGTHS

model_dir, items_path, out_path, mode = sys.argv[1:5]
T_override = float(sys.argv[5]) if len(sys.argv) > 5 else None
items = [json.loads(l) for l in open(items_path)]
t0 = time.time()
m = JevK5(model_dir, graphs=(mode == "graph"), temperature=T_override)
load_s = time.time() - t0
T = m.temperature


def prep(it):
    q = it["question"]
    opts = decision_options(q)
    ids = m.encode(it["state"], q.get("instructions", ""), [t for _, t in opts])
    return ids, [k for k, _ in opts]


def prod_probs(z):
    row = np.asarray(z, dtype=np.float32) / T
    e = np.exp(row - row.max())
    return [float(x) for x in e / e.sum()]


out = {}
t1 = time.time()
if mode in ("graph", "eager1"):
    for it in items:
        ids, keys = prep(it)
        z = m.letter_logits(ids, len(keys))
        out[it["id"]] = {"keys": keys, "z": [float(x) for x in z], "p": prod_probs(z), "tokens": len(ids)}
elif mode == "eager8":
    pre = [(it, *prep(it)) for it in items]
    for s in range(0, len(pre), 8):
        chunk = pre[s:s + 8]
        mx = max(len(ids) for _, ids, _ in chunk)
        fits = [n for n in GRAPH_LENGTHS if n >= mx]
        L = fits[0] if fits else mx
        ids_t = torch.zeros((len(chunk), L), dtype=torch.long, device=m.device)
        for i, (_, ids, _) in enumerate(chunk):
            ids_t[i, :len(ids)] = torch.tensor(ids, device=m.device)
        last = torch.tensor([len(ids) - 1 for _, ids, _ in chunk], device=m.device)
        with torch.inference_mode():
            logits = m._slot_logits(ids_t, last).float().cpu().numpy()
        for i, (it, ids, keys) in enumerate(chunk):
            z = logits[i, :len(keys)]
            out[it["id"]] = {"keys": keys, "z": [float(x) for x in z], "p": prod_probs(z), "tokens": len(ids)}
else:
    sys.exit("unknown mode")
run_s = time.time() - t1
meta = {"mode": mode, "model_dir": model_dir, "temperature": T, "n": len(items), "load_s": round(load_s, 1),
        "run_s": round(run_s, 2), "dec_per_s": round(len(items) / run_s, 1),
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
print(json.dumps(meta), flush=True)
json.dump({"meta": meta, "items": out}, open(out_path, "w"))
