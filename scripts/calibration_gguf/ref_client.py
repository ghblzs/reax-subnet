#!/usr/bin/env python3
"""Ask a running runtime sidecar (any model) for the same seeded items as run.py; write run.py's JSON lines.

usage: ref_client.py <model for parsing, e.g. s1-fast> <sidecar url> <token file> <out.jsonl> [--n 200] [--seed 7]
Used to put the bf16 s1-fast reference next to the s1-lite GGUF runs (same items, same ids).
"""
import argparse, json, os, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from reax_subnet.s1contract import parse_decision  # noqa: E402
from reax_subnet.s1items import generate_s1_items  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("url"); ap.add_argument("token"); ap.add_argument("out")
ap.add_argument("--n", type=int, default=200); ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()
tok = open(a.token).read().strip()
with open(a.out, "w") as out:
    out.write(json.dumps({"meta": {"sidecar": a.url, "model": a.model, "n": a.n, "seed": a.seed}}) + "\n")
    for it in generate_s1_items(a.n, seed=a.seed):
        d = parse_decision(it.decision(), a.model)
        body = json.dumps({"request_id": it.item_id[:24], "deadline_unix_ms": int(time.time() * 1000) + 60000,
                           "state": d.state, "questions": {"q": d.question.author_dict()}}).encode()
        req = urllib.request.Request(a.url.rstrip("/") + "/internal/v1/decide", data=body, method="POST",
                                     headers={"Content-Type": "application/json", "X-S1M-Internal-Token": tok})
        t0 = time.monotonic()
        res = json.load(urllib.request.urlopen(req, timeout=120))
        dt = time.monotonic() - t0
        probs = res["answers"]["q"]["probabilities"]
        keys = list(d.question.keys)
        out.write(json.dumps({"id": it.item_id, "kind": d.question.kind, "keys": keys,
                              "probs": [float(probs[k]) for k in keys], "tokens": res.get("usage", {}).get("input_tokens", 0),
                              "seconds": round(dt, 4)}) + "\n")
