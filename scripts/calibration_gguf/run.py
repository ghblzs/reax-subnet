#!/usr/bin/env python3
"""Run seeded subnet items through a pinned GGUF pool model with the runtime's own readout; write JSON lines.

usage: run.py <s1-lite|s1-cpu> <gguf file> <out.jsonl> [--n 200] [--seed 7] [--ngl 0] [--threads 0] [--gguf-name NAME]
Each line: {"id", "kind", "keys", "probs", "tokens", "seconds"}. Compare runs with compare.py.
The --gguf-name option lets a different quantisation of the same model run under the pool's readout.
"""
import argparse, json, os, platform, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "runtimes", "s1-gguf"))
import readouts  # noqa: E402
from reax_subnet.manifest import load_manifest  # noqa: E402
from reax_subnet.s1contract import parse_decision  # noqa: E402
from reax_subnet.s1items import generate_s1_items  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("gguf"); ap.add_argument("out")
ap.add_argument("--n", type=int, default=200); ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--ngl", type=int, default=0); ap.add_argument("--threads", type=int, default=0)
a = ap.parse_args()

import numpy as np  # noqa: E402
import llama_cpp  # noqa: E402
from llama_cpp import Llama  # noqa: E402

man = load_manifest(a.model).data
ro = man["readout"]
threads = a.threads or max(1, (os.cpu_count() or 2) // 2)
llm = Llama(model_path=a.gguf, n_ctx=man["limits"]["max_input_tokens"] + 64, n_batch=512, n_threads=threads,
            n_threads_batch=threads, n_gpu_layers=a.ngl, seed=0, logits_all=False, verbose=False)
ctx = getattr(llm, "ctx", None) or llm._ctx.ctx
nv = llm.n_vocab()
lid = {x: llm.tokenize(x.encode(), add_bos=False, special=False)[0] for x in readouts.LETTERS + "Z"}
meta = {"meta": {"model": a.model, "gguf": os.path.basename(a.gguf), "machine": platform.machine(),
                 "processor": platform.processor(), "ngl": a.ngl, "threads": threads, "n": a.n, "seed": a.seed,
                 "llama_cpp_python": llama_cpp.__version__}}
with open(a.out, "w") as out:
    out.write(json.dumps(meta) + "\n")
    t_all = time.monotonic()
    for it in generate_s1_items(a.n, seed=a.seed):
        dec = parse_decision(it.decision(), a.model)  # what the sidecar receives: Question.author_dict()
        q = dict(dec.question.author_dict())
        q.setdefault("instructions", "")
        prompt, keys, letters = readouts.build(ro["kind"], dec.state, q)
        ids = list(llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True))
        t0 = time.monotonic()
        llm.reset(); llm.eval(ids)
        row = np.ctypeslib.as_array(llama_cpp.llama_get_logits_ith(ctx, -1), shape=(nv,))
        logits = [float(row[lid[x]]) for x in letters]
        dt = time.monotonic() - t0
        probs = readouts.probabilities(logits, keys, readouts.temperature(ro, q["type"]))
        out.write(json.dumps({"id": it.item_id, "kind": q["type"],
                              "keys": keys, "probs": [probs[k] for k in keys], "logits": logits,
                              "tokens": len(ids), "seconds": round(dt, 4)}) + "\n")
    print(f"{a.n} items in {time.monotonic() - t_all:.1f}s", file=sys.stderr)
