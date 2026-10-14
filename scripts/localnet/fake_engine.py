#!/usr/bin/env python3
"""Synthetic competition engines for local tests of the engine competition (docs/ENGINE-COMPETITION.md).

Each variant serves the engine contract (``POST /internal/v1/decide``, ``GET /health``) on the unix socket named by
RUNTIME_UDS, answers with the deterministic hash test model (the same answers as the validator's ``hash`` runtime)
and models GPU time as ``--delay`` seconds per decision on each of its lanes:

* reference: one lane (the reference engine: one decision at a time);
* fast:      two lanes, same answers (a legitimate 2x throughput improvement);
* tiny:      one lane, 1 % less time per decision (below the minimum gain);
* cache:     one lane at 0.4x the time, and an answer cache keyed by the state only (wrong when the question changes);
* broken:    one lane at 0.25x the time, logits off by up to +-0.6 (a lossy shortcut, like aggressive quantisation);
* jitter:    two lanes, logits off by a fresh random +-0.08 (``--jitter``) on every call (non-deterministic);
* lookup:    two lanes, and tries to reach the network for answers first (must not matter: no network in the sandbox).

Local tests only. Usage: RUNTIME_UDS=/path/engine.sock S1M_INTERNAL_TOKEN_FILE=... fake_engine.py --variant fast
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import os
import random

from aiohttp import web

from reax_subnet.s1backends import S1HashBackend
from reax_subnet.s1contract import parse_decision

LANES = {"reference": 1, "fast": 2, "tiny": 1, "cache": 1, "broken": 1, "jitter": 2, "lookup": 2}
SPEED = {"tiny": 0.99, "cache": 0.4, "broken": 0.25}


def make_app(variant: str, token: str, delay: float, model: str = "s1-fast", jitter: float = 0.08) -> web.Application:
    backend = S1HashBackend(model)
    lanes = asyncio.Semaphore(LANES[variant])
    cost = delay * SPEED.get(variant, 1.0)
    cache: dict[str, dict] = {}
    rng = random.Random()

    def logits_off(base: str, keys, amp: float, fresh: bool) -> list[float]:
        out = []
        for k in keys:
            if fresh:
                out.append(rng.uniform(-amp, amp))
            else:
                h = hashlib.sha256(f"broken\x00{base}\x00{k}".encode()).digest()
                out.append((int.from_bytes(h[:4], "big") / 2**32 - 0.5) * 2 * amp)
        return out

    async def decide(request: web.Request) -> web.Response:
        if not hmac.compare_digest(request.headers.get("X-S1M-Internal-Token", ""), token):
            return web.json_response({"error": "unauthorized"}, status=401)
        body = await request.json()
        (qid, question), = body["questions"].items()
        key = json.dumps(body["state"], sort_keys=True)
        if variant == "cache" and key in cache:
            return web.json_response({"answers": {qid: cache[key]}})
        if variant == "lookup":
            try:  # an answer oracle on the internet; the sandbox has no network, so this must fail fast
                await asyncio.wait_for(asyncio.open_connection("1.1.1.1", 443), 0.2)
            except Exception:  # noqa: BLE001
                pass
        d = parse_decision({"id": "d", "state": body["state"], "questions": {qid: question}}, model)
        async with lanes:
            await asyncio.sleep(cost)
            z = backend.logits(d)
        if variant in ("broken", "jitter"):
            base = json.dumps([d.state, d.question.author_dict()], sort_keys=True)
            off = logits_off(base, d.question.keys, 0.6 if variant == "broken" else jitter, variant == "jitter")
            z = [a + b for a, b in zip(z, off)]
        import math

        T = backend.manifest.temperature
        m = max(z)
        e = [math.exp((x - m) / T) for x in z]
        s = math.fsum(e)
        probs = {k: v / s for k, v in zip(d.question.keys, e)}
        raw = {"type": d.question.kind, "probabilities": probs}
        if d.question.kind == "noul":
            raw["noul"] = probs["true"]
        if variant == "cache":
            cache[key] = raw
        return web.json_response({"answers": {qid: raw}})

    async def health(request: web.Request) -> web.Response:
        return web.json_response({"ok": True, "variant": variant})

    app = web.Application(client_max_size=1 << 20)
    app.router.add_post("/internal/v1/decide", decide)
    app.router.add_get("/health", health)
    return app


async def serve(variant: str, delay: float, jitter: float) -> None:
    token = open(os.environ["S1M_INTERNAL_TOKEN_FILE"], encoding="ascii").read().strip()
    path = os.environ["RUNTIME_UDS"]
    runner = web.AppRunner(make_app(variant, token, delay, jitter=jitter), access_log=None)
    await runner.setup()
    await web.UnixSite(runner, path).start()
    os.chmod(path, 0o666)
    await asyncio.Event().wait()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=sorted(LANES))
    ap.add_argument("--delay", type=float, default=0.02)
    ap.add_argument("--jitter", type=float, default=0.08, help="jitter variant: logit noise amplitude")
    args = ap.parse_args()
    asyncio.run(serve(args.variant, args.delay, args.jitter))


if __name__ == "__main__":
    main()
