#!/usr/bin/env python3
"""Throughput-limited fake s1-fast runtime sidecar for the capacity-v2 localnet e2e. LOCAL TEST ONLY.

Speaks the sidecar contract (GET /health, POST /internal/v1/decide with X-S1M-Internal-Token) and answers with the
deterministic hash model (the same answers as the validators' ``hash`` reference), but at most ``--capacity``
decisions per second in total: every decision occupies the one simulated machine for 1/capacity seconds, whoever
asked. Several miner hotkeys pointing at one instance share that capacity, like several hotkeys on one GPU box.

Usage: fake_capacity_backend.py --port P --capacity C --token-file F [--name N]
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import time

from aiohttp import web

from reax_subnet.s1backends import S1HashBackend
from reax_subnet.s1contract import Decision, parse_question


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--capacity", type=float, required=True, help="decisions per second, all callers together")
    ap.add_argument("--token-file", required=True)
    ap.add_argument("--name", default="machine")
    args = ap.parse_args()
    token = open(args.token_file).read().strip()
    backend = S1HashBackend("s1-fast")
    service = 1.0 / args.capacity
    state = {"next_free": 0.0, "served": 0, "t0": time.monotonic()}

    async def health(_request):
        return web.json_response({"ok": True, "model": "s1-fast", "machine": args.name, "capacity": args.capacity})

    async def stats(_request):
        return web.json_response({"machine": args.name, "capacity": args.capacity, "served": state["served"],
                                  "seconds": round(time.monotonic() - state["t0"], 1)})

    async def decide(request):
        if not hmac.compare_digest(request.headers.get("x-s1m-internal-token", ""), token):
            return web.json_response({"error": {"message": "unauthorized"}}, status=401)
        try:
            body = await request.json()
            (qid, q), = body["questions"].items()
            decision = Decision(qid, body["state"], parse_question(qid, q, "s1-fast"))
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"error": {"message": f"bad request: {type(exc).__name__}"}}, status=400)
        loop = asyncio.get_running_loop()
        now = loop.time()
        done_at = max(now, state["next_free"]) + service  # one simulated machine, first come first served
        state["next_free"] = done_at
        await asyncio.sleep(done_at - now)
        (answer,) = await backend.decide([decision])
        state["served"] += 1
        return web.json_response({"model": "s1-fast", "answers": {qid: answer},
                                  "usage": {"input_tokens": 1, "output_tokens": 0}})

    app = web.Application(client_max_size=1 << 20)
    app.add_routes([web.get("/health", health), web.get("/stats", stats), web.post("/internal/v1/decide", decide)])
    web.run_app(app, host="127.0.0.1", port=args.port, print=None, access_log=None)


if __name__ == "__main__":
    main()
