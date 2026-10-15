#!/usr/bin/env python3
"""Write the ``reax-selfcheck`` reference answers for a pool from its REFERENCE runtime.

Run it once per release, against the reference runtime on the release calibration GPU (for s1-pro: the
deterministic runtimes/s1-pro sidecar on the RTX PRO 6000 96 GB), then commit the file it writes:

    python3 scripts/make_selfcheck_reference.py --model s1-pro --url http://127.0.0.1:8102 \\
        --token-file deploy/secrets/runtime-token --reference "<runtime / GPU description>"

Items are the 640 generated calibration items (seed 20261003, scripts/calibration_s1/README.md). Output:
src/reax_subnet/selfcheck_data/<model>-<manifest_id>.json (schema ``reax-selfcheck/1``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from reax_subnet.manifest import load_manifest
from reax_subnet.s1backends import SidecarBackend
from reax_subnet.s1contract import parse_decision
from reax_subnet.s1items import generate_s1_items


async def main_async(args) -> int:
    manifest = load_manifest(args.model)
    items = generate_s1_items(args.items, seed=args.seed)
    decisions = [parse_decision(it.decision(), args.model) for it in items]
    with open(os.path.expanduser(args.token_file), encoding="ascii") as fh:
        token = fh.read().strip()
    backend = SidecarBackend(args.model, args.url, token, max_inflight=args.concurrency)
    try:
        if not await backend.ready():
            print(f"runtime at {args.url} is not healthy", file=sys.stderr)
            return 2
        answers = await backend.decide_each(decisions, retries=1)
    finally:
        await backend.close()
    out = []
    for it, d, a in zip(items, decisions, answers):
        if isinstance(a, BaseException):
            print(f"item {it.item_id} failed: {a}", file=sys.stderr)
            return 1
        out.append({"id": it.item_id, "state": it.state, "question": it.question(),
                    "p": [round(float(a["probabilities"][k]), 10) for k in d.question.keys]})
    doc = {"schema": "reax-selfcheck/1", "model": args.model, "manifest_id": manifest.manifest_id,
           "weights_revision": manifest.data["weights"].get("revision", ""), "reference": args.reference, "items": out}
    path = os.path.join(os.path.dirname(__file__), "..", "src", "reax_subnet", "selfcheck_data",
                        f"{args.model}-{manifest.manifest_id}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, separators=(",", ":"))
    print(f"wrote {os.path.normpath(path)} ({len(out)} items)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, choices=("s1-fast", "s1-pro"))
    ap.add_argument("--url", required=True)
    ap.add_argument("--token-file", required=True)
    ap.add_argument("--reference", required=True, help="free text: runtime build and GPU the answers come from")
    ap.add_argument("--items", type=int, default=640)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--concurrency", type=int, default=1)
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
