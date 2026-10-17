"""``reax-selfcheck``: check a runtime sidecar against published reference answers before mining.

Sends the published reference decisions (scripts/calibration_s1 items, reference path on the
release calibration GPU) through the sidecar exactly as the miner does, and scores the answers
with the validator's release constants: per-item logit distance, and the share of simulated
validator rounds (random draws of the pool's verified items) that would score or be struck.

Uses the miner's settings: ``REAX_MODEL``, ``REAX_S1_FAST_RUNTIME_URL`` / ``REAX_S1_PRO_RUNTIME_URL`` (or the
``..._RUNTIME_URLS`` list for several machines or GPUs behind one hotkey: every backend is checked and reported on its
own, and one that runs a different model fails the check) and ``REAX_RUNTIME_TOKEN_FILE``.
Exit code 0 = pass, 1 = fail, 2 = cannot run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import sys
import logging
import time
from importlib import resources
from typing import Any, Optional

from .manifest import load_manifest
from .s1backends import SidecarBackend
from .s1contract import parse_decision
from .s1scoring import PoolParams, logit_distance

log = logging.getLogger("reax.selfcheck")
SCHEMA = "reax-selfcheck/1"
SIMULATED_ROUNDS = 20000
# Pass line: the share of simulated rounds that score. The release calibration measured 0.999-1.0 across GPU types.
MIN_ROUND_PASS = 0.99


def load_reference(model: str) -> dict[str, Any]:
    manifest = load_manifest(model)
    name = f"{model}-{manifest.manifest_id}.json"
    try:
        raw = resources.files("reax_subnet.selfcheck_data").joinpath(name).read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ValueError(f"no reference answers for {model} manifest {manifest.manifest_id} in this release") from None
    doc = json.loads(raw)
    if doc.get("schema") != SCHEMA or doc.get("model") != model or doc.get("manifest_id") != manifest.manifest_id:
        raise ValueError(f"reference file {name} does not match this release's {model} manifest")
    return doc


def simulate_rounds(distances: list[float], params: PoolParams, rounds: int = SIMULATED_ROUNDS,
                    seed: int = 1) -> tuple[float, float]:
    """(share of rounds that score, share of rounds that strike) for random draws of verified items."""
    k = params.verified_per_round
    if len(distances) < k:
        return 0.0, 1.0
    rng = random.Random(seed)
    scored = struck = 0
    for _ in range(rounds):
        sample = rng.sample(distances, k)
        mean = math.fsum(sample) / k
        if max(sample) > params.tau_item or mean > params.tau_strike:
            struck += 1
        elif mean <= params.tau_mean:
            scored += 1
    return scored / rounds, struck / rounds


def evaluate(reference: dict[str, Any], answers: list[Any], params: PoolParams) -> dict[str, Any]:
    """``answers``: per reference item, probabilities in the item's key order or the exception it raised."""
    distances: list[float] = []
    failures: list[str] = []
    for item, answer in zip(reference["items"], answers):
        if isinstance(answer, BaseException):
            failures.append(f"{item['id']}: {type(answer).__name__}: {answer}"[:200])
            continue
        distances.append(logit_distance(answer, item["p"], params.temperature))
    report: dict[str, Any] = {"items": len(reference["items"]), "answered": len(distances), "failed": len(failures),
                              "failures": failures[:10]}
    if distances:
        s = sorted(distances)
        n = len(s)
        scored, struck = simulate_rounds(distances, params)
        report.update({
            "distance_median": round(s[n // 2], 4), "distance_p99": round(s[min(n - 1, int(0.99 * n))], 4),
            "distance_max": round(s[-1], 4), "items_over_tau_item": sum(1 for d in s if d > params.tau_item),
            "round_scores": round(scored, 5), "round_strikes": round(struck, 5),
        })
    report["pass"] = bool(distances) and not failures and report.get("round_strikes", 1.0) == 0.0 \
        and report.get("round_scores", 0.0) >= MIN_ROUND_PASS
    return report


async def run(model: str, url: str, token: str, limit: Optional[int], concurrency: int) -> dict[str, Any]:
    from .s1validator import POOL_PARAMS

    reference = load_reference(model)
    if limit:
        reference = dict(reference, items=reference["items"][:limit])
    decisions = [parse_decision({"id": it["id"], "state": it["state"], "questions": {"q": it["question"]}}, model)
                 for it in reference["items"]]
    backend = SidecarBackend(model, url, token, max_inflight=concurrency)
    try:
        if not await backend.ready():
            raise ConnectionError(f"runtime sidecar at {url} is not healthy")
        started = time.monotonic()
        answers = await backend.decide_each(decisions, retries=0)
        elapsed = time.monotonic() - started
    finally:
        await backend.close()
    keyed = [a if isinstance(a, BaseException) else [float(a["probabilities"][k]) for k in d.question.keys]
             for d, a in zip(decisions, answers)]
    report = evaluate(reference, keyed, POOL_PARAMS[model])
    report.update({"model": model, "manifest_id": reference["manifest_id"], "seconds": round(elapsed, 1),
                   "decisions_per_s": round(len(decisions) / elapsed, 1) if elapsed > 0 else None})
    return report


async def run_many(model: str, endpoints: list[tuple[str, str]], limit: Optional[int], concurrency: int) -> dict[str, Any]:
    """``run`` against every backend of one hotkey. ``endpoints``: (url, token). The overall verdict needs every
    backend to pass: a backend with other weights would be struck by the validators for the share of requests it
    gets."""
    reports = []
    for url, token in endpoints:
        try:
            rep = await run(model, url, token, limit, concurrency)
        except ConnectionError as exc:
            rep = {"model": model, "pass": False, "error": str(exc), "items": 0, "answered": 0, "failed": 0,
                   "failures": []}
        rep["url"] = url
        reports.append(rep)
    return {"model": model, "pass": all(r["pass"] for r in reports), "backends": reports}


IDENTITY_ITEMS = 24


async def check_backend_identity(backend: Any, model: str, items: int = IDENTITY_ITEMS) -> list[str]:
    """Smoke test of ONE backend against the published reference decisions: the first ``items`` of them must come back
    within the release's per-item and mean logit-distance lines. Returns the problems (empty = it answered like the
    reference model). This is not a proof of the backend's weights (a sidecar could recognise the public items); the
    validators' scoring stays the enforcement. Fails closed: a model without published reference answers cannot be
    checked, which is reported as a problem."""
    from .s1validator import POOL_PARAMS

    try:
        reference = load_reference(model)
    except ValueError as exc:
        return [f"{backend.url}: identity not checkable ({exc})"[:200]]
    params = POOL_PARAMS[model]
    ref_items = reference["items"][:items]
    decisions = [parse_decision({"id": it["id"], "state": it["state"], "questions": {"q": it["question"]}}, model)
                 for it in ref_items]
    answers = await backend.decide_each(decisions, retries=0)
    distances = []
    for d, it, a in zip(decisions, ref_items, answers):
        if isinstance(a, BaseException):
            return [f"{backend.url}: reference decision failed ({type(a).__name__}: {a})"[:200]]
        distances.append(logit_distance([float(a["probabilities"][k]) for k in d.question.keys], it["p"],
                                        params.temperature))
    mean = math.fsum(distances) / len(distances)
    if max(distances) > params.tau_item or mean > params.tau_mean:
        return [f"{backend.url}: answers differ from the reference {model} model (mean distance "
                f"{mean:.3f}, max {max(distances):.3f}); a backend with other weights or settings "
                "would be struck by validators"]
    return []


async def check_pool_identity(pool: Any, model: str, items: int = IDENTITY_ITEMS) -> list[str]:
    """Start-up smoke test of every backend of a ``PooledBackend`` against the published reference decisions (see
    ``check_backend_identity``). Returns the problems (empty = every backend answered like the reference model)."""
    problems: list[str] = []
    for backend in pool.backends:
        problems += await check_backend_identity(backend, model, items)
    return problems


def _print_report(report: dict[str, Any]) -> None:
    print(f"model {report['model']} manifest {report['manifest_id']}: {report['answered']}/{report['items']} answered "
          f"in {report['seconds']} s ({report['decisions_per_s']} decisions/s)")
    if report["answered"]:
        print(f"logit distance to the reference: median {report['distance_median']}, p99 {report['distance_p99']}, "
              f"max {report['distance_max']} ({report['items_over_tau_item']} items over the strike line per item)")
        print(f"simulated validator rounds: {report['round_scores']:.2%} score, {report['round_strikes']:.3%} strike")
    for f in report["failures"]:
        print(f"failed: {f}")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="reax-selfcheck", description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default=os.environ.get("REAX_MODEL", "s1-fast"), choices=("s1-fast", "s1-pro", "s1-lite", "s1-cpu"))
    ap.add_argument("--url", help="runtime sidecar URL (default: REAX_<MODEL>_RUNTIME_URL); several backends of one hotkey: "
                                  "comma-separated, each URL or URL|TOKEN_FILE (default: REAX_<MODEL>_RUNTIME_URLS). Every backend is checked "
                                  "against the published reference decisions: a smoke test, not a proof of its weights; validator scoring is the enforcement")
    ap.add_argument("--token-file", default=os.environ.get("REAX_RUNTIME_TOKEN_FILE", ""))
    ap.add_argument("--limit", type=int, default=0, help="use only the first N reference items")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="parallel requests (the validator path sends one decision at a time per item)")
    ap.add_argument("--json", action="store_true", help="print one JSON object")
    args = ap.parse_args(argv)
    var = f"REAX_{args.model.upper().replace('-', '_')}_RUNTIME_URL"
    try:
        from .config import ConfigError, _runtime_endpoints

        env = dict(os.environ)
        if args.url:
            env[var + "S"] = args.url
            env.pop(var, None)
        endpoints = _runtime_endpoints(env, args.model, env.get("REAX_ALLOW_PUBLIC_RUNTIME", "") in ("1", "true"))
        if not endpoints:
            if not env.get(var):
                raise ValueError(f"no runtime URL: pass --url or set {var} (or {var}S)")
            endpoints = ((env[var], ""),)
        if not args.token_file and not all(tf for _u, tf in endpoints):
            raise ValueError("no runtime token: pass --token-file or set REAX_RUNTIME_TOKEN_FILE")
        pairs = []
        for u, tf in endpoints:
            with open(os.path.expanduser(tf or args.token_file), encoding="ascii") as fh:
                token = fh.read().strip()
            if not token:
                raise ValueError("runtime token file is empty")
            pairs.append((u, token))
        concurrency = max(1, args.concurrency)
        if len(pairs) == 1:
            report = asyncio.run(run(args.model, pairs[0][0], pairs[0][1], args.limit or None, concurrency))
        else:
            report = asyncio.run(run_many(args.model, pairs, args.limit or None, concurrency))
    except (OSError, ValueError, ConnectionError) as exc:  # ConfigError is a ValueError
        if args.json:
            print(json.dumps({"pass": False, "error": str(exc)}))
        else:
            print(f"reax-selfcheck: cannot run: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, sort_keys=True))
    elif "backends" in report:
        for rep in report["backends"]:
            print(f"== backend {rep['url']}: {'PASS' if rep['pass'] else 'FAIL'}")
            if "error" in rep:
                print(f"failed: {rep['error']}")
            else:
                _print_report(rep)
        print(f"PASS: all {len(report['backends'])} backends match the reference" if report["pass"] else
              "FAIL: at least one backend does not match; do not mine with this setup (docs/MINER.md, "
              "'Several GPUs or machines'; docs/CALIBRATION.md)")
    else:
        _print_report(report)
        print("PASS: this runtime matches the reference" if report["pass"] else
              "FAIL: do not mine with this setup; see docs/MINER.md (troubleshooting) and docs/CALIBRATION.md")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
