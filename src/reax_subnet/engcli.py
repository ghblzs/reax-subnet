"""Engine competition tools (unreleased; docs/ENGINE-COMPETITION.md).

``reax-engine-eval``: measure one engine image against the reference on this machine, exactly as a validator does,
without any chain access. Competitors use it before submitting; validators use it to calibrate.

    reax-engine-eval --reference <image>@sha256:<d> --candidate <image>@sha256:<d> --weights /path/to/model [--gpu 0]

``reax-engine-submit``: publish ``reax-eng/1:<track>:<image>@sha256:<digest>`` as the hotkey's commitment. While the
competition runs on finney (from COMPETITIONS_START_UTC, docs/COMPETITIONS.md); on other networks it needs
REAX_FEATURE_ENGINE_COMPETITION=1. While the engine track is held on finney (competition_split.ENGINE_TRACK_START_UTC is
None) it publishes nothing.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import sys
from datetime import datetime, timezone

from .competition_split import engine_track_open
from .engcomp import commitment_text, parse_engine_commitment
from .engvalidator import TRACKS


def eval_main(argv=None) -> None:
    from .engrunner import DevRunner, DockerRunner, InfraError, ReferenceError_, evaluate

    ap = argparse.ArgumentParser(prog="reax-engine-eval", description=__doc__.split("\n\n")[1])
    ap.add_argument("--reference", required=True)
    ap.add_argument("--candidate", required=True, action="append", help="repeatable")
    ap.add_argument("--track", default="s1-fast", choices=sorted(TRACKS))
    ap.add_argument("--weights", default="", help="pinned model weights directory (docker runner)")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--docker-runtime", default="")
    ap.add_argument("--memory", default="64g", help="container memory limit")
    ap.add_argument("--cpus", default="8", help="container CPU limit")
    ap.add_argument("--registries", default="ghcr.io,docker.io,quay.io", help="allowed registries (comma-separated)")
    ap.add_argument("--no-pull", action="store_true", help="use local images (still addressed by digest)")
    ap.add_argument("--dev-map", default="", help="JSON map image -> command (tests only)")
    ap.add_argument("--workdir", default=os.path.expanduser("~/.reax/engines"))
    ap.add_argument("--params", default="", help="JSON overrides of the track parameters (calibration only)")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    params = TRACKS[args.track]
    if args.params:
        params = dataclasses.replace(params, **json.loads(args.params))
    if args.dev_map:
        runner = DevRunner.from_file(args.workdir, args.dev_map)
    else:
        if not os.path.isdir(args.weights):
            sys.exit("--weights must be the pinned model weights directory")
        runner = DockerRunner(args.workdir, args.weights, gpu=args.gpu, docker_runtime=args.docker_runtime,
                              pull=not args.no_pull, memory=args.memory, cpus=args.cpus,
                              registries=[r for r in args.registries.split(",") if r])
    results = []
    for cand in args.candidate:
        try:
            res = asyncio.run(asyncio.wait_for(evaluate(runner, args.reference, cand, args.track, params),
                                               params.eval_budget_s))
            row = {"candidate": cand, **res.to_json()}
        except (ReferenceError_, InfraError) as exc:
            row = {"candidate": cand, "inconclusive": str(exc)}
        except asyncio.TimeoutError:
            row = {"candidate": cand, "ok": False, "reason": f"evaluation exceeded {params.eval_budget_s:.0f} s"}
        print(json.dumps(row), flush=True)
        results.append(row)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"track": args.track, "params": dataclasses.asdict(params), "reference": args.reference,
                       "results": results}, fh, indent=1)


def submit_main(argv=None) -> None:
    from .cli import _wallet
    from .config import ConfigError, MAINNET_NAMES, load_common

    ap = argparse.ArgumentParser(prog="reax-engine-submit")
    ap.add_argument("image", help="<image>@sha256:<digest>")
    ap.add_argument("--track", default="s1-fast", choices=sorted(TRACKS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    network = os.environ.get("REAX_NETWORK", "finney") or "finney"
    if network not in MAINNET_NAMES and os.environ.get("REAX_FEATURE_ENGINE_COMPETITION") != "1":
        sys.exit("on a development network or the testnet the engine competition needs REAX_FEATURE_ENGINE_COMPETITION=1")
    if network in MAINNET_NAMES and not engine_track_open(datetime.now(timezone.utc)):
        sys.exit("the engine track is not open on finney (held by the subnet owner); no engine submission is accepted")
    image, sep, digest = args.image.partition("@sha256:")
    if not sep:
        sys.exit("pin the image by digest: <image>@sha256:<64 hex>")
    try:
        text = commitment_text(args.track, image, digest)
    except ValueError as exc:
        sys.exit(str(exc))
    assert parse_engine_commitment(text) is not None
    if args.dry_run:
        print(text)
        return
    try:
        common = load_common()
        wallet, hotkey = _wallet(common)
    except ConfigError as exc:
        sys.exit(f"configuration error: {exc}")
    from .chain import SdkChain

    chain = SdkChain(common.network, common.netuid, wallet)
    chain.set_commitment(text)
    print(f"published {text} for {hotkey}")
