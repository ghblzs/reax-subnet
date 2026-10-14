"""Engine competition tools (unreleased; docs/ENGINE-COMPETITION.md).

``reax-engine-eval``: measure one engine image against the reference on this machine, exactly as a validator does,
without any chain access. Competitors use it before submitting; validators use it to calibrate.

    reax-engine-eval --reference <image>@sha256:<d> --candidate <image>@sha256:<d> --weights /path/to/model [--gpu 0]

``reax-engine-submit``: publish ``reax-eng/1:<track>:<image>@sha256:<digest>`` as the hotkey's commitment. While the
competition is unreleased this is refused unless REAX_FEATURE_ENGINE_COMPETITION=1, REAX_ALLOW_UNRELEASED_FEATURES=1
and REAX_NETWORK is a local chain or the testnet.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import sys

from .engcomp import commitment_text, parse_engine_commitment
from .engvalidator import TRACKS


def eval_main(argv=None) -> None:
    from .engrunner import DevRunner, DockerRunner, ReferenceError_, evaluate

    ap = argparse.ArgumentParser(prog="reax-engine-eval", description=__doc__.split("\n\n")[1])
    ap.add_argument("--reference", required=True)
    ap.add_argument("--candidate", required=True, action="append", help="repeatable")
    ap.add_argument("--track", default="s1-fast", choices=sorted(TRACKS))
    ap.add_argument("--weights", default="", help="pinned model weights directory (docker runner)")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--docker-runtime", default="")
    ap.add_argument("--memory", default="64g", help="container memory limit")
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
                              pull=not args.no_pull, memory=args.memory)
    results = []
    for cand in args.candidate:
        try:
            res = asyncio.run(evaluate(runner, args.reference, cand, args.track, params))
            row = {"candidate": cand, **res.to_json()}
        except ReferenceError_ as exc:
            row = {"candidate": cand, "inconclusive": str(exc)}
        print(json.dumps(row), flush=True)
        results.append(row)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"track": args.track, "params": dataclasses.asdict(params), "reference": args.reference,
                       "results": results}, fh, indent=1)


def submit_main(argv=None) -> None:
    from .cli import _wallet
    from .config import ConfigError, load_common, unreleased_feature_allowed

    ap = argparse.ArgumentParser(prog="reax-engine-submit")
    ap.add_argument("image", help="<image>@sha256:<digest>")
    ap.add_argument("--track", default="s1-fast", choices=sorted(TRACKS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if not unreleased_feature_allowed(os.environ, "REAX_FEATURE_ENGINE_COMPETITION") or \
            os.environ.get("REAX_FEATURE_ENGINE_COMPETITION") != "1":
        sys.exit("the engine competition is not released: submissions are only accepted on a local chain or the "
                 "testnet with REAX_FEATURE_ENGINE_COMPETITION=1 and REAX_ALLOW_UNRELEASED_FEATURES=1")
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

    SdkChain(common.network, common.netuid, wallet).set_commitment(text)
    print(f"published {text} for {hotkey}")
