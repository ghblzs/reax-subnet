#!/usr/bin/env python3
"""Engine competition proof on a local subtensor (docs/ENGINE-COMPETITION.md).

Roles (all real CLIs, hotkey-only wallets):

* v1, v2: ``reax-validator`` with REAX_FEATURE_ENGINE_COMPETITION=1 (local chains only) and the dev engine runner;
* m_serve: an honest ``reax-miner`` serving the hash test model (the serving pool keeps working);
* competitors that publish ``reax-eng/1`` commitments with ``reax-engine-submit`` (fake_engine.py variants):
  c_fast (legitimate 2x), c_ref (the reference engine itself), c_tiny (+1 %), c_cache (state-keyed answer cache),
  c_broken (lossy shortcut), c_jitter (non-deterministic), c_copy (c_fast's digest, committed later).

Pass: on chain, c_fast holds about ``share`` of the incentive (champion; runner-ups get nothing because no other
submission is valid AND beats the reference by the minimum gain), m_serve about 1 - share, and every other
competitor none.

Prerequisite: start.sh is running. Usage:
    REAX_NETWORK=ws://127.0.0.1:9981 e2e_engine.py <work_dir> [--minutes 15] [--share 0.25] [--delay 0.02]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import bittensor as bt

HERE = os.path.dirname(os.path.abspath(__file__))
VALIDATORS = ["v1", "v2"]
COMPETITORS = {"c_fast": "fast", "c_ref": "reference", "c_tiny": "tiny", "c_cache": "cache", "c_broken": "broken",
               "c_jitter": "jitter"}
COPIER = "c_copy"
WITHIN = 0.25


def image_ref(variant: str, registry: str = "localhost") -> str:
    return f"{registry}/reax-eng-{variant}@sha256:" + hashlib.sha256(variant.encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--minutes", type=float, default=15)
    ap.add_argument("--share", type=float, default=0.25)
    ap.add_argument("--delay", type=float, default=0.02)
    ap.add_argument("--host", required=True,
                    help="a private interface address of this machine (the chain refuses loopback for an axon)")
    ap.add_argument("--skip-bootstrap", action="store_true")
    args = ap.parse_args()
    endpoint = os.environ.get("REAX_NETWORK", "ws://127.0.0.1:9951")
    if not endpoint.startswith(("ws://127.0.0.1", "ws://localhost")):
        sys.exit("e2e_engine.py only runs against a local chain")
    work = os.path.abspath(args.work)
    os.makedirs(work, exist_ok=True)
    py = sys.executable
    bindir = os.path.dirname(py)
    roles = [*VALIDATORS, "m_serve", *COMPETITORS, COPIER]
    if not args.skip_bootstrap:
        subprocess.run([py, os.path.join(HERE, "bootstrap.py"), work, *roles], check=True,
                       env={**os.environ, "REAX_NETWORK": endpoint})
    subnet = json.load(open(os.path.join(work, "subnet.json")))
    netuid = subnet["netuid"]
    dev_map = {image_ref(v): [py, os.path.join(HERE, "fake_engine.py"), "--variant", v, "--delay", str(args.delay)]
               for v in set(COMPETITORS.values())}
    dev_map[image_ref("fast", "ghcr.io/copier")] = dev_map[image_ref("fast")]
    map_file = os.path.join(work, "engine-dev-map.json")
    with open(map_file, "w") as fh:
        json.dump(dev_map, fh, indent=1)
    base_env = {**os.environ, "REAX_NETWORK": subnet["endpoint"], "REAX_NETUID": str(netuid),
                "REAX_WALLET_PATH": os.path.join(work, "wallets-server"), "REAX_MECHANISM": "s1",
                "REAX_ALLOW_DEV_MODELS": "1", "PYTHONUNBUFFERED": "1", "OPENAI_API_KEY": "",
                "REAX_S1_FAST_RUNTIME_URL": "hash"}
    flag_env = {"REAX_FEATURE_ENGINE_COMPETITION": "1", "REAX_ALLOW_UNRELEASED_FEATURES": "1"}
    procs: list[tuple[str, subprocess.Popen]] = []

    def spawn(name, cmd, env):
        log = open(os.path.join(work, f"{name}.log"), "w")
        procs.append((name, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)))

    def submit(role, ref):
        r = subprocess.run([os.path.join(bindir, "reax-engine-submit"), ref],
                           env={**base_env, **flag_env, "REAX_WALLET_NAME": role}, capture_output=True, text=True)
        print(role, r.stdout.strip() or r.stderr.strip()[-300:], flush=True)
        if r.returncode:
            sys.exit(f"submission of {role} failed")

    spawn("m_serve", [os.path.join(bindir, "reax-miner")],
          {**base_env, "REAX_MODEL": "s1-fast", "REAX_WALLET_NAME": "m_serve", "REAX_MINER_HOST": args.host,
           "REAX_MINER_PORT": "18291", "REAX_EXTERNAL_IP": args.host, "REAX_MIN_VALIDATOR_STAKE": "10",
           "REAX_METAGRAPH_REFRESH_S": "30"})
    for role, variant in COMPETITORS.items():
        submit(role, image_ref(variant))
    time.sleep(15)  # the copy is committed in a later block than the original
    submit(COPIER, image_ref("fast", "ghcr.io/copier"))
    time.sleep(30)
    over = json.dumps({"reward": {"share": args.share},
                       "track": {"timed_items": 256, "start_timeout_s": 60}})
    for v in VALIDATORS:
        spawn(v, [os.path.join(bindir, "reax-validator")],
              {**base_env, **flag_env, "REAX_WALLET_NAME": v, "REAX_ALLOW_PRIVATE_IPS": "1",
               "REAX_ROUND_INTERVAL_S": "20", "REAX_WEIGHTS_INTERVAL_BLOCKS": "1", "REAX_EMA_ALPHA": "0.3",
               "REAX_WARMUP_ROUNDS": "2", "REAX_STATE_PATH": os.path.join(work, f"{v}-state.json"),
               "REAX_LOG_JSON": "1", "REAX_ENGINE_RUNNER": "dev", "REAX_ENGINE_DEV_MAP": map_file,
               "REAX_ENGINE_REFERENCE": image_ref("reference"), "REAX_ENGINE_WORKDIR": os.path.join(work, f"{v}-eng"),
               "REAX_ENGINE_STATE_PATH": os.path.join(work, f"{v}-engine.json"),
               "REAX_ENGINE_EVAL_INTERVAL_S": "120", "REAX_ENGINE_MAX_EVALS": "8", "REAX_ENGINE_DEV_PARAMS": over})

    st = bt.Subtensor(network=subnet["endpoint"])
    uid_of = {role: info["uid"] for role, info in subnet["roles"].items()}
    deadline = time.time() + args.minutes * 60
    history, verdict, streak = [], None, 0

    def check() -> bool:
        mg = st.subnets.metagraph(netuid)
        inc = {r: float(next(n.incentive for n in mg.neurons if n.uid == uid_of[r])) for r in uid_of}
        total = sum(inc[r] for r in ["m_serve", *COMPETITORS, COPIER]) or 1.0
        rel = {r: inc[r] / total for r in inc}
        others = [r for r in [*COMPETITORS, COPIER] if r != "c_fast"]
        ok = (abs(rel["c_fast"] - args.share) <= WITHIN * args.share
              and abs(rel["m_serve"] - (1 - args.share)) <= WITHIN * (1 - args.share)
              and all(rel[r] < 0.01 for r in others))
        history.append({"t": round(time.time()), "block": mg.block, "incentive": inc, "ok": ok})
        print(json.dumps(history[-1]), flush=True)
        return ok

    try:
        while time.time() < deadline:
            time.sleep(30)
            for name, p in procs:
                if p.poll() is not None:
                    print(f"process {name} exited with {p.returncode}", flush=True)
            try:
                streak = streak + 1 if check() else 0
            except Exception as exc:  # noqa: BLE001
                print("metagraph read failed:", exc, flush=True)
            if streak >= 3:
                verdict = "PASS"
                break
        verdict = verdict or "FAIL"
    finally:
        for _, p in procs:
            p.terminate()
        for _, p in procs:
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()
        engines = {}
        for v in VALIDATORS:
            try:
                engines[v] = json.load(open(os.path.join(work, f"{v}-engine.json")))
            except OSError:
                engines[v] = None
        with open(os.path.join(work, "e2e_engine_result.json"), "w") as fh:
            json.dump({"verdict": verdict, "share": args.share, "uids": uid_of, "history": history,
                       "engine_state": engines}, fh, indent=1)
    print("VERDICT", verdict)
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
