#!/usr/bin/env python3
"""Tiered pools (REAX_FEATURE_TIERED_POOLS) end-to-end on a local subtensor: 2 validators, GPU / small / CPU miners.

Prerequisites: start.sh is running. s1-fast and s1-lite use the deterministic test model ("hash"); s1-cpu uses REAL
Gutsy 0.8B GGUF sidecars on loopback, given as URLs:
    --cpu-ref-v1 / --cpu-ref-v2   the validators' own s1-cpu reference sidecars
    --cpu-solo                    the sidecar of one CPU miner (its own machine)
    --cpu-shared                  ONE sidecar behind three CPU miner hotkeys (a sybil group)
Usage:
    e2e_tiered.py <work_dir> --token <runtime token file> --cpu-ref-v1 URL --cpu-ref-v2 URL --cpu-solo URL
                  --cpu-shared URL [--minutes 40]

Passes when, on chain, every honest miner holds incentive, every adversary holds none, the GPU pool holds the largest
share, and the three sybil hotkeys together hold no more than ~1.35x the solo CPU miner (they share one machine; they
may also hold nothing when their shared machine misses the latency bound).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time

import bittensor as bt

HERE = os.path.dirname(os.path.abspath(__file__))
VALIDATORS = ["v1", "v2"]
MINERS = {  # role -> (model, runtime URL key or "hash", adversary kind or None)
    "m_fast": ("s1-fast", "hash", None),
    "m_lite1": ("s1-lite", "hash", None),
    "m_lite2": ("s1-lite", "hash", None),
    "m_cpu_solo": ("s1-cpu", "cpu_solo", None),
    "m_cpu_sybil1": ("s1-cpu", "cpu_shared", None),
    "m_cpu_sybil2": ("s1-cpu", "cpu_shared", None),
    "m_cpu_sybil3": ("s1-cpu", "cpu_shared", None),
    "m_cpu_other_model": ("s1-cpu", "hash", "other_model"),
    "m_lite_quantised": ("s1-lite", "hash", "quantised"),
}
SYBILS = ["m_cpu_sybil1", "m_cpu_sybil2", "m_cpu_sybil3"]
# Honest single-machine miners must earn; the sybil group may earn (together no more than the solo node) or not at all
# (three hotkeys on one machine can also miss the latency bound together).
HONEST = [r for r, (_m, _u, adv) in MINERS.items() if adv is None and r not in SYBILS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--minutes", type=float, default=40)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--token", required=True)
    ap.add_argument("--cpu-ref-v1", required=True)
    ap.add_argument("--cpu-ref-v2", required=True)
    ap.add_argument("--cpu-solo", required=True)
    ap.add_argument("--cpu-shared", required=True)
    ap.add_argument("--skip-bootstrap", action="store_true")
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))  # run the cleanup below on kill
    urls = {"cpu_solo": args.cpu_solo, "cpu_shared": args.cpu_shared, "hash": "hash"}
    work = os.path.abspath(args.work)
    os.makedirs(work, exist_ok=True)
    py = sys.executable
    if not args.skip_bootstrap:
        subprocess.run([py, os.path.join(HERE, "bootstrap.py"), work, *VALIDATORS, *MINERS], check=True)
    subnet = json.load(open(os.path.join(work, "subnet.json")))
    netuid = subnet["netuid"]
    base_env = {
        **os.environ,
        "REAX_NETWORK": subnet["endpoint"],
        "REAX_NETUID": str(netuid),
        "REAX_WALLET_PATH": os.path.join(work, "wallets-server"),
        "REAX_MECHANISM": "s1",
        "REAX_ALLOW_DEV_MODELS": "1",
        "REAX_FEATURE_TIERED_POOLS": "1",
        "REAX_RUNTIME_TOKEN_FILE": args.token,
        "PYTHONUNBUFFERED": "1",
    }
    procs: list[tuple[str, subprocess.Popen]] = []

    def spawn(name, cmd, env):
        log = open(os.path.join(work, f"{name}.log"), "w")
        procs.append((name, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)))

    port = 18191
    for role, (model, url_key, adversary) in MINERS.items():
        env = {**base_env, "REAX_MODEL": model, "REAX_WALLET_NAME": role, "REAX_MINER_HOST": args.host,
               "REAX_MINER_PORT": str(port), "REAX_EXTERNAL_IP": args.host, "REAX_MIN_VALIDATOR_STAKE": "10",
               "REAX_METAGRAPH_REFRESH_S": "30", "REAX_PER_CALLER_PER_MINUTE": "60",
               f"REAX_{model.upper().replace('-', '_')}_RUNTIME_URL": urls[url_key]}
        if adversary:
            spawn(role, [py, os.path.join(HERE, "adversary_miner_s1.py")], {**env, "REAX_ADVERSARY": adversary})
        else:
            spawn(role, [os.path.join(os.path.dirname(py), "reax-miner")], env)
        port += 1
    time.sleep(60)
    validator_cmd = [os.path.join(os.path.dirname(py), "reax-validator")]
    cpu_ref = {"v1": args.cpu_ref_v1, "v2": args.cpu_ref_v2}
    for v in VALIDATORS:
        env = {**base_env, "REAX_WALLET_NAME": v, "REAX_ALLOW_PRIVATE_IPS": "1",
               "REAX_ROUND_INTERVAL_S": "60", "REAX_WEIGHTS_INTERVAL_BLOCKS": "1", "REAX_EMA_ALPHA": "0.3",
               "REAX_WARMUP_ROUNDS": "2", "REAX_DEV_PROBE_EVERY_ROUNDS": "2",
               "REAX_S1_FAST_RUNTIME_URL": "hash", "REAX_S1_LITE_RUNTIME_URL": "hash",
               "REAX_S1_CPU_RUNTIME_URL": cpu_ref[v],
               "REAX_STATE_PATH": os.path.join(work, f"{v}-state.json"), "REAX_LOG_JSON": "1"}
        spawn(v, validator_cmd, env)

    st = bt.Subtensor(network=subnet["endpoint"])
    uid_of = {role: info["uid"] for role, info in subnet["roles"].items()}
    honest_uids = {uid_of[r] for r in HONEST}
    deadline = time.time() + args.minutes * 60
    verdict, history, passes = None, [], 0

    def check() -> bool:
        mg = st.subnets.metagraph(netuid)
        inc = {n.uid: round(float(n.incentive), 4) for n in mg.neurons}
        by_role = {r: inc.get(uid_of[r], 0.0) for r in MINERS}
        honest_ok = all(inc.get(u, 0) > 0 for u in honest_uids)
        adv_zero = all(by_role[r] == 0 for r, (_m, _u, adv) in MINERS.items() if adv)
        gpu_top = by_role["m_fast"] >= max(v for r, v in by_role.items() if r != "m_fast")
        sybil_sum = sum(by_role[r] for r in SYBILS)
        sybil_ok = by_role["m_cpu_solo"] > 0 and sybil_sum <= 1.35 * by_role["m_cpu_solo"]
        history.append({"t": round(time.time()), "block": mg.block, "incentive_by_role": by_role,
                        "honest_ok": honest_ok, "adversaries_zero": adv_zero, "gpu_top": gpu_top,
                        "sybil_sum": round(sybil_sum, 4), "sybil_ok": sybil_ok})
        print(f"block {mg.block}: {by_role} honest>0={honest_ok} adv=0={adv_zero} gpu_top={gpu_top} "
              f"sybils={sybil_sum:.4f} vs solo={by_role['m_cpu_solo']:.4f}", flush=True)
        return honest_ok and adv_zero and gpu_top and sybil_ok

    try:
        while time.time() < deadline:
            time.sleep(30)
            dead = [n for n, p in procs if p.poll() is not None]
            if dead:
                print("processes exited:", dead)
                break
            if check():
                passes += 1
                if passes >= 3:
                    verdict = "PASS"
                    break
            else:
                passes = 0
    finally:
        for _, p in procs:
            p.terminate()
        for _, p in procs:
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()
    result = {"verdict": verdict or "FAIL", "netuid": netuid,
              "roles": {r: {"uid": i["uid"]} for r, i in subnet["roles"].items()},
              "final": history[-1] if history else None, "history": history}
    with open(os.path.join(work, "E2E-RESULT.json"), "w") as fh:
        json.dump(result, fh, indent=1)
    print(json.dumps(result["final"], indent=1), result["verdict"])
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
