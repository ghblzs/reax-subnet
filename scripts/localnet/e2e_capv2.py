#!/usr/bin/env python3
"""Capacity-v2 end-to-end on a local subtensor: does verified-throughput pay close the split-hotkey loophole? ON-CHAIN incentive.

Prerequisite: a local chain is running (REAX_NETWORK=ws://127.0.0.1:<port>). Everything runs on the hash test model with
throughput-limited fake s1-fast sidecars (fake_capacity_backend.py): a backend serves at most C decisions/s, shared by all
hotkeys that point to it.

    machine A: 160/s behind ONE hotkey        machine B: 160/s split over FOUR hotkeys (one shared backend)
    machine C:  40/s behind one hotkey        machine D: 320/s behind one hotkey

All hotkeys are s1-fast miners, two real validators (cli.validator_main, capacity probes every second round). Expected incentive
(fractions of the miners' total):

    v2   (capacity v2, linear factor):   A : sum(B) : C : D = 160 : 160 : 40 : 320   (D = 2 A, C = A/4, sum(B) = A)
    rc2  (1.2.0rc2, factor capped at 1): A : sum(B) : C : D = 1 : 4 : 1 : 1           (the loophole: B = 4 A)

Usage: REAX_NETWORK=ws://127.0.0.1:<port> e2e_capv2.py <work_dir> --host <private ip> --expect v2|rc2 [--minutes 40]
The code under test is whatever ``reax_subnet`` the interpreter (PYTHONPATH) imports.
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
MACHINES = {"A": 160.0, "B": 160.0, "C": 40.0, "D": 320.0}  # decisions/s of each backend
HOTKEYS = {"A": ["mA"], "B": ["mB1", "mB2", "mB3", "mB4"], "C": ["mC"], "D": ["mD"]}
WEIGHTS = {"v2": {"A": 160.0, "B": 160.0, "C": 40.0, "D": 320.0}, "rc2": {"A": 1.0, "B": 4.0, "C": 1.0, "D": 1.0}}
TOL_REL, TOL_ABS = 0.12, 0.012  # a machine passes when |on-chain - expected| <= max(TOL_ABS, TOL_REL * expected)


def expected(kind: str) -> dict[str, float]:
    w = WEIGHTS[kind]
    return {m: v / sum(w.values()) for m, v in w.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--host", required=True, help="a private interface address (the chain refuses loopback axons)")
    ap.add_argument("--expect", choices=sorted(WEIGHTS), required=True)
    ap.add_argument("--minutes", type=float, default=40, help="maximum run time")
    ap.add_argument("--min-minutes", type=float, default=12, help="no verdict before this (capacity averages settle)")
    ap.add_argument("--streak", type=int, default=6, help="consecutive passing readings (30 s apart) that end the run")
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit("terminated"))
    endpoint = os.environ.get("REAX_NETWORK", "")
    if not endpoint.startswith(("ws://127.0.0.1", "ws://localhost")):
        sys.exit("e2e_capv2.py only runs against a local chain")
    work = os.path.abspath(args.work)
    os.makedirs(work, exist_ok=True)
    py = sys.executable
    bindir = os.path.dirname(py)
    miners = [h for hs in HOTKEYS.values() for h in hs]
    subprocess.run([py, os.path.join(HERE, "bootstrap.py"), work, *VALIDATORS, *miners], check=True,
                   env={**os.environ, "REAX_NETWORK": endpoint})
    subnet = json.load(open(os.path.join(work, "subnet.json")))
    netuid = subnet["netuid"]
    uid_of = {role: info["uid"] for role, info in subnet["roles"].items()}
    token = os.path.join(work, "runtime-token")
    with open(token, "w") as fh:
        fh.write("e2e-capv2-token\n")
    os.chmod(token, 0o600)
    base_env = {**os.environ, "REAX_NETWORK": subnet["endpoint"], "REAX_NETUID": str(netuid),
                "REAX_WALLET_PATH": os.path.join(work, "wallets-server"), "REAX_MECHANISM": "s1",
                "REAX_ALLOW_DEV_MODELS": "1", "PYTHONUNBUFFERED": "1", "OPENAI_API_KEY": "",
                "REAX_RUNTIME_TOKEN_FILE": token}
    procs: dict[str, subprocess.Popen] = {}

    def spawn(name, cmd, env):
        procs[name] = subprocess.Popen(cmd, env=env, stdout=open(os.path.join(work, f"{name}.log"), "w"),
                                       stderr=subprocess.STDOUT)

    history: list[dict] = []
    exp = expected(args.expect)
    streak, verdict, started = 0, "FAIL", time.time()
    st = bt.Subtensor(network=subnet["endpoint"])
    try:
        backend_port = {m: 18600 + i for i, m in enumerate(MACHINES)}
        for m, cap in MACHINES.items():
            spawn(f"backend_{m}", [py, os.path.join(HERE, "fake_capacity_backend.py"), "--port", str(backend_port[m]),
                                   "--capacity", str(cap), "--token-file", token, "--name", m], base_env)
        time.sleep(3)
        port = 18391
        for m, hotkeys in HOTKEYS.items():
            for role in hotkeys:  # the miner's own limits are the code's defaults (they are part of what is tested)
                spawn(role, [os.path.join(bindir, "reax-miner")],
                      {**base_env, "REAX_MODEL": "s1-fast", "REAX_WALLET_NAME": role, "REAX_MINER_HOST": args.host,
                       "REAX_MINER_PORT": str(port), "REAX_EXTERNAL_IP": args.host, "REAX_MIN_VALIDATOR_STAKE": "10",
                       "REAX_METAGRAPH_REFRESH_S": "30",
                       "REAX_S1_FAST_RUNTIME_URL": f"http://127.0.0.1:{backend_port[m]}"})
                port += 1
        time.sleep(45)
        for i, v in enumerate(VALIDATORS):
            spawn(v, [os.path.join(bindir, "reax-validator")],
                  {**base_env, "REAX_WALLET_NAME": v, "REAX_ALLOW_PRIVATE_IPS": "1", "REAX_BURN_FRACTION": "0.0",
                   "REAX_ROUND_INTERVAL_S": "30", "REAX_WEIGHTS_INTERVAL_BLOCKS": "1", "REAX_EMA_ALPHA": "0.3",
                   "REAX_WARMUP_ROUNDS": "2", "REAX_DEV_PROBE_EVERY_ROUNDS": "2", "REAX_LOG_JSON": "1",
                   "REAX_S1_FAST_RUNTIME_URL": "hash", "REAX_S1_PRO_RUNTIME_URL": "hash",
                   "REAX_S1_LITE_RUNTIME_URL": "hash", "REAX_S1_CPU_RUNTIME_URL": "hash",
                   "REAX_FEATURE_ENGINE_COMPETITION": "0",
                   "REAX_STATE_PATH": os.path.join(work, f"{v}-state.json")})
            if i == 0:
                time.sleep(15)  # the validators' probes must not coincide: a shared backend would be measured twice over

        def incentive() -> dict[str, float]:
            mg = st.subnets.metagraph(netuid)
            by_uid = {n.uid: float(n.incentive) for n in mg.neurons}
            return {r: by_uid.get(uid_of[r], 0.0) for r in miners}

        while time.time() - started < args.minutes * 60:
            time.sleep(30)
            for name, p in procs.items():
                if p.poll() is not None:
                    print(f"process {name} exited with {p.returncode}", flush=True)
            try:
                inc = incentive()
            except Exception as exc:  # noqa: BLE001
                print("metagraph read failed:", exc, flush=True)
                continue
            total = sum(inc.values())
            if total <= 0:  # an epoch boundary: the chain shows no incentive for a moment; not a reading
                continue
            per_hotkey = {r: v / total for r, v in inc.items()}
            share = {m: sum(per_hotkey[r] for r in hs) for m, hs in HOTKEYS.items()}
            ok = all(abs(share[m] - exp[m]) <= max(TOL_ABS, TOL_REL * exp[m]) for m in HOTKEYS)
            minutes = (time.time() - started) / 60
            history.append({"t": round(time.time()), "minutes": round(minutes, 1), "ok": ok,
                            "machine": {m: round(s, 4) for m, s in share.items()},
                            "hotkey": {r: round(s, 4) for r, s in per_hotkey.items()}})
            print(f"{minutes:5.1f} min ok={ok} " + " ".join(f"{m}={share[m]:.3f}({exp[m]:.3f})" for m in HOTKEYS), flush=True)
            streak = streak + 1 if ok else 0
            if streak >= args.streak and minutes >= args.min_minutes:
                verdict = "PASS"
                break
        stats = {}
        for m in MACHINES:  # work actually served by every simulated machine
            try:
                import urllib.request
                stats[m] = json.load(urllib.request.urlopen(f"http://127.0.0.1:{backend_port[m]}/stats", timeout=5))
            except Exception as exc:  # noqa: BLE001
                stats[m] = str(exc)
    finally:
        for p in procs.values():
            p.terminate()
        for p in procs.values():
            try:
                p.wait(15)
            except subprocess.TimeoutExpired:
                p.kill()
    capacity = {}
    for v in VALIDATORS:
        try:
            state = json.load(open(os.path.join(work, f"{v}-state.json")))
            miners_state = state.get("miners", {})
            capacity[v] = {r: miners_state.get(str(uid_of[r]), {}).get("capacity") for r in miners}
        except (OSError, ValueError) as exc:
            capacity[v] = str(exc)
    tail = history[-args.streak:]
    avg = {m: round(sum(h["machine"][m] for h in tail) / len(tail), 4) for m in HOTKEYS} if tail else {}
    json.dump({"expect": args.expect, "verdict": verdict, "tolerance": {"relative": TOL_REL, "absolute": TOL_ABS},
               "machines_dps": MACHINES, "expected_share": exp, "average_last_readings": avg, "uids": uid_of,
               "backend_stats": stats, "validator_capacity_ema": capacity, "history": history},
              open(os.path.join(work, "E2E-CAPV2-RESULT.json"), "w"), indent=1)
    print("VERDICT", verdict, "average of the last readings:", avg)
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
