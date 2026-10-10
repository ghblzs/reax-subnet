#!/usr/bin/env python3
"""reax-s1 end-to-end run on a local subtensor: 2 validators, honest and adversarial miners.

Prerequisite: start.sh is running, and the model runtimes are reachable on loopback
(REAX_S1_FAST_RUNTIME_URL / REAX_S1_PRO_RUNTIME_URL, e.g. SSH tunnels to a GPU host; "hash" = test model).
Usage:
    e2e_s1.py <work_dir> [--minutes 25] [--host 127.0.0.1] [--pro]

Every process runs the real CLI (reax-validator / reax-miner) or the
adversary wrapper, with hotkey-only wallets. The run passes when, on chain,
every honest miner holds incentive and every adversary holds none.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import bittensor as bt

HERE = os.path.dirname(os.path.abspath(__file__))
HONEST = ["m_fast1", "m_fast2"]
HONEST_PRO = ["m_pro1"]
ADVERSARIES = {"m_other_model": "other_model", "m_quantised": "quantised", "m_uniform": "uniform",
               "m_inconsistent": "inconsistent", "m_slow": "slow"}
ADVERSARIES_PRO = {"m_pro_partial": "partial"}
SPECIAL = ["m_axon_copy", "m_offline"]
VALIDATORS = ["v1", "v2"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--minutes", type=float, default=25)
    ap.add_argument("--host", default="127.0.0.1", help="interface the miners bind to and publish (loopback is enough for a single-machine localnet)")
    ap.add_argument("--pro", action="store_true", help="enable the s1-pro pool (needs an s1-pro runtime)")
    ap.add_argument("--skip-bootstrap", action="store_true")
    args = ap.parse_args()
    global HONEST, ADVERSARIES
    if args.pro:
        HONEST = HONEST + HONEST_PRO
        ADVERSARIES = {**ADVERSARIES, **ADVERSARIES_PRO}
    model_of = {r: ("s1-pro" if r in HONEST_PRO or r in ADVERSARIES_PRO else "s1-fast") for r in HONEST + list(ADVERSARIES)}
    work = os.path.abspath(args.work)
    os.makedirs(work, exist_ok=True)
    py = sys.executable
    if not args.skip_bootstrap:
        roles = VALIDATORS + HONEST + list(ADVERSARIES) + SPECIAL
        subprocess.run([py, os.path.join(HERE, "bootstrap.py"), work, *roles], check=True)
    subnet = json.load(open(os.path.join(work, "subnet.json")))
    netuid = subnet["netuid"]
    base_env = {
        **os.environ,
        "REAX_NETWORK": subnet["endpoint"],
        "REAX_NETUID": str(netuid),
        "REAX_WALLET_PATH": os.path.join(work, "wallets-server"),
        "REAX_MECHANISM": "s1",
        "REAX_ALLOW_DEV_MODELS": "1",
        "PYTHONUNBUFFERED": "1",
    }
    if args.pro:
        base_env["REAX_FEATURE_S1_PRO_POOL"] = "1"
    procs: list[tuple[str, subprocess.Popen]] = []

    def spawn(name, cmd, env):
        log = open(os.path.join(work, f"{name}.log"), "w")
        procs.append((name, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)))

    port = 18091
    ports = {}
    for role in HONEST + list(ADVERSARIES):
        env = {**base_env, "REAX_MODEL": model_of[role], "REAX_WALLET_NAME": role, "REAX_MINER_HOST": args.host,
               "REAX_MINER_PORT": str(port),
               "REAX_EXTERNAL_IP": args.host, "REAX_MIN_VALIDATOR_STAKE": "10", "REAX_METAGRAPH_REFRESH_S": "30"}
        if role in ADVERSARIES:
            spawn(role, [py, os.path.join(HERE, "adversary_miner_s1.py")], {**env, "REAX_ADVERSARY": ADVERSARIES[role]})
        else:
            spawn(role, [os.path.join(os.path.dirname(py), "reax-miner")], env)
        ports[role] = port
        port += 1
    # Axon copier: publishes honest1's endpoint under its own hotkey, runs no server.
    copier = bt.Wallet(name="m_axon_copy", hotkey="default", path=base_env["REAX_WALLET_PATH"])
    r = bt.Subtensor(network=subnet["endpoint"]).execute(bt.ServeAxon(netuid, args.host, ports["m_fast1"]), copier)
    print("axon copier published m_fast1's endpoint:", r.success, r.message)
    from reax_subnet.chain import SdkChain
    from reax_subnet.manifest import commitment_text, load_manifest
    SdkChain(subnet["endpoint"], netuid, copier).set_commitment(commitment_text("s1-fast", load_manifest("s1-fast").manifest_id))
    time.sleep(60)  # let miners load models and publish endpoints
    validator_cmd = [os.path.join(os.path.dirname(py), "reax-validator")]
    validator_env = {}
    for v in VALIDATORS:
        validator_env[v] = {**base_env, "REAX_WALLET_NAME": v, "REAX_ALLOW_PRIVATE_IPS": "1",
                            "REAX_ROUND_INTERVAL_S": "30", "REAX_WEIGHTS_INTERVAL_BLOCKS": "1", "REAX_EMA_ALPHA": "0.3",
                            "REAX_WARMUP_ROUNDS": "2",
                            "REAX_STATE_PATH": os.path.join(work, f"{v}-state.json"), "REAX_LOG_JSON": "1"}
        spawn(v, validator_cmd, validator_env[v])

    st = bt.Subtensor(network=subnet["endpoint"])
    uid_of = {role: info["uid"] for role, info in subnet["roles"].items()}
    owner_uid = next(n.uid for n in st.subnets.metagraph(netuid).neurons if n.hotkey == subnet["owner_hotkey"])
    honest_uids = {uid_of[r] for r in HONEST}
    deadline = time.time() + args.minutes * 60
    verdict = None
    history = []
    restarted = False
    passes_after_restart = 0

    def check() -> bool:
        mg = st.subnets.metagraph(netuid)
        weights = st.weights.weights(netuid)
        inc = {n.uid: round(float(n.incentive), 4) for n in mg.neurons}
        rows_ok = all(
            uid_of[v] in weights and set(weights[uid_of[v]]) - {owner_uid} == honest_uids for v in VALIDATORS
        )
        honest_ok = all(inc.get(u, 0) > 0 for u in honest_uids)
        adv_zero = all(inc.get(uid_of[r], 0) == 0 for r in list(ADVERSARIES) + SPECIAL)
        history.append({"t": round(time.time()), "block": mg.block, "restarted": restarted,
                        "weights_rows": {str(k): v for k, v in weights.items()}, "incentive": inc,
                        "rows_ok": rows_ok, "honest_ok": honest_ok, "adversaries_zero": adv_zero})
        print(f"block {mg.block}: inc={inc} rows_ok={rows_ok} honest>0={honest_ok} adversaries=0={adv_zero} restarted={restarted}",
              flush=True)
        return rows_ok and honest_ok and adv_zero

    try:
        while time.time() < deadline:
            time.sleep(30)
            dead = [n for n, p in procs if p.poll() is not None]
            if dead:
                print("processes exited:", dead)
                break
            if not check():
                continue
            if not restarted:
                # Restart safety: kill validator v1 hard and start it again on the same state file.
                name, proc = next((n, p) for n, p in procs if n == "v1")
                proc.kill()
                proc.wait()
                procs.remove((name, proc))
                spawn("v1-restarted", validator_cmd, validator_env["v1"])
                restarted = True
                continue
            passes_after_restart += 1
            if passes_after_restart >= 3:
                verdict = "PASS"
                break
    finally:
        for _, p in procs:
            p.terminate()
        for _, p in procs:
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()
    result = {
        "verdict": verdict or "FAIL",
        "netuid": netuid,
        "roles": {r: {"uid": i["uid"], "hotkey": i["hotkey"]} for r, i in subnet["roles"].items()},
        "owner_hotkey": subnet["owner_hotkey"],
        "owner_uid": owner_uid,
        "validator_restarted_midrun": restarted,
        "final": history[-1] if history else None,
        "history": history,
    }
    with open(os.path.join(work, "E2E-RESULT.json"), "w") as fh:
        json.dump(result, fh, indent=1)
    print(json.dumps(result, indent=1))
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
