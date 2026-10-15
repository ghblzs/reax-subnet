#!/usr/bin/env python3
"""Release 1.2 end-to-end on a local subtensor: tiered pools + s1-pro + competitions, 2 validators, ON-CHAIN incentive.

Prerequisite: start.sh is running (local chain only). Everything runs on the hash test model (no GPU). Phases on ONE
subnet, each checked against the expected weights on chain (3 consecutive passes, 30 s apart):

 1. no champions: the whole competition share must be back with the inference part (tiers 80/15/5, s1-fast/s1-pro 35/65
    inside the GPU tier);
 2. an engine champion (reax-eng/1 candidate through the dev runner) and the fake specialist results of both tracks:
    inference 0.50, engine 0.20, tracks 0.15 + 0.15 (ranks 75/17/8), the tiers 80/15/5 inside the inference part;
 3. the s1-pro miner stops: its pool empties and the GPU tier's share goes to s1-fast (no burn), competitions unchanged.

Usage: REAX_NETWORK=ws://127.0.0.1:<port> e2e_release.py <work_dir> --host <private ip> [--minutes 40]
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e2e_engine import image_ref  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
VALIDATORS = ["v1", "v2"]
MINERS = {  # role -> pool
    "m_fast": "s1-fast", "m_pro": "s1-pro", "m_lite1": "s1-lite", "m_lite2": "s1-lite", "m_cpu1": "s1-cpu",
    "m_cpu2": "s1-cpu",
}
SPEC = {"demand_forecasting": ["sd1", "sd2", "sd3"], "risk_assessment": ["sr1", "sr2", "sr3"]}
SPEC_SCORES = (0.9, 0.8, 0.7)
RANKS = (0.75, 0.17, 0.08)
ENGINE = "c_fast"


def serving(pro: bool) -> dict[str, float]:
    """Inference part (sums to 1): tier budgets 80/15/5; s1-fast/s1-pro 35/65 inside the GPU tier (empty pool -> the
    other pool); two miners in each of the small and CPU tiers, so the tier-order rule does not bind."""
    gpu = {"m_fast": 0.80 * (0.35 if pro else 1.0)}
    if pro:
        gpu["m_pro"] = 0.80 * 0.65
    return {**gpu, "m_lite1": 0.075, "m_lite2": 0.075, "m_cpu1": 0.025, "m_cpu2": 0.025}


def expected(phase: int) -> dict[str, float]:
    if phase == 1:
        return serving(True)
    base = {r: 0.5 * s for r, s in serving(phase == 2).items()}
    base[ENGINE] = 0.20
    for roles in SPEC.values():
        for role, rank in zip(roles, RANKS):
            base[role] = 0.15 * rank
    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--minutes", type=float, default=40)
    ap.add_argument("--host", required=True, help="a private interface address (the chain refuses loopback axons)")
    ap.add_argument("--delay", type=float, default=0.02)
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit("terminated"))
    endpoint = os.environ.get("REAX_NETWORK", "")
    if not endpoint.startswith(("ws://127.0.0.1", "ws://localhost")):
        sys.exit("e2e_release.py only runs against a local chain")
    work = os.path.abspath(args.work)
    os.makedirs(work, exist_ok=True)
    py = sys.executable
    bindir = os.path.dirname(py)
    roles = [*VALIDATORS, *MINERS, ENGINE, *[r for rs in SPEC.values() for r in rs]]
    subprocess.run([py, os.path.join(HERE, "bootstrap.py"), work, *roles], check=True,
                   env={**os.environ, "REAX_NETWORK": endpoint})
    subnet = json.load(open(os.path.join(work, "subnet.json")))
    netuid = subnet["netuid"]
    uid_of = {role: info["uid"] for role, info in subnet["roles"].items()}
    token = os.path.join(work, "runtime-token")
    with open(token, "w") as fh:
        fh.write("e2e-release-token\n")
    os.chmod(token, 0o600)
    dev_map = {image_ref(v): [py, os.path.join(HERE, "fake_engine.py"), "--variant", v, "--delay", str(args.delay)]
               for v in ("fast", "reference")}
    map_file = os.path.join(work, "engine-dev-map.json")
    json.dump(dev_map, open(map_file, "w"), indent=1)
    spec_file = os.path.join(work, "spec-results.json")  # absent until phase 2
    sealed = os.path.join(work, "sealed")
    os.makedirs(sealed, exist_ok=True)
    base_env = {**os.environ, "REAX_NETWORK": subnet["endpoint"], "REAX_NETUID": str(netuid),
                "REAX_WALLET_PATH": os.path.join(work, "wallets-server"), "REAX_MECHANISM": "s1",
                "REAX_ALLOW_DEV_MODELS": "1", "PYTHONUNBUFFERED": "1", "OPENAI_API_KEY": "",
                "REAX_RUNTIME_TOKEN_FILE": token}
    procs: dict[str, subprocess.Popen] = {}

    def spawn(name, cmd, env):
        procs[name] = subprocess.Popen(cmd, env=env, stdout=open(os.path.join(work, f"{name}.log"), "w"),
                                       stderr=subprocess.STDOUT)

    history: dict[int, list] = {1: [], 2: [], 3: []}
    verdicts: dict[int, str] = {}
    st = bt.Subtensor(network=subnet["endpoint"])
    try:
        port = 18391
        for role, pool in MINERS.items():
            spawn(role, [os.path.join(bindir, "reax-miner")],
                  {**base_env, "REAX_MODEL": pool, "REAX_WALLET_NAME": role, "REAX_MINER_HOST": args.host,
                   "REAX_MINER_PORT": str(port), "REAX_EXTERNAL_IP": args.host, "REAX_MIN_VALIDATOR_STAKE": "10",
                   "REAX_METAGRAPH_REFRESH_S": "30", "REAX_PER_CALLER_PER_MINUTE": "120",
                   f"REAX_{pool.upper().replace('-', '_')}_RUNTIME_URL": "hash"})
            port += 1
        time.sleep(45)
        over = json.dumps({"reward": {}, "track": {"timed_items": 256, "start_timeout_s": 60,
                                                   "confirm_gap_blocks": 50}})
        for v in VALIDATORS:
            spawn(v, [py, os.path.join(HERE, "e2e_release_validator.py")],
                  {**base_env, "REAX_WALLET_NAME": v, "REAX_ALLOW_PRIVATE_IPS": "1", "REAX_BURN_FRACTION": "0.0",
                   "REAX_ROUND_INTERVAL_S": "30", "REAX_WEIGHTS_INTERVAL_BLOCKS": "1", "REAX_EMA_ALPHA": "0.3",
                   "REAX_WARMUP_ROUNDS": "2", "REAX_DEV_PROBE_EVERY_ROUNDS": "2", "REAX_LOG_JSON": "1",
                   "REAX_S1_FAST_RUNTIME_URL": "hash", "REAX_S1_PRO_RUNTIME_URL": "hash",
                   "REAX_S1_LITE_RUNTIME_URL": "hash", "REAX_S1_CPU_RUNTIME_URL": "hash",
                   "REAX_STATE_PATH": os.path.join(work, f"{v}-state.json"),
                   "REAX_FEATURE_ENGINE_COMPETITION": "1", "REAX_ENGINE_RUNNER": "dev", "REAX_ENGINE_DEV_MAP": map_file,
                   "REAX_ENGINE_REFERENCE": image_ref("reference"),
                   "REAX_ENGINE_WORKDIR": os.path.join(work, f"{v}-eng"),
                   "REAX_ENGINE_STATE_PATH": os.path.join(work, f"{v}-engine.json"),
                   "REAX_ENGINE_EVAL_INTERVAL_S": "120", "REAX_ENGINE_MAX_EVALS": "8", "REAX_ENGINE_DEV_PARAMS": over,
                   "REAX_SPEC_SEALED_DIR": sealed, "REAX_E2E_SPEC_RESULTS": spec_file})

        def incentive() -> dict[str, float]:
            mg = st.subnets.metagraph(netuid)
            by_uid = {n.uid: float(n.incentive) for n in mg.neurons}
            return {r: by_uid.get(u, 0.0) for r, u in uid_of.items() if r not in VALIDATORS}

        def check(phase: int) -> bool:
            inc = incentive()
            exp = expected(phase)
            total = sum(inc.values())
            rel = {r: (inc.get(r, 0.0) / total if total else 0.0) for r in inc}
            rows = {r: {"expected": round(exp.get(r, 0.0), 4), "onchain": round(rel[r], 4)} for r in sorted(inc)}
            ok = total > 0 and all(abs(rel[r] - exp.get(r, 0.0)) <= max(0.006, 0.12 * exp.get(r, 0.0)) for r in inc)
            history[phase].append({"t": round(time.time()), "ok": ok, "rows": rows})
            print(f"phase {phase} ok={ok}", json.dumps({r: (v['expected'], v['onchain']) for r, v in rows.items()}),
                  flush=True)
            return ok

        def wait_phase(phase: int, minutes: float) -> bool:
            deadline, streak = time.time() + minutes * 60, 0
            while time.time() < deadline:
                time.sleep(30)
                for name, p in procs.items():
                    if p.poll() is not None and name != "m_pro":
                        print(f"process {name} exited with {p.returncode}", flush=True)
                try:
                    streak = streak + 1 if check(phase) else 0
                except Exception as exc:  # noqa: BLE001
                    print("metagraph read failed:", exc, flush=True)
                if streak >= 3:
                    verdicts[phase] = "PASS"
                    return True
            verdicts[phase] = "FAIL"
            return False

        total_min = args.minutes
        ok = wait_phase(1, total_min * 0.3)
        if ok or True:  # run on: later phases are informative even if an earlier one failed
            json.dump({t: [[uid_of[r], s] for r, s in zip(rs, SPEC_SCORES)] for t, rs in SPEC.items()},
                      open(spec_file, "w"))
            r = subprocess.run([os.path.join(bindir, "reax-engine-submit"), image_ref("fast")],
                               env={**base_env, "REAX_FEATURE_ENGINE_COMPETITION": "1", "REAX_WALLET_NAME": ENGINE},
                               capture_output=True, text=True)
            print(ENGINE, (r.stdout.strip() or r.stderr.strip())[-300:], flush=True)
            wait_phase(2, total_min * 0.4)
            procs["m_pro"].terminate()
            wait_phase(3, total_min * 0.3)
    finally:
        for p in procs.values():
            p.terminate()
        for p in procs.values():
            try:
                p.wait(15)
            except subprocess.TimeoutExpired:
                p.kill()
    engines = {}
    for v in VALIDATORS:
        try:
            engines[v] = json.load(open(os.path.join(work, f"{v}-engine.json")))
        except OSError:
            engines[v] = None
    verdict = "PASS" if all(verdicts.get(p) == "PASS" for p in (1, 2, 3)) else "FAIL"
    json.dump({"verdict": verdict, "phases": verdicts, "uids": uid_of, "expected": {p: expected(p) for p in (1, 2, 3)},
               "history": history, "engine_state": engines}, open(os.path.join(work, "E2E-RELEASE-RESULT.json"), "w"),
              indent=1)
    print("VERDICT", verdict, verdicts)
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
