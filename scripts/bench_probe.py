#!/usr/bin/env python3
"""How many probe decisions per second can this validator host measure? (capacity review C1)

Starts fake s1-fast miners in separate processes (hash answers, no model, throwaway dev keys on 127.0.0.1) and runs
the real validator capacity probe against them with the release s1-fast probe parameters, every UID budgeted at the
pool cap. It prints, per probe, the total throughput the validator counted, its event-loop lag and whether the
saturation guard dropped the probe:

    python scripts/bench_probe.py --uids 120 --procs 8            # instant miners: the validator is the limit
    python scripts/bench_probe.py --uids 120 --procs 8 --dps 226  # each miner a 226/s machine (vLLM bf16, H100)

Run it on the validator host itself, with the validator stopped or on spare cores (miner processes need CPU too; use
about half the cores for --procs). Read the result like this:

* ``guard: tripped`` on instant miners: the event loop lagged; such a probe would record nothing. Expected at some
  load. The highest ``total`` of a probe that was NOT skipped is what the host can measure.
* With --dps, ``total`` should reach uids x dps. If it stops short while the guard is quiet, set
  ``REAX_PROBE_CEILING_DPS`` to about 0.8 x the largest total this host reached, so a probe at that load is dropped
  instead of recording numbers bounded by the validator.

Nothing here touches a chain, a wallet or the network beyond 127.0.0.1.
"""

from __future__ import annotations

import argparse
import asyncio
import multiprocessing as mp
import os
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass, field, replace

DEV = {"REAX_NETUID": "9", "REAX_NETWORK": "ws://127.0.0.1:9951", "REAX_ALLOW_DEV_MODELS": "1",
       "REAX_ALLOW_PRIVATE_IPS": "1", "REAX_EMA_ALPHA": "0.5", "REAX_BURN_FRACTION": "0.0",
       "REAX_WARMUP_ROUNDS": "0", "REAX_FEATURE_TIERED_POOLS": "1", "REAX_FEATURE_S1_PRO_POOL": "0",
       "REAX_SET_WEIGHTS": "0"}


class _Wallet:
    """Throwaway dev keypairs (//Bench/...): never a real wallet."""

    def __init__(self, uri: str):
        from bittensor.keyfiles import Keypair

        self.hotkey = Keypair.create_from_uri(uri)
        self.coldkey = Keypair.create_from_uri(uri + "/cold")
        self.coldkeypub = self.coldkey


@dataclass
class _Chain:
    neurons: list
    owner_hotkey: str
    block: int = 1000
    set_calls: list = field(default_factory=list)

    def snapshot(self):
        from reax_subnet.chain import Snapshot

        self.block += 10
        return Snapshot(block=self.block, neurons=tuple(self.neurons), owner_hotkey=self.owner_hotkey)

    def blocks_since_last_update(self, uid: int) -> int:
        return 10_000

    def weights_rate_limit(self) -> int:
        return 0

    def set_weights(self, uids, weights, version_key) -> None:
        self.set_calls.append((list(uids), list(weights), version_key))


def _miner_process(uids: list[int], dps: float, ports, ready) -> None:
    from aiohttp import web

    from reax_subnet.miner import S1Miner
    from reax_subnet.s1backends import S1HashBackend

    class Backend(S1HashBackend):
        def __init__(self):
            super().__init__("s1-fast", salt="reference")
            self.lock = asyncio.Lock()

        async def decide(self, decisions):
            if dps > 0:  # one machine per UID at dps decisions/s
                async with self.lock:
                    await asyncio.sleep(len(decisions) / dps)
            return await super().decide(decisions)

    async def main() -> None:
        vk = _Wallet("//Bench/Validator").hotkey.ss58_address
        runners = []
        for uid in uids:
            m = S1Miner(_Wallet(f"//Bench/Miner/{uid}"), Backend(), max_concurrent=16, per_caller_concurrency=16,
                        per_caller_per_minute=100_000, max_pending_per_ip=256)
            m.set_validators(frozenset({vk}))
            runner = web.AppRunner(m.app(), access_log=None)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            runners.append(runner)
            ports[uid] = site._server.sockets[0].getsockname()[1]
        ready.set()
        await asyncio.sleep(10 ** 6)

    asyncio.run(main())


async def _bench(args, ports) -> int:
    import aiohttp

    from reax_subnet import s1validator, tiers
    from reax_subnet.chain import Neuron
    from reax_subnet.config import load_validator
    from reax_subnet.manifest import commitment_text, load_manifest
    from reax_subnet.s1backends import S1HashBackend

    release = tiers.CAPACITY["s1-fast"]
    cap = replace(release, alpha=1.0, **({"window_s": args.window} if args.window else {}))
    tiers.CAPACITY = {"s1-fast": cap}
    s1validator.CAPACITY = tiers.CAPACITY
    s1validator.PROBE_EVERY_ROUNDS = 1
    owner, vw = _Wallet("//Bench/Owner"), _Wallet("//Bench/Validator")
    man = load_manifest("s1-fast")
    neurons = [Neuron(0, owner.hotkey.ss58_address, "c0", False, 0.0, None, None),
               Neuron(1, vw.hotkey.ss58_address, "c1", True, 5000.0, None, None)]
    for uid in sorted(ports):
        neurons.append(Neuron(uid, _Wallet(f"//Bench/Miner/{uid}").hotkey.ss58_address, f"c{uid}", False, 0.0,
                              "127.0.0.1", ports[uid], commitment=commitment_text("s1-fast", man.manifest_id)))
    state_dir = tempfile.mkdtemp(prefix="reax-bench-probe-")
    cfg = load_validator({**DEV, "REAX_STATE_PATH": os.path.join(state_dir, "state.json")})
    p = s1validator.POOL_PARAMS["s1-fast"]
    pools = {"s1-fast": replace(p, items_per_round=4, verified_per_round=4)}
    val = s1validator.S1Validator(cfg, vw, _Chain(neurons, owner.hotkey.ss58_address),
                                  {"s1-fast": S1HashBackend("s1-fast", salt="reference")}, pools=pools)
    print(f"{len(ports)} UIDs, miners at {'instant' if args.dps <= 0 else f'{args.dps:g}/s'}, window {cap.window_s:g} s, "
          f"{cap.inflight} lanes x {cap.items_per_request} decisions, lag limit "
          f"{1000 * s1validator.PROBE_LAG_P95_MAX_S:.0f} ms, ceiling "
          f"{s1validator._probe_ceiling_dps() or 'off'}", flush=True)
    best = 0.0
    async with aiohttp.ClientSession(trust_env=False) as session:
        await val.run_round(session)  # confirms the roles (no queries yet)
        await val.run_round(session)  # first answered round: a warm-up probe at the target budget
        val.last_probe_stats = {}
        for r in range(args.probes):
            for uid in ports:  # every UID budgeted at the pool cap
                val._probe_last[uid] = cap.cap_dps
                if val.state.miners[uid].capacity is None:
                    val.state.miners[uid].capacity = cap.cap_dps
            t0, c0 = time.monotonic(), time.process_time()
            summary = await val.run_round(session)
            wall, cpu = time.monotonic() - t0, time.process_time() - c0
            st = val.last_probe_stats
            if not st:
                print(f"probe {r + 1}: did not run", flush=True)
                continue
            per = [v["dps"] for v in (summary or {}).get("capacity", {}).values()]
            med = (f", per UID median {statistics.median(per):.0f}/s (min {min(per):.0f}, max {max(per):.0f})"
                   if per else "")
            print(f"probe {r + 1}: total {st['total_dps']:.0f} decisions/s{med}; event-loop lag p95 "
                  f"{st['lag_p95_ms']:.0f} ms; guard: {'TRIPPED (probe dropped)' if st['skipped'] else 'quiet'}; "
                  f"validator CPU {cpu:.1f} s in {wall:.1f} s", flush=True)
            if not st["skipped"]:
                best = max(best, st["total_dps"])
            val.last_probe_stats = {}
    want = len(ports) * args.dps if args.dps > 0 else None
    print(f"largest total of a probe that was kept: {best:.0f} decisions/s"
          + (f" (miners could do {want:.0f}/s)" if want else ""))
    if best:
        print(f"suggested REAX_PROBE_CEILING_DPS (only if totals stop short of the miners' speed): {0.8 * best:.0f}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uids", type=int, default=120, help="fake s1-fast UIDs (default 120)")
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="miner processes")
    ap.add_argument("--dps", type=float, default=0.0, help="decisions/s per fake machine; 0 = instant (default)")
    ap.add_argument("--probes", type=int, default=3, help="probes to run (default 3)")
    ap.add_argument("--window", type=float, default=0.0, help="override the probe window in seconds (default: release)")
    args = ap.parse_args(argv)
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    ports = manager.dict()
    uids = list(range(2, 2 + args.uids))
    procs = []
    for i in range(args.procs):
        ready = ctx.Event()
        proc = ctx.Process(target=_miner_process, args=(uids[i::args.procs], args.dps, ports, ready), daemon=True)
        proc.start()
        procs.append((proc, ready))
    try:
        for proc, ready in procs:
            if not ready.wait(300):
                print("a miner process did not start", file=sys.stderr)
                return 1
        return asyncio.run(_bench(args, dict(ports)))
    finally:
        for proc, _ready in procs:
            proc.terminate()


if __name__ == "__main__":
    sys.exit(main())
