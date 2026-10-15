#!/usr/bin/env python3
"""How many probe decisions per second can this validator host measure, and how close to each miner's real speed?
(capacity review C1; the 1.2.0rc3 "validator-bound probe" report)

Starts fake miners in separate processes (hash answers, no model, throwaway dev keys on 127.0.0.1) and runs the real
validator round (normal queries, then the capacity probe) against them with the release probe parameters, every UID
budgeted at the pool cap. Each fake machine serves its requests one at a time at its own decisions/s (a GPU), and
each request and answer can be delayed by half a round-trip time (the internet), so the bench sees what a mainnet
validator sees:

    python scripts/bench_probe.py --uids 120 --procs 8                    # instant miners: the validator is the limit
    python scripts/bench_probe.py --uids 120 --procs 8 --dps 226          # each miner a 226/s machine (vLLM bf16, H100)
    python scripts/bench_probe.py --uids 62 --pro-uids 113 --dps 30-250 --pro-dps 10-60 --rtt-ms 20-200 --trace
                                                                          # mainnet-like: 175 UIDs, mixed speed and RTT

Run it on the validator host itself, with the validator stopped or on spare cores (miner processes need CPU too; use
about half the cores for --procs). Read the result like this:

* ``guard: tripped`` on instant miners: the validator was the bottleneck; such a probe records nothing. The highest
  ``total`` of a probe that was NOT skipped is what the host can measure.
* With --dps, ``measured/true`` per UID should be close to 1 (the window loses about one round-trip time). If it
  stops short while the guard is quiet, set ``REAX_PROBE_CEILING_DPS`` to about 0.8 x the largest total this host
  reached, so a probe at that load is dropped instead of recording numbers bounded by the validator.
* --trace prints where the time goes: how late each UID's first request leaves after the window opens (send skew),
  when it reaches the miner, and how long a lane waits between an answer and its next request.
* --siblings K puts the first K s1-fast UIDs (spread over the UID range) on one machine: together they must not measure more than it.

Nothing here touches a chain, a wallet or the network beyond 127.0.0.1.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import multiprocessing as mp
import os
import random
import statistics
import sys
import tempfile
import threading
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


@dataclass(frozen=True)
class _Spec:
    uid: int
    model: str
    machine: int  # UIDs with the same machine share one backend (siblings)
    dps: float  # the machine's decisions/s; 0 = instant
    rtt_s: float  # round-trip time between validator and miner


def _range(text: str) -> tuple[float, float]:
    lo, _, hi = text.partition("-")
    return float(lo), float(hi or lo)


def _specs(args) -> list[_Spec]:
    rnd = random.Random(args.seed)
    rtt = _range(args.rtt_ms)
    models = ["s1-fast"] * args.uids + ["s1-pro"] * args.pro_uids
    rnd.shuffle(models)  # on mainnet the pools' UIDs are interleaved
    speeds = {"s1-fast": _range(args.dps), "s1-pro": _range(args.pro_dps)}
    out = []
    fast_seen = 0
    sibling_machine = None
    for i, model in enumerate(models):
        uid = 2 + i
        machine = uid
        if model == "s1-fast":
            fast_seen += 1
            if args.siblings > 1 and fast_seen <= args.siblings:
                sibling_machine = sibling_machine or uid
                machine = sibling_machine
        out.append(_Spec(uid, model, machine, rnd.uniform(*speeds[model]), rnd.uniform(*rtt) / 1000.0))
    # siblings share one machine speed
    first = {}
    for s in out:
        first.setdefault(s.machine, s.dps)
    return [replace(s, dps=first[s.machine]) for s in out]


def _miner_process(specs: list[_Spec], ports, ready, events_path: str, jitter: float, light: bool) -> None:
    from aiohttp import web

    from reax_subnet import auth
    from reax_subnet.miner import S1Miner
    from reax_subnet.s1backends import S1HashBackend

    if light:  # 20+ fake miners share one process here; a real miner checks only its own 16 lanes
        auth.verify_request = lambda headers, body, *a, **kw: auth.claimed_hotkey(headers)
    events: list[tuple[int, float, float, float]] = []  # (uid, miner app saw it, request "arrived", answer left)

    class Backend(S1HashBackend):
        def __init__(self, model: str, dps: float, lock: asyncio.Lock):
            super().__init__(model, salt="reference")
            self.dps, self.lock = dps, lock

        async def decide(self, decisions):
            if self.dps > 0:  # one GPU per machine: requests are served one at a time at dps decisions/s
                async with self.lock:
                    await asyncio.sleep(len(decisions) / self.dps)
            return await super().decide(decisions)

    def link(spec: _Spec):
        rnd = random.Random(spec.uid)

        def one_way() -> float:
            return max(0.0, spec.rtt_s / 2 * (1 + rnd.uniform(-jitter, jitter)))

        @web.middleware
        async def wan(request, handler):  # half an RTT before the miner sees the request and after its answer leaves
            entered = time.monotonic()  # this fake miner process got to it (its own loop lag is a bench artefact)
            if spec.rtt_s > 0:
                await asyncio.sleep(one_way())
            arrived = time.monotonic()
            resp = await handler(request)
            if request.path != "/health":
                events.append((spec.uid, entered, arrived, time.monotonic()))
            if spec.rtt_s > 0:
                await asyncio.sleep(one_way())
            return resp

        return wan

    async def dump() -> None:
        while True:
            await asyncio.sleep(0.5)
            if events:
                batch = events[:]
                del events[:len(batch)]
                with open(events_path, "a") as f:
                    for e in batch:
                        f.write(json.dumps(e) + "\n")

    async def main() -> None:
        vk = _Wallet("//Bench/Validator").hotkey.ss58_address
        locks: dict[int, asyncio.Lock] = {}
        runners = []
        for spec in specs:
            backend = Backend(spec.model, spec.dps, locks.setdefault(spec.machine, asyncio.Lock()))
            m = S1Miner(_Wallet(f"//Bench/Miner/{spec.uid}"), backend, max_concurrent=16, per_caller_concurrency=16,
                        per_caller_per_minute=100_000, max_pending_per_ip=256)
            m.set_validators(frozenset({vk}))
            app = m.app()
            app.middlewares.append(link(spec))
            runner = web.AppRunner(app, access_log=None)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            runners.append(runner)
            ports[spec.uid] = site._server.sockets[0].getsockname()[1]
        ready.set()
        asyncio.ensure_future(dump())
        await asyncio.sleep(10 ** 6)

    asyncio.run(main())


async def _loop_load(busy_ms: float, period_ms: float) -> None:
    """Other round work on the validator's event loop: busy_ms of CPU every period_ms."""
    while True:
        end = time.perf_counter() + busy_ms / 1000
        while time.perf_counter() < end:
            pass
        await asyncio.sleep(period_ms / 1000)


def _gil_load(stop: threading.Event, busy_ms: float, period_ms: float) -> None:
    """A Python worker thread (reference checks, chain reads) holding the GIL busy_ms out of every period_ms."""
    while not stop.is_set():
        end = time.perf_counter() + busy_ms / 1000
        while time.perf_counter() < end:
            pass
        time.sleep(period_ms / 1000)


def _pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))] if xs else float("nan")


def _read_events(paths) -> dict[int, list[tuple[float, float, float]]]:
    out: dict[int, list[tuple[float, float, float]]] = {}
    for p in paths:
        with open(p) as f:
            for line in f:
                uid, e, a, b = json.loads(line)
                out.setdefault(uid, []).append((e, a, b))
        os.truncate(p, 0)
    return out


def _trace_report(trace: dict, miner_events: dict, specs: dict[int, _Spec], order: list[int]) -> None:
    started = trace["started"]
    first_send, first_arrival, turn = {}, {}, []
    for uid, ev in trace["uids"].items():
        if ev:
            first_send[uid] = min(e[1] for e in ev) - started
        by_lane: dict[int, list] = {}
        for e in sorted(ev, key=lambda e: e[1]):
            by_lane.setdefault(e[0], []).append(e)
        for lane in by_lane.values():
            for prev, nxt in zip(lane, lane[1:]):
                turn.append(nxt[1] - prev[2])  # answer received -> next request handed to aiohttp
    miner_lag, gaps = [], []
    for uid, ev in miner_events.items():
        ev = sorted(e for e in ev if e[0] >= started - 0.5)
        if not ev:
            continue
        first_arrival[uid] = ev[0][1] - started
        if uid in first_send:
            miner_lag.append(ev[0][0] - started - first_send[uid])
        # the j-th answer to reach the validator's socket (left the miner + RTT/2) is the j-th one its lanes read:
        # the difference is how long the answer waited for the validator's event loop
        got = sorted(e[2] for e in trace["uids"].get(uid, []) if e[3])
        reached = sorted(e[2] + specs[uid].rtt_s / 2 for e in ev)
        gaps.extend(g - r for r, g in zip(reached, got))
    fs = [first_send[u] for u in order if u in first_send]
    fa = [first_arrival[u] for u in order if u in first_arrival]
    print(f"  trace: first send after the probe opened  p50 {1000 * _pct(fs, .5):.0f} ms, p95 {1000 * _pct(fs, .95):.0f} ms,"
          f" max {1000 * max(fs, default=0):.0f} ms (UIDs in probe order: first {1000 * fs[0]:.0f} ms, last "
          f"{1000 * fs[-1]:.0f} ms)" if fs else "  trace: no sends")
    if fa:
        print(f"  trace: first request at the miner p50 {1000 * _pct(fa, .5):.0f} ms, max {1000 * max(fa):.0f} ms after "
              f"the probe opened (incl. RTT/2; fake-miner process lag p50 {1000 * _pct(miner_lag, .5):.0f} ms, max "
              f"{1000 * max(miner_lag, default=0):.0f} ms is a bench artefact)")
    if gaps:
        print(f"  trace: answer at the validator's socket -> read by its lane (event-loop wait): p50 "
              f"{1000 * _pct(gaps, .5):.0f} ms, p95 {1000 * _pct(gaps, .95):.0f} ms, max {1000 * max(gaps):.0f} ms")
    if turn:
        print(f"  trace: answer read -> next request sent on that lane  p50 {1000 * _pct(turn, .5):.1f} ms, p95 "
              f"{1000 * _pct(turn, .95):.1f} ms ({len(turn)} turns)")


async def _bench(args, ports, specs: list[_Spec], events_glob: str) -> int:
    import aiohttp

    from reax_subnet import s1validator, tiers
    from reax_subnet.chain import Neuron
    from reax_subnet.config import load_validator
    from reax_subnet.manifest import commitment_text, load_manifest
    from reax_subnet.s1backends import S1HashBackend

    models = sorted({s.model for s in specs})
    caps = {}
    for m in models:
        release = tiers.CAPACITY[m]
        caps[m] = replace(release, alpha=1.0, **({"window_s": args.window} if args.window else {}))
    tiers.CAPACITY = caps
    s1validator.CAPACITY = tiers.CAPACITY
    s1validator.PROBE_EVERY_ROUNDS = 1
    owner, vw = _Wallet("//Bench/Owner"), _Wallet("//Bench/Validator")
    mans = {m: load_manifest(m) for m in models}
    neurons = [Neuron(0, owner.hotkey.ss58_address, "c0", False, 0.0, None, None),
               Neuron(1, vw.hotkey.ss58_address, "c1", True, 5000.0, None, None)]
    by_uid = {s.uid: s for s in specs}
    for uid in sorted(ports):
        m = by_uid[uid].model
        neurons.append(Neuron(uid, _Wallet(f"//Bench/Miner/{uid}").hotkey.ss58_address, f"c{uid}", False, 0.0,
                              "127.0.0.1", ports[uid], commitment=commitment_text(m, mans[m].manifest_id)))
    state_dir = tempfile.mkdtemp(prefix="reax-bench-probe-")
    env = {**DEV, "REAX_STATE_PATH": os.path.join(state_dir, "state.json"),
           "REAX_FEATURE_S1_PRO_POOL": "1" if "s1-pro" in models else "0"}
    cfg = load_validator(env)
    pools = {m: replace(s1validator.POOL_PARAMS[m], items_per_round=4, verified_per_round=4)
             for m in ("s1-fast", "s1-pro") if m in models or m == "s1-fast"}
    backends = {m: S1HashBackend(m, salt="reference") for m in pools}
    val = s1validator.S1Validator(cfg, vw, _Chain(neurons, owner.hotkey.ss58_address), backends, pools=pools)
    val.probe_trace = args.trace
    for m in models:
        cap = caps[m]
        n = sum(1 for s in specs if s.model == m)
        print(f"{m}: {n} UIDs, machines at {args.dps if m == 's1-fast' else args.pro_dps}/s, window {cap.window_s:g} s, "
              f"{cap.inflight} lanes x {cap.items_per_request} decisions", flush=True)
    print(f"RTT {args.rtt_ms} ms, siblings {args.siblings}, loop load {args.loop_load or 'off'}, GIL thread "
          f"{args.gil_load or 'off'}, lag limit {1000 * s1validator.PROBE_LAG_P95_MAX_S:.0f} ms, ceiling "
          f"{s1validator._probe_ceiling_dps() or 'off'}", flush=True)
    best = 0.0
    ratios_all: list[float] = []
    load = stop = None
    if args.loop_load:
        busy, period = _range(args.loop_load.replace("/", "-"))
        load = asyncio.ensure_future(_loop_load(busy, period))
    if args.gil_load:
        busy, period = _range(args.gil_load.replace("/", "-"))
        stop = threading.Event()
        threading.Thread(target=_gil_load, args=(stop, busy, period), daemon=True).start()
    async with aiohttp.ClientSession(trust_env=False) as session:
        await val.run_round(session)  # confirms the roles (no queries yet)
        await val.run_round(session)  # first answered round: a warm-up probe at the target budget
        val.last_probe_stats = {}
        await asyncio.sleep(1.0)
        _read_events(glob.glob(events_glob))
        for r in range(args.probes):
            for s in specs:  # every UID budgeted at the pool cap
                cap = caps[s.model]
                val._probe_last[s.uid] = cap.cap_dps
                if val.state.miners[s.uid].capacity is None:
                    val.state.miners[s.uid].capacity = cap.cap_dps
            t0, c0 = time.monotonic(), time.process_time()
            if args.profile:
                import cProfile
                import pstats

                prof = cProfile.Profile()
                orig = val._probe_capacity

                async def profiled(*a, **kw):
                    prof.enable()
                    try:
                        return await orig(*a, **kw)
                    finally:
                        prof.disable()

                val._probe_capacity = profiled
            summary = await val.run_round(session)
            wall, cpu = time.monotonic() - t0, time.process_time() - c0
            if args.profile:
                val._probe_capacity = orig
                pstats.Stats(prof).sort_stats("tottime").print_stats(25)
            st = val.last_probe_stats
            if not st:
                print(f"probe {r + 1}: did not run", flush=True)
                continue
            report = (summary or {}).get("capacity", {})
            print(f"probe {r + 1}: total {st['total_dps']:.0f} decisions/s; event-loop lag p95 {st['lag_p95_ms']:.0f} ms"
                  + (f", send skew {st['send_skew_ms']:.0f} ms" if "send_skew_ms" in st else "")
                  + f"; guard: {'TRIPPED (probe dropped)' if st['skipped'] else 'quiet'}; validator CPU {cpu:.1f} s "
                  f"in {wall:.1f} s", flush=True)
            for m in models:
                rows = []
                for s in specs:
                    if s.model != m or s.uid not in report:
                        continue
                    if sum(1 for o in specs if o.machine == s.machine) > 1:
                        continue  # siblings are reported together below
                    true = min(caps[m].cap_dps, s.dps) if s.dps > 0 else None
                    if true:
                        rows.append((report[s.uid]["dps"] / true, s))
                if rows:
                    rs = [x for x, _ in rows]
                    ratios_all.extend(rs)
                    near = [x for x, s in rows if s.rtt_s <= 0.06]
                    far = [x for x, s in rows if s.rtt_s >= 0.15]
                    order = sorted(rows, key=lambda t: t[1].uid)
                    k = max(1, len(order) // 4)
                    print(f"  {m}: measured/true median {statistics.median(rs):.2f}, p10 {_pct(rs, .1):.2f}, min "
                          f"{min(rs):.2f}, max {max(rs):.2f}; RTT<=60ms {statistics.median(near) if near else float('nan'):.2f},"
                          f" RTT>=150ms {statistics.median(far) if far else float('nan'):.2f}; first quarter of UIDs "
                          f"{statistics.median(x for x, _ in order[:k]):.2f}, last quarter "
                          f"{statistics.median(x for x, _ in order[-k:]):.2f}", flush=True)
            if args.siblings > 1:
                sib = [s for s in specs if sum(1 for o in specs if o.machine == s.machine) > 1]
                got = sum(report.get(s.uid, {}).get("dps", 0.0) for s in sib)
                print(f"  siblings: {len(sib)} UIDs on one {sib[0].dps:.0f}/s machine measured {got:.0f}/s together "
                      f"({got / sib[0].dps:.2f} x the machine)", flush=True)
            if args.trace and val.last_probe_trace:
                await asyncio.sleep(1.0)
                miner_events = _read_events(glob.glob(events_glob))
                order = [n.uid for n in neurons if n.uid in by_uid]
                _trace_report(val.last_probe_trace, miner_events, by_uid, order)
            if not st["skipped"]:
                best = max(best, st["total_dps"])
            val.last_probe_stats = {}
    if load is not None:
        load.cancel()
    if stop is not None:
        stop.set()
    want = sum(min(caps[s.model].cap_dps, s.dps) for s in specs if s.machine == s.uid) \
        if all(s.dps > 0 for s in specs) else None
    print(f"largest total of a probe that was kept: {best:.0f} decisions/s"
          + (f" (miners could do {want:.0f}/s)" if want else ""))
    if ratios_all:
        print(f"all probes: measured/true median {statistics.median(ratios_all):.3f}, p10 {_pct(ratios_all, .1):.3f}, "
              f"min {min(ratios_all):.3f}")
    if best:
        print(f"suggested REAX_PROBE_CEILING_DPS (only if totals stop short of the miners' speed): {0.8 * best:.0f}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uids", type=int, default=120, help="fake s1-fast UIDs (default 120)")
    ap.add_argument("--pro-uids", type=int, default=0, help="fake s1-pro UIDs (default 0), probed in the same probe")
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="miner processes")
    ap.add_argument("--dps", default="0", help="s1-fast decisions/s per machine, N or LO-HI; 0 = instant (default)")
    ap.add_argument("--pro-dps", default="0", help="s1-pro decisions/s per machine, N or LO-HI (default instant)")
    ap.add_argument("--rtt-ms", default="0", help="round-trip time per miner in ms, N or LO-HI (default 0)")
    ap.add_argument("--jitter", type=float, default=0.1, help="one-way delay jitter as a fraction (default 0.1)")
    ap.add_argument("--siblings", type=int, default=0, help="the first K s1-fast UIDs share one machine")
    ap.add_argument("--loop-load", default="", help="BUSY/PERIOD ms of other work on the validator loop, e.g. 3/10")
    ap.add_argument("--gil-load", default="", help="BUSY/PERIOD ms of a Python thread holding the GIL, e.g. 2/10")
    ap.add_argument("--trace", action="store_true", help="per-lane timing: send skew, arrival at the miner, turns")
    ap.add_argument("--full-miners", action="store_true",
                    help="fake miners check request signatures too (default: skipped, their CPU is a bench artefact)")
    ap.add_argument("--profile", action="store_true", help="cProfile the probe (slows it; for the breakdown only)")
    ap.add_argument("--probes", type=int, default=3, help="probes to run (default 3)")
    ap.add_argument("--window", type=float, default=0.0, help="override the probe window in seconds (default: release)")
    ap.add_argument("--seed", type=int, default=1, help="seed for the per-UID speed and RTT")
    args = ap.parse_args(argv)
    specs = _specs(args)
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    ports = manager.dict()
    events_dir = tempfile.mkdtemp(prefix="reax-bench-events-")
    machines = sorted({s.machine for s in specs})
    procs = []
    for i in range(args.procs):  # siblings stay in one process (they share one backend lock)
        mine = set(machines[i::args.procs])
        group = [s for s in specs if s.machine in mine]
        if not group:
            continue
        ready = ctx.Event()
        proc = ctx.Process(target=_miner_process, args=(group, ports, ready, os.path.join(events_dir, f"{i}.jsonl"),
                                                         args.jitter, not args.full_miners), daemon=True)
        proc.start()
        procs.append((proc, ready))
    try:
        for proc, ready in procs:
            if not ready.wait(300):
                print("a miner process did not start", file=sys.stderr)
                return 1
        return asyncio.run(_bench(args, dict(ports), specs, os.path.join(events_dir, "*.jsonl")))
    finally:
        for proc, _ready in procs:
            proc.terminate()


if __name__ == "__main__":
    sys.exit(main())
