"""Capacity probe shards (1.2.0rc5, probeshard.py): on mainnet the probe's one event loop sat at 100 % of a core for
the whole window and clipped fast miners. The window can now run in forked worker processes; these tests check the
shard plan, that a sharded probe measures like the one-process probe (siblings included), and that a failing or
hanging shard drops the whole probe instead of crediting only the other shards' UIDs.

The probe tests reuse the fake-machine arena of test_tiers.py (real aiohttp miners on 127.0.0.1, a faked chain); the
miners keep running on the test's event loop while the shards send to them.
"""

from __future__ import annotations

import asyncio
import math
import os
import sys
import threading
import time

import pytest

from reax_subnet import probeshard, s1validator
from reax_subnet.chain import Neuron
from reax_subnet.s1validator import S1Validator
from test_tiers import (PROBE_V2, Machine, _probe_arena, _release_floors_off, probe_v2)  # noqa: F401 (fixtures)

linux_only = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="probe shards need Linux fork")


def _n(uid: int, coldkey: str, ip: str) -> Neuron:
    return Neuron(uid, f"hk{uid}", coldkey, False, 0.0, ip, 8000 + uid)


def test_shard_count_follows_the_setting_the_lanes_and_the_cores(monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(30)))
    monkeypatch.setattr(sys, "platform", "linux")
    lanes = 85 * 16 + 145 * 16 + 10 * 8  # the 1.2.0rc4 mainnet probe
    assert probeshard.shard_count(lanes, 240, {}) == 20  # 192 lanes per shard
    assert probeshard.shard_count(lanes, 240, {"REAX_PROBE_SHARDS": "auto"}) == 20
    assert probeshard.shard_count(lanes, 240, {"REAX_PROBE_SHARDS": "1"}) == 1  # the one-process probe
    assert probeshard.shard_count(lanes, 240, {"REAX_PROBE_SHARDS": "6"}) == 6
    assert probeshard.shard_count(lanes, 4, {"REAX_PROBE_SHARDS": "6"}) == 4  # never more shards than UIDs
    assert probeshard.shard_count(lanes, 240, {"REAX_PROBE_SHARDS": "nonsense"}) == 20
    assert probeshard.shard_count(6 * 16, 6, {}) == 1  # small probes stay in the validator's process
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(8)))
    assert probeshard.shard_count(lanes, 240, {}) == 6  # two cores stay with the validator
    monkeypatch.setattr(sys, "platform", "darwin")
    assert probeshard.shard_count(lanes, 240, {"REAX_PROBE_SHARDS": "6"}) == 1


def test_one_coldkey_shares_a_shard_and_shards_are_balanced():
    targets = [_n(2, "ck-a", "93.184.216.34"), _n(3, "ck-b", "151.101.1.69"), _n(4, "ck-a", "151.101.65.69"),
               _n(5, "ck-c", "151.101.1.69"), _n(6, "ck-d", "127.0.0.1"), _n(7, "ck-e", "127.0.0.1"),
               _n(8, "ck-f", "10.0.0.8"), _n(9, "ck-g", "10.0.0.8")]
    groups = [[n.uid for n in g] for g in probeshard.sibling_groups(targets)]
    assert groups == [[2, 4], [3], [5], [6], [7], [8], [9]]  # coldkey only: a declared IP never groups (review H1)
    plan = probeshard.plan_shards(targets, lambda n: 16.0, 3)
    uids = [[n.uid for n in s] for s in plan]
    assert sorted(u for s in uids for u in s) == [2, 3, 4, 5, 6, 7, 8, 9]
    assert any({2, 4} <= set(s) for s in uids)
    assert all(s == sorted(s) for s in uids)  # target order inside a shard
    assert max(map(len, uids)) - min(map(len, uids)) <= 1
    assert [len(g) for g in probeshard.sibling_groups([_n(10 + i, "", None) for i in range(3)])] == [1, 1, 1]


def _live_like(groups: dict[str, int], singles: int) -> list[Neuron]:
    """A metagraph like SN92 on 2026-10-15: a few multi-UID coldkeys (33, 21, 20, 16 UIDs) and many single UIDs."""
    out, uid = [], 2
    for ck, size in groups.items():
        for _ in range(size):
            out.append(_n(uid, ck, f"203.0.{uid // 250}.{uid % 250}"))
            uid += 1
    for i in range(singles):
        out.append(_n(uid, f"solo{i}", f"198.51.{uid // 250}.{uid % 250}"))
        uid += 1
    return out


def test_a_big_coldkey_group_is_split_over_shards():
    """Review H1: the live metagraph has a 33-UID coldkey. A group heavier than a fair shard is split, so the busiest
    shard stays near the average load (the common start and the skew guard bound siblings in different shards)."""
    targets = _live_like({"big": 33, "b": 21, "c": 20, "d": 16, "e": 40}, 110)
    for weight in (lambda n: 1.0, lambda n: 1.0 + (n.uid % 7)):  # equal UIDs, and uneven expected requests/s
        plan = probeshard.plan_shards(targets, weight, 20)
        loads = [sum(weight(n) for n in s) for s in plan]
        mean = sum(loads) / 20
        assert len(plan) == 20 and max(loads) <= 1.5 * mean, (max(loads), mean)
        assert sorted(n.uid for s in plan for n in s) == sorted(n.uid for n in targets)
        for ck in ("big", "e"):
            shards_used = sum(1 for s in plan if any(n.coldkey == ck for n in s))
            assert shards_used >= 3, (ck, shards_used)
    small = [n for n in targets if n.coldkey == "d"]  # a group within a fair share stays together
    plan = probeshard.plan_shards(small + targets[-60:], lambda n: 1.0, 4)
    assert sum(1 for s in plan if any(n.coldkey == "d" for n in s)) == 1


def test_sybils_declaring_competitors_ips_cannot_merge_them_into_one_shard():
    """Review H1 attack: 6 cheap UIDs of one coldkey declare the IPs of the big competitor groups (axon IPs are
    self-declared on chain). In 1.2.0rc5-rc6 union-find over coldkey OR IP glued all of them into one 118-UID shard."""
    targets = _live_like({"v1": 33, "v2": 21, "v3": 20, "v4": 16}, 144)
    victims_ip = {ck: next(n.ip for n in targets if n.coldkey == ck) for ck in ("v1", "v2", "v3", "v4")}
    declared = list(victims_ip.values()) + list(victims_ip.values())[:2]
    sybils = [_n(900 + i, "attacker", ip) for i, ip in enumerate(declared)]
    targets += sybils
    groups = probeshard.sibling_groups(targets)
    assert max(len(g) for g in groups) == 33  # nothing merged beyond the biggest real coldkey
    assert [n.uid for g in groups for n in g if n.coldkey == "attacker"] == [n.uid for n in sybils]
    assert all(len({n.coldkey for n in g}) == 1 for g in groups)
    plan = probeshard.plan_shards(targets, lambda n: 1.0, 20)
    sizes = [len(s) for s in plan]
    assert max(sizes) <= 1.5 * len(targets) / 20, sizes
    # the victims (90 UIDs) are spread out, not one shard of them
    assert max(sum(1 for n in s if n.coldkey.startswith("v")) for s in plan) <= 1.5 * len(targets) / 20


def test_one_coldkey_with_many_uids_still_gets_shards(monkeypatch):
    """The shard count is bounded by the UIDs, not the coldkeys: one operator's 100 UIDs are not one loop."""
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(30)))
    monkeypatch.setattr(sys, "platform", "linux")
    targets = [_n(2 + i, "one-operator", None) for i in range(100)]
    shards = probeshard.shard_count(100 * 16, len(targets), {})
    assert shards == 9
    plan = probeshard.plan_shards(targets, lambda n: 1.0, shards)
    assert len(plan) == 9 and max(map(len, plan)) <= 12


@linux_only
async def test_a_sharded_probe_measures_like_one_process(probe_v2, make_wallet, fake_chain_cls, tmp_path,
                                                         monkeypatch, caplog):
    probe_v2()
    monkeypatch.setenv("REAX_PROBE_SHARDS", "3")
    groups = [(Machine(160.0), [{}]) for _ in range(6)]
    with caplog.at_level("INFO", logger="reax.s1validator"):
        val, by_group, summaries, _ = await _probe_arena(make_wallet, fake_chain_cls, tmp_path, groups)
    stats = val.last_probe_stats
    assert stats["shards"] == 3 and not stats["skipped"], stats
    assert stats["send_skew_ms"] < 1000 * s1validator.PROBE_SEND_SKEW_MAX_S
    assert stats["total_rps"] > 0 and 0.0 < stats["cpu_share"] <= 1.5 and "cpu_saturated" in stats
    report = summaries[-1]["capacity"]
    for g in by_group.values():
        assert report[g[0]]["dps"] >= 0.6 * 160.0, report
        assert not report[g[0]]["invalid"]
    assert "capacity probe: 6 UIDs in 3 process(es)" in caplog.text


@linux_only
async def test_siblings_in_different_shards_still_share_one_machine(probe_v2, make_wallet, fake_chain_cls, tmp_path,
                                                                    monkeypatch):
    """Loopback siblings with different coldkeys are not grouped, so they can land in different shards: the common
    start keeps them loaded together, and together they measure no more than the machine."""
    probe_v2()
    monkeypatch.setenv("REAX_PROBE_SHARDS", "4")
    shared = Machine(160.0)
    groups = [(shared, [{}])] + [(Machine(160.0), [{}]) for _ in range(6)] + [(shared, [{}])]
    seen = []
    plan = probeshard.plan_shards

    def spy(targets, lanes, shards):
        out = plan(targets, lanes, shards)
        seen.append([[n.uid for n in s] for s in out])
        return out

    monkeypatch.setattr(probeshard, "plan_shards", spy)
    val, by_group, summaries, _ = await _probe_arena(make_wallet, fake_chain_cls, tmp_path, groups)
    a, b = by_group[0][0], by_group[7][0]
    assert not any(a in s and b in s for s in seen[-1]), seen[-1]  # really in different shards
    report = summaries[-1]["capacity"]
    pair = report[a]["dps"] + report[b]["dps"]
    solo = max(report[by_group[g][0]]["dps"] for g in range(1, 7))
    skew = val.last_probe_stats["send_skew_ms"] / 1000
    assert pair <= solo * (1 + skew / PROBE_V2.window_s) * 1.05 + 2.0, (pair, solo, skew)


@linux_only
@pytest.mark.parametrize("how", ["crash", "hang", "raise"])
async def test_a_failing_shard_drops_the_whole_probe(how, probe_v2, make_wallet, fake_chain_cls, tmp_path,
                                                     monkeypatch, caplog):
    probe_v2()
    monkeypatch.setenv("REAX_PROBE_SHARDS", "3")
    monkeypatch.setattr(probeshard, "PROBE_SHARD_RESULT_SLACK_S", 1.0)
    parent = os.getpid()
    window = S1Validator._probe_window
    victim = []

    async def failing(self, targets, *a, **kw):
        if os.getpid() != parent and victim and victim[0] in {n.uid for n in targets}:
            if how == "crash":
                os._exit(3)
            if how == "hang":
                time.sleep(60)
            raise RuntimeError("shard broke")
        return await window(self, targets, *a, **kw)

    monkeypatch.setattr(S1Validator, "_probe_window", failing)
    groups = [(Machine(160.0), [{}]) for _ in range(6)]
    victim.append(2)  # the first miner UID of the arena
    started = time.monotonic()
    with caplog.at_level("ERROR", logger="reax.s1validator"):
        val, by_group, summaries, _ = await _probe_arena(make_wallet, fake_chain_cls, tmp_path, groups, rounds=2)
    assert summaries[-1]["capacity"] == {}  # nobody is credited, also not the UIDs of the healthy shards
    assert all(val.state.miners[u].capacity is None for g in by_group.values() for u in g)
    assert val.last_probe_stats["skipped"]
    assert "capacity probe skipped" in caplog.text and "shard" in caplog.text
    assert time.monotonic() - started < 40.0
    with pytest.raises(ChildProcessError):  # every shard was reaped (none left running or as a zombie)
        os.waitpid(-1, os.WNOHANG)


async def test_a_saturated_window_loop_drops_the_whole_probe(probe_v2, make_wallet, fake_chain_cls, tmp_path,
                                                              monkeypatch, caplog):
    """Review M2: a loop at ~100 % of a core read s1-fast ~0.00 on the live host while the lag (1-2 ms) and skew guards
    stayed quiet. cpu_saturated now drops the probe: capacities are kept, nobody is measured at what the loop left."""
    probe_v2()
    monkeypatch.setattr(s1validator, "PROBE_CPU_SATURATED", 0.0)
    groups = [(Machine(160.0), [{}]) for _ in range(3)]
    with caplog.at_level("INFO", logger="reax.s1validator"):
        val, by_group, summaries, _ = await _probe_arena(make_wallet, fake_chain_cls, tmp_path, groups, rounds=2)
    stats = val.last_probe_stats
    assert stats["shards"] == 1 and stats["cpu_saturated"] and stats["skipped"]
    assert summaries[-1]["capacity"] == {}
    assert all(val.state.miners[u].capacity is None for g in by_group.values() for u in g)
    assert "(cpu_saturated)" in caplog.text and "busiest window loop at" in caplog.text


@linux_only
async def test_one_saturated_shard_drops_the_whole_probe_and_shards_are_logged(probe_v2, make_wallet, fake_chain_cls,
                                                                               tmp_path, monkeypatch, caplog):
    probe_v2()
    monkeypatch.setenv("REAX_PROBE_SHARDS", "3")
    parent, window = os.getpid(), S1Validator._probe_window

    async def one_busy(self, targets, *a, **kw):
        win = await window(self, targets, *a, **kw)
        if os.getpid() != parent and 2 in {n.uid for n in targets}:
            win.cpu_share = 0.97  # this shard's loop was saturated
        return win

    monkeypatch.setattr(S1Validator, "_probe_window", one_busy)
    groups = [(Machine(160.0), [{}]) for _ in range(6)]
    with caplog.at_level("INFO", logger="reax.s1validator"):
        val, by_group, summaries, _ = await _probe_arena(make_wallet, fake_chain_cls, tmp_path, groups, rounds=2)
    stats = val.last_probe_stats
    assert stats["skipped"] and stats["cpu_saturated"] and summaries[-1]["capacity"] == {}
    assert stats["shard_uids"] == [2, 2, 2] and abs(sum(stats["shard_load"]) - 1.0) < 0.01
    assert len(stats["shard_cpu"]) == 3 and max(stats["shard_cpu"]) == 0.97
    assert "capacity probe shards: UIDs [2, 2, 2]" in caplog.text


async def test_answers_do_not_turn_stale_while_the_validator_checks_them(probe_v2, make_wallet, fake_chain_cls,
                                                                         tmp_path, monkeypatch, caplog):
    """Answers are checked after the slowest pool's window and one after the other; in 1.2.0rc4 an s1-fast answer
    that waited longer than its own pool's window + latency bounds + 10 s was 'stale' and the UID measured 0 (seen in
    the bench with s1-lite's 16 s window in the same probe). The allowed age now covers the probe's own duration."""
    probe_v2()
    monkeypatch.setattr(s1validator, "PROBE_RESPONSE_AGE_SLACK_S", 0.5)
    per_pool = PROBE_V2.window_s + 2 * 4.0  # the arena's latency_slow_s
    parse, slept = s1validator.parse_s1_request, []

    def slow_parse(*a, **kw):
        if not slept and threading.current_thread() is not threading.main_thread():  # the probe's check thread
            slept.append(True)
            time.sleep(per_pool + 1.0)  # the validator is slow: longer than the pool's bounds + the slack
        return parse(*a, **kw)

    monkeypatch.setattr(s1validator, "parse_s1_request", slow_parse)
    groups = [(Machine(160.0), [{}]) for _ in range(2)]
    with caplog.at_level("WARNING", logger="reax.s1validator"):
        val, by_group, summaries, _ = await _probe_arena(make_wallet, fake_chain_cls, tmp_path, groups, rounds=2)
    assert slept
    assert "StaleRequest" not in caplog.text
    assert all(summaries[-1]["capacity"][g[0]]["dps"] > 0 for g in by_group.values())


def test_shards_are_balanced_by_expected_requests():
    targets = [_n(2 + i, f"ck{i}", None) for i in range(6)]
    rate = {2: 40.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 38.0}  # two fast UIDs, four slow ones
    plan = probeshard.plan_shards(targets, lambda n: rate[n.uid], 2)
    loads = [sum(rate[n.uid] for n in s) for s in plan]
    assert max(loads) - min(loads) <= 1.0, loads
    assert all(any(n.uid in (2, 7) for n in s) for s in plan)  # the two fast UIDs on different shards


def test_a_shard_result_can_only_hold_plain_values():
    """The parent unpickles a shard's result with an allow-list: ProbeWindow and builtin values, nothing else."""
    import io
    import pickle

    good = probeshard.ProbeWindow(results=[(2, [(0, b"raw", [("X-A", "1"), ("X-A", "2")])], 5.0, 16, 1.0, 1)])
    got = probeshard._ResultUnpickler(io.BytesIO(pickle.dumps(good, protocol=pickle.HIGHEST_PROTOCOL))).load()
    assert got.results == good.results

    class Evil:
        def __reduce__(self):
            return (os.system, ("true",))

    with pytest.raises(pickle.UnpicklingError):
        probeshard._ResultUnpickler(io.BytesIO(pickle.dumps(probeshard.ProbeWindow(results=[Evil()])))).load()


def _fds() -> int:
    return len(os.listdir("/proc/self/fd"))


async def _plain_window(ready, go):
    from multidict import CIMultiDict

    ready()
    await go()
    return probeshard.ProbeWindow(results=[(2, [(0, b"raw", CIMultiDict([("X-A", "1")]))], 5.0, 16, time.monotonic(),
                                            1)])


@linux_only
async def test_shards_leave_no_file_descriptors_or_processes_behind():
    before = _fds()
    got = await probeshard.run_shards([_plain_window] * 3, ready_timeout_s=5.0, result_timeout_s=5.0)
    assert [w.results[0][1][0][1] for w in got] == [b"raw"] * 3
    assert dict(got[0].results[0][1][0][2]) == {"X-A": "1"}

    async def broken(ready, go):
        raise RuntimeError("broken shard")

    failed = await probeshard.run_shards([_plain_window, broken], ready_timeout_s=5.0, result_timeout_s=5.0)
    assert len(failed) == 1 and "broken shard" in failed[0]
    await asyncio.sleep(0.1)  # a closed pipe transport releases its fd on the next loop pass
    assert _fds() == before
    with pytest.raises(ChildProcessError):
        os.waitpid(-1, os.WNOHANG)


@linux_only
async def test_a_probe_cancelled_while_forking_reaps_the_forked_shard(monkeypatch):
    """The probe's time budget may cancel it between a fork and the shard's registration: that shard is killed and
    reaped too, not left as a zombie."""
    loop = asyncio.get_running_loop()
    connect, calls = loop.connect_read_pipe, []

    async def cancelled_second(*a, **kw):
        calls.append(1)
        if len(calls) == 2:
            raise asyncio.CancelledError
        return await connect(*a, **kw)

    monkeypatch.setattr(loop, "connect_read_pipe", cancelled_second)
    before = _fds()
    with pytest.raises(asyncio.CancelledError):
        await probeshard.run_shards([_plain_window] * 3, ready_timeout_s=5.0, result_timeout_s=5.0)
    assert len(calls) == 2
    await asyncio.sleep(0.1)  # a closed pipe transport releases its fd on the next loop pass
    assert _fds() == before
    with pytest.raises(ChildProcessError):
        os.waitpid(-1, os.WNOHANG)


def test_the_shard_unpickler_refuses_every_other_global():
    import io
    import pickle

    for module, name in (("builtins", "eval"), ("copyreg", "_reconstructor"), ("posix", "system"),
                         ("reax_subnet.probeshard", "_child")):
        payload = pickle.PROTO + b"\x04" + pickle.GLOBAL + f"{module}\n{name}\n".encode() + pickle.STOP
        with pytest.raises(pickle.UnpicklingError):
            probeshard._ResultUnpickler(io.BytesIO(payload)).load()


@linux_only
async def test_a_shard_that_exits_between_ready_and_go_is_named():
    """Review L1: the go message to a shard that already exited is a clear failure, not a broken pipe."""
    async def gone(ready, go):
        ready()
        os._exit(0)

    async def late(ready, go):
        await asyncio.sleep(0.5)  # the other shard has exited by now
        ready()
        await go()
        return probeshard.ProbeWindow(results=[])

    failed = await probeshard.run_shards([gone, late], ready_timeout_s=5.0, result_timeout_s=5.0)
    assert failed == ["shard 0 exited before the window opened (BrokenPipeError)"], failed
    with pytest.raises(ChildProcessError):
        os.waitpid(-1, os.WNOHANG)


@linux_only
async def test_a_shard_writes_no_core_dump_and_cannot_be_traced():
    """Review L3: RLIMIT_CORE 0 and not dumpable in every shard; the validator process itself is unchanged."""
    import ctypes
    import resource

    async def report(ready, go):
        ready()
        await go()
        dumpable = ctypes.CDLL(None).prctl(3, 0, 0, 0, 0)  # PR_GET_DUMPABLE
        return probeshard.ProbeWindow(results=[], failed=[f"{resource.getrlimit(resource.RLIMIT_CORE)} {dumpable}"])

    before = resource.getrlimit(resource.RLIMIT_CORE)
    got = await probeshard.run_shards([report, report], ready_timeout_s=5.0, result_timeout_s=5.0)
    assert [w.failed for w in got] == [["(0, 0) 0"], ["(0, 0) 0"]]
    assert resource.getrlimit(resource.RLIMIT_CORE) == before


async def test_the_outer_probe_budget_covers_the_shard_slack(probe_v2, make_wallet, fake_chain_cls, tmp_path,
                                                             monkeypatch):
    """Review L2: the shards' own start, go and result slack fit inside the probe's outer time budget, so a hanging
    shard is reported by its own timeout first."""
    probe_v2()
    seen, real = [], asyncio.wait_for

    async def spy(aw, timeout, *a, **kw):
        if getattr(aw, "__qualname__", "").endswith("_probe_capacity"):
            seen.append(timeout)
        return await real(aw, timeout, *a, **kw)

    monkeypatch.setattr(s1validator.asyncio, "wait_for", spy)
    groups = [(Machine(160.0), [{}]) for _ in range(2)]
    await _probe_arena(make_wallet, fake_chain_cls, tmp_path, groups, rounds=2)
    assert seen
    slack = (probeshard.PROBE_SHARD_START_SLACK_S + probeshard.PROBE_SHARD_GO_DELAY_S
             + probeshard.PROBE_SHARD_RESULT_SLACK_S)
    floor = s1validator.PROBE_PREPARE_BUDGET_S + s1validator.PROBE_WARMUP_S + 1.0 + 5.0 + PROBE_V2.window_s + slack
    assert min(seen) >= floor, (seen, floor)


def test_a_coldkey_with_a_high_past_capacity_cannot_saturate_its_own_shard():
    """Review N1: shards were balanced by each UID's expected requests/s, but a UID is sent every request prepared for
    its budget (up to headroom x its last measurement) if it answers junk instantly. 24 s1-fast UIDs of one coldkey
    with a moving average of 48/s and a last probe of 160/s put ~3x the mean shard load on one shard, the cpu_saturated
    guard dropped every probe and all capacities froze. The plan now weighs each UID by its prepared requests."""
    import random

    from reax_subnet.tiers import CAPACITY

    rng = random.Random(92)
    targets, roles, ema, last = [], {}, {}, {}
    uid = 2
    for i, (role, count, lo, hi) in enumerate((("s1-fast", 85, 20.0, 80.0), ("s1-pro", 145, 5.0, 20.0),
                                               ("s1-lite", 10, 4.0, 16.0))):
        for k in range(count):
            targets.append(_n(uid, f"{role}-{k % 60}", None))  # a few multi-UID coldkeys per pool
            roles[uid] = role
            ema[uid] = last[uid] = rng.uniform(lo, hi)
            uid += 1
    attackers = []
    for _ in range(24):
        targets.append(_n(uid, "attacker", None))
        roles[uid], ema[uid], last[uid] = "s1-fast", 48.0, 160.0
        attackers.append(uid)
        uid += 1
    budgets = {u: CAPACITY[roles[u]].budget_dps(max(ema[u], last[u])) for u in roles}

    def expected(n):  # the 1.2.0rc7 plan weight
        cap = CAPACITY[roles[n.uid]]
        return max(cap.target_dps, ema[n.uid]) / cap.items_per_request

    def prepared(n):
        return s1validator.probe_request_load(CAPACITY[roles[n.uid]], budgets[n.uid])

    def sent(n):  # what a shard really sends: the attacker answers instantly, an honest UID at its own speed
        cap = CAPACITY[roles[n.uid]]
        return prepared(n) if n.uid in attackers else min(prepared(n), ema[n.uid] / cap.items_per_request)

    def busiest(weight):
        return max(math.fsum(sent(n) for n in s) for s in probeshard.plan_shards(targets, weight, 20))

    # Review calibration (extrapolated, not benched): ~1.9 ms of a shard's core per request, so ~470 requests/s
    # reach the 90 % cpu_saturated limit.
    saturation = 470.0
    fair = math.fsum(prepared(n) for n in targets) / 20  # mean prepared requests/s per shard
    mean_sent = math.fsum(sent(n) for n in targets) / 20
    # The 1.2.0rc7 plan: one shard ~5.5x the mean, ~92 % of saturation (434 requests/s with this seed).
    assert busiest(expected) > max(4 * mean_sent, 0.9 * saturation), (busiest(expected), mean_sent)
    # Planned by prepared requests: ~176 requests/s, ~37 % of saturation.
    assert busiest(prepared) < 1.5 * fair < 0.75 * saturation, (busiest(prepared), fair)
    # Even if every UID answers instantly, no shard carries much more than its fair share of the prepared requests.
    loads = [math.fsum(prepared(n) for n in s) for s in probeshard.plan_shards(targets, prepared, 20)]
    assert max(loads) < 1.5 * fair, loads


async def test_probes_dropped_in_a_row_are_counted_and_logged(probe_v2, make_wallet, fake_chain_cls, tmp_path,
                                                              monkeypatch, caplog):
    """Review N2: from the third probe dropped in a row each drop logs one greppable ERROR line with the reasons, the
    count is in last_probe_stats, and a kept probe resets it."""
    probe_v2()
    probe, seen = S1Validator._probe_capacity, []

    async def three_saturated(self, *a, **kw):
        monkeypatch.setattr(s1validator, "PROBE_CPU_SATURATED", 0.0 if len(seen) < 3 else 0.90)
        try:
            return await probe(self, *a, **kw)
        finally:
            seen.append(dict(self.last_probe_stats))

    monkeypatch.setattr(S1Validator, "_probe_capacity", three_saturated)
    groups = [(Machine(160.0), [{}]) for _ in range(2)]
    with caplog.at_level("ERROR", logger="reax.s1validator"):
        val, by_group, summaries, _ = await _probe_arena(make_wallet, fake_chain_cls, tmp_path, groups, rounds=5)
    assert [s["drops_in_row"] for s in seen] == [1, 2, 3, 0], seen
    assert summaries[-1]["capacity"] and all(val.state.miners[u].capacity for g in by_group.values() for u in g)
    alerts = [r.getMessage() for r in caplog.records if "probes dropped in a row" in r.getMessage()]
    assert len(alerts) == 1 and alerts[0].startswith("capacity probe: 3 probes dropped in a row")
    assert alerts[0].count("busiest window loop at") == 3
    assert all(r.levelname == "ERROR" for r in caplog.records if "probes dropped in a row" in r.getMessage())


def test_the_drop_alert_names_the_reasons_of_the_last_drops(caplog):
    """Every drop from the third in a row logs one line with the reasons of the last three; the count keeps growing."""
    import collections

    val = S1Validator.__new__(S1Validator)
    val._probe_drops, val.last_probe_stats = 0, {}
    val._probe_drop_reasons = collections.deque(maxlen=s1validator.PROBE_DROPS_ALERT)
    with caplog.at_level("ERROR", logger="reax.s1validator"):
        for k in range(5):
            val._probe_dropped([f"reason {k}"])
    assert val.last_probe_stats["drops_in_row"] == 5
    msgs = [r.getMessage() for r in caplog.records]
    assert len(msgs) == 3 and msgs[-1].startswith("capacity probe: 5 probes dropped in a row")
    assert "reason 2 | reason 3 | reason 4" in msgs[-1] and "reason 1" not in msgs[-1]
