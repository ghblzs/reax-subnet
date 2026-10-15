"""Capacity probe timing (1.2.0rc3 mainnet report): the validator needs ~0.2 ms of CPU per lane it opens, so with
~175 UIDs x 16 lanes the last UIDs' first requests left 0.6-2.5 s after the one common window had opened and fast
miners were measured at 75-90 % of their speed. Each UID's window now opens at the validator's first send to it,
lanes open one per UID per loop pass, and a probe whose first sends spread too far is dropped.

The probe tests reuse the fake-machine arena of test_tiers.py (real aiohttp miners on 127.0.0.1, a faked chain).
"""

from __future__ import annotations

import asyncio
import math

import pytest

from reax_subnet import s1validator
from reax_subnet.s1validator import S1Validator
from test_tiers import (PROBE_V2, Machine, _probe_arena, _release_floors_off, probe_v2)  # noqa: F401 (fixtures)


@pytest.fixture
def slow_sends(monkeypatch):
    """The validator's CPU sends one request at a time and each costs ``cost`` seconds (signing, aiohttp request
    setup on a busy host), for the first ``lanes`` sends of each probe: the burst that opens the lanes. (In this arena
    the miners share the validator's event loop, so the cost is a FIFO lock and an asyncio sleep, not a blocking one:
    the miners keep serving while the validator is busy, as on mainnet.)"""

    def use(cost: float, lanes: int):
        state = {"in_probe": False, "sends": 0, "cpu": asyncio.Lock()}
        post, probe = S1Validator._post_s1, S1Validator._probe_capacity

        async def slow_post(self, *a, **kw):
            if state["in_probe"] and state["sends"] < lanes:
                state["sends"] += 1
                async with state["cpu"]:
                    await asyncio.sleep(cost)
            return await post(self, *a, **kw)

        async def probing(self, *a, **kw):
            state.update(in_probe=True, sends=0)
            try:
                return await probe(self, *a, **kw)
            finally:
                state["in_probe"] = False

        monkeypatch.setattr(S1Validator, "_post_s1", slow_post)
        monkeypatch.setattr(S1Validator, "_probe_capacity", probing)

    return use


def _report(summaries):
    return summaries[-1]["capacity"]


async def test_a_uid_the_validator_reaches_last_is_not_under_counted(probe_v2, slow_sends, make_wallet,
                                                                      fake_chain_cls, tmp_path):
    """8 UIDs x 16 lanes at 4 ms per send: in 1.2.0rc3 the last UID's lanes opened ~0.5 s into its 2 s window
    and it measured ~25 % less than the first UID on an identical machine."""
    probe_v2()
    slow_sends(0.004, 8 * PROBE_V2.inflight)
    groups = [(Machine(160.0), [{}]) for _ in range(8)]
    val, by_group, summaries, _ = await _probe_arena(make_wallet, fake_chain_cls, tmp_path, groups)
    report = _report(summaries)
    first, last = report[by_group[0][0]]["dps"], report[by_group[7][0]]["dps"]
    assert first >= 0.6 * 160.0, report
    assert last >= 0.92 * first, (first, last)
    assert val.last_probe_stats["send_skew_ms"] < 1000 * s1validator.PROBE_SEND_SKEW_MAX_S


async def test_siblings_far_apart_in_send_order_still_share_one_machine(probe_v2, slow_sends, make_wallet,
                                                                         fake_chain_cls, tmp_path):
    """Per-UID windows differ by the send skew only: two hotkeys on one machine, sent to first and last of 8 UIDs
    during a slow burst, together measure no more than the machine (plus skew / window)."""
    probe_v2()
    slow_sends(0.004, 8 * PROBE_V2.inflight)
    shared = Machine(160.0)
    groups = [(shared, [{}])] + [(Machine(160.0), [{}]) for _ in range(6)] + [(shared, [{}])]
    val, by_group, summaries, _ = await _probe_arena(make_wallet, fake_chain_cls, tmp_path, groups)
    report = _report(summaries)
    pair = report[by_group[0][0]]["dps"] + report[by_group[7][0]]["dps"]
    solo = max(report[by_group[g][0]]["dps"] for g in range(1, 7))
    skew = val.last_probe_stats["send_skew_ms"] / 1000
    assert pair <= solo * (1 + skew / PROBE_V2.window_s) * 1.05 + 2.0, (pair, solo, skew)


async def test_every_uid_gets_its_first_request_before_any_uid_gets_a_second_lane(probe_v2, make_wallet,
                                                                                   fake_chain_cls, tmp_path,
                                                                                   monkeypatch):
    probe_v2()
    monkeypatch.setenv("REAX_PROBE_TRACE", "1")
    groups = [(Machine(160.0), [{}]) for _ in range(6)]
    val, by_group, _summaries, _ = await _probe_arena(make_wallet, fake_chain_cls, tmp_path, groups, rounds=2)
    trace = val.last_probe_trace
    firsts, seconds = [], []
    for uid, events in trace["uids"].items():
        lane_start = {}
        for lane, sent, _got, _ok in events:
            lane_start[lane] = min(sent, lane_start.get(lane, math.inf))
        firsts.append(lane_start[0])
        seconds.append(lane_start[1])
        assert trace["windows"][uid] == min(lane_start.values())  # the UID's window opens at its first send
    assert max(firsts) <= min(seconds)


async def test_a_probe_whose_first_sends_spread_too_far_is_dropped(probe_v2, make_wallet, fake_chain_cls, tmp_path,
                                                                   monkeypatch, caplog):
    probe_v2()
    monkeypatch.setattr(s1validator, "PROBE_SEND_SKEW_MAX_S", 0.0)  # any spread counts as too far
    groups = [(Machine(160.0), [{}]), (Machine(160.0), [{}])]
    with caplog.at_level("ERROR", logger="reax.s1validator"):
        val, by_group, summaries, _ = await _probe_arena(make_wallet, fake_chain_cls, tmp_path, groups, rounds=2)
    assert summaries[-1]["capacity"] == {}
    assert all(val.state.miners[u].capacity is None for g in by_group.values() for u in g)
    assert val.last_probe_stats["skipped"] and val.last_probe_stats["send_skew_ms"] > 0
    assert "first sends spread over" in caplog.text


async def test_the_send_skew_guard_is_quiet_on_a_normal_probe(probe_v2, make_wallet, fake_chain_cls, tmp_path):
    probe_v2()
    groups = [(Machine(160.0), [{}]) for _ in range(4)]
    val, by_group, summaries, _ = await _probe_arena(make_wallet, fake_chain_cls, tmp_path, groups, rounds=2)
    assert not val.last_probe_stats["skipped"]
    assert val.last_probe_stats["send_skew_ms"] < 1000 * s1validator.PROBE_SEND_SKEW_MAX_S
    assert all(summaries[-1]["capacity"][g[0]]["dps"] > 0 for g in by_group.values())
