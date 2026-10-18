"""1.2.0rc6 worker fixes from review-rc6b: one first-draw group per confirmation (F1), an exhausted pool is a logged
"no draw" (F1), a transient re-fetch error of a pending entry does not reject it (F2), and a pending entry fetched again
with the weights of its first check is served without the copy checks again (F3). The keep-alive tests no longer read
the wall clock (F4: test_keepalive.NOW)."""

import logging
import os

import pytest
from spec_helpers import HOTKEYS, make_weights, synthetic_pool, write_pool
from test_spec_evaluator import TRACK, Env

from reax_subnet.spec import config as C
from reax_subnet.spec import evaluator as E
from reax_subnet.spec import sampling as S
from reax_subnet.spec import state as ST

# the state machine on the fake models' score scale (conftest.py)
pytestmark = pytest.mark.usefixtures("rc5_entry_floors")

DECIDED = (ST.CHAMPION, ST.FORMER, ST.NOT_BETTER)


def _tiered_pool(counts):
    rows = synthetic_pool(sum(counts.values()))
    i = 0
    for tier, n in counts.items():
        for _ in range(n):
            rows[i]["tier"] = tier
            i += 1
    return rows


# the deployed DF tier counts (review-rc6b sim/confirm_union.py): a first draw takes 255 non-proc items of 520
LIVE_DF = {"proc_sealed": 900, "same_family": 102, "new_domain": 127, "new_family": 291}


def _env(tmp_path, counts):
    env = Env(tmp_path, {})
    write_pool(str(tmp_path / "sealed"), TRACK, _tiered_pool(counts))    # read again every cycle
    return env


def _statuses(env, uids):
    return {u: env.sub(u).status for u in uids}


# ------------------------------------------------------------------------------------------------ F1
async def test_two_first_draw_groups_due_together_confirm_one_group_per_cycle(tmp_path):
    env = _env(tmp_path, LIVE_DF)
    commits = {}
    for u in range(1, 10):
        env.add(u, f"m/u{u}", 0.80 + 0.01 * u)
    for grp in ((1, 2, 3, 4), (5, 6, 7, 8)):                          # two first draws two hours apart
        commits.update({u: f"m/u{u}" for u in grp})
        await env.cycle(commits)
        env.clock.advance(2 * 3600)
    ts = env.ev.state.track(TRACK)
    a, b = env.sub(1).draw1, env.sub(5).draw1
    assert a != b and {env.sub(u).draw1 for u in (1, 2, 3, 4)} == {a} and {env.sub(u).draw1 for u in (5, 6, 7, 8)} == {b}
    assert set(_statuses(env, range(1, 9)).values()) == {ST.PENDING_CONFIRM}
    # both first draws excluded together leave too few non-proc items: the draw the old grouping asked for
    union = set(ts.draws[a].item_ids) | set(ts.draws[b].item_ids)
    with pytest.raises(S.SampleError):
        S.draw(env.ev._pool(TRACK), n=300, seed=1, use_counts=ts.used, exclude=union)

    commits[9] = "m/u9"                                              # a queued entry behind both groups
    env.clock.t = env.sub(5).confirm_after_ts + 1800                  # both groups due in the same cycle
    env.server.served.clear()
    await env.cycle(commits)
    # group A confirmed; all entries share one block, so on the vacant track they wait for the cohort playoff (1.2.0rc7)
    assert all(env.sub(u).history[-1]["kind"] == "confirm" for u in (1, 2, 3, 4))
    assert set(_statuses(env, (1, 2, 3, 4)).values()) <= {ST.PENDING_CONFIRM, ST.NOT_BETTER}
    assert not any(env.sub(u).confirm_draw for u in (5, 6, 7, 8))
    assert set(_statuses(env, (5, 6, 7, 8)).values()) == {ST.PENDING_CONFIRM} and env.sub(9).status == ST.QUEUED
    assert sorted(env.server.served) == [f"m/u{u}" for u in (1, 2, 3, 4)]
    d2 = ts.draws[env.sub(1).history[-1]["draw"]]
    assert not set(d2.item_ids) & set(ts.draws[a].item_ids)          # disjoint from the draw it confirms

    env.clock.advance(3600)
    await env.cycle(commits)
    assert all(env.sub(u).history[-1]["kind"] == "confirm" for u in (5, 6, 7, 8))   # then group B
    assert ST.CHAMPION not in _statuses(env, range(1, 9)).values()    # uid 9 (same block, queued) holds the cohort back
    assert env.sub(9).status == ST.QUEUED
    d2 = ts.draws[env.sub(5).history[-1]["draw"]]
    assert not set(d2.item_ids) & set(ts.draws[b].item_ids)

    env.clock.advance(3600)
    env.server.served.clear()
    await env.cycle(commits)
    assert env.sub(9).history and env.sub(9).status != ST.QUEUED and "m/u9" in env.server.served   # then the queue: no stall


async def test_exhausted_pool_is_a_logged_no_draw_and_the_track_returns(tmp_path, caplog):
    # a first draw leaves 4 non-proc items: its confirmation cannot be drawn
    env = _env(tmp_path, {"proc_sealed": 900, "same_family": 102, "new_domain": 127, "new_family": 30})
    env.add(1, "a/m1", 0.9)
    await env.cycle({1: "a/m1"})
    assert env.sub(1).status == ST.PENDING_CONFIRM
    env.clock.advance(13 * 3600)
    env.server.served.clear()
    with caplog.at_level(logging.WARNING, logger=E.log.name):
        assert await env.cycle({1: "a/m1"}) == {TRACK: []}
    ts = env.ev.state.track(TRACK)
    assert ts.events[-1].split(" ", 1)[1].startswith("no draw: only ") and "refill the sealed pool" in ts.events[-1].split(" ", 1)[1]
    assert env.sub(1).status == ST.PENDING_CONFIRM and env.server.served == []
    recs = [r for r in caplog.records if r.name == E.log.name]
    assert recs and all(r.levelno == logging.WARNING and r.exc_info is None for r in recs)   # no traceback
    assert not any("evaluation failed" in r.getMessage() for r in caplog.records)
    # the pool refilled: the next cycle confirms
    write_pool(str(tmp_path / "sealed"), TRACK, _tiered_pool(LIVE_DF))
    env.clock.advance(3600)
    await env.cycle({1: "a/m1"})
    assert env.sub(1).status == ST.CHAMPION


# ------------------------------------------------------------------------------------------------ F2
async def _pending(env):
    env.add(1, "a/m1", 0.9)
    await env.cycle({1: "a/m1"}, block=1000)
    s = env.sub(1)
    assert s.status == ST.PENDING_CONFIRM and s.hash_named and s.attempts == 0
    env.clock.advance(13 * 3600)
    env.hub.unavailable.add("a/m1")                                  # miner-side missing/private repo at refetch
    return s


async def test_one_transient_refetch_error_at_confirmation_does_not_reject(tmp_path):
    env = Env(tmp_path, {})
    s = await _pending(env)
    block = 1000 + C.PUBLISH_DEADLINE_BLOCKS + 1                     # past the publish deadline of a first fetch
    await env.cycle({1: "a/m1"}, block=block)
    assert s.status == ST.PENDING_CONFIRM and s.attempts == 0 and s.reason.startswith("waiting for the repo")
    env.hub.unavailable.clear()
    env.clock.advance(3600)
    await env.cycle({1: "a/m1"}, block=block + 300)
    assert s.status == ST.CHAMPION and s.attempts == 0


async def test_repeated_unavailable_refetches_reject_without_serve_strikes(tmp_path):
    env = Env(tmp_path, {})
    s = await _pending(env)
    block = 1000 + 10
    for i in range(1, C.MAX_ATTEMPTS + 1):                           # inside the publish deadline: it does not apply either
        await env.cycle({1: "a/m1"}, block=block + i)
        env.clock.advance(3600)
        assert s.attempts == 0
    assert s.status == ST.REJECTED and s.reason == "no longer available (3 fetches failed)"
    assert s.unavailable_fetches == C.MAX_ATTEMPTS and s.attempts == 0


async def test_a_never_fetched_entry_still_expires_at_the_publish_deadline(tmp_path):
    env = Env(tmp_path, {})
    env.add(1, "a/m1", 0.9)
    env.hub.unavailable.add("a/m1")
    await env.cycle({1: "a/m1"}, block=1000)
    assert env.sub(1).status == ST.QUEUED and not env.sub(1).hash_named
    env.clock.advance(3600)
    await env.cycle({1: "a/m1"}, block=1000 + C.PUBLISH_DEADLINE_BLOCKS + 1)
    assert env.sub(1).status == ST.REJECTED and "not available within" in env.sub(1).reason


# ------------------------------------------------------------------------------------------------ F3
async def test_refetched_pending_entry_with_its_hash_skips_copy_checks_and_renewal(tmp_path, monkeypatch):
    env = Env(tmp_path, {})
    env.add(1, "a/m1", 0.9)
    await env.cycle({1: "a/m1"})
    s = env.sub(1)
    assert s.status == ST.PENDING_CONFIRM and s.delta_checked and not os.path.exists(env.ev._model_dir(s))
    calls = []

    async def no_renewal(*a, **k):
        calls.append("renewal")
        return False                                                 # low disk: would hold the entry (PENDING_VERIFY)

    def no_copy_check(*a, **k):
        calls.append("copy")
        return ST.PENDING_VERIFY, "must not run"

    monkeypatch.setattr(env.ev, "_refresh_sketches", no_renewal)
    monkeypatch.setattr(env.ev, "_copy_check", no_copy_check)
    env.hub.fetches.clear()
    env.server.served.clear()
    env.clock.advance(13 * 3600)
    await env.cycle({1: "a/m1"})
    assert env.hub.fetches == ["a/m1"] and env.server.served == ["a/m1"] and calls == []
    assert s.status == ST.CHAMPION and s.delta_checked and len(s.history) == 2


async def test_refetched_pending_entry_with_other_weights_is_rejected(tmp_path):
    env = Env(tmp_path, {})
    env.add(1, "a/m1", 0.9)
    await env.cycle({1: "a/m1"})
    s = env.sub(1)
    assert s.status == ST.PENDING_CONFIRM and s.delta_checked
    env.hub.add("a/m1", hotkey=HOTKEYS[1], weights=make_weights(777))   # other weights under the same pinned revision
    env.server.served.clear()
    env.clock.advance(13 * 3600)
    await env.cycle({1: "a/m1"})
    assert s.status == ST.REJECTED and s.reason == "fetched again: other weights than at its first check"
    assert env.server.served == [] and not os.path.exists(env.ev._model_dir(s))
