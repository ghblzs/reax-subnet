"""Mechanism v2: sealed items are reused by rule (ITEM_MAX_USES) and a close first-champion playoff gets one more draw."""

import logging

import pytest
from test_cohort import H
from test_spec_evaluator import TRACK, Env
from spec_helpers import synthetic_pool

from reax_subnet.spec import evaluator as E
from reax_subnet.spec import sampling as S
from reax_subnet.spec import state as ST

pytestmark = pytest.mark.usefixtures("rc5_entry_floors")


# ------------------------------------------------------------------------------------------ A: reuse and retirement
def test_a_retired_item_is_never_drawn_again_and_a_used_one_below_the_limit_is():
    pool = synthetic_pool(1500)
    retired = {r["id"] for r in pool[:900]}
    use = {i: S.ITEM_MAX_USES for i in retired}
    use.update({r["id"]: 1 for r in pool[900:1100]})        # reusable
    rows, fresh = S.draw(pool, n=300, seed=1, use_counts=use)
    assert not {r["id"] for r in rows} & retired
    assert S.ITEM_MAX_USES == 24


def test_below_the_limit_least_used_first_is_kept():
    pool = synthetic_pool(1500)
    use = {r["id"]: (2 if i < 600 else 1) for i, r in enumerate(pool)}
    rows, fresh = S.draw(pool, n=300, seed=3, use_counts=use)
    assert not fresh and all(use[r["id"]] == 1 for r in rows)


def test_a_pool_of_only_retired_items_gives_no_draw():
    pool = synthetic_pool(600)
    with pytest.raises(S.SampleError):
        S.draw(pool, n=300, seed=1, use_counts={r["id"]: S.ITEM_MAX_USES for r in pool})


async def test_confirmation_stays_disjoint_from_the_first_draw_while_items_are_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "ITEM_MAX_USES", 100)
    env = Env(tmp_path, {})
    env.add(1, "a/m1", 0.9)
    ts = env.ev.state.track(TRACK)
    ts.used = {r["id"]: 1 for r in synthetic_pool(1500)}   # no fresh item left: every draw reuses
    await env.cycle({1: "a/m1"})
    env.clock.advance(13 * H)
    await env.cycle({1: "a/m1"})
    d1, d2 = (ts.draws[d["draw"]] for d in env.sub(1).history)
    assert not set(d1.item_ids) & set(d2.item_ids)
    assert not any("exhausted" in str(e) for e in ts.events)   # reuse is normal, no event for it


async def test_refill_warning_only_when_nonproc_uses_fall_below_refill_draws(tmp_path, caplog):
    env = Env(tmp_path, {})
    env.add(1, "a/m1", 0.9)
    ts = env.ev.state.track(TRACK)
    with caplog.at_level(logging.WARNING, logger="reax.spec"):
        await env.cycle({1: "a/m1"})
        assert "refill needed" not in caplog.text
        pool = env.ev._pool(TRACK_CATEGORY)
        nonproc = [r["id"] for r in pool if r.get("tier") != "proc_sealed"]    # 900 of 1500
        assert len(nonproc) == 900
        for i in nonproc:
            ts.used[i] = 0
        for i in nonproc[:300]:                              # 600 x 24 uses / 255 per draw = 56.5 draws: still enough
            ts.used[i] = S.ITEM_MAX_USES
        assert S.nonproc_draws_left(pool, ts.used, 300) > S.REFILL_DRAWS
        env.ev._record_draw(ts, pool, [], True, 99, _Snap(), env.clock())
        assert "refill needed" not in caplog.text
        for i in nonproc[300:450]:                           # 450 x 24 / 255 = 42.4 draws < 48
            ts.used[i] = S.ITEM_MAX_USES
        env.ev._record_draw(ts, env.ev._pool(TRACK_CATEGORY), [], True, 98, _Snap(), env.clock())
        assert "refill needed" in caplog.text


TRACK_CATEGORY = "demand_forecasting"


class _Snap:
    block = 1


# ------------------------------------------------------------------------------------------ B: close playoff
async def _to_playoff(env, skills, rest=600, before_playoff=None):
    commits = {}
    for u, sk in skills.items():
        env.add(u, f"m/u{u}", sk)
        commits[u] = f"m/u{u}"
    await env.cycle(commits, block=1000)
    env.clock.advance(13 * H)
    await env.cycle(commits, block=1100)                              # the confirmations
    env.clock.advance(rest)
    if before_playoff:
        before_playoff()
    await env.cycle(commits, block=1200)                              # the playoff
    return commits


def _playoff_entries(ts):
    return [s for s in ts.submissions.values() if any(h["kind"] == "playoff" for h in s.history)]


async def test_a_clear_playoff_winner_gets_no_extension(tmp_path):
    env = Env(tmp_path, {})
    await _to_playoff(env, {1: 0.60, 2: 0.80, 3: 0.97})
    ts = env.ev.state.track(TRACK)
    assert ts.champion == env.sub(3).key
    assert all("draw2" not in h for s in ts.submissions.values() for h in s.history)


CLOSE = {1: 0.30, 2: 0.96, 3: 0.965}     # 85.0 vs 90.0 on the first draw, far more than one SE apart


async def test_the_gap_is_measured_in_paired_standard_errors(tmp_path, monkeypatch):
    env = Env(tmp_path, {})
    await _to_playoff(env, CLOSE)                                      # 1.0 SE: a five-point gap is clear
    ts = env.ev.state.track(TRACK)
    assert ts.champion == env.sub(3).key and not any("extension" in e for e in ts.events)
    env2 = Env(tmp_path / "wide", {})
    monkeypatch.setattr(E, "PLAYOFF_MIN_GAP_SE", 100.0)                # the same scores are "close" at 100 SE
    await _to_playoff(env2, CLOSE)
    assert any("playoff extension on draw" in e for e in env2.ev.state.track(TRACK).events)


async def test_a_close_playoff_runs_one_more_draw_for_the_tied_group_only(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "PLAYOFF_MIN_GAP_SE", 100.0)
    env = Env(tmp_path, {})
    await _to_playoff(env, CLOSE)
    ts = env.ev.state.track(TRACK)
    ext = {s.uid: s for s in ts.submissions.values() if any("draw2" in h for h in s.history)}
    assert set(ext) == {2, 3}                                          # uid 1 (far below) is not within the distance
    h = ext[2].history[-1]
    first, second = ts.draws[h["draw"]], ts.draws[h["draw2"]]
    assert first.draw_id != second.draw_id and not set(first.item_ids) & set(second.item_ids)   # disjoint, pool allows
    assert abs(h["score"] - (h["score1"] + h["score2"]) / 2) < 1e-9    # decided on the mean of both draws
    assert len([d for d in ts.draws.values() if d.draw_id in (h["draw"], h["draw2"])]) == 2
    assert env.sub(1).history[-1].get("draw2") is None
    assert sum("playoff extension on draw" in e for e in ts.events) == 1   # one extension only


async def test_the_extension_runs_without_hourly_budget_and_is_charged_to_it(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "PLAYOFF_MIN_GAP_SE", 100.0)
    env = Env(tmp_path, {}, max_evals=3)
    await _to_playoff(env, CLOSE, rest=H)                              # playoff of 2 uses 2 of 3, the extension needs 2 again
    ts = env.ev.state.track(TRACK)
    assert not any("extension skipped" in e for e in ts.events)
    assert sum("playoff extension on draw" in e for e in ts.events) == 1
    assert {s.uid for s in ts.submissions.values() if any("draw2" in h for h in s.history)} == {2, 3}
    assert len(env.ev.state.eval_times) > 3 and env.ev._budget(env.clock()) == 0   # charged: the others wait next cycle


async def test_still_tied_after_the_extension_the_earlier_commitment_wins(tmp_path, monkeypatch):
    env = Env(tmp_path, {})
    monkeypatch.setattr(E, "PLAYOFF_MIN_GAP_SE", 100.0)                # tied before and after the extension
    await _to_playoff(env, CLOSE)
    ts = env.ev.state.track(TRACK)
    assert ts.champion == env.sub(2).key                               # uid 2 committed first although uid 3 scores higher


async def test_a_serving_failure_in_the_extension_falls_back_to_the_first_draw(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "PLAYOFF_MIN_GAP_SE", 100.0)
    env = Env(tmp_path, {})
    orig = env.ev._serve
    calls = {"n": 0}

    async def flaky(p, rows, now):
        calls["n"] += (p.sub.uid == 2)      # first pass, confirmation, playoff, then the extension: the 4th serve fails
        if p.sub.uid == 2 and calls["n"] == 4:
            return None
        return await orig(p, rows, now)
    monkeypatch.setattr(env.ev, "_serve", flaky)
    await _to_playoff(env, CLOSE)
    ts = env.ev.state.track(TRACK)
    assert ts.champion == env.sub(3).key
    assert any("playoff extension abandoned" in e for e in ts.events)


# ------------------------------------------------------------------------------------------ review fixes (F2, F3)
async def test_the_tied_group_includes_a_runner_up_below_the_floor_on_draw_one(tmp_path, monkeypatch):
    """Review F2: the group is formed among all served entries; the floor applies to the two-draw means."""
    from reax_subnet import competition_split as cs
    monkeypatch.setattr(E.SC, "paired_se", lambda *a, **k: 6.0)        # uid 2 (85.0) is within 1 SE of uid 3 (90.0)

    def floor(v):
        return lambda: monkeypatch.setitem(cs.TRACK_ENTRY_FLOOR, TRACK, v)
    env = Env(tmp_path / "a", {})
    await _to_playoff(env, {2: 0.96, 3: 0.965}, before_playoff=floor(85.8))   # uid 2: 85.0 on draw 1, 86.8 on the mean
    ts = env.ev.state.track(TRACK)
    h2, h3 = env.sub(2).history[-1], env.sub(3).history[-1]
    assert h2["score1"] < 85.8 < h2["score"] and "draw2" in h2 and "draw2" in h3     # the runner-up below the floor got draw 2
    assert h3["score"] - h2["score"] < 6.0
    assert ts.champion == env.sub(2).key                               # clears the floor on the mean, tie goes to the earlier commitment
    floor(24.8)()                                                       # back to the release floor until the playoff
    env2 = Env(tmp_path / "b", {})
    await _to_playoff(env2, {2: 0.96, 3: 0.965}, before_playoff=floor(87.5))  # the mean is still below the floor
    assert any("draw2" in h for h in env2.sub(2).history)
    assert env2.ev.state.track(TRACK).champion == env2.sub(3).key


async def test_an_extended_entry_stores_the_two_draw_nonproc_and_tier_scores(tmp_path, monkeypatch):
    """Review F3: nonproc / by_tier follow the two-draw score; per-draw values stay in the history; keep-alive reads the mean."""
    from test_keepalive import DF, NOW, _snap, _sub
    monkeypatch.setattr(E, "PLAYOFF_MIN_GAP_SE", 100.0)
    env = Env(tmp_path, {})
    await _to_playoff(env, CLOSE)
    s = env.sub(2)
    h = s.history[-1]
    assert s.status == ST.CHAMPION and "draw2" in h
    assert abs(s.nonproc - (h["nonproc1"] + h["nonproc2"]) / 2) < 1e-9 and s.nonproc != h["nonproc1"]
    for t_, v in s.by_tier.items():
        assert abs(v - (h["by_tier1"][t_] + h["by_tier2"][t_]) / 2) < 1e-3
    assert env.sub(1).history[-1].get("nonproc1") is None            # a non-extended entry keeps the single-draw fields
    floor = (h["nonproc1"] + s.nonproc) / 2                           # draw one alone fails it, the two-draw mean clears it
    assert h["nonproc1"] < floor < s.nonproc or s.nonproc < floor < h["nonproc1"]
    for nonproc, expect in ((h["nonproc1"], s.nonproc > h["nonproc1"]), (s.nonproc, True)):
        st = ST.SpecState()
        sub = _sub(st, 1, ST.CHAMPION, 60.0, nonproc=nonproc)
        got = compute_keepalive_uids(st, _snap([sub]), floor, DF, NOW)
        assert (got == [1]) == (nonproc > floor)
    assert (s.nonproc > floor) != (h["nonproc1"] > floor)             # draw two changed the eligibility


def compute_keepalive_uids(st, snap, nonproc_floor, track, now):
    return [u for u, _ in E.compute_keepalive(st, [track], snap, {track: 0.0}, now=now, nonproc_floors={track: nonproc_floor})[track]]
