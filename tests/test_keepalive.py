"""Participation keep-alive for specialist entrants (docs/SPECIALIST-TRACKS.md, "Keep-alive"; 1.2.0rc6)."""

import asyncio
import math

import pytest
from spec_helpers import HOTKEYS, commit, finetune, make_weights, neuron, snapshot, write_model
from test_remote_competitions import NOW0, SERVING, Clock, snap, stamp, write_engine_state
from test_spec_evaluator import TRACK, Env

from reax_subnet import competition_split as CS
from reax_subnet import remote as RM
from reax_subnet.competition_split import RankedLeaders, combine_weights, keepalive_uids
from reax_subnet.competitions import SpecialistAdapter
from reax_subnet.config import ENGINE_STATE_FILE, REMOTE_MAX_AGE_S, SPEC_STATE_FILE
from reax_subnet.engvalidator import EngineCompetition
from reax_subnet.spec import config as C
from reax_subnet.spec import machine as M
from reax_subnet.spec import state as ST
from reax_subnet.spec.commitment import SpecCommitment
from reax_subnet.spec.evaluator import compute_keepalive

# the hand-written scores below sit around the 1.2.0rc5 entry floors (conftest.py); the release floors have their own tests
pytestmark = pytest.mark.usefixtures("rc5_entry_floors")

KA = CS.KEEPALIVE_SHARE
DF, RA = "demand_forecasting", "risk_assessment"
BASE = {0: 0.10, 1: 0.30, 2: 0.25, 3: 0.20, 4: 0.15}         # uid 0 = owner (burn 0.10)
# the evaluation clock of every keep-alive test (the remote tests read their files at NOW0): never the wall clock, so the
# 24 h window and the future-start rule do not depend on when the suite runs
NOW = NOW0.timestamp()


def test_release_constants():
    assert KA == 0.05 and 0.0 < KA < 1.0
    assert C.MAX_OPEN_PER_COLDKEY_TRACK == 2


# ------------------------------------------------------------------------------------------------ share layer
@pytest.mark.parametrize("ka", [None, {}, {DF: []}, {DF: [(50, 10.0), (51, 24.8)]}, {DF: [(50, float("nan")), (51, -1.0)]},
                                {DF: [(0, 99.0)]}])
def test_no_eligible_entry_is_bit_identical_to_rc5(ka):
    """Nobody eligible (none, empty, at/below the floor, not finite, only the owner): exactly the rc5 weights."""
    leaders = {DF: RankedLeaders([(1, 40.0), (2, 30.0)]), RA: [(3, 50.0)]}
    for champ, frac in ((None, 0.0), (4, 0.7)):
        for gate in (None, lambda t, u: u != 2):
            rc5 = combine_weights(BASE, champ, frac, leaders, owner_uid=0, burn_fraction=0.10, award_ok=gate)
            assert combine_weights(BASE, champ, frac, leaders, owner_uid=0, burn_fraction=0.10, award_ok=gate,
                                   keepalive=ka) == rc5
    assert combine_weights(SERVING, None, 0.0, {}, owner_uid=0, keepalive=ka) == combine_weights(SERVING, None, 0.0, {}, owner_uid=0)


def test_off_switch_is_bit_identical():
    leaders = {DF: RankedLeaders([(1, 40.0)])}
    alive = {DF: [(1, 40.0), (5, 30.0)]}
    assert combine_weights(BASE, None, 0.0, leaders, owner_uid=0, keepalive=alive, keepalive_share=0.0) == \
        combine_weights(BASE, None, 0.0, leaders, owner_uid=0)


def test_keepalive_is_reserved_before_the_ranks_and_sums_to_one():
    serving = {11: 0.6, 12: 0.4}
    leaders = {DF: RankedLeaders([(20, 40.0), (21, 30.0), (22, 28.0)])}
    alive = {DF: [(20, 40.0), (21, 30.0), (22, 28.0), (23, 26.0)]}     # 23: pending confirmation, no rank yet
    w = combine_weights(serving, None, 0.0, leaders, keepalive=alive)
    per = 0.15 * KA / 4
    assert w[23] == pytest.approx(per)
    assert w[20] == pytest.approx(0.15 * (1 - KA) * 0.75 + per)
    assert w[21] == pytest.approx(0.15 * (1 - KA) * 0.17 + per) and w[22] == pytest.approx(0.15 * (1 - KA) * 0.08 + per)
    # the track still takes exactly its 15 %; serving keeps the rest pro rata
    assert sum(w[u] for u in (20, 21, 22, 23)) == pytest.approx(0.15)
    assert w[11] == pytest.approx(0.6 * 0.85) and w[12] == pytest.approx(0.4 * 0.85)
    assert math.fsum(w.values()) == pytest.approx(1.0)


def test_track_without_leaders_reserves_only_the_keepalive_part():
    """Nobody crowned yet (all pending confirmation): only 5 % of the track share is taken, 95 % stays with inference."""
    serving = {11: 0.6, 12: 0.4}
    w = combine_weights(serving, None, 0.0, {}, keepalive={DF: [(20, 30.0), (21, 29.0)]})
    assert w[20] == w[21] == pytest.approx(0.15 * KA / 2)
    assert w[11] + w[12] == pytest.approx(1 - 0.15 * KA)


def test_keepalive_is_not_gated_by_the_owner_approval_but_ranks_are():
    serving = {11: 1.0}
    leaders = {DF: RankedLeaders([(20, 40.0), (21, 30.0)])}
    w = combine_weights(serving, None, 0.0, leaders, award_ok=lambda t, u: False, keepalive={DF: [(20, 40.0), (21, 30.0)]})
    assert w[20] == w[21] == pytest.approx(0.15 * KA / 2)           # no approval: no rank share, only the keep-alive
    assert w[11] == pytest.approx(1 - 0.15 * KA)


def test_owner_burn_is_kept_and_the_owner_never_holds_a_slot():
    w = combine_weights(BASE, None, 0.0, {}, owner_uid=0, burn_fraction=0.10, keepalive={DF: [(0, 50.0), (9, 30.0)]})
    assert w[0] == pytest.approx(0.10)                               # the owner's burn stays exactly as it is
    assert w[9] == pytest.approx(0.15 * KA * 0.90)                 # the only slot (the owner is not counted), miner part 0.90
    assert keepalive_uids([(0, 50.0), (9, 30.0), (9, 31.0)], None, owner_uid=0) == [9]


def test_two_tracks_are_separate_and_bounded():
    serving = {11: 1.0}
    alive = {DF: [(20, 30.0)], RA: [(30 + i, 40.0) for i in range(10)]}
    w = combine_weights(serving, None, 0.0, {}, keepalive=alive)
    assert w[20] == pytest.approx(0.15 * KA)
    assert all(w[30 + i] == pytest.approx(0.15 * KA / 10) for i in range(10))
    assert w[11] == pytest.approx(1 - 2 * 0.15 * KA)                # never more than 5 % of each track share


def test_floors_apply_to_keepalive_entries(monkeypatch):
    monkeypatch.setitem(CS.TRACK_ENTRY_FLOOR, DF, 24.8)
    w = combine_weights({11: 1.0}, None, 0.0, {}, keepalive={DF: [(20, 24.8), (21, 24.9)]})
    assert 20 not in w and w[21] == pytest.approx(0.15 * KA)


def test_keepalive_beats_the_lowest_serving_weight_with_19_entries():
    """RESULT.md math: 19 entrants across two tracks; the lowest non-zero live incentive is ~0.00008 (block 9295813)."""
    serving = {u: 1.0 for u in range(100, 340)}                      # 240 serving uids
    alive = {DF: [(u, 30.0) for u in range(11)], RA: [(u, 40.0) for u in range(11, 19)]}
    w = combine_weights(serving, None, 0.0, {}, keepalive=alive)
    low = min(w[u] for u in range(19))
    assert low == pytest.approx(0.15 * KA / 11) and low > 0.0006   # > 8e-5, the lowest live non-zero incentive


# ------------------------------------------------------------------------------------------------ eligibility (pure)
REV = "a" * 40


def _sub(st, uid, status, score, *, coldkey=None, checked=True, derived="", block=1, track=DF, repo=None, nonproc=...):
    """``nonproc`` defaults to a score above every non-proc floor when the entry has a score (the rc6 rules alone)."""
    repo = repo or f"m/m{uid}"
    key = f"{HOTKEYS[uid]}|{SpecCommitment(track, repo, REV).text()}"
    s = ST.Submission(key=key, uid=uid, hotkey=HOTKEYS[uid], coldkey=coldkey or f"c{uid}", track_id=track, repo=repo,
                      revision=REV, first_seen_block=block, first_seen_ts=1.0, status=status, score=score,
                      delta_checked=checked, derived_from=derived,
                      nonproc=(99.0 if score is not None else None) if nonproc is ... else nonproc)
    s.last_eval_ts = s.champion_since_ts = NOW - 3600.0             # decided/scored an hour ago (inside the 24 h window)
    s.confirm_after_ts = NOW - 3600.0 + M.CONFIRM_DELAY_S          # the first pass, as the evaluator records it
    st.track(track).submissions[key] = s
    if status == ST.CHAMPION:
        st.track(track).champion = key
    return s


def _snap(subs, coldkeys=None):
    coldkeys = coldkeys or {}
    return snapshot(1000, [neuron(s.uid, SpecCommitment(s.track_id, s.repo, REV).text(), coldkey=coldkeys.get(s.uid, s.coldkey))
                           for s in subs])


def test_eligibility_by_status_copy_check_and_floor():
    st = ST.SpecState()
    subs = [
        _sub(st, 1, ST.CHAMPION, 40.0), _sub(st, 2, ST.PENDING_CONFIRM, 30.0), _sub(st, 3, ST.FORMER, 35.0),
        _sub(st, 4, ST.NOT_BETTER, 28.0),                              # scored above the floor, did not beat the incumbent
        _sub(st, 5, ST.QUEUED, None),                                  # not evaluated yet
        _sub(st, 6, ST.PENDING_VERIFY, None, checked=False),           # base missing
        _sub(st, 7, ST.REJECTED, 50.0, checked=True),                  # exact copy (rejected)
        _sub(st, 8, ST.NOT_BETTER, 60.0, derived="x"),                 # near copy that did not beat its parent
        _sub(st, 9, ST.NOT_BETTER, 20.0),                              # below the floor
        _sub(st, 10, ST.FORMER, 45.0, checked=False),                  # crowned before rc5 without the copy check
        _sub(st, 11, ST.WITHDRAWN, 45.0),
        _sub(st, 12, ST.PENDING_CONFIRM, 32.0, derived="y"),           # near copy that beat its parent: fine
        _sub(st, 13, ST.QUEUED, 99.0),                                 # re-queued with an old score: still nothing
    ]
    out = compute_keepalive(st, [DF], _snap(subs), {DF: 24.8}, now=NOW)
    assert sorted(u for u, _ in out[DF]) == [1, 2, 12]                # decided entries (former, not better) do not qualify
    assert out[DF][0] == (1, 40.0)                                    # champion first
    # inactive / unknown track: nothing
    assert compute_keepalive(st, [], _snap(subs), now=NOW) == {}


def test_the_hotkey_must_still_commit_to_the_submission():
    st = ST.SpecState()
    a, b = _sub(st, 1, ST.CHAMPION, 40.0), _sub(st, 2, ST.PENDING_CONFIRM, 30.0)
    moved = snapshot(1000, [neuron(1, SpecCommitment(DF, "m/m1", REV).text()), neuron(2, commit(DF, "m/other"))])
    assert compute_keepalive(st, [DF], moved, {DF: 0.0}, now=NOW)[DF] == [(1, 40.0)]
    gone = snapshot(1000, [neuron(2, SpecCommitment(DF, "m/m2", REV).text())])
    assert compute_keepalive(st, [DF], gone, {DF: 0.0}, now=NOW)[DF] == [(2, 30.0)]
    # the uid follows the hotkey (renumbered)
    renum = snapshot(1000, [neuron(77, SpecCommitment(DF, "m/m2", REV).text(), hotkey=HOTKEYS[2])])
    assert compute_keepalive(st, [DF], renum, {DF: 0.0}, now=NOW)[DF] == [(77, 30.0)]
    assert a and b


def test_at_most_two_slots_per_coldkey_by_priority():
    st = ST.SpecState()
    subs = [_sub(st, 1, ST.PENDING_CONFIRM, 50.0, block=1), _sub(st, 2, ST.PENDING_CONFIRM, 30.0, block=5),
            _sub(st, 3, ST.CHAMPION, 40.0, block=9), _sub(st, 4, ST.PENDING_CONFIRM, 35.0, block=2),
            _sub(st, 5, ST.PENDING_CONFIRM, 29.0)]
    same = {u: "cold-A" for u in (1, 2, 3, 4)}                       # four hotkeys of one coldkey
    out = compute_keepalive(st, [DF], _snap(subs, same), {DF: 0.0}, now=NOW)[DF]
    assert [u for u, _ in out] == [3, 1, 5]                           # champion, earliest pending; other coldkey 5 unaffected
    # the live coldkey counts, not the one stored at first sight
    out = compute_keepalive(st, [DF], _snap(subs), {DF: 0.0}, now=NOW)[DF]
    assert sorted(u for u, _ in out) == [1, 2, 3, 4, 5]


def test_keepalive_expires_24_hours_after_the_first_pass_or_the_crowning():
    st = ST.SpecState()
    subs = [_sub(st, 1, ST.CHAMPION, 40.0), _sub(st, 2, ST.PENDING_CONFIRM, 30.0), _sub(st, 3, ST.PENDING_CONFIRM, 31.0)]
    now = NOW
    subs[0].champion_since_ts = now - 25 * 3600.0                     # crowned 25 h ago, approval still open: expired
    subs[1].confirm_after_ts = now - 23 * 3600.0 + M.CONFIRM_DELAY_S  # first pass 23 h ago: still inside
    subs[2].confirm_after_ts = None                                   # no first-pass time: nothing (fail closed)
    subs[2].last_eval_ts = now                                        # a re-fetch moves last_eval_ts: it does not count
    out = compute_keepalive(st, [DF], _snap(subs), {DF: 0.0}, now=now)[DF]
    assert [u for u, _ in out] == [2]
    assert compute_keepalive(st, [DF], _snap(subs), {DF: 0.0}, now=now + 2 * 3600.0)[DF] == []


def test_default_floor_is_the_release_constant(monkeypatch):
    monkeypatch.setitem(CS.TRACK_ENTRY_FLOOR, DF, 33.0)
    st = ST.SpecState()
    subs = [_sub(st, 1, ST.CHAMPION, 40.0), _sub(st, 2, ST.PENDING_CONFIRM, 33.0)]
    assert compute_keepalive(st, [DF], _snap(subs), now=NOW)[DF] == [(1, 40.0)]


# ------------------------------------------------------------------------------------------------ evaluator, end to end
async def test_sybil_junk_below_the_floor_and_copies_get_nothing(tmp_path, monkeypatch):
    """10 Sybil hotkeys with junk models (below the 24.8 floor), an exact copy of a good entry and a near copy that does
    not beat its parent: only the genuine entry and the improving near copy hold a keep-alive slot."""
    monkeypatch.setitem(CS.TRACK_ENTRY_FLOOR, DF, 24.8)
    base = make_weights(1)
    write_model(str(tmp_path / "bases" / ("Qwen__Qwen3-4B@" + "b" * 40)), base)
    env = Env(tmp_path, {}, base_dir=str(tmp_path / "bases"), max_evals=40)
    good = finetune(base, 10)
    env.add(1, "a/good", 0.9, weights=good)
    env.add(2, "s/copy", 0.9, weights=good, shards=3)                    # identical tensors
    env.add(3, "s/near", 0.9, weights={k: v + 1e-4 for k, v in good.items()})   # near copy, no gain over its parent
    commits = {1: "a/good", 2: "s/copy", 3: "s/near"}
    for u in range(4, 14):
        env.add(u, f"s/junk{u}", 0.02, weights=finetune(base, 100 + u))
        commits[u] = f"s/junk{u}"
    await env.cycle({1: "a/good"}, block=100)
    await env.cycle(commits, block=110)
    s = snapshot(110, [neuron(u, commit(TRACK, r)) for u, r in commits.items()])
    active = env.cfg.active_tracks(env.clock())
    assert env.sub(2).status == ST.REJECTED and env.sub(3).derived_from == env.sub(1).key
    assert all(env.sub(u).score is not None and env.sub(u).score <= 24.8 for u in range(4, 14))
    alive = env.ev.keepalive(active, s)[TRACK]
    assert [u for u, _ in alive] == [1]
    w = combine_weights({50: 1.0}, None, 0.0, {}, keepalive={TRACK: alive})
    assert set(w) == {1, 50} and w[1] == pytest.approx(0.15 * KA)


async def test_queued_gets_nothing_first_pass_gets_a_slot(tmp_path):
    env = Env(tmp_path, {})
    env.add(1, "a/m1", 0.9)
    env.add(2, "a/m2", 0.9)
    env.hub.unavailable.add("a/m2")                                   # not public yet: stays queued
    commits = {1: "a/m1", 2: "a/m2"}
    s = snapshot(1000, [neuron(u, commit(TRACK, r)) for u, r in commits.items()])
    active = env.cfg.active_tracks(env.clock())
    assert env.ev.keepalive(active, s) == {TRACK: []}               # nothing evaluated yet
    await env.cycle(commits)
    assert env.sub(1).status == ST.PENDING_CONFIRM and env.sub(2).status == ST.QUEUED
    assert [u for u, _ in env.ev.keepalive(active, s)[TRACK]] == [1]


# ------------------------------------------------------------------------------------------------ adapters and validator
class FakeEval:
    def __init__(self, tracks, leaders, alive, fail=False):
        self.cfg = C.SpecConfig(sealed_dir="", work_dir="", state_path="", tracks=tracks)
        self.clock = lambda: NOW0.timestamp()
        self._l, self._a, self.fail = leaders, alive, fail

    def leaders(self, active, snap):
        return self._l

    def keepalive(self, active, snap):
        if self.fail:
            raise ValueError("bad")
        return self._a

    async def evaluate_cycle(self, snap):
        return self._l


def test_local_adapter_hands_out_keepalive_after_a_cycle_and_after_refresh():
    ev = FakeEval(C.DEFAULT_TRACKS, {DF: RankedLeaders()}, {DF: [(20, 30.0)]})
    ad = SpecialistAdapter(ev, free_mib=lambda g: asyncio.sleep(0, result=None))
    assert ad.keepalive([DF], NOW0) == {}
    assert asyncio.run(ad.run_cycle(snap())) is True
    assert ad.keepalive([DF, RA], NOW0) == {DF: [(20, 30.0)]}
    ev.fail = True
    ad.refresh(snap())
    assert ad.keepalive([DF], NOW0) == {}                             # fail closed


def test_local_validator_weights_include_keepalive(tmp_path):
    ev = FakeEval(C.DEFAULT_TRACKS, {}, {DF: [(20, 30.0), (21, 30.0)]})
    ad = SpecialistAdapter(ev, free_mib=lambda g: asyncio.sleep(0, result=None))
    ad.refresh(snap())

    class R:
        binds_hotkey = True

    comp = EngineCompetition(R(), {"s1-fast": "ref"}, str(tmp_path / "e.json"), specialist=ad, utcnow=lambda: NOW0)
    w = comp.weights(SERVING, snap(), {0})
    assert w[20] == w[21] == pytest.approx(0.15 * KA / 2)
    assert math.fsum(w.values()) == pytest.approx(1.0)
    # nobody eligible: exactly the serving weights (the rc5 early return)
    ev._a = {}
    ad.refresh(snap())
    assert comp.weights(SERVING, snap(), {0}) == SERVING


def _spec_file(path):
    st = ST.SpecState()
    subs = [_sub(st, 1, ST.CHAMPION, 40.0, repo="a/m1"), _sub(st, 2, ST.PENDING_CONFIRM, 30.0, repo="a/m2"),
            _sub(st, 3, ST.QUEUED, None, repo="a/m3"), _sub(st, 4, ST.NOT_BETTER, 10.0, repo="a/m4")]
    ST.save(st, str(path))
    return st, subs


def test_remote_json_path_equals_local_and_pays(tmp_path, monkeypatch):
    monkeypatch.setitem(CS.TRACK_ENTRY_FLOOR, DF, 24.8)
    sp = tmp_path / SPEC_STATE_FILE
    st, subs = _spec_file(sp)
    stamp(sp, NOW0.timestamp())
    clock = Clock(NOW0.timestamp())
    ev = RM.RemoteSpecEvaluator(str(sp), clock=clock)
    s = _snap(subs)
    active = ev.cfg.active_tracks(NOW0.timestamp())
    assert ev.keepalive(active, s) == {DF: [], RA: []}               # before any leaders() read: nothing
    ev.leaders(active, s)
    assert ev.keepalive(active, s) == compute_keepalive(st, [DF, RA], s, now=NOW) == {DF: [(1, 40.0), (2, 30.0)], RA: []}
    adapter = RM.RemoteSpecAdapter(ev, free_mib=None)
    ep = tmp_path / ENGINE_STATE_FILE
    write_engine_state(ep, champion=False)
    stamp(ep, NOW0.timestamp())
    comp = RM.RemoteEngineCompetition(str(ep), specialist=adapter, utcnow=lambda: NOW0, wall=clock)
    adapter.refresh(s)
    w = comp.weights({50: 1.0}, s, set())
    assert w[1] == pytest.approx(0.15 * ((1 - KA) * 0.75 + KA / 2)) and w[2] == pytest.approx(0.15 * KA / 2)
    assert 3 not in w and 4 not in w and math.fsum(w.values()) == pytest.approx(1.0)
    # stale file: no leaders and no keep-alive
    clock.t += REMOTE_MAX_AGE_S + 60
    adapter.refresh(s)
    assert comp.weights({50: 1.0}, s, set()) == {50: 1.0}
    # garbage file: the same
    clock.t = NOW0.timestamp()
    sp.write_text("{not json")
    adapter.refresh(s)
    assert comp.weights({50: 1.0}, s, set()) == {50: 1.0}


def test_remote_awaiting_approval_still_keeps_the_champion_alive(tmp_path):
    sp = tmp_path / SPEC_STATE_FILE
    _, subs = _spec_file(sp)
    stamp(sp, NOW0.timestamp())

    class NoApproval:
        def approved(self, *a):
            return False

    ev = RM.RemoteSpecEvaluator(str(sp), clock=lambda: NOW0.timestamp())
    adapter = RM.RemoteSpecAdapter(ev, free_mib=None)
    ep = tmp_path / ENGINE_STATE_FILE
    write_engine_state(ep, champion=False)
    stamp(ep, NOW0.timestamp())
    comp = RM.RemoteEngineCompetition(str(ep), specialist=adapter, utcnow=lambda: NOW0, wall=lambda: NOW0.timestamp(),
                                      award_gate=NoApproval())
    s = _snap(subs)
    adapter.refresh(s)
    w = comp.weights({50: 1.0}, s, set())
    assert w[1] == w[2] == pytest.approx(0.15 * KA / 2)              # crowned, awaiting approval: keep-alive only
    assert w[50] == pytest.approx(1 - 0.15 * KA)


def test_refetch_at_confirmation_does_not_extend_the_keepalive_window():
    """Review finding: a re-fetch at each confirmation attempt moves last_eval_ts; the window still counts from the first pass."""
    st = ST.SpecState()
    s = _sub(st, 2, ST.PENDING_CONFIRM, 30.0)
    now = NOW
    s.confirm_after_ts = now - 39 * 3600.0 + M.CONFIRM_DELAY_S       # first pass 39 h ago
    s.last_eval_ts = now                                              # re-fetched just now for another confirmation try
    assert compute_keepalive(st, [DF], _snap([s]), {DF: 0.0}, now=now)[DF] == []
