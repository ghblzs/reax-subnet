"""Mechanism v2 anti-churn: provisional keep-alive for entries that wait for their first evaluation."""

import json
import math

import pytest
from spec_helpers import HOTKEYS, REV, neuron, snapshot
from test_keepalive import DF, NOW, RA, _snap, _sub

from reax_subnet import competition_split as CS
from reax_subnet.chain import Neuron, Snapshot
from reax_subnet.competition_split import RankedLeaders, combine_weights
from reax_subnet.spec import evaluator as E
from reax_subnet.spec import state as ST
from reax_subnet.spec.commitment import SpecCommitment
from reax_subnet.spec.evaluator import compute_keepalive, compute_provisional, mark_provisional

pytestmark = pytest.mark.usefixtures("rc5_entry_floors")
IMM, LEAD = E.IMMUNITY_BLOCKS_ASSUMED, CS.PROVISIONAL_KA_LEAD_BLOCKS
BLOCK = 20000
REG_DUE = BLOCK - IMM + LEAD - 10       # immunity ends in 590 blocks: inside the lead
SERVING = {11: 0.4, 12: 0.3, 13: 0.2, 14: 0.1}   # median of positives 0.25
H = 3600.0


def _p(st, uid, *, status=ST.QUEUED, since=NOW - H, track=DF, coldkey=None, **kw):
    s = _sub(st, uid, status, None, coldkey=coldkey, track=track, **kw)
    s.hash_named = f"h{uid}"
    s.provisional_since_ts = since
    s.wait_kind = ST.WAIT_INFRA if status == ST.PENDING_VERIFY else ""
    return s


def _snapr(subs, regs=None, cold=None, cblocks=None, block=BLOCK):
    regs, cold, cblocks = regs or {}, cold or {}, cblocks or {}
    ns = []
    for s in subs:
        n = neuron(s.uid, SpecCommitment(s.track_id, s.repo, REV).text(), coldkey=cold.get(s.uid, s.coldkey))
        ns.append(Neuron(n.uid, n.hotkey, n.coldkey, False, 0.0, None, None, commitment=n.commitment,
                         commitment_block=cblocks.get(s.uid, 0), registered_block=regs.get(s.uid, REG_DUE)))
    return snapshot(block, ns)


def _uids(st, snap, now=NOW):
    return compute_provisional(st, [DF], snap, now=now)[DF]


def test_release_constants():
    assert CS.PROVISIONAL_KA is True
    assert (CS.PROVISIONAL_KA_LEAD_BLOCKS, CS.PROVISIONAL_KA_MAX_HOURS) == (600, 12.0)
    assert (CS.PROVISIONAL_KA_FRACTION, CS.PROVISIONAL_KA_MAX_SHARE) == (0.5, 0.05)


# ------------------------------------------------------------------------------------------ eligibility
def test_queued_and_infra_waiting_entries_are_eligible():
    st = ST.SpecState()
    subs = [_p(st, 1), _p(st, 2, status=ST.PENDING_VERIFY, checked=False)]
    assert _uids(st, _snapr(subs)) == [1, 2]


@pytest.mark.parametrize("mutate", [
    lambda s: setattr(s, "status", ST.REJECTED),                       # failed a rule/copy check
    lambda s: setattr(s, "status", ST.WITHDRAWN),
    lambda s: setattr(s, "status", ST.NOT_BETTER),
    lambda s: setattr(s, "status", ST.PENDING_CONFIRM),                # already scored: the normal keep-alive
    lambda s: setattr(s, "status", ST.CHAMPION),
    lambda s: setattr(s, "delta_checked", False),                      # queued, copy check not passed yet
    lambda s: setattr(s, "score", 55.0),                               # scored on a first draw
    lambda s: setattr(s, "provisional_since_ts", None),                # window start unknown: fail closed
])
def test_each_condition_is_needed(mutate):
    st = ST.SpecState()
    s = _p(st, 1)
    assert _uids(st, _snapr([s])) == [1]
    mutate(s)
    assert _uids(st, _snapr([s])) == []


def _unfetched(st, uid, **kw):
    s = _p(st, uid, **kw)
    s.hash_named, s.delta_checked = "", False
    return s


def test_unfetched_queued_entry_is_eligible():
    st = ST.SpecState()
    s = _unfetched(st, 1)
    assert _uids(st, _snapr([s])) == [1]
    assert mark_provisional(st, [DF], _snapr([s]), NOW + 5) is False   # window start already set; nothing new
    s.provisional_since_ts = None
    assert mark_provisional(st, [DF], _snapr([s]), NOW + 5) is True and s.provisional_since_ts == NOW + 5


@pytest.mark.parametrize("status", [ST.REJECTED, ST.NOT_BETTER, ST.WITHDRAWN, ST.PENDING_VERIFY, ST.PENDING_CONFIRM])
def test_unfetched_entry_needs_to_be_queued_and_a_rejected_key_never_is(status):
    st = ST.SpecState()
    s = _unfetched(st, 1)
    s.wait_kind = ""
    s.status = status
    assert _uids(st, _snapr([s])) == []
    s.provisional_since_ts = None
    assert mark_provisional(st, [DF], _snapr([s]), NOW) is False and s.provisional_since_ts is None


def test_unfetched_entry_with_an_invalid_or_foreign_commitment_is_not_eligible():
    st = ST.SpecState()
    s = _unfetched(st, 1)
    snap = _snapr([s])

    def with_commitment(text):
        return snapshot(BLOCK, [Neuron(n.uid, n.hotkey, n.coldkey, False, 0.0, None, None, commitment=text,
                                       registered_block=REG_DUE) for n in snap.neurons])
    assert _uids(st, with_commitment("not a commitment")) == []
    assert _uids(st, with_commitment("")) == []
    assert _uids(st, with_commitment(SpecCommitment(DF, "m/other", REV).text())) == []   # changed commitment
    assert _uids(st, snapshot(BLOCK, [])) == []                                         # hotkey gone


def test_unfetched_entry_over_the_open_limit_is_not_eligible_and_still_obeys_the_other_limits():
    from reax_subnet.spec import config as C
    st = ST.SpecState()
    subs = [_unfetched(st, u, coldkey="ck") for u in range(1, C.MAX_OPEN_PER_COLDKEY_TRACK + 2)]
    assert _uids(st, _snapr(subs)) == []        # more open entries than the queue limit lets in: nobody is paid
    st = ST.SpecState()
    a, b = _unfetched(st, 1, coldkey="ck"), _unfetched(st, 2, coldkey="ck")
    assert len(_uids(st, _snapr([a, b]))) == 1  # one per coldkey and track
    a.provisional_since_ts = NOW - 13 * H
    assert _uids(st, _snapr([a])) == []         # 12 h limit
    s = _unfetched(st, 3, coldkey="ck3")
    assert _uids(st, _snapr([s], regs={3: BLOCK - IMM + LEAD + 50})) == []   # immunity not close to its end
    s.score = 50.0
    assert _uids(st, _snapr([s])) == []


def test_fetched_entry_that_then_fails_a_check_stops_at_once():
    st = ST.SpecState()
    s = _p(st, 1)
    assert _uids(st, _snapr([s])) == [1]
    s.status, s.reason = ST.REJECTED, "identical tensors to an earlier commitment"
    assert _uids(st, _snapr([s])) == []
    t = _unfetched(st, 2)                       # unfetched, then fetched and rejected: same
    assert _uids(st, _snapr([t])) == [2]
    t.hash_named, t.status = "h2", ST.REJECTED
    assert _uids(st, _snapr([t])) == []


def test_fetched_entries_keep_the_stricter_checks():
    st = ST.SpecState()
    s = _p(st, 1)
    s.delta_checked = False                     # fetched but copy check not done: nothing
    assert _uids(st, _snapr([s])) == []


def test_pending_verification_on_the_miners_side_is_not_eligible():
    st = ST.SpecState()
    s = _p(st, 1, status=ST.PENDING_VERIFY, checked=False)
    s.wait_kind = ST.WAIT_MINER
    assert _uids(st, _snapr([s])) == []


def test_current_commitment_must_be_the_submissions():
    st = ST.SpecState()
    s = _p(st, 1)
    snap = _snapr([s])
    other = snapshot(BLOCK, list(
        Neuron(n.uid, n.hotkey, n.coldkey, False, 0.0, None, None,
               commitment=SpecCommitment(DF, "m/other", REV).text(), registered_block=REG_DUE) for n in snap.neurons))
    assert _uids(st, other) == []
    gone = snapshot(BLOCK, [])
    assert _uids(st, gone) == []


def test_immunity_window():
    st = ST.SpecState()
    s = _p(st, 1)
    end = lambda reg: reg + IMM
    assert _uids(st, _snapr([s], {1: BLOCK - IMM + LEAD})) == [1]           # ends in exactly LEAD blocks
    assert _uids(st, _snapr([s], {1: BLOCK - IMM + LEAD + 1})) == []        # 601 blocks left: too early
    assert _uids(st, _snapr([s], {1: BLOCK - IMM - 5000})) == [1]           # long over
    assert _uids(st, _snapr([s], {1: 0})) == []                              # unknown registration
    assert end(BLOCK - IMM) == BLOCK


def test_twelve_hour_limit():
    st = ST.SpecState()
    s = _p(st, 1, since=NOW - 12 * H)
    assert _uids(st, _snapr([s])) == [1]
    assert _uids(st, _snapr([s]), now=NOW + 1) == []
    s.provisional_since_ts = NOW + 2 * H                                    # clock jump: fail closed
    assert _uids(st, _snapr([s])) == []


def test_one_per_coldkey_and_track():
    st = ST.SpecState()
    subs = [_p(st, 1), _p(st, 2), _p(st, 3)]
    snap = _snapr(subs, cold={1: "A", 2: "A", 3: "B"}, cblocks={1: 50, 2: 40, 3: 60})
    assert _uids(st, snap) == [2, 3]                                         # A: earlier commitment block wins
    ra = _p(st, 4, track=RA)
    out = compute_provisional(st, [DF, RA], _snapr(subs + [ra], cold={1: "A", 2: "A", 3: "B", 4: "A"}), now=NOW)
    assert out[RA] == [4]                                                    # the limit is per track


def test_does_not_use_the_top8_slots():
    st = ST.SpecState()
    ka = [_sub(st, u, ST.PENDING_CONFIRM, 50.0, nonproc=70.0 + u, block=u) for u in range(10, 19)]
    pv = [_p(st, 1), _p(st, 2)]
    snap = _snapr(ka + pv)
    assert len(compute_keepalive(st, [DF], snap, {DF: 0.0}, now=NOW)[DF]) == CS.KEEPALIVE_MAX_ENTRIES
    assert _uids(st, snap) == [1, 2]


def test_off_switch(monkeypatch):
    st = ST.SpecState()
    s = _p(st, 1)
    monkeypatch.setattr(E, "PROVISIONAL_KA", False)
    assert _uids(st, _snapr([s])) == []
    assert mark_provisional(st, [DF], _snapr([s]), NOW) is False


# ------------------------------------------------------------------------------------------ the window start
def test_mark_sets_the_start_once_and_only_for_eligible_entries():
    st = ST.SpecState()
    a, b, c = _p(st, 1, since=None), _p(st, 2, since=None), _p(st, 3, since=None)
    c.status = ST.REJECTED
    snap = _snapr([a, b, c], regs={2: BLOCK - IMM + LEAD + 100})            # b: immunity far from over
    assert mark_provisional(st, [DF], snap, NOW) is True
    assert a.provisional_since_ts == NOW and b.provisional_since_ts is None and c.provisional_since_ts is None
    assert mark_provisional(st, [DF], snap, NOW + 100) is False             # set once
    assert a.provisional_since_ts == NOW


def test_old_state_files_load_and_pay_nothing_until_marked(tmp_path):
    st = ST.SpecState()
    s = _p(st, 1, since=None)
    raw = ST._to_json(st)
    for t in raw["tracks"].values():
        for sub in t["submissions"].values():
            del sub["provisional_since_ts"]                                   # a file written before this release
    path = str(tmp_path / "state.json")
    with open(path, "w") as fh:
        json.dump(raw, fh)
    loaded = ST.load(path)
    sub = next(iter(loaded.track(DF).submissions.values()))
    assert sub.provisional_since_ts is None
    assert compute_provisional(loaded, [DF], _snapr([s]), now=NOW)[DF] == []
    # and a newer file with unknown keys loads, the start survives a save/load round trip
    sub.provisional_since_ts = NOW
    ST.save(loaded, path)
    again = ST.load(path)
    assert next(iter(again.track(DF).submissions.values())).provisional_since_ts == NOW


# ------------------------------------------------------------------------------------------ pay
MANY = {100 + i: 1.0 for i in range(100)}       # median serving weight 0.01 of the miner part
CAP = 0.15 * CS.PROVISIONAL_KA_MAX_SHARE         # one track's provisional total at most 0.0075 of the weights


def test_pay_is_half_the_median_and_comes_from_the_track_share():
    w = combine_weights(MANY, None, 0.0, {}, provisional={DF: [5]})
    assert w[5] == pytest.approx(0.5 * 0.01)
    assert math.fsum(w.values()) == pytest.approx(1.0)
    # the other track and the inference part keep what is theirs: only 0.005 left the serving miners, pro rata
    assert w[100] == pytest.approx(0.01 * (1 - 0.005))


def test_pay_is_the_same_median_as_the_keepalive_entries():
    alive = {DF: [(5, 50.0)]}
    w = combine_weights(MANY, None, 0.0, {}, keepalive=alive, provisional={DF: [6]})
    assert w[5] == pytest.approx(0.01) and w[6] == pytest.approx(0.005)


def test_cap_splits_evenly():
    w = combine_weights({100: 1.0}, None, 0.0, {}, provisional={DF: [5, 6, 7]})   # no typical miner: the cap decides
    assert w[5] == pytest.approx(w[6]) == pytest.approx(w[7])
    assert w[5] + w[6] + w[7] == pytest.approx(CAP)


def test_provisional_total_is_taken_before_the_ranks_and_never_from_another_track():
    leaders = {DF: RankedLeaders([(20, 40.0)]), RA: RankedLeaders([(21, 40.0)])}
    base = combine_weights(MANY, None, 0.0, leaders)
    w = combine_weights(MANY, None, 0.0, leaders, provisional={DF: [5]})
    assert w[5] == pytest.approx(0.005)
    assert w[21] == pytest.approx(base[21], rel=1e-3)        # RA's winner: only the serving normalisation moved it
    # DF's winner is paid from the rest of DF's share: 0.15 x 0.75 less its 0.005 / 0.15 part
    assert w[20] == pytest.approx(base[20] - 0.75 * 0.005, rel=1e-3)
    assert math.fsum(w.values()) == pytest.approx(1.0)


def test_it_does_not_count_against_the_top8_and_leaves_their_pay_alone():
    alive = {DF: [(u, 50.0) for u in range(1, 9)]}
    a = combine_weights(MANY, None, 0.0, {}, keepalive=alive)
    b = combine_weights(MANY, None, 0.0, {}, keepalive=alive, provisional={DF: [30]})
    assert all(b[u] == pytest.approx(a[u]) for u in range(1, 9)) and b[30] > 0


def test_owner_and_already_paid_uids_get_no_double_slot():
    alive = {DF: [(5, 50.0)]}
    w = combine_weights(MANY, None, 0.0, {}, keepalive=alive, provisional={DF: [5, 0]}, owner_uid=0)
    assert w[5] == pytest.approx(0.01) and 0 not in w


def test_unclaimed_goes_to_serving_and_nothing_is_bit_identical():
    assert combine_weights(MANY, None, 0.0, {}, provisional=None) == combine_weights(MANY, None, 0.0, {})
    assert combine_weights(MANY, None, 0.0, {}, provisional={DF: []}) == combine_weights(MANY, None, 0.0, {})


def test_off_switches(monkeypatch):
    plain = combine_weights(MANY, None, 0.0, {})
    assert combine_weights(MANY, None, 0.0, {}, provisional={DF: [5]}, keepalive_share=0.0) == plain
    monkeypatch.setattr(CS, "PROVISIONAL_KA", False)
    assert combine_weights(MANY, None, 0.0, {}, provisional={DF: [5]}) == plain


# ------------------------------------------------------------------------------------------ plumbing
def test_remote_state_file_to_weights(tmp_path):
    from test_remote_competitions import NOW0, Clock, stamp, write_engine_state
    from reax_subnet import remote as RM
    from reax_subnet.config import ENGINE_STATE_FILE

    st = ST.SpecState()
    subs = [_p(st, 1), _p(st, 2, since=NOW - 13 * H)]                       # uid 2: its 12 hours are over
    sp = tmp_path / "spec-state.json"
    ST.save(st, str(sp))
    stamp(sp, NOW0.timestamp())
    clock = Clock(NOW0.timestamp())
    ev = RM.RemoteSpecEvaluator(str(sp), clock=clock)
    adapter = RM.RemoteSpecAdapter(ev, free_mib=None)
    ep = tmp_path / ENGINE_STATE_FILE
    write_engine_state(ep, champion=False)
    stamp(ep, NOW0.timestamp())
    comp = RM.RemoteEngineCompetition(str(ep), specialist=adapter, utcnow=lambda: NOW0, wall=clock)
    s = _snapr(subs, block=BLOCK)
    adapter.refresh(s)
    assert adapter.provisional([DF], NOW0) == {DF: [1]}
    w = comp.weights(dict(MANY), s, set())
    assert w[1] == pytest.approx(0.005) and 2 not in w
    assert math.fsum(w.values()) == pytest.approx(1.0)
    # stale file: nothing is paid
    clock.t += 10**6
    adapter.refresh(s)
    assert adapter.provisional([DF], NOW0) == {}


async def test_evaluator_persists_the_window_start_and_never_pays_a_scored_entry(tmp_path):
    from test_spec_evaluator import TRACK, Env, T0
    from spec_helpers import commit

    env = Env(tmp_path, {})
    env.add(1, "a/m1", 0.9)
    await env.cycle({1: "a/m1"})                                # fetched, checked, scored on its first draw
    s = env.sub(1)
    assert s.score is not None and s.provisional_since_ts is None
    n = neuron(1, commit(TRACK, "a/m1"))
    snap = snapshot(BLOCK, [Neuron(n.uid, n.hotkey, n.coldkey, False, 0.0, None, None, commitment=n.commitment,
                                   registered_block=REG_DUE)])
    active = env.cfg.active_tracks(env.clock())
    assert env.ev.provisional(active, snap) == {TRACK: []}      # scored: the normal keep-alive decides, not this
    # an entry that waits for its first evaluation (fetched and checked, not scored)
    s.score, s.status = None, ST.QUEUED
    s.delta_checked, s.hash_named = True, s.hash_named or "h1"
    assert env.ev.provisional(active, snap) == {TRACK: [1]}
    assert s.provisional_since_ts == env.clock()
    on_disk = ST.load(env.cfg.state_path)
    assert next(iter(on_disk.track(TRACK).submissions.values())).provisional_since_ts == env.clock()
    env.clock.advance(12 * H + 1)
    assert env.ev.provisional(active, snap) == {TRACK: []}      # 12 hours are over


# ------------------------------------------------------------------ review delta: one window per hotkey (provka3)
def _recommit(st, old, repo, track=DF):
    """The hotkey commits again: a new Submission under the same hotkey, the old one withdrawn (as ``_sync`` does)."""
    old.status = ST.WITHDRAWN
    s = _sub(st, old.uid, ST.QUEUED, None, track=track, repo=repo, coldkey=old.coldkey)
    s.hash_named, s.delta_checked, s.wait_kind = f"h{repo}", True, ""
    return s


def test_recommit_does_not_restart_the_hotkey_window():
    st = ST.SpecState()
    a = _p(st, 1, since=NOW)
    snap = _snapr([a])
    assert _uids(st, snap, now=NOW + 11 * H) == [1]
    b = _recommit(st, a, "m/new")
    snap = _snapr([b])
    assert mark_provisional(st, [DF], snap, NOW + 11 * H) is True
    assert b.provisional_since_ts == NOW                                   # inherited, not NOW + 11 h
    assert _uids(st, snap, now=NOW + 11.9 * H) == [1]
    assert _uids(st, snap, now=NOW + 12.1 * H) == []                       # paid time stays <= 12 h in total
    c = _recommit(st, b, "m/newer")
    assert mark_provisional(st, [DF], _snapr([c]), NOW + 13 * H) is True
    assert _uids(st, _snapr([c]), now=NOW + 13 * H) == []                  # every later recommit is dead too


def test_window_is_also_shared_across_tracks_and_a_stale_own_start_cannot_extend_it():
    st = ST.SpecState()
    a = _p(st, 1, since=NOW)
    other = _p(st, 1, since=NOW + 10 * H, track=RA, repo="m/ra")           # same hotkey, later start in another track
    snap = _snapr([a, other])
    out = compute_provisional(st, [DF, RA], snap, now=NOW + 13 * H)
    assert out == {DF: [], RA: []}                                         # the hotkey's earliest start rules
    fresh = _recommit(st, other, "m/ra2", track=RA)
    assert mark_provisional(st, [DF, RA], _snapr([fresh]), NOW + 13 * H) is True
    assert fresh.provisional_since_ts == NOW


def test_other_hotkey_of_the_same_coldkey_is_unaffected_except_for_the_one_per_coldkey_rule():
    st = ST.SpecState()
    a = _p(st, 1, since=NOW - 20 * H, coldkey="ck")                        # hotkey 1: window long over
    b = _unfetched(st, 2, since=None, coldkey="ck")                        # another hotkey, same coldkey, new
    snap = _snapr([a, b])
    assert mark_provisional(st, [DF], snap, NOW) is True and b.provisional_since_ts == NOW
    assert _uids(st, snap) == [2]                                          # hotkey 2 gets its own fresh 12 h
    a.status = ST.WITHDRAWN                                                # keeps the open-entry count low
    c = _p(st, 3, since=NOW - H, coldkey="ck")
    assert _uids(st, _snapr([b, c])) == [2]                                # still one per coldkey and track (earlier uid)


def test_new_hotkey_registration_is_a_new_hotkey():
    st = ST.SpecState()
    old = _p(st, 1, since=NOW - 20 * H)
    old.status = ST.WITHDRAWN
    new = _unfetched(st, 2, since=None, coldkey=old.coldkey)               # fresh registration: other hotkey string
    assert mark_provisional(st, [DF], _snapr([new]), NOW) is True
    assert new.provisional_since_ts == NOW and _uids(st, _snapr([new])) == [2]


# ------------------------------------------------------------------ review delta: "waiting for the repo" (provka3)
def test_unfetched_entry_waiting_for_the_repo_is_not_eligible_until_a_fetch_succeeds():
    st = ST.SpecState()
    s = _unfetched(st, 1)
    assert _uids(st, _snapr([s])) == [1]                                   # never attempted: eligible
    s.reason = "waiting for the repo: 404 not found"                       # a fetch attempt found no repo
    assert _uids(st, _snapr([s])) == []
    s.provisional_since_ts = None
    assert mark_provisional(st, [DF], _snapr([s]), NOW) is False and s.provisional_since_ts is None
    # a later attempt succeeds: the fetched-entry rules apply (reason is irrelevant then)
    s.hash_named, s.delta_checked = "h1", True
    s.provisional_since_ts = NOW - H
    assert _uids(st, _snapr([s])) == [1]
    s.delta_checked = False                                                # fetched, but the checks have not passed
    assert _uids(st, _snapr([s])) == []


async def test_evaluator_marks_after_sync_before_the_tracks_are_evaluated(tmp_path):
    from test_spec_evaluator import TRACK, Env
    from spec_helpers import commit

    env = Env(tmp_path, {})
    env.add(1, "a/m1", 0.9)
    n = neuron(1, commit(TRACK, "a/m1"))
    snap = snapshot(BLOCK, [Neuron(n.uid, n.hotkey, n.coldkey, False, 0.0, None, None, commitment=n.commitment,
                                   registered_block=REG_DUE)])
    seen = []
    real = env.ev._run_track

    async def spy(tdef, snapshot_, now):
        seen.append([x.provisional_since_ts for x in env.ev.state.track(TRACK).submissions.values()])
        return await real(tdef, snapshot_, now)

    env.ev._run_track = spy
    await env.ev.evaluate_cycle(snap)
    assert seen and seen[0] == [env.clock()]            # already marked when the first track starts
