"""1.2.0rc7: keep-alive for entries last scored before 1.2.0rc6 (no non-proc score) and re-registration of a withdrawn
model by the same coldkey (docs/SPECIALIST-TRACKS.md, "Keep-alive while you wait" and "Re-registering")."""

import pytest
from spec_helpers import HOTKEYS, commit, make_weights, neuron, snapshot
from test_keepalive import DF, NOW, RA, _snap, _sub
from test_spec_evaluator import TRACK, Env

from reax_subnet import competition_split as CS
from reax_subnet.spec import machine as M
from reax_subnet.spec import state as ST
from reax_subnet.spec.evaluator import compute_keepalive

pytestmark = pytest.mark.usefixtures("rc5_entry_floors")


def _legacy(st, uid, status, score, **kw):
    """An entry as rc5 left it: a total score, no non-proc score, no per-tier scores."""
    s = _sub(st, uid, status, score, nonproc=None, **kw)
    assert s.by_tier == {}
    return s


def _uids(out, track=DF):
    return sorted(u for u, _ in out[track])


# ------------------------------------------------------------------------------------------------ legacy keep-alive
def test_release_constants():
    assert CS.KEEPALIVE_LEGACY_FLOOR == {DF: 60.0, RA: 50.0}
    # above what a rules-only model reaches under the rc5 mix (perfect on proc, uniform elsewhere: 49.8 / 31.8)
    assert CS.KEEPALIVE_LEGACY_FLOOR[DF] > 49.8 and CS.KEEPALIVE_LEGACY_FLOOR[RA] > 31.8
    for t in (DF, RA):
        assert CS.KEEPALIVE_LEGACY_FLOOR[t] > CS.TRACK_ENTRY_FLOOR[t]
        assert CS.KEEPALIVE_LEGACY_FLOOR[t] > CS.KEEPALIVE_NONPROC_FLOOR[t]


def test_legacy_entry_qualifies_above_the_legacy_floor():
    """The live case: DF entries scored 79.5 to 89.5 on their rc5 first draw, RA 27 the same shape."""
    st = ST.SpecState()
    subs = [_legacy(st, 1, ST.PENDING_CONFIRM, 89.5), _legacy(st, 2, ST.PENDING_CONFIRM, 60.0),    # at the floor: no
            _legacy(st, 3, ST.PENDING_CONFIRM, 59.9), _legacy(st, 4, ST.CHAMPION, 79.5),
            _legacy(st, 5, ST.PENDING_CONFIRM, float("nan")),
            _legacy(st, 6, ST.PENDING_CONFIRM, 55.0, track=RA), _legacy(st, 7, ST.PENDING_CONFIRM, 50.0, track=RA),
            _legacy(st, 8, ST.PENDING_CONFIRM, 49.9, track=RA)]
    out = compute_keepalive(st, [DF, RA], _snap(subs), {DF: 24.8, RA: 33.7}, now=NOW)
    assert out[DF] == [(4, 79.5), (1, 89.5)]                       # champion first, then pending
    assert out[RA] == [(6, 55.0)]


def test_an_rc6_draw_without_nonproc_items_is_not_legacy():
    """``nonproc`` None but per-tier scores present: scored under rc6+ on a draw without non-proc items, fail closed."""
    st = ST.SpecState()
    s = _legacy(st, 1, ST.PENDING_CONFIRM, 95.0)
    s.by_tier = {"proc_sealed": 95.0}
    assert compute_keepalive(st, [DF], _snap([s]), {DF: 24.8}, now=NOW) == {DF: []}


def test_legacy_entry_keeps_every_other_condition():
    st = ST.SpecState()
    subs = [_legacy(st, 1, ST.PENDING_CONFIRM, 80.0, checked=False),            # not copy-checked
            _legacy(st, 2, ST.NOT_BETTER, 80.0), _legacy(st, 3, ST.FORMER, 80.0), _legacy(st, 4, ST.WITHDRAWN, 80.0),
            _legacy(st, 5, ST.QUEUED, 80.0), _legacy(st, 6, ST.PENDING_VERIFY, 80.0),
            _legacy(st, 7, ST.PENDING_CONFIRM, 80.0),                            # window over (first pass 25 h ago)
            _legacy(st, 8, ST.PENDING_CONFIRM, 80.0),                            # no first-pass time
            _legacy(st, 9, ST.PENDING_CONFIRM, 80.0, repo="m/old"),               # hotkey commits to something else now
            _legacy(st, 10, ST.PENDING_CONFIRM, 80.0)]                           # the one that qualifies
    subs[6].confirm_after_ts = NOW - 25 * 3600.0 + M.CONFIRM_DELAY_S
    subs[7].confirm_after_ts = None
    snap = _snap(subs)
    snap = snapshot(snap.block, [n if n.uid != 9 else neuron(9, commit(DF, "m/new")) for n in snap.neurons])
    assert compute_keepalive(st, [DF], snap, {DF: 24.8}, now=NOW) == {DF: [(10, 80.0)]}
    # the 24 h window starts at the first pass (confirm_after_ts), not at the last fetch
    subs[9].last_eval_ts = NOW
    subs[9].confirm_after_ts = NOW - 24 * 3600.0 - 1 + M.CONFIRM_DELAY_S
    assert compute_keepalive(st, [DF], snap, {DF: 24.8}, now=NOW) == {DF: []}


def test_legacy_entries_share_the_coldkey_limit_with_rc6_entries():
    st = ST.SpecState()
    subs = [_legacy(st, 1, ST.PENDING_CONFIRM, 80.0, coldkey="c", block=1),
            _sub(st, 2, ST.PENDING_CONFIRM, 50.0, coldkey="c", block=2),          # rc6 shape (non-proc 99)
            _legacy(st, 3, ST.PENDING_CONFIRM, 85.0, coldkey="c", block=3)]
    assert compute_keepalive(st, [DF], _snap(subs), {DF: 24.8}, now=NOW) == {DF: [(1, 80.0), (2, 50.0)]}


def test_legacy_floor_parameter_and_missing_floor():
    st = ST.SpecState()
    subs = [_legacy(st, 1, ST.PENDING_CONFIRM, 70.0)]
    assert _uids(compute_keepalive(st, [DF], _snap(subs), {DF: 24.8}, legacy_floors={DF: 75.0}, now=NOW)) == []
    assert _uids(compute_keepalive(st, [DF], _snap(subs), {DF: 24.8}, legacy_floors={DF: 65.0}, now=NOW)) == [1]
    # a track without a legacy floor: a legacy entry gets no slot (fail closed)
    assert _uids(compute_keepalive(st, [DF], _snap(subs), {DF: 24.8}, legacy_floors={}, now=NOW)) == []
    # the non-proc floors do not apply to a legacy entry; an rc6 entry is unaffected by the legacy floor
    assert _uids(compute_keepalive(st, [DF], _snap(subs), {DF: 24.8}, nonproc_floors={DF: 99.0}, now=NOW)) == [1]
    s2 = _sub(st, 2, ST.PENDING_CONFIRM, 40.0, nonproc=55.0)
    assert _uids(compute_keepalive(st, [DF], _snap(subs + [s2]), {DF: 24.8}, legacy_floors={DF: 99.0}, now=NOW)) == [2]


def test_legacy_entry_through_the_state_file(tmp_path):
    """A state written by rc5 (no ``nonproc``/``by_tier`` keys) loads into the legacy shape."""
    st = ST.SpecState()
    s = _legacy(st, 1, ST.PENDING_CONFIRM, 83.8)
    path = str(tmp_path / "spec.json")
    ST.save(st, path)
    import json
    raw = json.load(open(path))
    sub = next(iter(raw["tracks"][DF]["submissions"].values()))
    del sub["nonproc"], sub["by_tier"]
    json.dump(raw, open(path, "w"))
    st2 = ST.load(path)
    s2 = st2.track(DF).submissions[s.key]
    assert s2.nonproc is None and s2.by_tier == {}
    assert compute_keepalive(st2, [DF], _snap([s2]), {DF: 24.8}, now=NOW) == {DF: [(1, 83.8)]}


# ------------------------------------------------------------------------------------------------ re-registration
COLD = "5ColdSame" + "z" * 40


def _n(uid, repo, *, hotkey_of=None, coldkey=COLD):
    return neuron(uid, commit(TRACK, repo), coldkey=coldkey, hotkey=HOTKEYS[hotkey_of if hotkey_of is not None else uid])


async def _first_pass_then_dereg(env, w):
    env.add(1, "a/m1", 0.9, weights=w)
    await env.ev.evaluate_cycle(snapshot(100, [_n(1, "a/m1")]))
    first = env.sub(1)
    assert first.status == ST.PENDING_CONFIRM
    await env.ev.evaluate_cycle(snapshot(200, []))                   # deregistered: the hotkey left
    assert first.status == ST.WITHDRAWN
    return first


def _subs(env):
    return list(env.ev.state.track(TRACK).submissions.values())


@pytest.fixture
def env(tmp_path):
    return Env(tmp_path, {})


async def test_same_hotkey_re_registers_on_a_new_uid(env):
    w = make_weights(7)
    first = await _first_pass_then_dereg(env, w)
    first_key = first.key
    await env.ev.evaluate_cycle(snapshot(300, [_n(5, "a/m1", hotkey_of=1)]))
    new = env.sub(5)
    assert new.key == first_key and new is not first                 # a new entry under the commitment's key
    assert new.status == ST.PENDING_CONFIRM and new.first_seen_block == 300 and new.draw1 == first.draw1
    assert "identical tensors" not in new.reason
    # the withdrawn entry stays on record, with its carried first pass linked
    old = env.ev.state.track(TRACK).submissions[f"{first_key}#withdrawn@100"]
    assert old is first and old.status == ST.WITHDRAWN and old.key == f"{first_key}#withdrawn@100"
    assert set(env.ev.state.track(TRACK).hashes.values()) == {new.key}
    assert any("committed again after its withdrawal" in e for e in env.ev.state.track(TRACK).events)


async def test_same_hotkey_and_commitment_is_not_stuck_withdrawn_after_a_reload(env):
    w = make_weights(7)
    key = (await _first_pass_then_dereg(env, w)).key
    ST.save(env.ev.state, env.cfg.state_path)
    env.ev = env.make()
    await env.ev.evaluate_cycle(snapshot(300, [_n(1, "a/m1")]))
    subs = env.ev.state.track(TRACK).submissions
    assert subs[key].status == ST.PENDING_CONFIRM and subs[key].first_seen_block == 300
    assert subs[f"{key}#withdrawn@100"].status == ST.WITHDRAWN


async def test_new_hotkey_same_coldkey_same_tensors_passes_the_copy_check(env):
    w = make_weights(7)
    first = await _first_pass_then_dereg(env, w)
    env.add(2, "a/m1-again", 0.9, weights=w, shards=3)                 # same tensors, new repo bound to the new hotkey
    await env.ev.evaluate_cycle(snapshot(300, [_n(2, "a/m1-again")]))
    new = env.sub(2)
    assert new.status == ST.PENDING_CONFIRM, new.reason
    assert new.hash_named == first.hash_named and new.first_seen_block == 300
    assert first.status == ST.WITHDRAWN                               # not re-labelled as a copy
    assert env.ev.state.track(TRACK).hashes[new.hash_named] == new.key
    # the hash now belongs to the new entry: another coldkey re-using it is refused
    env.add(3, "a/thief", 0.9, weights=w)
    await env.ev.evaluate_cycle(snapshot(400, [_n(2, "a/m1-again"), _n(3, "a/thief", coldkey="5ColdOther" + "q" * 40)]))
    assert env.sub(3).status == ST.REJECTED and "identical tensors to an earlier commitment (uid 2)" in env.sub(3).reason
    assert env.sub(2).status == ST.PENDING_CONFIRM


async def test_other_coldkey_re_using_a_withdrawn_hash_stays_rejected(env):
    w = make_weights(7)
    await _first_pass_then_dereg(env, w)
    env.add(2, "a/copy", 0.9, weights=w)
    await env.ev.evaluate_cycle(snapshot(300, [_n(2, "a/copy", coldkey="5ColdOther" + "q" * 40)]))
    assert env.sub(2).status == ST.REJECTED and "identical tensors to an earlier commitment (uid 1)" in env.sub(2).reason


async def test_same_repo_under_another_coldkey_proves_control(env):
    """The checked manifest binds the recommit to the same HF repo, even after a coldkey change."""
    w = make_weights(7)
    await _first_pass_then_dereg(env, w)
    await env.ev.evaluate_cycle(snapshot(300, [_n(5, "a/m1", hotkey_of=1, coldkey="5ColdOther" + "q" * 40)]))
    assert env.sub(5).status == ST.PENDING_CONFIRM and env.sub(5).carried_from


async def test_live_entry_of_the_same_coldkey_still_blocks_a_second_copy(env):
    """Only a WITHDRAWN entry is released: a coldkey cannot enter the same tensors twice side by side."""
    w = make_weights(7)
    env.add(1, "a/m1", 0.9, weights=w)
    env.add(2, "a/m1-twin", 0.9, weights=w)
    await env.ev.evaluate_cycle(snapshot(100, [_n(1, "a/m1")]))
    await env.ev.evaluate_cycle(snapshot(110, [_n(1, "a/m1"), _n(2, "a/m1-twin")]))
    assert env.sub(2).status == ST.REJECTED and "identical tensors" in env.sub(2).reason
    assert env.sub(1).status == ST.PENDING_CONFIRM
