"""Rule 17 Oct: a pruned pending confirmation can reclaim its first draw within 72 hours."""
import shutil
import dataclasses

import pytest
from spec_helpers import HOTKEYS, commit, make_weights, neuron, snapshot
from test_spec_evaluator import TRACK, Env

from reax_subnet.spec import evaluator as E
from reax_subnet.spec import state as ST

pytestmark = pytest.mark.usefixtures("rc5_entry_floors")
H = 3600


async def pruned(tmp_path):
    env = Env(tmp_path, {})
    env.add(1, "a/old", 0.9, weights=make_weights(7))
    await env.cycle({1: "a/old"}, block=100)
    old = env.sub(1)
    old.provisional_since_ts = env.clock.t - H
    env.clock.advance(13 * H)
    await env.cycle({}, block=200)
    assert old.withdrawn_kind == "pruned" and old.withdrawn_status == ST.PENDING_CONFIRM
    return env, old


def new_snapshot(repo, coldkey=None, block=300):
    return snapshot(block, [neuron(2, commit(TRACK, repo, rev="b" * 40), coldkey=coldkey)])


@pytest.mark.parametrize("same_repo", [False, True])
async def test_carries_first_draw_after_checks_and_confirms_next_cycle(tmp_path, same_repo):
    env, old = await pruned(tmp_path)
    repo = old.repo if same_repo else "a/recommit"
    coldkey = "different-coldkey" if same_repo else old.coldkey
    if same_repo:
        shutil.rmtree(env.hub.repos[repo])  # replace the fixture revision, including its old shard files
    env.add(2, repo, 0.9, weights=make_weights(7), shards=3)
    snap = new_snapshot(repo, coldkey)
    env.server.served.clear()
    await env.ev.evaluate_cycle(snap)
    new = env.sub(2)
    assert new.status == ST.PENDING_CONFIRM and new.draw1 == old.draw1, new.reason
    assert new.carried_from == old.key and old.carried_to == new.key
    assert new.confirm_after_ts == env.clock.t and not env.server.served
    for field in ("score", "delta", "first_gain", "first_incumbent", "nonproc", "by_tier", "provisional_since_ts"):
        assert getattr(new, field) == getattr(old, field)
    assert new.history == old.history and new.history is not old.history
    assert new.by_tier is not old.by_tier
    # Cohorts remain first-seen blocks; sharing the first draw only groups confirmations.
    assert new.first_seen_block == 300 and old.first_seen_block == 100
    assert any("first draw carried over (rule 17 Oct)" in e for e in env.ev.state.track(TRACK).events)
    await env.ev.evaluate_cycle(snap)
    assert new.status == ST.CHAMPION
    assert [h["kind"] for h in new.history] == ["first", "confirm"]
    draws = env.ev.state.track(TRACK).draws
    assert not set(draws[new.draw1].item_ids) & set(draws[new.history[-1]["draw"]].item_ids)
    assert not E.carryover_eligible(old, new, {HOTKEYS[2]}, env.clock.t)


@pytest.mark.parametrize("expired", [False, True])
async def test_another_repo_and_coldkey_remains_a_copy(tmp_path, expired):
    env, old = await pruned(tmp_path)
    if expired:
        env.clock.advance(72 * H + 1)
    env.add(2, "b/copy", 0.9, weights=make_weights(7))
    await env.ev.evaluate_cycle(new_snapshot("b/copy", "another-coldkey"))
    assert env.sub(2).status == ST.REJECTED
    assert "identical tensors" in env.sub(2).reason and not old.carried_to


async def test_expired_prune_needs_new_first_draw(tmp_path):
    env, old = await pruned(tmp_path)
    env.clock.advance(72 * H + 1)
    env.add(2, "a/recommit", 0.9, weights=make_weights(7))
    await env.ev.evaluate_cycle(new_snapshot("a/recommit", old.coldkey))
    new = env.sub(2)
    assert new.status == ST.PENDING_CONFIRM and new.draw1 != old.draw1 and not new.carried_from
    assert not old.carried_to


async def test_changed_commitment_does_not_carry(tmp_path):
    env = Env(tmp_path, {})
    env.add(1, "a/old", 0.9, weights=make_weights(7))
    await env.cycle({1: "a/old"}, block=100)
    old = env.sub(1)
    await env.ev.evaluate_cycle(snapshot(200, [neuron(1)]))  # still registered, no specialist commitment
    assert old.withdrawn_kind == "changed"
    env.add(2, "a/recommit", 0.9, weights=make_weights(7))
    await env.ev.evaluate_cycle(new_snapshot("a/recommit", old.coldkey))
    assert env.sub(2).draw1 != old.draw1 and not env.sub(2).carried_from


async def test_legacy_state_and_retro_boundary(tmp_path):
    env, old = await pruned(tmp_path)
    raw = ST._to_json(env.ev.state)
    for entry in raw["tracks"][TRACK]["submissions"].values():
        for field in ("withdrawn_ts", "withdrawn_kind", "withdrawn_status", "carried_from", "carried_to"):
            entry.pop(field)
    loaded = ST.from_json(raw)
    old = loaded.track(TRACK).submissions[old.key]
    new = dataclasses.replace(old, key="new", status=ST.QUEUED, hotkey=HOTKEYS[2])
    assert E.carryover_eligible(old, new, {HOTKEYS[2]}, E.CARRYOVER_RETRO_END_TS)
    assert not E.carryover_eligible(old, new, {old.hotkey}, E.CARRYOVER_RETRO_END_TS)
    assert not E.carryover_eligible(old, new, set(), E.CARRYOVER_RETRO_END_TS + 1)
    env.ev.state = loaded
    env.clock.t = E.CARRYOVER_RETRO_END_TS - 1
    env.add(2, "a/recommit", 0.9, weights=make_weights(7))
    await env.ev.evaluate_cycle(new_snapshot("a/recommit", old.coldkey))
    assert env.sub(2).carried_from == old.key
    ST.save(loaded, str(tmp_path / "roundtrip.json"))
    assert ST._to_json(ST.load(str(tmp_path / "roundtrip.json")))["tracks"] == ST._to_json(loaded)["tracks"]


async def test_only_pending_unconfirmed_first_pass_within_window(tmp_path):
    env, old = await pruned(tmp_path)
    new = dataclasses.replace(old, key="new", status=ST.QUEUED, hotkey=HOTKEYS[2])
    end = old.withdrawn_ts + 72 * H
    assert E.carryover_eligible(old, new, set(), end)
    assert not E.carryover_eligible(old, new, set(), end + 0.1)
    for changes in ({"withdrawn_status": ST.NOT_BETTER}, {"first_gain": None}, {"draw1": ""},
                    {"confirm_after_ts": None}, {"confirm_score": 80}, {"confirm_draw": "done"},
                    {"hash_named": "different"}, {"carried_to": "already"}):
        assert not E.carryover_eligible(dataclasses.replace(old, **changes), new, set(), env.clock.t)


async def test_draw_retained_through_prune_window(tmp_path):
    env, old = await pruned(tmp_path)
    ts = env.ev.state.track(TRACK)
    for i in range(E.MAX_UNREFERENCED_DRAWS + 1):
        ts.draws[str(i)] = ST.Draw(str(i), env.clock.t + i, 300, i, [])
    E.SpecEvaluator._prune_draws(ts, old.withdrawn_ts + 72 * H + E.CARRYOVER_DRAW_GRACE_S)
    assert old.draw1 in ts.draws
    E.SpecEvaluator._prune_draws(ts, old.withdrawn_ts + 72 * H + E.CARRYOVER_DRAW_GRACE_S + 1)
    assert old.draw1 not in ts.draws


async def test_keepalive_uses_original_first_pass_and_no_provisional_restart(tmp_path):
    env, old = await pruned(tmp_path)
    env.add(2, "a/recommit", 0.9, weights=make_weights(7))
    snap = new_snapshot("a/recommit", old.coldkey)
    env.clock.advance(20 * H)  # original first pass >24 h old, new confirm_after_ts is now
    await env.ev.evaluate_cycle(snap)
    new = env.sub(2)
    assert new.carried_from == old.key and new.confirm_after_ts == env.clock.t
    assert E.compute_keepalive(env.ev.state, [TRACK], snap, now=env.clock.t)[TRACK] == []
    assert E.compute_provisional(env.ev.state, [TRACK], snap, now=env.clock.t)[TRACK] == []
    # Within the original window it has exactly the normal pending entry's eligibility.
    floors = {TRACK: 0.0}
    assert E.compute_keepalive(env.ev.state, [TRACK], snap, floors=floors, nonproc_floors=floors,
                              now=old.confirm_after_ts + H)[TRACK] == [(2, new.score)]


async def test_not_better_when_pruned_has_nothing_to_restore(tmp_path):
    env = Env(tmp_path, {})
    env.add(1, "a/failed", 0.0, weights=make_weights(7))
    await env.cycle({1: "a/failed"}, block=100)
    old = env.sub(1)
    assert old.status == ST.NOT_BETTER
    await env.cycle({}, block=200)
    env.add(2, "a/recommit", 0.9, weights=make_weights(7))
    await env.ev.evaluate_cycle(new_snapshot("a/recommit", old.coldkey))
    new = env.sub(2)
    assert new.status == ST.REJECTED and not new.carried_from
    assert old.status == ST.NOT_BETTER and not old.carried_to


async def test_checks_must_pass_before_carryover(tmp_path, monkeypatch):
    env, old = await pruned(tmp_path)
    env.add(2, "a/recommit", 0.9, weights=make_weights(7))
    monkeypatch.setattr(env.ev, "_near_check", lambda *args: (ST.REJECTED, "test rule refusal"))
    await env.ev.evaluate_cycle(new_snapshot("a/recommit", old.coldkey))
    assert env.sub(2).status == ST.REJECTED and not env.sub(2).carried_from and not old.carried_to


async def test_shared_repo_does_not_bypass_live_exact_copy_check(tmp_path):
    env = Env(tmp_path, {})
    env.add(1, "a/model", 0.9, weights=make_weights(7))
    await env.cycle({1: "a/model"}, block=100)
    ts = env.ev.state.track(TRACK)
    old = env.sub(1)
    new = dataclasses.replace(old, key="later", uid=2, coldkey="another", first_seen_block=200, status=ST.QUEUED)
    ts.submissions[new.key] = new
    assert "identical tensors" in env.ev._exact_check(ts, new)


async def test_archiving_a_reused_hotkey_keeps_carry_links_stable(tmp_path):
    env, old = await pruned(tmp_path)
    env.add(2, "a/recommit", 0.9, weights=make_weights(7))
    await env.ev.evaluate_cycle(new_snapshot("a/recommit", old.coldkey))
    new = env.sub(2)
    original_key = old.key
    snap = snapshot(400, [neuron(2, commit(TRACK, "a/recommit", "b" * 40), coldkey=old.coldkey),
                          neuron(5, commit(TRACK, old.repo), hotkey=old.hotkey, coldkey=old.coldkey)])
    env.ev._sync(env.cfg.tracks[0], snap, env.clock.t)
    assert old.key == original_key + "#withdrawn@100"
    assert new.carried_from == old.key and old.carried_to == new.key
    assert env.ev.state.track(TRACK).submissions[original_key].uid == 5


# critic review rc8.post2 (B1-B4): regressions
async def test_carried_entry_as_parent_gets_no_second_first_draw(tmp_path):
    from spec_helpers import finetune
    env, old = await pruned(tmp_path)
    w = make_weights(7)
    env.add(2, "a/recommit", 0.9, weights=w)
    env.add(3, "c/child", 0.9, weights=finetune(w, 3, scale=0.001))
    snap = snapshot(300, [neuron(2, commit(TRACK, "a/recommit"), coldkey=old.coldkey),
                          neuron(3, commit(TRACK, "c/child"))])
    await env.ev.evaluate_cycle(snap)
    p = env.sub(2)
    assert p.carried_from == old.key and p.draw1 == old.draw1
    assert [h["kind"] for h in p.history].count("first") == 1


async def test_near_copy_in_window_does_not_take_the_originals_place(tmp_path):
    from spec_helpers import finetune
    env, old = await pruned(tmp_path)
    w = make_weights(7)
    env.add(3, "m/copy", 0.9, weights=finetune(w, 5, scale=0.001))
    await env.ev.evaluate_cycle(snapshot(300, [neuron(3, commit(TRACK, "m/copy"), coldkey="mallory")]))
    m = env.sub(3)
    env.add(2, "a/recommit", 0.9, weights=w)
    await env.ev.evaluate_cycle(snapshot(400, [neuron(3, commit(TRACK, "m/copy"), coldkey="mallory"),
                                               neuron(2, commit(TRACK, "a/recommit"), coldkey=old.coldkey)]))
    a = env.sub(2)
    assert a.carried_from == old.key and a.derived_from != m.key


async def test_window_counts_to_the_recommit_not_the_fetch(tmp_path):
    env, old = await pruned(tmp_path)
    env.clock.advance(72 * H - 2 * H)  # recommit 2 h before the window ends; repo not yet fetchable
    snap = new_snapshot("a/recommit", old.coldkey)
    await env.ev.evaluate_cycle(snap)
    env.clock.advance(3 * H)  # repo public 1 h after the window
    env.add(2, "a/recommit", 0.9, weights=make_weights(7))
    await env.ev.evaluate_cycle(snap)
    assert env.sub(2).carried_from == old.key


async def test_pruned_and_reregistered_same_hotkey_counts_as_pruned(tmp_path):
    env = Env(tmp_path, {})
    env.add(1, "a/old", 0.9, weights=make_weights(7))
    await env.cycle({1: "a/old"}, block=100)
    old = env.sub(1)
    env.clock.advance(13 * H)
    await env.ev.evaluate_cycle(snapshot(200, [neuron(9, hotkey=HOTKEYS[1], registered_block=150)]))
    assert old.withdrawn_kind == "pruned"
    env.add(1, "a/old2", 0.9, weights=make_weights(7))
    await env.ev.evaluate_cycle(snapshot(300, [neuron(9, commit(TRACK, "a/old2"), hotkey=HOTKEYS[1], coldkey=old.coldkey,
                                                      registered_block=150)]))
    new = next(s for s in env.ev.state.track(TRACK).submissions.values() if s.repo == "a/old2")
    assert new.carried_from == old.key


# critic delta (C1/C2): a withdrawn original is never rejected or re-scored through other entries
async def test_near_copies_cannot_get_the_pruned_original_or_its_recommit_rejected(tmp_path):
    from spec_helpers import finetune
    env, old = await pruned(tmp_path)
    w = make_weights(7)
    score0 = old.score
    env.hub.unavailable.add("a/old")          # the author moved to a new repo
    neurons = []
    for i, u in enumerate((3, 4, 5)):          # near-copies by others, one per cycle
        env.add(u, f"m{u}/copy", 0.99, weights=finetune(w, u, scale=0.001))
        neurons.append(neuron(u, commit(TRACK, f"m{u}/copy"), coldkey=f"mallory{u}"))
        await env.ev.evaluate_cycle(snapshot(300 + i, list(neurons)))
        env.clock.advance(H)
    assert old.status == ST.WITHDRAWN and old.score == score0
    env.add(2, "a/new", 0.9, weights=w)
    a_n = neuron(2, commit(TRACK, "a/new"), coldkey=old.coldkey)
    await env.ev.evaluate_cycle(snapshot(400, neurons + [a_n]))
    a = env.sub(2)
    assert a.carried_from == old.key and a.score == score0
    assert all(env.sub(u).derived_from != old.key for u in (3, 4, 5)) and a.derived_from == old.derived_from
    for _ in range(3):
        env.clock.advance(13 * H)
        await env.ev.evaluate_cycle(snapshot(500, neurons + [a_n]))
    assert a.status != ST.REJECTED and old.status == ST.WITHDRAWN


async def test_withdrawn_parent_of_an_earlier_child_is_not_rejected(tmp_path):
    from spec_helpers import finetune
    env = Env(tmp_path, {})
    w = make_weights(7)
    env.add(1, "a/old", 0.9, weights=w)
    await env.cycle({1: "a/old"}, block=100)
    old = env.sub(1)
    env.add(3, "c/child", 0.99, weights=finetune(w, 3, scale=0.001))
    c_n = neuron(3, commit(TRACK, "c/child"), coldkey="carol")
    await env.ev.evaluate_cycle(snapshot(150, [neuron(1, commit(TRACK, "a/old")), c_n]))
    c = env.sub(3)
    env.clock.advance(13 * H)
    env.hub.unavailable.add("a/old")
    for _ in range(4):   # the child's evaluations need the pruned, unfetchable parent
        await env.ev.evaluate_cycle(snapshot(200, [c_n]))
        env.clock.advance(13 * H)
    assert c.derived_from == old.key and old.withdrawn_kind == "pruned"
    assert old.status == ST.WITHDRAWN and old.attempts == 0, (old.status, old.reason)


async def test_exact_copy_fetched_after_the_recommit_is_refused(tmp_path):
    # critic delta 2 D1: a copier commits the pruned original's exact tensors (repo private), the author recommits and is
    # carried, the copier publishes later -> the copier is refused, the carried recommit stays
    env, old = await pruned(tmp_path)
    w = make_weights(7)
    env.add(3, "m/copy", 0.9, weights=w)
    env.hub.unavailable.add("m/copy")
    m_n = neuron(3, commit(TRACK, "m/copy"), coldkey="mallory")
    await env.ev.evaluate_cycle(snapshot(250, [m_n]))
    env.add(2, "a/new", 0.9, weights=w)
    a_n = neuron(2, commit(TRACK, "a/new"), coldkey=old.coldkey)
    await env.ev.evaluate_cycle(snapshot(300, [m_n, a_n]))
    a, m = env.sub(2), env.sub(3)
    assert a.carried_from == old.key
    env.hub.unavailable.clear()
    for _ in range(3):
        env.clock.advance(2 * H)
        await env.ev.evaluate_cycle(snapshot(310, [m_n, a_n]))
    assert a.status != ST.REJECTED and m.status == ST.REJECTED and "identical tensors" in m.reason
