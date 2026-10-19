"""1.2.0rc9.post3: critic-v3int #2 (flag-off exclusions), #3 (carry-over of the extra-confirmation stage), #4 (stage reset)."""

import copy

import pytest
from spec_helpers import commit, make_weights, neuron, snapshot
from test_spec_evaluator import TRACK, Env, crown
from test_v3_decisions import champ_and_challenger, confirm_scenario, zs  # noqa: F401  (zs: fixture)

from reax_subnet.spec import evaluator as E
from reax_subnet.spec import machine as M
from reax_subnet.spec import state as ST

pytestmark = pytest.mark.usefixtures("rc5_entry_floors")
H = 3600
CHAMP = {1: "a/champ", 2: "a/chal"}


def v3_env(tmp_path):
    env = Env(tmp_path, {})
    env.cfg.v3_decisions = True
    return env


def items(ts, draw):
    return set(ts.draws[draw].item_ids)


def disjoint(ts, draws):
    sets = [items(ts, d) for d in draws]
    return all(not a & b for i, a in enumerate(sets) for b in sets[i + 1:])


# ------------------------------------------------------------------------------------------------ #4 stage reset
async def test_hold_and_requeue_starts_a_fresh_stage(tmp_path, zs):
    env = v3_env(tmp_path)
    sub, ts = await champ_and_challenger(env, zs, 1.4)
    env.clock.advance(1800)
    zs[:] = [2.6]
    await env.cycle(CHAMP)
    assert sub.status == ST.PENDING_CONFIRM and len(sub.close_draws) == 2
    stale = list(sub.close_draws)
    E.SpecEvaluator._hold(ts, sub, E.RENEWAL_WAIT, env.clock(), ST.WAIT_INFRA)
    assert sub.status == ST.PENDING_VERIFY and sub.close_draws == [] and sub.confirm_draws == []
    env.add(3, "a/fresh", 0.80)
    env.clock.advance(1800)
    zs[:] = [2.5, 2.5]
    await env.cycle({**CHAMP, 3: "a/fresh"})
    fresh = env.sub(3)
    # no close priority: the requeued entry and the fresh one share one draw, and it is a new first stage
    assert sub.status == fresh.status == ST.PENDING_CONFIRM and sub.draw1 == fresh.draw1
    assert sub.draw1 not in stale
    # no endless close-draw appends, no pooling with the rows of the abandoned stage
    assert sub.close_draws == [] and fresh.close_draws == []
    assert sub.history[-1]["pooled_z"] == pytest.approx(2.5) and sub.history[-1]["draw"] == sub.draw1
    assert [h["draw"] for h in sub.history if h["kind"] == "first"][:2] == stale      # the old rows stay as history


async def test_requeue_after_hold_from_a_close_call_does_not_keep_the_close_stage(tmp_path, zs):
    env = v3_env(tmp_path)
    sub, ts = await champ_and_challenger(env, zs, 1.4)
    assert sub.status == ST.QUEUED and len(sub.close_draws) == 1
    E.SpecEvaluator._hold(ts, sub, E.RENEWAL_WAIT, env.clock(), ST.WAIT_INFRA)
    assert sub.close_draws == []
    env.clock.advance(1800)
    zs[:] = [1.4]
    await env.cycle(CHAMP)                       # a close call again, as a first stage of its own
    assert sub.status == ST.QUEUED and len(sub.close_draws) == 1 and sub.close_draws[0] == sub.history[-1]["draw"]


def test_confirmation_pools_only_the_rows_of_the_current_stage(tmp_path):
    env = v3_env(tmp_path)
    ev = E.SpecEvaluator.__new__(E.SpecEvaluator)

    def row(kind, draw, gain, se):
        return {"kind": kind, "draw": draw, "gain": gain, "se": se, "inc_key": "I"}

    ch = ST.Submission.__new__(ST.Submission)
    ch.draw1, ch.close_draws, ch.confirm_draw, ch.confirm_draws = "d1", [], "", ["c1"]
    ch.history = [row("first", "old-first", 40.0, 1.0), row("confirm", "old-confirm", 50.0, 1.0),     # abandoned stage
                  row("first", "d1", 3.0, 2.0), row("confirm", "c1", 1.0, 3.0),
                  row("first", "other", 9.0, 0.5)]                                                    # a newer first row of another stage
    new = row("confirm", "c2", 0.5, 3.0)
    ch.history.append(new)
    verdict, _ = ev._confirm_v3(ch, new, "I", 0.5, 3.0, True, "")
    gains, ses = [3.0, 1.0, 0.5], [2.0, 3.0, 3.0]
    mean, se = M.pooled(gains, ses)
    assert new["pooled_z"] == pytest.approx(M.z_of(mean, se))
    assert verdict == M.confirm_v3(list(zip(gains, ses)), [(1.0, 3.0), (0.5, 3.0)], extra_used=True) == "fail"


def test_confirm_v3_takes_gain_se_pairs():
    # a confirmation row older than a first-stage row: the tail slice of the old code paired the wrong SEs with the gains
    draws = [(0.6, 3.0), (8.0, 1.0), (0.4, 0.3)]            # confirmation, first stage, confirmation (newest)
    confirm = [(0.6, 3.0), (0.4, 0.3)]
    mean, se = M.pooled([g for g, _ in draws], [e for _, e in draws])
    assert M.reaches(mean, se, M.Z_CROWN) and mean >= M.MARGIN_FLOOR       # a strong pooled result ...
    cmean, cse = M.pooled([0.6, 0.4], [3.0, 0.3])
    assert not M.reaches(cmean, cse, M.Z_CONFIRM_MIN)                        # ... but the confirmation draws do not confirm
    assert M.confirm_v3(draws, confirm, extra_used=True) == "fail"
    tail_cmean, tail_cse = M.pooled([0.6, 0.4], [1.0, 0.3])               # what ``ses[-2:]`` would have given
    assert M.reaches(tail_cmean, tail_cse, M.Z_CONFIRM_MIN)


# ------------------------------------------------------------------------------------------------ #2 flag off keeps exclusions
async def test_flag_off_keeps_the_exclusions_of_a_confirmation_draw_taken(tmp_path, zs):
    env = v3_env(tmp_path)
    sub, ts = await confirm_scenario(env, zs, 2.05, 1.5)
    assert len(sub.confirm_draws) == 1
    env.cfg.v3_decisions = False
    env.clock.advance(1800)
    zs[:] = [3.0]
    await env.cycle(CHAMP)
    assert sub.status == ST.CHAMPION        # the entry's v3 first stage is confirmed under the old rule
    assert disjoint(ts, [sub.draw1, sub.confirm_draws[0], sub.history[-1]["draw"]])


async def test_flag_off_keeps_the_exclusions_of_a_close_call(tmp_path, zs):
    env = v3_env(tmp_path)
    sub, ts = await champ_and_challenger(env, zs, 1.4)
    first = sub.close_draws[0]
    env.cfg.v3_decisions = False
    env.clock.advance(1800)
    zs[:] = [2.5]
    await env.cycle(CHAMP)
    assert sub.status == ST.PENDING_CONFIRM
    assert disjoint(ts, [first, sub.history[-1]["draw"]])


# ------------------------------------------------------------------------------------------------ #3, #5 carry-over
GAPS = {"first": (60.0, 20.0, 4.0), "confirm": (50.0, 18.0, 5.0)}


def add_gap_fields(sub):
    for h in sub.history:
        if h["kind"] in GAPS:
            g, x, se = GAPS[h["kind"]]
            h.update(s_gen=g, s_x=x, se_gap=se, gap=g - x,
                     gap_binding={"penalty": 3.0, "g_med": 5.0, "raw": g, "s_eff": g - 3.0})


async def test_carryover_keeps_the_extra_confirmation_allowance_exclusions_and_binding_rows(tmp_path, zs):
    env = v3_env(tmp_path)
    env.cfg.g_med = {TRACK: 5.0}
    old, ts = await confirm_scenario(env, zs, 2.05, 1.5)
    assert old.status == ST.PENDING_CONFIRM and len(old.confirm_draws) == 1 and not old.confirm_draw
    add_gap_fields(old)
    penalty, details = env.ev._binding_penalty(old, TRACK)
    assert penalty > 0 and details["draws"] == 2
    old.provisional_since_ts = env.clock.t - H
    await env.cycle({1: "a/champ"}, block=2000)                         # the challenger's hotkey leaves: pruned
    assert old.withdrawn_kind == "pruned" and old.withdrawn_status == ST.PENDING_CONFIRM
    env.add(3, "a/recom", 0.82, weights=make_weights(102))              # the same tensors, committed by the same coldkey
    snap = snapshot(3000, [neuron(1, commit(TRACK, "a/champ")), neuron(3, commit(TRACK, "a/recom"), coldkey=old.coldkey)])
    await env.ev.evaluate_cycle(snap)
    new = env.sub(3)
    assert new.carried_from == old.key and new.status == ST.PENDING_CONFIRM
    assert new.confirm_draws == old.confirm_draws and new.confirm_draws is not old.confirm_draws
    assert new.close_draws == old.close_draws and new.draw1 == old.draw1
    assert [h["kind"] for h in new.history] == ["first", "confirm"]
    for a, b in zip(new.history, [h for h in old.history if h["kind"] in GAPS]):
        assert a == b and a is not b
        assert {"s_gen", "s_x", "se_gap", "gap_binding"} <= a.keys()
    assert env.ev._binding_penalty(new, TRACK) == (penalty, details)    # the recommit does not change the binding penalty
    # the extra confirmation of the recommit excludes all earlier draws and is the LAST one allowed
    zs[:] = [8.0]
    await env.ev.evaluate_cycle(snap)
    assert new.status == ST.CHAMPION and len(new.confirm_draws) == 1
    last = new.history[-1]
    assert last["kind"] == "confirm" and last["pooled_z"] >= M.Z_CROWN
    assert disjoint(ts, [new.draw1, new.confirm_draws[0], last["draw"]])


async def test_carryover_of_an_extra_pending_entry_cannot_take_a_second_extra(tmp_path, zs):
    env = v3_env(tmp_path)
    old, ts = await confirm_scenario(env, zs, 2.05, 1.5)
    old.provisional_since_ts = env.clock.t - H
    await env.cycle({1: "a/champ"}, block=2000)
    env.add(3, "a/recom", 0.82, weights=make_weights(102))
    snap = snapshot(3000, [neuron(1, commit(TRACK, "a/champ")), neuron(3, commit(TRACK, "a/recom"), coldkey=old.coldkey)])
    await env.ev.evaluate_cycle(snap)
    new = env.sub(3)
    assert len(new.confirm_draws) == 1
    zs[:] = [0.4]                                                        # the allowance is used up: weak = fail, not another extra
    await env.ev.evaluate_cycle(snap)
    assert new.status == ST.NOT_BETTER and len(new.confirm_draws) == 1
