"""1.2.0rc6 review fixes (review-rc6 F1, F2, F3, F5): an approved champion holds no keep-alive slot, a far-future window
start gives none, sketch renewal failures do not block the track, and the behavioural order rule ignores yes/no rows."""

import dataclasses
import math
import os

import numpy as np
import pytest
from spec_helpers import finetune, synthetic_pool
from test_copycheck2 import _base, _rng, logit_probs, temperature
from test_keepalive import BASE, DF, KA, NOW, RA, _snap, _sub
from test_rc5_fixes import _env

from reax_subnet.competition_split import RankedLeaders, combine_weights
from reax_subnet.spec import evaluator as E
from reax_subnet.spec import machine as M
from reax_subnet.spec import state as ST
from reax_subnet.spec import tensors as T
from reax_subnet.spec.evaluator import compute_keepalive

# the fake models' score scale and the hand-written leader scores (conftest.py)
pytestmark = pytest.mark.usefixtures("rc5_entry_floors")


# ------------------------------------------------------------------------------------------------ F1
def test_approved_champion_is_paid_by_its_rank_only():
    serving = {11: 1.0}
    leaders = {DF: RankedLeaders([(20, 40.0), (21, 30.0)])}
    alive = {DF: [(20, 40.0), (23, 30.0)]}                            # 20: champion inside its 24 h, 23: pending
    w = combine_weights(serving, None, 0.0, leaders, award_ok=lambda t, u: True, keepalive=alive)
    assert w[20] == pytest.approx(0.15 * (1 - KA) * 0.75)            # rank only, no slot on top
    assert w[23] == pytest.approx(0.15 * KA)                         # the pending entry takes the whole keep-alive part
    # awaiting approval: no rank share, the slot stays (shared with the pending entry)
    w = combine_weights(serving, None, 0.0, leaders, award_ok=lambda t, u: u != 20, keepalive=alive)
    assert w[20] == pytest.approx(0.15 * KA / 2) and w[23] == pytest.approx(0.15 * KA / 2)
    # no gate (None): unchanged rc6 behaviour, both slots
    w = combine_weights(serving, None, 0.0, leaders, keepalive=alive)
    assert w[20] == pytest.approx(0.15 * (1 - KA) * 0.75 + 0.15 * KA / 2)


def test_only_the_approved_champion_alive_pays_exactly_like_rc5():
    leaders = {DF: RankedLeaders([(1, 40.0), (2, 30.0)]), RA: [(3, 50.0)]}
    gate = lambda t, u: True                                         # noqa: E731
    rc5 = combine_weights(BASE, 4, 0.7, leaders, owner_uid=0, burn_fraction=0.10, award_ok=gate)
    assert combine_weights(BASE, 4, 0.7, leaders, owner_uid=0, burn_fraction=0.10, award_ok=gate,
                           keepalive={DF: [(1, 40.0)], RA: [(3, 50.0)]}) == rc5
    # the gate is asked per track: approved in RA does not drop a slot in DF
    w = combine_weights({11: 1.0}, None, 0.0, {DF: RankedLeaders([(1, 40.0)])}, award_ok=lambda t, u: t == RA,
                        keepalive={DF: [(1, 40.0)]})
    assert w[1] == pytest.approx(0.15 * KA)


# ------------------------------------------------------------------------------------------------ F2
def test_future_window_start_fails_closed():
    st = ST.SpecState()
    subs = [_sub(st, 1, ST.CHAMPION, 40.0), _sub(st, 2, ST.PENDING_CONFIRM, 30.0), _sub(st, 3, ST.PENDING_CONFIRM, 31.0)]
    now = NOW
    subs[0].champion_since_ts = now + 10 * 365 * 86400.0             # a clock that jumped at crowning: never expires
    subs[1].confirm_after_ts = now + 1800.0 + M.CONFIRM_DELAY_S        # small skew between hosts: fine
    subs[2].confirm_after_ts = now + E.KEEPALIVE_FUTURE_S + 1.0 + M.CONFIRM_DELAY_S
    out = compute_keepalive(st, [DF], _snap(subs), {DF: 0.0}, now=now)[DF]
    assert [u for u, _ in out] == [2]


# ------------------------------------------------------------------------------------------------ F3
async def _legacy_parent_and_copy(tmp_path):
    """An earlier entry whose sketch must be renewed (rc5 sketch, weights gone) and a near copy of it waiting."""
    base = _base()
    env = _env(tmp_path, base)
    parent = finetune(base, 10)
    env.add(1, "a/parent", 0.6, weights=parent)
    await env.cycle({1: "a/parent"}, block=100)
    sp = env.ev._sketch_path(env.sub(1))
    T.save_sketch(T.load_sketch(sp), sp)                              # rc5: no seed id
    copy = dict(parent)
    copy["model.norm.weight"] = parent["model.norm.weight"] + 1e-3
    env.add(2, "b/copy", 0.95, weights=copy)
    assert env.sub(1).status == ST.PENDING_CONFIRM
    assert not os.path.exists(env.ev._model_dir(env.sub(1)))          # a pending entry's weights are not kept (1.2.0rc6)
    env.hub.fetches.clear()
    return env


async def test_renewal_error_is_skipped_and_the_track_goes_on(tmp_path):
    env = await _legacy_parent_and_copy(tmp_path)
    fetch = env.hub.fetch

    async def broken(c, dest, **kw):
        if c.repo == "a/parent":
            raise OSError(28, "No space left on device")
        return await fetch(c, dest, **kw)

    env.hub.fetch = broken
    await env.cycle({1: "a/parent", 2: "b/copy"}, block=110)
    s = env.sub(2)
    assert s.delta_checked and s.status != ST.QUEUED and "b/copy" in env.server.served
    assert not env.ev._sketch_current(env.sub(1))                     # stays old, tried again before the next candidate


async def test_damaged_sketch_file_is_renewed(tmp_path):
    env = await _legacy_parent_and_copy(tmp_path)
    with open(env.ev._sketch_path(env.sub(1)), "wb") as fh:
        fh.write(b"PK\x03\x04 not a zip")
    assert not env.ev._sketch_current(env.sub(1))
    await env.cycle({1: "a/parent", 2: "b/copy"}, block=110)
    assert env.ev._sketch_current(env.sub(1)) and env.sub(2).derived_from == env.sub(1).key


async def test_renewal_waits_for_disk_then_finds_the_copy(tmp_path):
    env = await _legacy_parent_and_copy(tmp_path)
    env.ev.cfg = dataclasses.replace(env.ev.cfg, min_free_bytes=10)
    free = iter([100, 5])                                             # enough at the cycle gate, too little at the renewal
    env.ev._free_bytes = lambda path: next(free, 100)
    await env.cycle({1: "a/parent", 2: "b/copy"}, block=110)
    s = env.sub(2)
    assert s.status == ST.PENDING_VERIFY and s.reason == E.RENEWAL_WAIT and not s.delta_checked
    assert "a/parent" not in env.hub.fetches and "b/copy" not in env.server.served
    assert not os.path.exists(env.ev._model_dir(s))                  # the candidate's download is freed while it waits
    await env.cycle({1: "a/parent", 2: "b/copy"}, block=120)         # disk freed: queued again, renewed, copy found
    assert env.ev._sketch_current(env.sub(1)) and env.sub(2).derived_from == env.sub(1).key


async def test_behaviour_check_error_keeps_the_group_results(tmp_path, monkeypatch):
    base = _base()
    env = _env(tmp_path, base)
    env.add(1, "a/one", 0.9, weights=finetune(base, 10))
    env.add(2, "b/two", 0.9, weights=finetune(base, 11))

    def boom(*a, **kw):
        raise ZeroDivisionError("measurement bug")

    monkeypatch.setattr(E, "behaviour_stats", boom)
    await env.cycle({1: "a/one", 2: "b/two"})
    assert env.sub(1).status == env.sub(2).status == ST.PENDING_CONFIRM
    assert env.sub(2).derived_from == ""


# ------------------------------------------------------------------------------------------------ F5
def _binary_heavy(n_binary=270, n_multi=30):
    pool = synthetic_pool(3 * max(n_binary, n_multi))
    return [r for r in pool if len(r["labels"]) == 2][:n_binary] + [r for r in pool if len(r["labels"]) > 2][:n_multi]


def _prob_noise(rows, seed, sd=0.08):
    """An independent fine-tune of the same task: shared per-item probabilities (confident on yes/no rows) plus its own
    probability noise of sd ``sd``."""
    out = []
    for r in rows:
        labels = [str(x) for x in r["labels"]]
        if len(labels) == 2:
            q = (0.8 if str(r["expected"]) == labels[1] else 0.2) + 0.1 * (_rng("item", r["id"]).rand() - 0.5)
            q = min(0.99, max(0.01, q + sd * _rng("bin", seed, r["id"]).randn()))
            out.append({labels[0]: 1 - q, labels[1]: q})
        else:
            out.append(logit_probs(r, seed, 0.3))
    return out


@pytest.mark.parametrize("n_multi", [0, 30])
def test_independent_binary_heavy_finetunes_are_not_twins(n_multi):
    rows = _binary_heavy(n_multi=n_multi)
    a, b = _prob_noise(rows, 1), _prob_noise(rows, 2)
    binary = [(pa, pb) for r, pa, pb in zip(rows, a, b) if len(r["labels"]) == 2]
    argmax = sum(max(pa, key=pa.get) == max(pb, key=pb.get) for pa, pb in binary) / len(binary)
    assert argmax >= E.TWIN_SAME_ORDER                                # the rc6 order rule alone would call them twins
    st = E.behaviour_stats(rows, a, b)
    assert st.order_items == n_multi and st.jsd_bits > E.TWIN_JSD_BITS and not st.twin(), st
    assert st.as_dict()["order_items"] == n_multi


def test_temperature_copy_on_a_binary_heavy_draw_is_still_a_twin():
    rows = _binary_heavy(n_multi=30)
    a = [logit_probs(r, 1, 0.5) for r in rows]
    t = temperature(1.6)                                              # beyond the JSD rule: the order rule on 30 rows
    copy = [t(p, r) for p, r in zip(a, rows)]
    st = E.behaviour_stats(rows, a, copy)
    assert E.TWIN_JSD_BITS < st.jsd_bits < E.TWIN_ORDER_JSD_BITS and st.same_order >= E.TWIN_SAME_ORDER and st.twin(), st


def test_order_rule_needs_enough_multi_label_items_and_a_bounded_jsd():
    st = E.BehaviourStats(items=300, jsd_bits=0.02, mean_abs=0.05, same_order=1.0, order_items=E.TWIN_MIN_ORDER_ITEMS - 1)
    assert not st.twin()
    assert dataclasses.replace(st, order_items=E.TWIN_MIN_ORDER_ITEMS).twin()
    assert not dataclasses.replace(st, order_items=200, jsd_bits=E.TWIN_ORDER_JSD_BITS).twin()
    assert dataclasses.replace(st, order_items=0, jsd_bits=E.TWIN_JSD_BITS / 2).twin()
    assert math.isclose(E.TWIN_ORDER_JSD_BITS, 0.05) and E.TWIN_SAME_ORDER == 0.99
