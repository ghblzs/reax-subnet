"""1.2.0rc7 cohort review fixes (F1 structured wait kind and cohort deadline, F5 full playoff field, state fields)."""

import pytest
from test_cohort import H
from test_spec_bases import GIB, OTHER_BASE, benv
from test_spec_evaluator import DEFAULT_BASE, TRACK, Env

from reax_subnet.spec import evaluator as E
from reax_subnet.spec import state as ST

pytestmark = pytest.mark.usefixtures("rc5_entry_floors")


async def test_a_legal_base_larger_than_the_cycle_budget_holds_the_cohort_until_its_deadline(tmp_path):
    """A legal base larger than the 40 GiB download budget (caps: 40 GiB weights, 44 GiB in total) is deferred on every cycle: this host's
    wait, so it holds the cohort back, but only for COHORT_MAX_WAIT_S after the first passed confirmation."""
    env = benv(tmp_path)
    env.basehub.add(DEFAULT_BASE)
    env.basehub.add(OTHER_BASE)
    # just under the 40 GiB weight cap of a base, with its tokenizer just over the 40 GiB download budget of a cycle
    env.basehub.sizes[OTHER_BASE] = {"model-00000-of-00001.safetensors": 40 * GIB - 2 ** 20, "tokenizer.json": 32 * 2 ** 20}
    env.add(1, "a/one", 0.85)
    env.add(2, "a/big", 0.95, base_model=OTHER_BASE)
    commits = {1: "a/one", 2: "a/big"}
    await env.cycle(commits, block=1000)
    big = env.sub(2)
    assert big.status == ST.PENDING_VERIFY and big.wait_kind == ST.WAIT_INFRA and "deferred" in big.reason
    assert env.bf.waits_here(OTHER_BASE)
    env.clock.advance(13 * H)
    await env.cycle(commits, block=1100)
    ts = env.ev.state.track(TRACK)
    assert env.sub(1).confirm_draw and not ts.champion
    confirmed = ts.draws[env.sub(1).confirm_draw].ts
    while env.clock.t + H < confirmed + E.COHORT_MAX_WAIT_S:
        env.clock.advance(H)
        await env.cycle(commits, block=1100)
        assert not ts.champion and big.status == ST.PENDING_VERIFY and big.wait_kind == ST.WAIT_INFRA
    env.clock.t = confirmed + E.COHORT_MAX_WAIT_S
    await env.cycle(commits, block=1100)
    assert ts.champion == env.sub(1).key                              # the deadline crowns; uid 2 still waits for its base
    assert big.status == ST.PENDING_VERIFY and env.basehub.downloads == [DEFAULT_BASE]


async def test_a_base_id_containing_deferred_that_is_not_on_the_hub_is_the_miners_wait(tmp_path):
    env = benv(tmp_path)
    env.basehub.add(DEFAULT_BASE)
    evil = "evil/deferred-base@" + "c" * 40                          # the miner chooses the text of the reason
    env.add(1, "a/one", 0.85)
    env.add(2, "a/evil", 0.95, base_model=evil)
    commits = {1: "a/one", 2: "a/evil"}
    await env.cycle(commits, block=1000)
    o = env.sub(2)
    assert o.status == ST.PENDING_VERIFY and "deferred" in o.reason and o.wait_kind == ST.WAIT_MINER
    assert not env.bf.waits_here(evil)
    env.clock.advance(13 * H)
    await env.cycle(commits, block=1100)
    assert env.ev.state.track(TRACK).champion == env.sub(1).key       # crowned at its confirmation, no wait
    assert o.wait_kind == ST.WAIT_MINER


async def test_the_playoff_waits_for_a_full_field_instead_of_cutting_it_by_the_hours_rest(tmp_path):
    env = Env(tmp_path, {}, max_evals=5)
    commits = {}
    for u, sk in {1: 0.80, 2: 0.86, 3: 0.92}.items():
        env.add(u, f"m/u{u}", sk)
        commits[u] = f"m/u{u}"
    await env.cycle(commits, block=1000)                              # three first passes, one draw
    env.clock.advance(13 * H)
    await env.cycle(commits, block=1100)                              # three confirmations: 3 of 5 used this hour
    ts = env.ev.state.track(TRACK)
    assert all(env.sub(u).confirm_draw for u in commits) and not ts.champion
    draws = len(ts.draws)
    env.clock.advance(600)
    await env.cycle(commits, block=1100)                              # 2 left, 3 needed: no playoff of two
    assert len(ts.draws) == draws and not ts.champion
    env.clock.advance(H)
    await env.cycle(commits, block=1100)
    assert ts.champion == env.sub(3).key
    assert ts.submissions[ts.champion].history[-1]["kind"] == "playoff" and ts.submissions[ts.champion].history[-1]["field"] == 3


def test_wait_kind_and_playoff_aborts_round_trip_and_default_in_an_older_file(tmp_path):
    st = ST.SpecState()
    ts = st.track(TRACK)
    ts.playoff_aborts = 2
    ts.submissions["k"] = ST.Submission("k", 1, "h", "c", TRACK, "r", "v", 5, 0.0, status=ST.PENDING_VERIFY,
                                        reason="x", wait_kind=ST.WAIT_INFRA)
    path = str(tmp_path / "s.json")
    ST.save(st, path)
    back = ST.load(path).track(TRACK)
    assert back.playoff_aborts == 2 and back.submissions["k"].wait_kind == ST.WAIT_INFRA
    raw = ST._to_json(st)                                             # a file written before these fields
    del raw["tracks"][TRACK]["playoff_aborts"], raw["tracks"][TRACK]["submissions"]["k"]["wait_kind"]
    old = ST._from_json(raw).track(TRACK)
    assert old.playoff_aborts == 0 and old.submissions["k"].wait_kind == ""
    assert not E.SpecEvaluator._cohort_open(old, old.submissions["k"])   # unknown kind: re-classified on the next cycle
