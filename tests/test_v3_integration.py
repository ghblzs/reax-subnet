"""v3 integration with the rc9.post1 scheduling and start-up rules."""
import copy
from pathlib import Path

import pytest

from test_spec_evaluator import Env, TRACK, crown
from test_startfail import champion, scripted
from test_v3_decisions import champ_and_challenger, confirm_scenario, zs
from reax_subnet.spec import config as C, state as ST, vllm as V

pytestmark = pytest.mark.usefixtures("rc5_entry_floors")
H = 3600


@pytest.mark.parametrize("v3", [False, True])
@pytest.mark.parametrize("whole_host", [False, True])
async def test_abandoned_incumbent_does_not_consume_items_or_draw(tmp_path, v3, whole_host):
    env = await champion(tmp_path)
    env.cfg.v3_decisions = v3
    ts = env.ev.state.track(TRACK)
    used, draws = copy.deepcopy(ts.used), copy.deepcopy(ts.draws)
    history = []  # the new challenger is first synced in this cycle
    rounds = len(env.sub(1).serve_log)
    calls = scripted(env, [V.ModelUnavailable("invalid config")] * 2)
    if whole_host:
        def fail(*args):
            raise V.HostUnavailable("docker unavailable")
        env.ev.backend_factory = fail
    env.clock.advance(H + 1)
    await env.cycle({1: "a/champ", 2: "a/next"})
    assert ts.used == used and ts.draws == draws
    assert env.sub(2).history == history and not env.sub(2).draw1
    assert env.sub(2).status == ST.QUEUED
    assert len(env.sub(1).serve_log) == rounds + 1
    assert env.sub(1).serve_log[-1][1] == ("infra" if whole_host else "model")
    assert len(env.ev.state.eval_times) == 3  # champion twice, one host probe; actual compute is still charged
    persisted = ST.load(env.cfg.state_path).track(TRACK)
    assert persisted.used == used and persisted.draws == draws
    if not whole_host:
        assert [repo for repo, _ in calls] == ["a/champ", "a/champ", "a/next"]


@pytest.mark.parametrize("stage", ["close", "confirmation"])
async def test_abandoned_v3_extra_draw_keeps_evidence_and_only_one_probe(tmp_path, zs, stage):
    env = Env(tmp_path, {})
    env.cfg.v3_decisions = True
    if stage == "close":
        sub, ts = await champ_and_challenger(env, zs, 1.4)
    else:
        sub, ts = await confirm_scenario(env, zs, 2.05, 1.5)
        assert len(sub.confirm_draws) == 1
    before = copy.deepcopy(sub)
    used, draws = copy.deepcopy(ts.used), copy.deepcopy(ts.draws)
    env.add(3, "a/other", 0.8)
    n_champ, n_sub = len(env.sub(1).serve_log), len(sub.serve_log)
    calls = scripted(env, [V.ModelUnavailable("invalid tokenizer")] * 2)
    env.clock.advance(H + 1)
    await env.cycle({1: "a/champ", 2: "a/chal", 3: "a/other"})
    assert ts.used == used and ts.draws == draws
    assert sub.history == before.history and sub.close_draws == before.close_draws
    assert sub.confirm_draws == before.confirm_draws and sub.status == before.status
    assert len(env.sub(1).serve_log) == n_champ + 1
    assert len(sub.serve_log) == n_sub + 1 and sub.serve_log[-1][1] == "ok"  # probe is a real serve
    assert not env.sub(3).serve_log
    assert [r for r, _ in calls] == ["a/champ", "a/champ", "a/chal"]
    assert len(env.ev.state.eval_times) == 3
    # Retry the still-pending extra draw next hour. The abandoned probe added no statistical evidence.
    scripted(env, [])
    env.clock.advance(H + 1)
    zs[:] = [2.6 if stage == "close" else 8.0] + ([3.0] if stage == "close" else [])
    await env.cycle({1: "a/champ", 2: "a/chal", 3: "a/other"})
    assert sub.status == (ST.PENDING_CONFIRM if stage == "close" else ST.CHAMPION)
    assert len(sub.history) > len(before.history)


@pytest.mark.parametrize("stage", ["close", "confirmation"])
async def test_extra_draw_retry_success_is_one_new_round_and_spends_budget(tmp_path, zs, stage):
    env = Env(tmp_path, {})
    env.cfg.v3_decisions = True
    if stage == "close":
        sub, ts = await champ_and_challenger(env, zs, 1.4)
    else:
        sub, ts = await confirm_scenario(env, zs, 2.05, 1.5)
    n_champ, n_sub, n_draw = len(env.sub(1).serve_log), len(sub.serve_log), len(ts.draws)
    calls = scripted(env, [V.ModelUnavailable("invalid config"), None])
    env.clock.advance(H + 1)
    zs[:] = [2.6 if stage == "close" else 8.0]
    await env.cycle({1: "a/champ", 2: "a/chal"})
    assert len(env.sub(1).serve_log) == n_champ + 1 and env.sub(1).serve_log[-1][1] == "ok"
    assert len(sub.serve_log) == n_sub + 1 and len(ts.draws) == n_draw + 1
    assert len(env.ev.state.eval_times) == 3
    assert [r for r, _ in calls] == ["a/champ", "a/champ", "a/chal"]
    assert sub.status == (ST.PENDING_CONFIRM if stage == "close" else ST.CHAMPION)


@pytest.mark.parametrize("stage", ["close", "confirmation"])
async def test_extra_draw_waits_for_shared_hourly_budget(tmp_path, zs, stage):
    env = Env(tmp_path, {})
    env.cfg.v3_decisions = True
    if stage == "close":
        sub, ts = await champ_and_challenger(env, zs, 1.4)
    else:
        sub, ts = await confirm_scenario(env, zs, 2.05, 1.5)
    env.clock.advance(H + 1)
    env.cfg.max_evals_per_hour = 2
    env.ev.state.eval_times[:] = [env.clock()]
    before = copy.deepcopy(sub)
    draws = copy.deepcopy(ts.draws)
    env.server.served.clear()
    await env.cycle({1: "a/champ", 2: "a/chal"})
    assert not env.server.served and sub.history == before.history and ts.draws == draws
    env.clock.advance(H + 1)
    zs[:] = [2.6 if stage == "close" else 8.0]
    await env.cycle({1: "a/champ", 2: "a/chal"})
    assert env.server.served == ["a/champ", "a/chal"]
    assert len(env.ev.state.eval_times) == 2


async def test_v3_keeps_three_abort_playoff_bound_then_uses_new_incumbent_evidence(tmp_path, zs):
    from test_cohort import _two_ready
    env, commits = await _two_ready(tmp_path, 0.65, 0.90)
    env.cfg.v3_decisions = True
    # The first passes were legacy: turn these into v3-admitted entries to exercise v3 after the playoff.
    for uid in (1, 2):
        env.sub(uid).decision_rule = "v3"
    ts = env.ev.state.track(TRACK)
    first = env.sub(2).draw1
    real = env.server

    def factory(model_dir, wb):
        if Path(model_dir, "skill_key.txt").read_text() == "a/two":
            raise V.HostUnavailable("host startup deadline")
        return real(model_dir, wb)
    env.ev.backend_factory = factory
    for i in range(C.PLAYOFF_MAX_ABORTS):
        env.clock.advance(H + 1)
        await env.cycle(commits, block=1100)
        if i < C.PLAYOFF_MAX_ABORTS - 1:
            assert not ts.champion and ts.playoff_aborts == i + 1
    assert ts.champion == env.sub(1).key and ts.playoff_aborts == 0
    sub = env.sub(2)
    assert sub.status == ST.PENDING_CONFIRM and not sub.confirm_draw and sub.draw1 == first
    assert sub.attempts == 0 and not sub.confirm_draws
    env.ev.backend_factory = real
    env.clock.advance(H + 1)
    zs[:] = [2.0]
    await env.cycle(commits, block=1100)
    assert sub.status == ST.PENDING_CONFIRM and len(sub.confirm_draws) == 1
    assert sub.history[-1]["inc_key"] == env.sub(1).key
    env.clock.advance(H + 1)
    zs[:] = [4.0]
    await env.cycle(commits, block=1100)
    assert sub.status == ST.CHAMPION


@pytest.mark.parametrize("v3", [False, True])
async def test_retry_retains_rc9post1_one_attempt_hourly_overrun(tmp_path, v3):
    """The retry is immediate even at a full group budget; it is charged and stops later work (legacy behavior)."""
    env = await champion(tmp_path)
    env.cfg.v3_decisions = v3
    env.cfg.max_evals_per_hour = 2
    env.clock.advance(H + 1)
    calls = scripted(env, [V.ModelUnavailable("invalid config"), None])
    await env.cycle({1: "a/champ", 2: "a/next"})
    assert len(calls) == 3 and len(env.ev.state.eval_times) == 3
    assert env.ev._budget(env.clock()) == 0
    env.sub(2).status = ST.QUEUED
    await env.cycle({1: "a/champ", 2: "a/next"})
    assert len(calls) == 3  # no further group runs until capacity returns


@pytest.mark.parametrize("v3", [False, True])
async def test_lone_incumbent_retry_failure_does_not_charge_draw(tmp_path, v3):
    env = await champion(tmp_path)
    env.cfg.v3_decisions = v3
    sub = env.sub(1)
    sub.serve_retry_pending = True
    ts = env.ev.state.track(TRACK)
    used, draws = copy.deepcopy(ts.used), copy.deepcopy(ts.draws)
    rounds = len(sub.serve_log)
    scripted(env, [V.ModelUnavailable("invalid tokenizer")] * 2)
    env.clock.advance(H + 1)
    await env.cycle({1: "a/champ"})
    assert ts.used == used and ts.draws == draws
    assert len(sub.serve_log) == rounds + 1 and sub.serve_log[-1][1] == "model"
    assert sub.serve_retry_pending and len(env.ev.state.eval_times) == 2
