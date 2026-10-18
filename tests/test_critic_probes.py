"""P1–P6 critic scenarios, with assertions for the bounded worker rules."""
import shutil
from pathlib import Path

import pytest
from test_spec_evaluator import Env, TRACK
from test_startfail import champion

from reax_subnet.spec import config as C, fetch as F, state as ST, vllm as V

pytestmark = pytest.mark.usefixtures("rc5_entry_floors")


async def test_P1_champion_repo_gone_is_rejected_without_strikes(tmp_path):
    """Lost local copy and private repo: reject on three refetches, then evaluate the challenger."""
    env = await champion(tmp_path)
    real = env.hub.fetch

    async def fetch(commitment, *a, **k):
        if commitment.repo == "a/champ":
            raise F.FetchError("unavailable", "repo or revision not found (still private?)")
        return await real(commitment, *a, **k)
    env.hub.fetch = fetch
    shutil.rmtree(Path(env.ev.cfg.work_dir, "models"))
    Path(env.ev.cfg.work_dir, "models").mkdir()
    for _ in range(30):
        env.clock.advance(3601)
        env.server.served.clear()
        await env.cycle({1: "a/champ", 2: "a/next"})
    print("champ", env.sub(1).status, env.sub(1).attempts, "next", env.sub(2).status, env.sub(2).draw1)
    assert env.sub(1).status == ST.REJECTED and env.ev.state.track(TRACK).champion != env.sub(1).key
    assert env.sub(1).unavailable_fetches == 3 and env.sub(1).attempts == 0
    assert env.sub(2).status == ST.NOT_BETTER and env.sub(2).history


async def test_P2_champion_always_infra_requests_pay_suspension(tmp_path):
    """Persistent host failures retain the crown but request suspension after the liveness bound."""
    env = await champion(tmp_path)
    real = env.server

    def factory(model_dir, wb):
        if Path(model_dir, "skill_key.txt").read_text() == "a/champ":
            raise V.HostUnavailable("vLLM not healthy before the start-up deadline")
        return real(model_dir, wb)
    env.ev.backend_factory = factory
    for _ in range(20):
        env.clock.advance(4 * 3600)
        await env.cycle({1: "a/champ", 2: "a/next"})
    print("champ", env.sub(1).status, env.sub(1).serve_log[-3:], "next", env.sub(2).status, env.sub(2).serve_log[-3:])
    assert env.sub(1).status == ST.CHAMPION and env.sub(1).serve_retry_pending
    assert env.sub(1).pay_suspended_since_ts is not None and env.sub(1).attempts == 0
    assert env.sub(2).status == ST.QUEUED and all(e[1] == "ok" for e in env.sub(2).serve_log)


async def test_P3_whole_group_failures_request_suspension_without_strikes(tmp_path):
    """Whole-group attribution cannot prevent the attribution-independent liveness signal."""
    env = await champion(tmp_path)
    def factory(model_dir, wb):
        raise V.ModelUnavailable("container exited during start-up: invalid tokenizer")
    env.ev.backend_factory = factory
    for _ in range(10):
        env.clock.advance(8 * 3600)
        await env.cycle({1: "a/champ", 2: "a/next"})
    print("champ", env.sub(1).status, env.sub(1).attempts, "sybil", env.sub(2).status, env.sub(2).attempts)
    assert env.sub(1).status == ST.CHAMPION and env.sub(2).status == ST.QUEUED
    assert env.sub(1).attempts == 0 and env.sub(1).pay_suspended_since_ts is not None


async def test_P4_single_model_host_fault_keeps_crown_without_strikes(tmp_path):
    """A lone champion with a host-side CUDA driver fault gets no model strike."""
    env = await champion(tmp_path)

    def factory(model_dir, wb):
        raise V.ModelUnavailable("container exited during start-up: RuntimeError: CUDA driver initialization failed")
    env.ev.backend_factory = factory
    env.clock.advance(3601)
    await env.cycle({1: "a/champ", 2: "a/next"})   # challenger present: both fail -> host_problem
    env.sub(2).status = ST.WITHDRAWN                 # challenger leaves; champion alone is re-measured
    for i in range(3):
        env.clock.advance(3601)
        await env.cycle({1: "a/champ"})
        print(i, env.sub(1).status, env.sub(1).attempts, env.sub(1).serve_retry_pending)
    print("final", env.sub(1).status, env.sub(1).serve_log)
    assert env.sub(1).status == ST.CHAMPION and env.sub(1).attempts == 0
    assert all(x[1] == "infra" for x in env.sub(1).serve_log[-4:])


async def test_P5_playoff_member_always_infra_returns_to_challenger_path(tmp_path):
    """Three infra playoff aborts crown a served entry and preserve the failed entry's first draw."""
    from test_cohort import H, _two_ready
    env, commits = await _two_ready(tmp_path, 0.85, 0.90)
    real = env.server

    def factory(model_dir, wb):
        if Path(model_dir, "skill_key.txt").read_text() == "a/two":
            raise V.HostUnavailable("vLLM not healthy before the start-up deadline")
        return real(model_dir, wb)
    env.ev.backend_factory = factory
    first_draw = env.sub(2).draw1
    for _ in range(C.PLAYOFF_MAX_ABORTS):
        env.clock.advance(H)
        await env.cycle(commits, block=1100)
    ts = env.ev.state.track(TRACK)
    print("champion", repr(ts.champion), "aborts", ts.playoff_aborts)
    assert ts.champion == env.sub(1).key and ts.playoff_aborts == 0
    assert env.sub(2).status == ST.PENDING_CONFIRM and not env.sub(2).confirm_draw
    assert env.sub(2).draw1 == first_draw and env.sub(2).attempts == 0


async def test_P6_private_pending_entry_cannot_block_confirmation_queue(tmp_path):
    """A private confirmation head is retried and rejected while later healthy entries progress."""
    env = Env(tmp_path, {}, sample=300)
    env.add(1, "a/champ", 0.95)
    await env.cycle({1: "a/champ"})
    env.clock.advance(13 * 3600)
    await env.cycle({1: "a/champ"})
    assert env.sub(1).status == ST.CHAMPION
    env.server.skills["a/champ"] = 0.55
    env.add(2, "a/dead", 0.99)
    env.clock.advance(3601)
    await env.cycle({1: "a/champ", 2: "a/dead"})
    env.add(3, "a/good", 0.99)
    env.clock.advance(3601)
    await env.cycle({1: "a/champ", 2: "a/dead", 3: "a/good"})
    print("after first passes", env.sub(2).status, env.sub(2).draw1, env.sub(3).status, env.sub(3).draw1)
    real = env.hub.fetch

    async def fetch(commitment, *a, **k):
        if commitment.repo == "a/dead":
            raise F.FetchError("unavailable", "repo or revision not found (still private?)")
        return await real(commitment, *a, **k)
    env.hub.fetch = fetch
    env.add(4, "a/later", 0.6)
    for _ in range(40):
        env.clock.advance(3601)
        await env.cycle({1: "a/champ", 2: "a/dead", 3: "a/good", 4: "a/later"})
    ts = env.ev.state.track(TRACK)
    print("dead", env.sub(2).status, env.sub(2).reason[:60], "| good", env.sub(3).status, env.sub(3).confirm_draw,
          "| later", env.sub(4).status, "| champion uid", ts.submissions[ts.champion].uid if ts.champion else None)

    assert env.sub(2).status == ST.REJECTED and env.sub(2).unavailable_fetches == 3
    assert env.sub(2).attempts == 0 and ts.champion == env.sub(3).key
