"""Review probes for the 1.2.0rc7 cohort rule (worker slice review-cohort). F1, F3 and F4 were strict xfails on the reviewed
commit 24185b4; fixed in 1.2.0rc7 (structured wait kind and cohort deadline, lost first draws skipped in the confirmation
queue, a playoff decided among the served entries after MAX_ATTEMPTS host failures), they are normal tests now."""

import json
from contextlib import asynccontextmanager

import pytest
from spec_helpers import HOTKEYS
from test_cohort import H, _one_pending_and
from test_spec_evaluator import TRACK, Env

from reax_subnet.spec import evaluator as E
from reax_subnet.spec import state as ST
from reax_subnet.spec.vllm import HostUnavailable

pytestmark = pytest.mark.usefixtures("rc5_entry_floors")


async def test_probe_base_name_with_deferred_does_not_block_the_cohort(tmp_path, monkeypatch):
    env = Env(tmp_path, {})
    evil = "evil/deferred-base@" + "c" * 40                           # any hub id that contains the word
    why = E._no_base(ST.Submission("k", 2, "h", "c", TRACK, "r", "v", 0, 0.0, base_model=evil),
                     "base not available on the hub: 404")          # a miner-side reason
    ts, o = await _one_pending_and(env, ST.PENDING_VERIFY, why, base_model=evil)
    monkeypatch.setattr(E.SpecEvaluator, "_sync", lambda self, tdef, snap, now: None)
    env.clock.advance(13 * H)
    await env.cycle({1: "a/one"}, block=1100)
    assert o.status == ST.PENDING_VERIFY and "deferred" in o.reason
    assert ts.champion == env.sub(1).key                             # the miner's wait does not hold the cohort back


async def test_probe_lost_first_draw_at_the_head_does_not_starve_other_confirmations(tmp_path):
    env = Env(tmp_path, {}, max_evals=1)
    env.add(1, "a/one", 0.85)
    env.add(2, "a/two", 0.86)
    commits = {1: "a/one", 2: "a/two"}
    await env.cycle(commits, block=1000)
    env.clock.advance(H)
    await env.cycle(commits, block=1000)
    ts = env.ev.state.track(TRACK)
    assert env.sub(1).draw1 != env.sub(2).draw1
    ts.draws.pop(env.sub(1).draw1)                                   # lost under rc6 (40-draw retention) before the deploy
    for _ in range(5):
        env.clock.advance(13 * H)
        await env.cycle(commits, block=1100)
    assert env.sub(2).confirm_draw or ts.champion == env.sub(2).key  # before the fix: uid 1's group was retried every cycle


async def test_probe_host_failure_of_one_participant_no_longer_stalls_the_playoff(tmp_path):
    env = Env(tmp_path, {})
    env.add(1, "a/one", 0.85)
    env.add(2, "a/two", 0.90)
    commits = {1: "a/one", 2: "a/two"}
    await env.cycle(commits, block=1000)
    env.clock.advance(13 * H)
    await env.cycle(commits, block=1100)
    assert env.sub(1).confirm_draw and env.sub(2).confirm_draw
    real = env.server

    def flaky(model_dir, wb):
        if open(f"{model_dir}/skill_key.txt").read() == "a/two":
            @asynccontextmanager
            async def boom():
                raise HostUnavailable("vLLM not healthy before the start-up deadline")
                yield
            return boom()
        return real(model_dir, wb)
    env.ev.backend_factory = flaky
    ts = env.ev.state.track(TRACK)
    draws0 = len(ts.draws)
    for i in range(E.C.MAX_ATTEMPTS):
        assert not ts.champion and ts.playoff_aborts == i
        env.clock.advance(H)
        await env.cycle(commits, block=1100)
    assert len(ts.draws) - draws0 == E.C.MAX_ATTEMPTS                # one playoff draw per abandoned attempt
    assert ts.champion == env.sub(1).key and ts.playoff_aborts == 0  # then decided among the entries served
    assert env.sub(2).status == ST.PENDING_CONFIRM and not env.sub(2).confirm_draw   # back on the challenger path


async def test_probe_forward_from_an_rc6_state_file(tmp_path):
    """The live rc6 shape: pending entries with draw1, no rc7 fields in the file; rc7 loads it and finishes the cohort."""
    env = Env(tmp_path, {}, max_evals=2)
    skills = {1: 0.80, 2: 0.79, 3: 0.84, 4: 0.78, 5: 0.86, 6: 0.93}
    commits = {}
    for u, sk in skills.items():
        env.add(u, f"m/u{u}", sk)
        commits[u] = f"m/u{u}"
    for _ in range(3):
        await env.cycle(commits, block=1000)
        env.clock.advance(H)
    raw = json.load(open(env.cfg.state_path))
    for s in raw["tracks"][TRACK]["submissions"].values():          # an rc6 file
        del s["confirm_draw"], s["confirm_score"]
    json.dump(raw, open(env.cfg.state_path, "w"))
    env.ev = env.make()
    ts = env.ev.state.track(TRACK)
    assert all(env.sub(u).status == ST.PENDING_CONFIRM and env.sub(u).draw1 in ts.draws for u in skills)
    env.clock.t = env.sub(1).confirm_after_ts + 1
    for _ in range(4):
        await env.cycle(commits, block=1100)
        env.clock.advance(H)
    assert ts.champion == env.sub(6).key


def test_probe_waiting_entries_keep_their_keepalive_until_24h_after_the_first_pass():
    s = ST.Submission(f"{HOTKEYS[1]}|x", 1, HOTKEYS[1], "c", TRACK, "r", "v", 1000, 0.0, status=ST.PENDING_CONFIRM,
                      draw1="d", confirm_draw="d2", confirm_score=70.0, confirm_after_ts=100_000.0)
    assert E._keepalive_since(s) == 100_000.0 - E.M.CONFIRM_DELAY_S  # the first pass, not the confirmation or the wait
