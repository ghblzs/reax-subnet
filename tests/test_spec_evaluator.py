import json
import os

import numpy as np
import pytest
from spec_helpers import (HOTKEYS, Clock, FakeHub, FakeServer, commit, finetune, make_weights, neuron, snapshot,
                          synthetic_pool, write_model, write_pool)

from reax_subnet.spec import config as C
from reax_subnet.spec import evaluator as E
from reax_subnet.spec import machine as M
from reax_subnet.spec import state as ST

T0 = C.DEFAULT_TRACKS[0].start_ts + 3600       # one hour into the first track
TRACK = "demand_forecasting"


class Env:
    def __init__(self, tmp_path, skills, *, max_evals=20, sample=300, base_dir=""):
        self.tmp = tmp_path
        write_pool(str(tmp_path / "sealed"), TRACK, synthetic_pool(1500))
        self.hub = FakeHub(str(tmp_path))
        self.server = FakeServer(skills)
        self.clock = Clock(T0)
        self.cfg = C.SpecConfig(sealed_dir=str(tmp_path / "sealed"), work_dir=str(tmp_path / "work"), state_path=str(tmp_path / "spec.json"),
                                sample_size=sample, max_evals_per_hour=max_evals, base_dir=base_dir,
                                tracks=C.DEFAULT_TRACKS[:1])  # the evaluator tests are per track; two-track wiring is in test_competitions.py
        self.seed = 0
        self.ev = self.make()

    def make(self):
        def seeds(track, block):
            self.seed += 1
            return self.seed
        return E.SpecEvaluator(self.cfg, fetcher=self.hub, backend_factory=self.server, clock=self.clock, seed_source=seeds)

    def add(self, uid, repo, skill=None, weights=None, **kw):
        if skill is not None:
            self.server.skills[repo] = skill
        self.hub.add(repo, hotkey=HOTKEYS[uid], weights=weights if weights is not None else make_weights(uid + 100), **kw)

    async def cycle(self, commits, block=1000):
        """commits: {uid: repo} -> neurons commit to that repo."""
        neurons = [neuron(u, commit(TRACK, r)) for u, r in commits.items()]
        return await self.ev.evaluate_cycle(snapshot(block, neurons))

    def sub(self, uid):
        return next(s for s in self.ev.state.track(TRACK).submissions.values() if s.uid == uid)


@pytest.fixture
def env(tmp_path):
    return Env(tmp_path, {})


async def test_inactive_track_is_ignored(env):
    env.clock.t = C.DEFAULT_TRACKS[0].start_ts - 10
    env.add(1, "a/m1", 0.9)
    assert await env.cycle({1: "a/m1"}) == {}
    assert env.hub.fetches == []
    env.clock.t = C.DEFAULT_TRACKS[0].start_ts + 14 * 86400 + 1
    assert await env.cycle({1: "a/m1"}) == {}


async def test_first_champion_needs_a_second_draw_after_12_hours(env):
    env.add(1, "a/m1", 0.9)
    assert await env.cycle({1: "a/m1"}) == {TRACK: []}
    assert env.sub(1).status == ST.PENDING_CONFIRM and len(env.server.served) == 1
    env.clock.advance(11 * 3600)
    assert await env.cycle({1: "a/m1"}) == {TRACK: []} and len(env.server.served) == 1     # too early: nothing served
    env.clock.advance(2 * 3600)
    out = await env.cycle({1: "a/m1"})
    assert [u for u, _ in out[TRACK]] == [1] and out[TRACK][0][1] > 50
    assert env.sub(1).status == ST.CHAMPION
    ts = env.ev.state.track(TRACK)
    d1, d2 = (ts.draws[d["draw"]] for d in env.sub(1).history)
    assert not set(d1.item_ids) & set(d2.item_ids)                                          # disjoint confirmation draw


async def crown(env, uid, repo, skill):
    env.add(uid, repo, skill)
    await env.cycle({uid: repo})
    env.clock.advance(13 * 3600)
    return await env.cycle({uid: repo})


async def test_margin_challenger_within_noise_does_not_dethrone(env):
    await crown(env, 1, "a/champ", 0.80)
    env.add(2, "a/close", 0.82)
    await env.cycle({1: "a/champ", 2: "a/close"})
    assert env.sub(2).status == ST.NOT_BETTER and "margin" in env.sub(2).reason
    assert env.ev.state.track(TRACK).champion == env.sub(1).key


async def test_clear_winner_is_confirmed_and_former_champion_stays_second(env):
    await crown(env, 1, "a/champ", 0.65)
    env.add(2, "a/better", 0.95)
    out = await env.cycle({1: "a/champ", 2: "a/better"})
    assert env.sub(2).status == ST.PENDING_CONFIRM and [u for u, _ in out[TRACK]] == [1]   # still the old champion meanwhile
    env.clock.advance(13 * 3600)
    out = await env.cycle({1: "a/champ", 2: "a/better"})
    assert [u for u, _ in out[TRACK]] == [2, 1] and out[TRACK][0][1] > out[TRACK][1][1]
    assert env.sub(1).status == ST.FORMER and env.sub(2).status == ST.CHAMPION


async def test_confirmation_can_fail(env):
    await crown(env, 1, "a/champ", 0.65)
    env.add(2, "a/lucky", 0.95)
    await env.cycle({1: "a/champ", 2: "a/lucky"})
    env.server.skills["a/lucky"] = 0.50                      # it was luck: the second draw disagrees
    env.clock.advance(13 * 3600)
    out = await env.cycle({1: "a/champ", 2: "a/lucky"})
    assert env.sub(2).status == ST.NOT_BETTER and [u for u, _ in out[TRACK]] == [1]


async def test_earlier_commitment_wins_when_two_challengers_are_close(env):
    await crown(env, 1, "a/champ", 0.55)
    env.add(2, "a/early", 0.90)
    env.add(3, "a/late", 0.91)
    commits = {1: "a/champ", 3: "a/late"}
    await env.cycle(commits, block=2000)                     # uid 3 commits first (block 2000)
    commits[2] = "a/early"
    await env.cycle(commits, block=2010)
    assert env.sub(3).first_seen_block < env.sub(2).first_seen_block
    env.clock.advance(13 * 3600)
    out = await env.cycle(commits, block=3000)
    # the earlier committer (uid 3) is confirmed first; uid 2 must then beat it by the full margin and cannot
    assert out[TRACK][0][0] == 3 and env.sub(3).status == ST.CHAMPION
    assert env.sub(2).status == ST.NOT_BETTER


async def test_processing_order_is_commitment_order(env):
    await crown(env, 1, "a/champ", 0.50)
    env.add(2, "a/two", 0.90)
    env.add(3, "a/three", 0.90)
    commits = {1: "a/champ", 3: "a/three", 2: "a/two"}
    await env.cycle(commits, block=2000)
    assert env.sub(3).first_seen_block == env.sub(2).first_seen_block == 2000
    env.clock.advance(13 * 3600)
    await env.cycle(commits, block=3000)
    # same block: lower uid first, and the second cannot beat the first by the full margin
    assert env.sub(2).status == ST.CHAMPION and env.sub(3).status == ST.NOT_BETTER


async def test_rate_limit_queues_by_commitment_block(tmp_path):
    env = Env(tmp_path, {}, max_evals=2)
    for u in (1, 2, 3):
        env.add(u, f"a/m{u}", 0.7)
    await env.cycle({3: "a/m3"}, block=100)                        # uid 3 commits first and takes the first slot
    commits = {3: "a/m3", 1: "a/m1", 2: "a/m2"}
    for b in (110, 120):
        await env.cycle(commits, block=b)
    assert env.server.served == ["a/m3", "a/m1"]                    # budget 2 per hour: uid 2 waits
    assert env.sub(2).status == ST.QUEUED
    env.clock.advance(3601)
    await env.cycle(commits, block=200)
    assert env.server.served == ["a/m3", "a/m1", "a/m2"] and env.sub(2).status == ST.PENDING_CONFIRM


async def test_incumbent_run_counts_against_the_budget(tmp_path):
    env = Env(tmp_path, {}, max_evals=2)
    await crown(env, 1, "a/champ", 0.6)
    env.clock.advance(3601)
    env.add(2, "a/c2", 0.9)
    env.add(3, "a/c3", 0.9)
    env.server.served.clear()
    await env.cycle({1: "a/champ", 2: "a/c2", 3: "a/c3"}, block=500)
    assert env.server.served == ["a/champ", "a/c2"] and env.sub(3).status == ST.QUEUED


async def test_identical_weights_rejected_before_any_gpu_time(env):
    w = make_weights(7)
    env.add(1, "a/orig", 0.8, weights=w, shards=1)
    env.add(2, "a/copy", 0.8, weights=w, shards=5)             # same tensors, different shard layout
    await env.cycle({1: "a/orig"}, block=100)
    served = len(env.server.served)
    await env.cycle({1: "a/orig", 2: "a/copy"}, block=110)
    assert env.sub(2).status == ST.REJECTED and "identical tensors" in env.sub(2).reason
    assert len(env.server.served) == served and "a/copy" in env.hub.fetches


async def test_earlier_commitment_keeps_the_hash_even_if_it_was_slow_to_publish(env):
    w = make_weights(7)
    env.add(1, "a/orig", 0.8, weights=w)
    env.add(2, "a/copy", 0.8, weights=w)
    env.hub.unavailable.add("a/orig")                           # original committed first but is still private
    await env.cycle({1: "a/orig"}, block=100)
    await env.cycle({1: "a/orig", 2: "a/copy"}, block=110)
    assert env.sub(2).status == ST.PENDING_CONFIRM             # the copy was fine so far ...
    env.hub.unavailable.clear()
    env.clock.advance(700)
    await env.cycle({1: "a/orig", 2: "a/copy"}, block=120)     # ... until the earlier commitment is published
    assert env.sub(2).status == ST.REJECTED and env.sub(1).status in (ST.PENDING_CONFIRM, ST.CHAMPION)


async def test_near_duplicate_is_flagged_and_must_beat_its_parent(tmp_path):
    base = make_weights(1)
    base_dir = write_model(str(tmp_path / "bases" / ("Qwen__Qwen3-4B@" + "b" * 40)), base)
    env = Env(tmp_path, {}, base_dir=str(tmp_path / "bases"))
    ft = finetune(base, 10)

    def noisy(seed):
        return {k: v + 0.001 * np.random.RandomState(seed).randn(*v.shape).astype(np.float32) for k, v in ft.items()}

    env.add(1, "a/parent", 0.6, weights=ft)
    env.add(2, "a/child-better", 0.95, weights=noisy(4))
    env.add(3, "a/child-same", 0.6, weights=noisy(5))
    env.add(4, "a/indep", 0.6, weights=finetune(base, 11))
    await env.cycle({1: "a/parent"}, block=100)
    await env.cycle({1: "a/parent", 2: "a/child-better", 3: "a/child-same", 4: "a/indep"}, block=110)
    assert env.sub(2).derived_from == env.sub(1).key == env.sub(3).derived_from and env.sub(2).delta_checked
    assert env.sub(4).derived_from == ""
    assert env.sub(2).status == ST.PENDING_CONFIRM                     # beats its parent by the margin
    assert env.sub(3).status == ST.NOT_BETTER and "parent" in env.sub(3).reason
    assert env.sub(4).status == ST.PENDING_CONFIRM                     # vacant track: the entry floor is the bar


async def test_submission_identical_to_base_is_rejected(tmp_path):
    base = make_weights(1)
    write_model(str(tmp_path / "bases" / ("Qwen__Qwen3-4B@" + "b" * 40)), base)
    env = Env(tmp_path, {}, base_dir=str(tmp_path / "bases"))
    env.add(1, "a/nothing", 0.7, weights=base)
    await env.cycle({1: "a/nothing"})
    assert env.sub(1).status == ST.REJECTED and "base model" in env.sub(1).reason


async def test_repo_must_be_public_within_the_deadline(env):
    env.add(1, "a/late", 0.9)
    env.hub.unavailable.add("a/late")
    await env.cycle({1: "a/late"}, block=100)
    assert env.sub(1).status == ST.QUEUED and "waiting" in env.sub(1).reason
    env.clock.advance(C.RETRY_UNAVAILABLE_S + 1)
    await env.cycle({1: "a/late"}, block=100 + C.PUBLISH_DEADLINE_BLOCKS + 1)
    assert env.sub(1).status == ST.REJECTED and "not available" in env.sub(1).reason


async def test_rule_violation_rejects_without_gpu(env):
    env.add(1, "a/gpl", 0.9, licence="gpl-3.0")
    await env.cycle({1: "a/gpl"})
    assert env.sub(1).status == ST.REJECTED and "licence" in env.sub(1).reason and env.server.served == []


async def test_changed_commitment_withdraws_champion(env):
    await crown(env, 1, "a/champ", 0.8)
    env.add(1, "a/other", 0.8, weights=make_weights(55))
    out = await env.cycle({1: "a/other"})
    assert out[TRACK] == [] and env.sub(1).status in (ST.QUEUED, ST.PENDING_CONFIRM, ST.WITHDRAWN) and env.ev.state.track(TRACK).champion == ""


async def test_coldkey_budget(env):
    for i in range(3):
        env.add(1, f"a/m{i}", 0.5, weights=make_weights(200 + i))
    ev = env.ev
    cold = "5ColdSame" + "z" * 40
    neurons = [neuron(1, commit(TRACK, "a/m0"), coldkey=cold), neuron(2, commit(TRACK, "a/m1"), coldkey=cold, hotkey=HOTKEYS[2]),
               neuron(3, commit(TRACK, "a/m2"), coldkey=cold, hotkey=HOTKEYS[3])]
    for r in ("a/m1", "a/m2"):
        env.hub.add(r, hotkey=HOTKEYS[2 if r == "a/m1" else 3], weights=make_weights(300 + len(r)))
        env.server.skills[r] = 0.5
    await ev.evaluate_cycle(snapshot(100, neurons))
    third = next(s for s in ev.state.track(TRACK).submissions.values() if s.uid == 3)
    assert third.status == ST.REJECTED and "open submissions" in third.reason


async def test_state_survives_restart_and_corrupt_file(env):
    await crown(env, 1, "a/champ", 0.65)
    env.add(2, "a/better", 0.95)
    await env.cycle({1: "a/champ", 2: "a/better"})
    ev2 = env.make()                                           # a restart: state is read from disk
    ts = ev2.state.track(TRACK)
    assert ts.champion == env.sub(1).key and ts.submissions[env.sub(2).key].status == ST.PENDING_CONFIRM
    assert ts.submissions[env.sub(2).key].confirm_after_ts == env.sub(2).confirm_after_ts
    env.ev = ev2
    env.clock.advance(13 * 3600)
    out = await env.cycle({1: "a/champ", 2: "a/better"})
    assert out[TRACK][0][0] == 2
    raw = json.load(open(env.cfg.state_path))
    assert raw["version"] == ST.SPEC_STATE_VERSION and "item_ids" in json.dumps(raw)
    assert "state" not in json.dumps(raw).replace("status", "")           # no sealed row content, only ids
    open(env.cfg.state_path, "w").write("{not json")
    ev3 = env.make()
    assert ev3.state.tracks == {} and any(f.startswith("spec.json.corrupt-") for f in os.listdir(env.tmp))


async def test_unservable_incumbent_does_not_crash_or_promote(env):
    await crown(env, 1, "a/champ", 0.8)
    env.add(2, "a/x", 0.95)
    from reax_subnet.spec.vllm import ModelUnavailable
    real = env.ev.backend_factory

    def flaky(model_dir, wb):
        if "a/champ" in open(os.path.join(model_dir, "skill_key.txt")).read():
            raise ModelUnavailable("out of memory")
        return real(model_dir, wb)

    env.ev.backend_factory = flaky
    out = await env.cycle({1: "a/champ", 2: "a/x"})
    assert [u for u, _ in out[TRACK]] == [1] and env.sub(2).status == ST.QUEUED


def test_machine_rules():
    assert M.required_margin(0.2) == 1.5 and M.required_margin(1.0) == 2.0
    assert M.first_pass(60.0, 58.4, 0.5).passed and not M.first_pass(59.4, 58.4, 0.5).passed
    assert not M.first_pass(60.0, 58.4, 1.0).passed                      # 2 x SE = 2.0 > gain 1.6
    assert M.first_pass(1.0, None, 0.0, entry_floor=0.5).passed and not M.first_pass(0.1, None, 0.0, entry_floor=0.5).passed
    assert M.confirm(59.3, 58.4, 1.5) and not M.confirm(59.0, 58.4, 1.5)  # needs delta / 2 = 0.75
    assert not M.confirm(58.0, 58.4, 1.5)
    assert not M.confirm(60.0, 58.4, 1.5, promoted_this_round=True, se=1.0) and M.confirm(60.5, 58.4, 1.5, promoted_this_round=True, se=1.0)
    assert M.confirmation_due(0, 12 * 3600, M.CONFIRM_DELAY_S) and not M.confirmation_due(0, 12 * 3600 - 1)
