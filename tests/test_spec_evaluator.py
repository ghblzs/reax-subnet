import dataclasses
import json
import os

import numpy as np
import pytest
from spec_helpers import (HOTKEYS, Clock, FakeHub, FakeServer, commit, finetune, make_weights, neuron, snapshot,
                          synthetic_pool, write_model, write_pool)

from reax_subnet.spec import config as C
from reax_subnet.spec import evaluator as E
from reax_subnet.spec import machine as M
from reax_subnet.spec import sampling as S
from reax_subnet.spec import state as ST

# the state machine / copy checks on the fake models' score scale (conftest.py)
pytestmark = pytest.mark.usefixtures("rc5_entry_floors")

T0 = C.DEFAULT_TRACKS[0].start_ts + 3600       # one hour into the first track
TRACK = "demand_forecasting"
DEFAULT_BASE = "Qwen/Qwen3-4B@" + "b" * 40      # spec_helpers.write_model's default base_model


class Env:
    def __init__(self, tmp_path, skills, *, max_evals=20, sample=300, base_dir=None):
        self.tmp = tmp_path
        if base_dir is None:    # the declared base of every test model is on the host (without it nothing is served)
            base_dir = str(tmp_path / "default-bases")
            write_model(os.path.join(base_dir, DEFAULT_BASE.replace("/", "__", 1)), make_weights(999))
        write_pool(str(tmp_path / "sealed"), TRACK, synthetic_pool(1500))
        self.hub = FakeHub(str(tmp_path))
        self.server = FakeServer(skills)
        self.clock = Clock(T0)
        self.cfg = C.SpecConfig(sealed_dir=str(tmp_path / "sealed"), work_dir=str(tmp_path / "work"), state_path=str(tmp_path / "spec.json"),
                                sample_size=sample, max_evals_per_hour=max_evals, base_dir=base_dir,
                                tracks=C.DEFAULT_TRACKS[:1], min_free_bytes=0)  # the evaluator tests are per track; two-track wiring is in test_competitions.py
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


async def test_pending_entry_is_fetched_again_at_confirmation_and_hash_checked(env):
    """1.2.0rc6: only a champion's files are kept; a pending entry is fetched again (pinned revision) at confirmation and
    must have the weights of its first check."""
    env.add(1, "a/m1", 0.9)
    env.add(2, "a/m2", 0.9)
    await env.cycle({1: "a/m1", 2: "a/m2"})
    assert env.sub(1).status == env.sub(2).status == ST.PENDING_CONFIRM
    assert not os.path.exists(env.ev._model_dir(env.sub(1))) and not os.path.exists(env.ev._model_dir(env.sub(2)))
    env.hub.add("a/m2", hotkey=HOTKEYS[2], weights=make_weights(777))    # other weights under the same pinned revision
    env.hub.fetches.clear()
    env.server.served.clear()
    env.clock.advance(13 * 3600)
    await env.cycle({1: "a/m1", 2: "a/m2"})
    assert env.hub.fetches.count("a/m1") == 1 and env.sub(1).status == ST.CHAMPION
    assert os.path.exists(env.ev._model_dir(env.sub(1)))                  # the champion's files are kept
    assert env.sub(2).status == ST.REJECTED and "other weights" in env.sub(2).reason
    assert "a/m2" not in env.server.served and not os.path.exists(env.ev._model_dir(env.sub(2)))


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
    # the earlier committer (uid 3) is confirmed first (alone: one first draw per confirmation group, 1.2.0rc6)
    assert out[TRACK][0][0] == 3 and env.sub(3).status == ST.CHAMPION
    assert env.sub(2).status == ST.PENDING_CONFIRM
    env.clock.advance(3600)
    out = await env.cycle(commits, block=3300)
    # uid 2 must then beat it by the full margin and cannot
    assert out[TRACK][0][0] == 3 and env.sub(2).status == ST.NOT_BETTER


async def test_processing_order_is_commitment_order(env):
    await crown(env, 1, "a/champ", 0.65)
    assert env.sub(1).status == ST.CHAMPION     # a champion: on a vacant track the cohort playoff decides (1.2.0rc7)
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
    await crown(env, 1, "a/champ", 0.7)      # clear of the floor on both 300-item draws (0.6 sat at its edge)
    assert env.sub(1).status == ST.CHAMPION
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
    assert not M.confirm(59.3, 58.4, 1.5, new_incumbent=True, se=0.5) and M.confirm(59.9, 58.4, 1.5, new_incumbent=True, se=0.5)
    assert M.confirm(36.0, 33.0, 33.7, new_incumbent=True, se=1.0)        # vacant first pass: full margin, not floor / 2
    assert M.confirmation_due(0, 12 * 3600, M.CONFIRM_DELAY_S) and not M.confirmation_due(0, 12 * 3600 - 1)


async def test_vacant_track_challenger_confirms_against_a_champion_crowned_meanwhile_by_the_full_margin(tmp_path, monkeypatch):
    # the margin rule, calibrated on the draws of the 1.2.0rc5 mix (50 % proc); the release mix draws other items and the
    # gain between the two fake skills lands at 16.7, just past half the floor, which says nothing about the rule
    monkeypatch.setattr(S, "TIER_MIX", {"proc_sealed": 0.5, "same_family": 0.25, "new_domain": 0.125, "new_family": 0.125})
    env = Env(tmp_path, {}, sample=600)                         # a larger draw: the fake models' SE is wide
    # decision rule v3 (test_v3_decisions.py) is stricter on this path; this test pins the pre-v3 rule (flag off)
    env.cfg = dataclasses.replace(env.cfg, entry_floor=30.0, v3_decisions=False)    # half the floor (15) is well above the full margin here
    env.ev = env.make()
    env.add(1, "a/first", 0.65)
    env.add(2, "a/second", 0.75)
    await env.cycle({1: "a/first"})
    env.clock.advance(3600)
    # both pass the floor while the track is vacant; uid 2 is first seen a block later (another cohort, 1.2.0rc7: an entry
    # of the same block would hold back uid 1's crowning until the cohort playoff)
    await env.cycle({1: "a/first", 2: "a/second"}, block=1001)
    assert env.sub(1).status == env.sub(2).status == ST.PENDING_CONFIRM
    assert env.sub(2).first_seen_block == env.sub(1).first_seen_block + 1
    assert env.sub(1).first_incumbent == env.sub(2).first_incumbent == ""
    env.clock.advance(11.5 * 3600)                               # uid 1 is due, uid 2 not yet
    await env.cycle({1: "a/first", 2: "a/second"})
    assert env.sub(1).status == ST.CHAMPION and env.sub(2).status == ST.PENDING_CONFIRM
    env.clock.advance(3600)
    out = await env.cycle({1: "a/first", 2: "a/second"})
    gain = env.sub(2).score - env.sub(1).score
    # a clear win over the champion crowned an hour earlier; it used to need half the entry floor (15 points here)
    assert M.MARGIN_FLOOR <= gain < 0.5 * env.ev._entry_floor(C.DEFAULT_TRACKS[0])
    assert env.sub(2).status == ST.CHAMPION and env.sub(1).status == ST.FORMER
    assert [u for u, _ in out[TRACK]] == [2, 1]


def test_first_incumbent_of_a_state_written_before_the_field():
    sub = ST.Submission(key="k", uid=1, hotkey=HOTKEYS[1], coldkey="c", track_id=TRACK, repo="a/m", revision="a" * 40,
                        first_seen_block=1, first_seen_ts=0.0, draw1="d1",
                        history=[{"draw": "d1", "kind": "first", "score": 40.0, "incumbent": None}])
    assert sub.first_incumbent is None and E.SpecEvaluator._first_incumbent(sub) == ""   # vacant at the first pass
    sub.history[0]["incumbent"] = 38.0
    assert E.SpecEvaluator._first_incumbent(sub) is None                                # unknown: the old rule applies
    sub.first_incumbent = "x"
    assert E.SpecEvaluator._first_incumbent(sub) == "x"


# ------------------------------------------------------------------------------------------------ S2: base verification
def bases(tmp_path, weights, *names):
    """The same base weights under one or more ``<hf id>@<rev>`` names (an alias is a second name for the same tensors)."""
    root = tmp_path / "bases"
    for n in names:
        write_model(str(root / n.replace("/", "__", 1)), weights)
    return str(root)


async def test_missing_base_is_pending_not_passed_and_is_checked_once_the_base_arrives(tmp_path):
    env = Env(tmp_path, {}, base_dir=str(tmp_path / "bases"))   # the operator has bases, but not this one
    os.makedirs(tmp_path / "bases", exist_ok=True)
    env.add(1, "a/m1", 0.9)
    out = await env.cycle({1: "a/m1"})
    sub = env.sub(1)
    assert sub.status == ST.PENDING_VERIFY and "not available" in sub.reason and not sub.delta_checked
    assert out == {TRACK: []} and env.server.served == []                     # no GPU time, no rank, no award
    assert any("pending verification" in e for e in env.ev.state.track(TRACK).events)
    env.clock.advance(3600)
    await env.cycle({1: "a/m1"})
    assert env.sub(1).status == ST.PENDING_VERIFY and env.server.served == []   # still kept, still waiting
    bases(tmp_path, make_weights(999), DEFAULT_BASE)                          # the operator adds the base
    env.clock.advance(3600)
    await env.cycle({1: "a/m1"})
    assert env.sub(1).delta_checked and env.sub(1).base_id and env.sub(1).status == ST.PENDING_CONFIRM
    assert env.server.served == ["a/m1"]


async def test_no_base_dir_at_all_crowns_nobody(tmp_path):
    env = Env(tmp_path, {}, base_dir="")
    env.add(1, "a/m1", 0.9)
    for _ in range(3):
        out = await env.cycle({1: "a/m1"})
        env.clock.advance(13 * 3600)
    assert env.sub(1).status == ST.PENDING_VERIFY and out == {TRACK: []} and env.ev.state.track(TRACK).champion == ""


async def test_exact_duplicate_is_rejected_even_while_the_base_is_missing(tmp_path):
    env = Env(tmp_path, {}, base_dir="")
    w = make_weights(7)
    env.add(1, "a/orig", 0.8, weights=w)
    env.add(2, "a/copy", 0.8, weights=w, shards=3)
    await env.cycle({1: "a/orig"}, block=100)
    await env.cycle({1: "a/orig", 2: "a/copy"}, block=110)
    assert env.sub(1).status == ST.PENDING_VERIFY
    assert env.sub(2).status == ST.REJECTED and "identical tensors" in env.sub(2).reason


async def test_near_copy_declaring_an_alias_of_the_base_is_still_compared(tmp_path):
    """The copier declares another name for the same base weights (a mirror): the comparison uses the resolved base
    (its tensor hash), not the declared string. An independent fine-tune declaring the alias is not flagged."""
    base = make_weights(1)
    alias = "mirror/Qwen3-4B@" + "c" * 40
    env = Env(tmp_path, {}, base_dir=bases(tmp_path, base, DEFAULT_BASE, alias))
    ft = finetune(base, 10)
    near = {k: v + 0.001 * np.random.RandomState(4).randn(*v.shape).astype(np.float32) for k, v in ft.items()}
    env.add(1, "a/parent", 0.6, weights=ft)
    env.add(2, "b/near-copy", 0.95, weights=near, base_model=alias)
    env.add(3, "c/indep", 0.6, weights=finetune(base, 11), base_model=alias)
    await env.cycle({1: "a/parent"}, block=100)
    await env.cycle({1: "a/parent", 2: "b/near-copy", 3: "c/indep"}, block=110)
    s1, s2, s3 = env.sub(1), env.sub(2), env.sub(3)
    assert s1.base_id == s2.base_id == s3.base_id != ""
    assert s2.derived_from == s1.key                                  # the alias did not hide the near copy
    assert s3.derived_from == "" and s3.status == ST.PENDING_CONFIRM  # a genuine independent fine-tune is not penalised
    assert s2.status == ST.PENDING_CONFIRM                            # derived, and beats its parent by the margin


async def test_a_base_under_an_unknown_alias_is_pending(tmp_path):
    env = Env(tmp_path, {}, base_dir=bases(tmp_path, make_weights(999), DEFAULT_BASE))
    env.add(1, "a/m1", 0.9, base_model="qwen/qwen3-4b@" + "b" * 40)  # not a name this validator has
    await env.cycle({1: "a/m1"})
    assert env.sub(1).status == ST.PENDING_VERIFY and env.server.served == []


async def test_a_base_that_shares_no_tensor_with_the_model_is_pending(tmp_path):
    other = {f"other.{i}": np.ones((8, 8), dtype=np.float32) for i in range(2)}
    env = Env(tmp_path, {}, base_dir=bases(tmp_path, other, DEFAULT_BASE))
    env.add(1, "a/m1", 0.9)
    await env.cycle({1: "a/m1"})
    assert env.sub(1).status == ST.PENDING_VERIFY and "no weight" in env.sub(1).reason and env.server.served == []


async def test_unverified_champion_of_an_older_state_loses_the_award_until_verified(tmp_path):
    env = Env(tmp_path, {})
    await crown(env, 1, "a/champ", 0.8)
    champ = env.sub(1)
    assert champ.status == ST.CHAMPION
    champ.delta_checked, champ.base_id = False, ""                   # as crowned by a release that let it pass
    env.ev.base_provider = C.dir_base_provider("")                   # and the base is not on this host
    env.add(2, "a/next", 0.5)
    out = await env.cycle({1: "a/champ", 2: "a/next"})
    assert env.sub(1).status == ST.PENDING_VERIFY and env.ev.state.track(TRACK).champion == ""
    assert out[TRACK] == []


async def test_a_verified_submission_stays_verified_when_the_base_is_removed(tmp_path):
    env = Env(tmp_path, {})
    await crown(env, 1, "a/champ", 0.8)
    env.ev.base_provider = C.dir_base_provider("")
    import shutil
    shutil.rmtree(env.ev._model_dir(env.sub(1)))                     # forces a fresh download and copy check
    env.add(2, "a/next", 0.5)
    await env.cycle({1: "a/champ", 2: "a/next"})
    assert env.sub(1).status == ST.CHAMPION and env.sub(2).status == ST.PENDING_VERIFY


# ------------------------------------------------------------------------------------------------ rollback compatibility
def _a_state(tmp_path):
    st = ST.SpecState(eval_times=[1.0])
    ts = st.track(TRACK)
    sub = ST.Submission(key="k1", uid=1, hotkey=HOTKEYS[1], coldkey="c", track_id=TRACK, repo="a/m", revision="a" * 40,
                        first_seen_block=1, first_seen_ts=0.0, status=ST.PENDING_CONFIRM, score=40.0, draw1="d1",
                        first_incumbent="", base_id="f" * 64, delta_checked=True,
                        history=[{"draw": "d1", "kind": "first", "score": 40.0, "incumbent": None}])
    ts.submissions[sub.key] = sub
    ts.draws["d1"] = ST.Draw("d1", 1.0, 1, 7, ["row-1"])
    path = tmp_path / "spec.json"
    ST.save(st, str(path))
    return path


def test_state_of_a_newer_release_loads_after_a_rollback(tmp_path):
    """A newer release adds keys at every level; this loader (the "older" one here) keeps everything it knows."""
    path = _a_state(tmp_path)
    raw = json.load(open(path))
    raw["future_top"] = {"x": 1}
    t = raw["tracks"][TRACK]
    t["future_track"] = [1, 2]
    t["submissions"]["k1"]["future_field"] = "y"
    t["draws"]["d1"]["future_draw"] = 3
    json.dump(raw, open(path, "w"))
    st = ST.load(str(path))
    assert not list(tmp_path.glob("spec.json.corrupt-*"))                  # not moved aside
    sub = st.track(TRACK).submissions["k1"]
    assert sub.status == ST.PENDING_CONFIRM and sub.first_incumbent == "" and sub.base_id == "f" * 64 and sub.delta_checked
    assert st.track(TRACK).draws["d1"].item_ids == ["row-1"] and st.eval_times == [1.0]
    assert ST.from_json(raw).track(TRACK).submissions["k1"].score == 40.0  # the remote reader's path


def test_state_of_an_older_release_loads_with_defaults(tmp_path):
    """1.2.0rc4 wrote no first_incumbent and no base_id: they get their defaults and the first-pass history decides."""
    path = _a_state(tmp_path)
    raw = json.load(open(path))
    for k in ("first_incumbent", "base_id"):
        del raw["tracks"][TRACK]["submissions"]["k1"][k]
    json.dump(raw, open(path, "w"))
    st = ST.load(str(path))
    assert not list(tmp_path.glob("spec.json.corrupt-*"))
    sub = st.track(TRACK).submissions["k1"]
    assert sub.first_incumbent is None and sub.base_id == "" and sub.status == ST.PENDING_CONFIRM
    assert E.SpecEvaluator._first_incumbent(sub) == ""                   # vacant at the first pass (from the history)


def test_fields_added_after_rc4_have_defaults():
    rc4_required = {"key", "uid", "hotkey", "coldkey", "track_id", "repo", "revision", "first_seen_block", "first_seen_ts"}
    required = {f.name for f in dataclasses.fields(ST.Submission)
                if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING}
    assert required == rc4_required and ST.SPEC_STATE_VERSION == 1


@pytest.mark.parametrize("child_first", [False, True])
async def test_shared_parent_and_challenger_served_once(tmp_path, monkeypatch, child_first):
    from reax_subnet.spec import scoring as SC

    env = Env(tmp_path, {}, max_evals=3)
    await crown(env, 1, "a/champ", 0.80)
    env.clock.advance(3600)
    parent = finetune(make_weights(999), 10)
    env.add(2, "a/parent", 0.1, weights=parent)
    env.add(3, "a/child", 0.95, weights=finetune(parent, 11, scale=0.001))
    served_items, decision_results, draw_rows = {}, {}, []
    original_serve, original_decide = env.ev._serve, env.ev._decide

    async def serve(prepared, rows, now):
        items = await original_serve(prepared, rows, now)
        served_items.setdefault(prepared.sub.key, []).append(items)
        draw_rows.append(tuple(r["id"] for r in rows))
        return items

    def decide(ts, tdef, challengers, results, *args, **kwargs):
        decision_results.update(results)
        return original_decide(ts, tdef, challengers, results, *args, **kwargs)

    monkeypatch.setattr(env.ev, "_serve", serve)
    monkeypatch.setattr(env.ev, "_decide", decide)
    env.server.served.clear()
    commits = {1: "a/champ", 2: "a/parent", 3: "a/child"}
    nodes = [dataclasses.replace(neuron(u, commit(TRACK, repo)),
                                 registered_block=100 if child_first and u == 3 else 0)
             for u, repo in commits.items()]
    snap = snapshot(1200, nodes)
    if child_first:
        # Establish the earlier parent sketch without serving it, then let immunity put its child first.
        env.cfg.max_evals_per_hour = 0
        await env.ev.evaluate_cycle(snap)
        assert await env.ev._prepare(env.cfg.tracks[0], env.sub(2), snap, env.clock.t) is not None
        env.cfg.max_evals_per_hour = 3
    await env.ev.evaluate_cycle(snap)
    assert env.sub(3).derived_from == env.sub(2).key
    assert sorted(env.server.served) == ["a/champ", "a/child", "a/parent"]
    assert len(env.ev.state.eval_times) == 3
    assert len(set(draw_rows)) == 1  # every role uses the same draw
    parent_key = env.sub(2).key
    assert len(served_items[parent_key]) == 1
    assert decision_results[parent_key] is served_items[parent_key][0]
    parent_score = SC.score_items(served_items[parent_key][0]).total
    assert env.sub(2).score == pytest.approx(parent_score)
    parent_first = next(h for h in env.sub(2).history if h["kind"] == "first")
    child_first = next(h for h in env.sub(3).history if h["kind"] == "first")
    assert parent_first["score"] == pytest.approx(parent_score)
    assert parent_first["draw"] == child_first["draw"]
    assert env.sub(3).status == ST.PENDING_CONFIRM


@pytest.mark.parametrize("failing_uid", [2, 3], ids=["first-candidate", "after-prepared-candidate"])
async def test_entry_local_fetch_io_failure_does_not_block_later_candidates(tmp_path, monkeypatch, failing_uid):
    import errno

    env = Env(tmp_path, {}, max_evals=6)  # retain spare capacity so the retry-delay check is not budget-blocked
    await crown(env, 1, "a/champ", 0.80)
    env.clock.advance(3600)
    commits = {1: "a/champ"}
    for uid in (2, 3, 4):
        env.add(uid, f"q/{uid}", 0.1)
        commits[uid] = f"q/{uid}"
    original_fetch = env.hub.fetch
    failures = []
    unavailable = True

    async def fetch(commitment, *args, **kwargs):
        if commitment.repo == f"q/{failing_uid}" and unavailable:
            failures.append(env.clock.t)
            raise FileNotFoundError(errno.ENOENT, "entry-local staging file disappeared")
        return await original_fetch(commitment, *args, **kwargs)

    monkeypatch.setattr(env.hub, "fetch", fetch)
    env.server.served.clear()
    await env.cycle(commits, block=1200)
    assert "q/4" in env.server.served  # a healthy candidate after the error must still run
    assert all(env.sub(u).status == ST.NOT_BETTER for u in (2, 3, 4) if u != failing_uid)
    failed = env.sub(failing_uid)
    # Proposed deferral contract: host attribution, queued/held rather than miner rejection, and a timed retry.
    assert failed.status in (ST.QUEUED, ST.PENDING_VERIFY)
    assert failed.wait_kind == ST.WAIT_INFRA and failed.reason
    assert failed.last_eval_ts is None and failed.last_fetch_ts == env.clock.t and failed.attempts == 0
    assert failures == [env.clock.t]
    env.clock.advance(C.RETRY_UNAVAILABLE_S - 1)
    await env.cycle(commits, block=1200)
    assert len(failures) == 1
    env.clock.advance(1)
    await env.cycle(commits, block=1200)
    assert len(failures) == 2  # one retry when due, not an unbounded loop inside the cycle
    assert failed.status in (ST.QUEUED, ST.PENDING_VERIFY) and failed.attempts == 0
    unavailable = False
    env.clock.advance(3600)
    await env.cycle(commits, block=1200)
    assert failed.status == ST.NOT_BETTER and f"q/{failing_uid}" in env.server.served


async def test_host_wide_fetch_io_failure_aborts_without_rejecting_miners(tmp_path, monkeypatch, caplog):
    import errno
    import logging

    env = Env(tmp_path, {}, max_evals=4)
    await crown(env, 1, "a/champ", 0.80)
    env.clock.advance(3600)
    commits = {1: "a/champ", 2: "q/2", 3: "q/3"}
    for uid in (2, 3):
        env.add(uid, f"q/{uid}", 0.1)
    # Record first-seen before the host runs out of disk, then pass the miner publish deadline.
    env.cfg.max_evals_per_hour = 0
    await env.cycle(commits, block=1200)
    env.cfg.max_evals_per_hour = 4
    original_fetch = env.hub.fetch
    attempts = []

    async def disk_full(commitment, *args, **kwargs):
        attempts.append(commitment.repo)
        raise OSError(errno.ENOSPC, "host model filesystem is full")

    monkeypatch.setattr(env.hub, "fetch", disk_full)
    env.server.served.clear()
    with caplog.at_level(logging.ERROR, logger="reax.spec"):
        for _ in range(2):
            env.clock.advance(3600)
            await env.cycle(commits, block=1200 + C.PUBLISH_DEADLINE_BLOCKS + 1)
            assert not env.server.served and not env.ev.state.eval_times
            assert all(env.sub(u).status == ST.QUEUED and env.sub(u).attempts == 0 for u in (2, 3))
            assert env.sub(1).status == ST.CHAMPION
    assert len(attempts) == 2  # host-wide failure stops preparation on each turn
    assert "evaluation failed" in caplog.text and "host model filesystem is full" in caplog.text
    monkeypatch.setattr(env.hub, "fetch", original_fetch)
    env.clock.advance(3600)
    await env.cycle(commits, block=1200 + C.PUBLISH_DEADLINE_BLOCKS + 1)
    assert all(env.sub(u).status == ST.NOT_BETTER for u in (2, 3))
    assert {"a/champ", "q/2", "q/3"} <= set(env.server.served)
