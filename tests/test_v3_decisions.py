"""Eval mechanism v3, stage 1: variance-aware decisions (machine.py constants, evaluator ``_decide``)."""

import dataclasses
import json
import math

import pytest
from spec_helpers import HOTKEYS, synthetic_pool, write_pool
from test_spec_evaluator import TRACK, Env, crown

from reax_subnet.spec import config as C
from reax_subnet.spec import machine as M
from reax_subnet.spec import scoring as SC
from reax_subnet.spec import state as ST

pytestmark = pytest.mark.usefixtures("rc5_entry_floors")
H = 3600
_BaseCfg = C.SpecConfig


@dataclasses.dataclass
class _V3Cfg(_BaseCfg):
    v3_decisions: bool = True   # the release default is off until the notice; these tests exercise the v3 rule


@pytest.fixture(autouse=True)
def _v3_on(monkeypatch):
    monkeypatch.setattr(C, "SpecConfig", _V3Cfg)


# ------------------------------------------------------------------------------------------------ pure functions
def test_constants():
    assert (M.Z_PASS, M.Z_CLOSE, M.Z_EXTRA, M.Z_CROWN, M.MARGIN_FLOOR) == (2.0, 1.0, 1.5, 2.75, 1.5)
    assert M.V3_DECISIONS is False and _BaseCfg("s", "w", "p").v3_decisions is False   # off until the dated notice


def test_pooled_one_draw_is_exact_and_two_draws_use_rms_over_sqrt_k():
    assert M.pooled([3.7], [1.3]) == (3.7, 1.3)
    mean, se = M.pooled([4.0, 2.0], [1.0, 3.0])
    assert mean == 3.0 and se == pytest.approx(math.sqrt((1 + 9) / 2) / math.sqrt(2))
    assert M.pooled([], []) == (0.0, math.inf)


def test_z_of_degenerate_cases():
    assert M.z_of(2.0, 1.0) == 2.0
    assert M.z_of(2.0, math.inf) == 0.0 and M.z_of(2.0, math.nan) == 0.0
    assert M.z_of(2.0, 0.0) == M.Z_RECORD_MAX and M.z_of(-2.0, 0.0) == -M.Z_RECORD_MAX and M.z_of(0.0, 0.0) == 0.0
    assert M.z_of(1e9, 1.0) == M.Z_RECORD_MAX   # finite, safe for the state JSON


@pytest.mark.parametrize("gain,se", [(1.5, 0.75), (1.5, 0.7499), (1.4999, 0.5), (3.0, 1.5), (2.9999, 1.5), (10.0, 4.99), (10.0, 5.0)])
def test_first_stage_one_draw_equals_the_old_margin_rule(gain, se):
    old = M.first_pass(gain + 10, 10.0, se).passed
    new = M.first_stage_v3(gain, se, gain, se, close_used=True) == "pass"
    assert old == new


@pytest.mark.parametrize("gain,se,verdict", [
    (3.0, 1.5, "pass"),          # z = 2.0 exactly, gain above the floor
    (2.99, 1.5, "close"),        # z just under 2
    (1.5, 1.5, "close"),         # z = 1.0 exactly
    (1.49, 1.5, "fail"),         # z just under 1
    (3.0, 1.0, "pass"),          # z 3
    (1.4, 0.5, "fail"),          # z 2.8 but under the floor: not close, not pass (as today)
    (-1.0, 1.0, "fail"),
])
def test_first_stage_boundaries(gain, se, verdict):
    assert M.first_stage_v3(gain, se, gain, se, close_used=False) == verdict


def test_first_stage_extra_draw_is_pooled_and_there_is_only_one():
    # second draw: pooled over two draws must reach Z_PASS and the floor; no further close call
    g, se = M.pooled([3.0, 3.0], [2.5, 2.5])      # z = 3 * sqrt(2) / 2.5 = 1.70
    assert M.first_stage_v3(3.0, 2.5, g, se, close_used=True) == "fail"
    g, se = M.pooled([4.0, 3.0], [2.0, 2.0])      # mean 3.5, se 1.414 -> z 2.47
    assert M.first_stage_v3(3.0, 2.0, g, se, close_used=True) == "pass"
    g, se = M.pooled([1.6, 1.0], [0.5, 0.5])      # z large but mean 1.3 < MARGIN_FLOOR
    assert M.first_stage_v3(1.0, 0.5, g, se, close_used=True) == "fail"
    assert M.first_stage_v3(1.5, 1.0, 1.5, 1.0, close_used=True) == "fail"     # would be close; the extra draw is used up


@pytest.mark.parametrize("gain,se,cg,verdict", [
    (2.75, 1.0, 1.0, "crown"),       # z = 2.75 exactly
    (2.7499, 1.0, 1.0, "extra"),
    (1.5 * 1.0, 1.0, 1.0, "extra"),  # z = 1.5 exactly, but mean 1.5 >= floor
    (1.49, 1.0, 1.0, "fail"),        # z under Z_EXTRA
    (4.0, 1.0, -0.1, "fail"),        # strong pooled evidence but the confirmation gain is not positive
    (4.0, 1.0, 0.0, "fail"),
    (1.4, 0.4, 1.4, "fail"),         # z 3.5 but pooled mean under the floor
])
def test_confirmation_boundaries(gain, se, cg, verdict):
    assert M.confirm_v3([(gain, se)], [(cg, se)], extra_used=False) == verdict


def test_extra_confirmation_needs_crown_z_and_positive_sum_of_confirmation_gains():
    gains, ses = [4.0, 1.0, 3.0], [1.0, 1.0, 1.0]           # first stage, confirmation, extra confirmation
    mean, se = M.pooled(gains, ses)
    assert mean / se >= M.Z_CROWN
    pairs = list(zip(gains, ses))
    assert M.confirm_v3(pairs, pairs[1:], extra_used=True) == "crown"
    assert M.confirm_v3(pairs, [(-3.0, 1.0), (0.0, 1.0)], extra_used=True) == "fail"   # sum of confirmation gains <= 0
    assert M.confirm_v3([(2.0, 1.0), (0.5, 1.0), (0.5, 1.0)], [(0.5, 1.0), (0.5, 1.0)], extra_used=True) == "fail"   # pooled z 1.5, below Z_CROWN
    # no third draw: a pooled z between Z_EXTRA and Z_CROWN after the extra draw is a failure, not another extra
    assert M.confirm_v3([(1.5, 1.0)] * 3, [(1.5, 1.0)] * 2, extra_used=True) == "fail"


# ------------------------------------------------------------------------------------------------ evaluator level
@pytest.fixture
def zs(monkeypatch):
    """Control the paired SE: each ``paired_se`` call pops the next target z and returns |gain| / z."""
    seq: list[float] = []

    def fake(a, b, seed, resamples=400):
        gain = SC.score_items(a).total - SC.score_items(b).total
        assert seq, "unexpected paired_se call"
        return abs(gain) / seq.pop(0)

    monkeypatch.setattr(SC, "paired_se", fake)
    return seq


def rows(sub, kind):
    return [h for h in sub.history if h["kind"] == kind]


async def champ_and_challenger(env, zs, first_z, *, skill=0.82):
    await crown(env, 1, "a/champ", 0.65)
    env.add(2, "a/chal", skill)
    zs[:] = [first_z]
    await env.cycle({1: "a/champ", 2: "a/chal"})
    return env.sub(2), env.ev.state.track(TRACK)


async def test_clear_first_draw_goes_straight_to_pending_like_today(tmp_path, zs):
    env = Env(tmp_path, {})
    sub, ts = await champ_and_challenger(env, zs, 2.5)
    assert sub.status == ST.PENDING_CONFIRM and sub.decision_rule == "v3" and not sub.close_draws
    row = rows(sub, "first")[0]
    assert row["inc_key"] == env.sub(1).key and row["se"] > 0
    assert row["z"] == pytest.approx(2.5) and row["pooled_z"] == pytest.approx(2.5)
    assert sub.first_incumbent == env.sub(1).key and sub.draw1 == row["draw"]


async def test_far_from_significant_first_draw_is_not_better_at_once(tmp_path, zs):
    env = Env(tmp_path, {})
    sub, _ = await champ_and_challenger(env, zs, 0.7)
    assert sub.status == ST.NOT_BETTER and "margin" in sub.reason and not sub.close_draws


async def test_close_call_gets_one_more_disjoint_draw_and_then_goes_pending(tmp_path, zs):
    env = Env(tmp_path, {})
    sub, ts = await champ_and_challenger(env, zs, 1.4)
    assert sub.status == ST.QUEUED and sub.reason == "close call (z 1.4): one more draw"
    assert len(sub.close_draws) == 1 and not sub.draw1
    first = sub.close_draws[0]
    t_first = ts.draws[first].ts
    env.clock.advance(1800)                                   # next cycle, 30 minutes later
    zs[:] = [2.6]
    await env.cycle({1: "a/champ", 2: "a/chal"})
    assert sub.status == ST.PENDING_CONFIRM and sub.decision_rule == "v3"
    assert sub.draw1 == first and sub.close_draws == [first, sub.history[-1]["draw"]]
    assert not set(ts.draws[sub.close_draws[0]].item_ids) & set(ts.draws[sub.close_draws[1]].item_ids)
    assert sub.confirm_after_ts == t_first + 12 * H           # counted from the FIRST draw: no extra delay
    last = sub.history[-1]
    assert last["pooled_z"] >= M.Z_PASS and last["z"] == pytest.approx(2.6)
    assert any("close call" in e for e in ts.events) and any("pooled z" in e for e in ts.events)


async def test_close_call_that_stays_close_is_not_better_and_gets_no_third_draw(tmp_path, zs):
    env = Env(tmp_path, {})
    sub, ts = await champ_and_challenger(env, zs, 1.4)
    zs[:] = [1.2]
    await env.cycle({1: "a/champ", 2: "a/chal"})
    assert sub.status == ST.NOT_BETTER and "pooled evidence" in sub.reason
    assert sub.history[-1]["pooled_z"] < M.Z_PASS


async def test_second_close_call_is_never_a_second_extra_draw(tmp_path, zs):
    env = Env(tmp_path, {})
    sub, ts = await champ_and_challenger(env, zs, 1.1)
    zs[:] = [1.9]                       # pooled about 1.9 * ... still under 2: fail, not a new close call
    await env.cycle({1: "a/champ", 2: "a/chal"})
    assert sub.status == ST.NOT_BETTER and len(sub.close_draws) == 1


async def test_close_call_entry_is_picked_before_fresh_queue_entries(tmp_path, zs):
    env = Env(tmp_path, {})
    await crown(env, 1, "a/champ", 0.65)
    env.add(2, "a/chal", 0.82)
    zs[:] = [1.4]
    await env.cycle({1: "a/champ", 2: "a/chal"})
    env.add(3, "a/fresh", 0.80)
    zs[:] = [2.6, 2.5]
    await env.cycle({1: "a/champ", 2: "a/chal", 3: "a/fresh"})
    ts = env.ev.state.track(TRACK)
    assert env.sub(2).status == ST.PENDING_CONFIRM and env.sub(3).status == ST.PENDING_CONFIRM
    # the fresh entry's draw is disjoint from the close-call entry's first draw (one shared draw for the group)
    d_close = set(ts.draws[env.sub(2).close_draws[0]].item_ids)
    assert not d_close & set(ts.draws[env.sub(3).draw1].item_ids)


async def confirm_scenario(env, zs, first_z, confirm_z, *, extra=()):
    sub, ts = await champ_and_challenger(env, zs, first_z)
    assert sub.status == ST.PENDING_CONFIRM
    env.clock.advance(13 * H)
    zs[:] = [confirm_z]
    await env.cycle({1: "a/champ", 2: "a/chal"})
    for z in extra:
        env.clock.advance(1800)
        zs[:] = [z]
        await env.cycle({1: "a/champ", 2: "a/chal"})
    return sub, ts


async def test_confirmation_crowns_on_pooled_evidence(tmp_path, zs):
    env = Env(tmp_path, {})
    sub, ts = await confirm_scenario(env, zs, 2.5, 3.0)
    assert sub.status == ST.CHAMPION and ts.champion == sub.key and env.sub(1).status == ST.FORMER
    assert sub.history[-1]["kind"] == "confirm" and sub.history[-1]["pooled_z"] >= M.Z_CROWN
    first, conf = rows(sub, "first")[0], rows(sub, "confirm")[0]
    assert not set(ts.draws[first["draw"]].item_ids) & set(ts.draws[conf["draw"]].item_ids)


async def test_confirmation_that_the_old_rule_accepts_but_pooled_evidence_does_not(tmp_path, zs):
    env = Env(tmp_path, {})
    sub, ts = await confirm_scenario(env, zs, 2.1, 0.5)
    assert sub.status == ST.NOT_BETTER and "gain too small" in sub.reason
    # the same scenario under the old rule (flag off): positive and >= delta/2 is enough
    env2 = Env(tmp_path / "off", {})
    env2.ev.cfg.v3_decisions = False
    sub2, _ = await confirm_scenario(env2, zs, 2.1, 0.5)
    assert sub2.status == ST.CHAMPION


async def test_confirmation_in_the_gray_zone_gets_one_extra_draw_then_crowns(tmp_path, zs):
    env = Env(tmp_path, {})
    sub, ts = await confirm_scenario(env, zs, 2.05, 1.5)
    assert sub.status == ST.PENDING_CONFIRM and len(sub.confirm_draws) == 1
    assert sub.reason.startswith("confirmation: pooled evidence") and "< 2.75, one more draw" in sub.reason
    assert ts.champion == env.sub(1).key
    env.clock.advance(1800)
    zs[:] = [8.0]
    await env.cycle({1: "a/champ", 2: "a/chal"})
    assert sub.status == ST.CHAMPION and ts.champion == sub.key
    draws = [sub.draw1, *sub.confirm_draws, sub.history[-1]["draw"]]
    sets = [set(ts.draws[d].item_ids) for d in draws]
    assert all(not a & b for i, a in enumerate(sets) for b in sets[i + 1:]), "every draw of the entry is disjoint"
    assert sub.history[-1]["pooled_z"] >= M.Z_CROWN


async def test_extra_confirmation_that_does_not_reach_the_crown_bar_fails(tmp_path, zs):
    env = Env(tmp_path, {})
    sub, ts = await confirm_scenario(env, zs, 2.05, 1.5, extra=[0.4])
    assert sub.status == ST.NOT_BETTER and ts.champion == env.sub(1).key


async def test_no_second_extra_confirmation_draw(tmp_path, zs):
    env = Env(tmp_path, {})
    sub, ts = await confirm_scenario(env, zs, 2.05, 1.5, extra=[2.2])
    assert sub.status in (ST.NOT_BETTER, ST.CHAMPION) and len(sub.confirm_draws) == 1


async def test_parent_must_still_be_beaten_by_the_full_margin_per_draw(tmp_path, zs):
    env = Env(tmp_path, {})
    sub, ts = await champ_and_challenger(env, zs, 2.5)
    env.clock.advance(13 * H)
    zs[:] = [5.0]
    env.ev.state.track(TRACK).submissions[sub.key].derived_from = env.sub(1).key    # twin of the incumbent
    await env.cycle({1: "a/champ", 2: "a/chal"})
    # the incumbent is the parent: nothing extra is asked of the pair beyond the margin already required
    assert sub.status == ST.CHAMPION


def vacant_pair(env):
    env.add(1, "a/first", 0.65)
    env.add(2, "a/second", 0.80)


async def vacant_then_new_incumbent(env, zs, z_new):
    """uid 2 passed its first draw on a vacant track; uid 1 is crowned first; uid 2 then meets the new champion alone."""
    vacant_pair(env)
    await env.cycle({1: "a/first"})
    env.clock.advance(H)
    await env.cycle({1: "a/first", 2: "a/second"}, block=1001)
    env.clock.advance(11.5 * H)
    await env.cycle({1: "a/first", 2: "a/second"})
    assert env.sub(1).status == ST.CHAMPION and env.sub(2).status == ST.PENDING_CONFIRM
    assert env.sub(2).first_incumbent == "" and env.sub(2).decision_rule == "v3"
    env.clock.advance(H)
    zs[:] = [z_new]
    await env.cycle({1: "a/first", 2: "a/second"})
    return env.sub(2), env.ev.state.track(TRACK)


async def test_new_incumbent_is_met_on_its_own_evidence_only(tmp_path, zs):
    env = Env(tmp_path / "a", {})
    sub, ts = await vacant_then_new_incumbent(env, zs, 3.2)
    assert sub.status == ST.CHAMPION and sub.history[-1]["pooled_z"] == pytest.approx(3.2)
    assert sub.history[-1]["inc_key"] == env.sub(1).key and rows(sub, "first")[0]["inc_key"] == ""
    env = Env(tmp_path / "b", {})
    sub, ts = await vacant_then_new_incumbent(env, zs, 2.0)       # old rule: full margin (z >= 2) would crown; v3 wants more
    assert sub.status == ST.PENDING_CONFIRM and len(sub.confirm_draws) == 1
    env.clock.advance(1800)
    zs[:] = [4.0]
    await env.cycle({1: "a/first", 2: "a/second"})
    # pooled over the two draws against the new incumbent: z about 4 * sqrt(2) / sqrt(1 + 4/16...) >= 2.75
    assert sub.status == ST.CHAMPION
    env = Env(tmp_path / "c", {})
    sub, ts = await vacant_then_new_incumbent(env, zs, 1.0)
    assert sub.status == ST.NOT_BETTER and ts.champion == env.sub(1).key


async def test_same_round_tie_goes_to_the_earlier_commitment_unless_the_later_is_clearly_better(tmp_path, zs):
    # two challengers due on the same confirmation draw; the first is crowned, the second meets it with the pooled bar
    for second_z, expect in ((2.0, ST.PENDING_CONFIRM), (3.0, ST.CHAMPION), (0.8, ST.NOT_BETTER)):
        env = Env(tmp_path / f"t{second_z}", {})
        await crown(env, 1, "a/champ", 0.65)
        env.add(2, "a/c2", 0.80)
        env.add(3, "a/c3", 0.95)
        zs[:] = [3.0, 3.0]
        await env.cycle({1: "a/champ", 2: "a/c2", 3: "a/c3"})
        assert env.sub(2).status == env.sub(3).status == ST.PENDING_CONFIRM
        env.clock.advance(13 * H)
        zs[:] = [5.0, second_z]
        await env.cycle({1: "a/champ", 2: "a/c2", 3: "a/c3"})
        events = env.ev.state.track(TRACK).events
        crowned = [e for e in events if "is the new champion" in e]
        assert env.sub(2).key[:12] in crowned[-2 if expect == ST.CHAMPION else -1], "the earlier commitment is crowned first"
        assert env.sub(3).status == expect
        # a tie inside the noise keeps the earlier commitment; only a clear win (pooled z >= Z_CROWN) takes the crown
        assert env.sub(2).status == (ST.FORMER if expect == ST.CHAMPION else ST.CHAMPION)


# ------------------------------------------------------------------------------------------------ flag and grandfathering
async def test_flag_off_restores_the_old_rule(tmp_path, zs):
    env = Env(tmp_path, {})
    env.ev.cfg.v3_decisions = False
    sub, _ = await champ_and_challenger(env, zs, 1.4)
    assert sub.status == ST.NOT_BETTER and "margin" in sub.reason and not sub.close_draws and sub.decision_rule == ""
    env2 = Env(tmp_path / "x", {})
    env2.ev.cfg.v3_decisions = False
    sub2, ts2 = await champ_and_challenger(env2, zs, 2.5)
    assert sub2.status == ST.PENDING_CONFIRM and sub2.decision_rule == "" and sub2.delta is not None
    assert sub2.confirm_after_ts == ts2.draws[sub2.draw1].ts + 12 * H


async def test_entry_pending_from_the_old_rule_is_confirmed_under_the_old_rule(tmp_path, zs):
    env = Env(tmp_path, {})
    env.ev.cfg.v3_decisions = False                  # first pass before the switch
    sub, ts = await champ_and_challenger(env, zs, 2.1)
    assert sub.status == ST.PENDING_CONFIRM and sub.decision_rule == ""
    env.ev.cfg.v3_decisions = True                   # the switch
    env.clock.advance(13 * H)
    zs[:] = [0.5]                                    # v3 would refuse this (see the pooled-evidence test above)
    await env.cycle({1: "a/champ", 2: "a/chal"})
    assert sub.status == ST.CHAMPION and sub.decision_rule == ""


async def test_v3_entry_confirms_under_the_old_rule_after_a_rollback(tmp_path, zs):
    env = Env(tmp_path, {})
    sub, ts = await champ_and_challenger(env, zs, 2.5)
    assert sub.decision_rule == "v3"
    env.ev.cfg.v3_decisions = False
    env.clock.advance(13 * H)
    zs[:] = [3.0]
    await env.cycle({1: "a/champ", 2: "a/chal"})
    assert sub.status == ST.CHAMPION


async def test_close_call_draws_stay_in_the_state_for_the_next_cycle(tmp_path, zs):
    env = Env(tmp_path, {})
    sub, ts = await champ_and_challenger(env, zs, 1.4)
    first = sub.close_draws[0]
    for i in range(60):               # far more than MAX_UNREFERENCED_DRAWS other draws
        ts.draws[f"junk{i}"] = ST.Draw(f"junk{i}", 1e9 + i, 1, i, [], True)
    env.ev._prune_draws(ts, env.clock())
    assert first in ts.draws


# ------------------------------------------------------------------------------------------------ state compatibility
def test_old_state_without_the_new_fields_loads_with_defaults():
    old = {"version": ST.SPEC_STATE_VERSION, "saved_at": 1, "eval_times": [],
           "tracks": {TRACK: {"champion": "", "submissions": {"k": {
               "key": "k", "uid": 1, "hotkey": HOTKEYS[1], "coldkey": "c", "track_id": TRACK, "repo": "a/m", "revision": "a" * 40,
               "first_seen_block": 1, "first_seen_ts": 0.0, "status": ST.PENDING_CONFIRM, "draw1": "d1",
               "history": [{"draw": "d1", "kind": "first", "score": 40.0, "incumbent": 38.0}]}}}}}
    sub = ST.from_json(old).tracks[TRACK].submissions["k"]
    assert sub.decision_rule == "" and sub.close_draws == [] and sub.confirm_draws == []


def test_a_new_state_loads_in_a_worker_that_does_not_know_the_new_fields():
    sub = ST.Submission(key="k", uid=1, hotkey=HOTKEYS[1], coldkey="c", track_id=TRACK, repo="a/m", revision="a" * 40,
                        first_seen_block=1, first_seen_ts=0.0, decision_rule="v3", close_draws=["d0", "d1"], confirm_draws=["d2"])
    state = ST.SpecState()
    state.track(TRACK).submissions["k"] = sub
    raw = json.loads(json.dumps(ST._to_json(state)))

    @dataclasses.dataclass
    class OldSubmission:         # the Submission of release 18e2597 (no v3 fields): the loader filters unknown keys
        key: str
        uid: int
        hotkey: str
        coldkey: str
        track_id: str
        repo: str
        revision: str
        first_seen_block: int
        first_seen_ts: float
        status: str = ST.QUEUED
        history: list = dataclasses.field(default_factory=list)

    old = ST._build(OldSubmission, raw["tracks"][TRACK]["submissions"]["k"])
    assert old.key == "k" and not hasattr(old, "decision_rule")
    again = ST.from_json(raw).tracks[TRACK].submissions["k"]
    assert again.decision_rule == "v3" and again.close_draws == ["d0", "d1"] and again.confirm_draws == ["d2"]


def test_env_flag(tmp_path):
    base = {"REAX_SPEC_SEALED_DIR": str(tmp_path)}
    assert C.from_env(base).v3_decisions is False
    assert C.from_env({**base, "REAX_SPEC_V3_DECISIONS": "0"}).v3_decisions is False
    assert C.from_env({**base, "REAX_SPEC_V3_DECISIONS": "off"}).v3_decisions is False
    assert C.from_env({**base, "REAX_SPEC_V3_DECISIONS": "1"}).v3_decisions is True
    assert C.from_env({**base, "REAX_SPEC_V3_DECISIONS": ""}).v3_decisions is False      # empty = release default (off)
    assert C.from_env({**base, "REAX_SPEC_V3_DECISIONS": " ON "}).v3_decisions is True
    for bad in ("disabled", "none", "2"):
        with pytest.raises(ValueError):
            C.from_env({**base, "REAX_SPEC_V3_DECISIONS": bad})


def test_close_call_needs_half_the_margin_floor():
    from reax_subnet.spec import machine as M
    # z = 1.33 but the gain (0.2) can never reach the 1.5 floor: no wasted extra draw
    assert M.first_stage_v3(0.2, 0.15, 0.2, 0.15, close_used=False) == "fail"
    assert M.first_stage_v3(0.8, 0.6, 0.8, 0.6, close_used=False) == "close"


def test_confirmation_draw_must_itself_confirm():
    # huge first stage, confirmation gain barely positive (z 0.1): pooled z is high, but the confirmation does not confirm
    assert M.confirm_v3([(12.0, 3.5), (0.4, 3.5)], [(0.4, 3.5)], extra_used=False) == "extra"
    assert M.confirm_v3([(12.0, 3.5), (0.4, 3.5), (0.3, 3.5)], [(0.4, 3.5), (0.3, 3.5)], extra_used=True) == "fail"
    assert M.confirm_v3([(12.0, 3.5), (2.0, 3.5)], [(2.0, 3.5)], extra_used=False) == "crown"   # z 0.57 >= 0.5


def test_scored_x_mode_fails_closed_to_shadow():
    base = {"REAX_SPEC_SEALED_DIR": "s", "REAX_SPEC_WORK_DIR": "w", "REAX_SPEC_STATE_PATH": "p"}
    assert C._x_mode("scored") == "shadow" and C._x_mode("bogus") == "off" and C._x_mode("shadow") == "shadow"
