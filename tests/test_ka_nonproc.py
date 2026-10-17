"""1.2.0rc6 keep-alive on the non-proc tiers, the per-tier score history and the same-tier shortfall of the sampler."""

import collections
import hashlib
import json
import os
from contextlib import asynccontextmanager

import pytest
from spec_helpers import commit, neuron, snapshot, synthetic_pool
from test_keepalive import DF, NOW, RA, _snap, _sub
from test_spec_evaluator import TRACK, Env

from reax_subnet import competition_split as CS
from reax_subnet.spec import evaluator as E
from reax_subnet.spec import sampling as S
from reax_subnet.spec import scoring as SC
from reax_subnet.spec import state as ST
from reax_subnet.spec.evaluator import compute_keepalive, nonproc_scores

# The sampler mechanism tests below were written for the 1.2.0rc5 mix (50 % proc); they test how shortfalls are filled,
# which does not depend on the shares, so they pin that mix. The release mix has its own test at the end.
RC5_MIX = {"proc_sealed": 0.5, "same_family": 0.25, "new_domain": 0.125, "new_family": 0.125}


@pytest.fixture(autouse=True)
def _rc5_mix(request, monkeypatch):
    if "release_mix" not in request.node.name:
        monkeypatch.setattr(S, "TIER_MIX", RC5_MIX)

# The deployed sealed pools per tier (full pools, 16 Oct 2026) and the reference model's (Plumb-4B = s1-fast) score per
# tier on them, measured with the validator's evaluator.
POOLS = {DF: {"proc_sealed": 900, "same_family": 102, "new_domain": 127, "new_family": 298},
         RA: {"proc_sealed": 600, "same_family": 127, "new_domain": 83, "new_family": 316}}
REFERENCE = {DF: {"proc_sealed": 10.97, "same_family": 40.35, "new_domain": 32.77, "new_family": 79.81},
             RA: {"proc_sealed": 12.46, "same_family": 36.03, "new_domain": 33.38, "new_family": 53.01}}


def tier_pool(counts):
    rows, i = [], 0
    for tier, c in counts.items():
        for _ in range(c):
            rows.append({"id": f"r{i:05d}", "tier": tier, "question": {"type": S.PRIMITIVES[i % 3]}})
            i += 1
    return rows


def rounds(pool, k, n=600, exclude_first=False):
    used, out = {}, []
    for j in range(k):
        rows, fresh = S.draw(pool, n=n, seed=100 + j, use_counts=used)
        for r in rows:
            used[r["id"]] = used.get(r["id"], 0) + 1
        out.append((collections.Counter(r["tier"] for r in rows), fresh, rows))
    return out


# ------------------------------------------------------------------------------------------------ sampler
@pytest.mark.parametrize("track", [DF, RA])
def test_six_rounds_on_the_deployed_pools_keep_the_proc_share(track):
    pool = tier_pool(POOLS[track])
    out = rounds(pool, 6)
    for j, (c, fresh, rows) in enumerate(out):
        assert len(rows) == 600 and len({r["id"] for r in rows}) == 600
        assert c["proc_sealed"] == 300                                    # the published 50 %, every round
        assert c["new_domain"] >= 75 and c["new_family"] >= 75            # at least their quota
        assert c["same_family"] == min(150, POOLS[track]["same_family"])  # the whole small tier, every round
        assert fresh == (j == 0)                                          # reuse from round 2 on, reported
    kinds = collections.Counter(r["question"]["type"] for r in out[-1][2])
    assert all(190 <= kinds[k] <= 210 for k in S.PRIMITIVES)


def test_rc6_sampler_let_proc_grow_beyond_its_share():
    """The measured problem: with the rc6 shortfall rule, proc took 62-70 % of DF draws 2..6; now exactly 50 %."""
    pool = tier_pool(POOLS[DF])
    assert all(c["proc_sealed"] == 300 for c, _, _ in rounds(pool, 6))


def test_a_short_tier_reuses_its_own_rows_before_other_fresh_rows():
    pool = synthetic_pool(1000)                       # 400 proc, 200 per non-proc tier
    sf = [r for r in pool if r["tier"] == "same_family"]
    used = {r["id"]: 1 for r in sf[:180]}             # 20 fresh same_family rows left
    rows, fresh = S.draw(pool, n=400, seed=7, use_counts=used)
    c = collections.Counter(r["tier"] for r in rows)
    assert not fresh
    assert c == {"proc_sealed": 200, "same_family": 100, "new_domain": 50, "new_family": 50}
    picked_sf = {r["id"] for r in rows if r["tier"] == "same_family"}
    assert {r["id"] for r in sf[180:]} <= picked_sf   # its fresh rows first


def test_least_used_rows_of_the_tier_come_first():
    pool = synthetic_pool(1000)
    sf = [r for r in pool if r["tier"] == "same_family"]
    used = {r["id"]: (1 if i < 150 else 3) for i, r in enumerate(sf)}
    rows, _ = S.draw(pool, n=400, seed=3, use_counts=used)
    got = [r for r in rows if r["tier"] == "same_family"]
    assert len(got) == 100 and all(used[r["id"]] == 1 for r in got)


def test_tier_too_small_overflows_to_non_proc_tiers_and_proc_never_grows():
    pool = tier_pool({"proc_sealed": 400, "same_family": 30, "new_domain": 200, "new_family": 200})
    rows, fresh = S.draw(pool, n=400, seed=1)
    c = collections.Counter(r["tier"] for r in rows)
    assert fresh and c["proc_sealed"] == 200 and c["same_family"] == 30 and c["new_domain"] + c["new_family"] == 170
    # the non-proc tiers exhausted (confirmation excludes): proc keeps its quota and the draw is smaller (1.2.0rc6)
    excl = {r["id"] for r in pool if r["tier"] in ("new_domain", "new_family")}
    rows, _ = S.draw(pool, n=400, seed=1, exclude=excl, min_n=50)
    c = collections.Counter(r["tier"] for r in rows)
    assert c == {"proc_sealed": 30, "same_family": 30}         # proc keeps its 50 % of the smaller draw
    with pytest.raises(S.SampleError):                          # fewer items than the caller's minimum: no draw at all
        S.draw(pool, n=400, seed=1, exclude=excl, min_n=100)



def test_confirmation_draw_stays_disjoint_and_keeps_the_shares():
    pool = tier_pool(POOLS[RA])
    first, _ = S.draw(pool, n=600, seed=1)
    ids = {r["id"] for r in first}
    used = {i: 1 for i in ids}
    second, fresh = S.draw(pool, n=600, seed=2, use_counts=used, exclude=ids)
    assert not ({r["id"] for r in second} & ids)
    c = collections.Counter(r["tier"] for r in second)
    assert c["proc_sealed"] <= 0.5 * sum(c.values()) and fresh            # proc never above its share, even here


# ------------------------------------------------------------------------------------------------ release constant
# Release floors, measured (competition_split.py): the reference model's (Plumb-4B = s1-fast) score with the validator's
# evaluator on six consecutive 300-item release draws from the deployed pools (live use counts): whole draw
# (TRACK_ENTRY_FLOOR) and its non-proc items (KEEPALIVE_NONPROC_FLOOR), each + 1.5 points. 1.2.0rc7.post2: the refilled
# pools of 17 Oct 2026 (1.2.0rc6, 16 Oct: 38.17 / 31.66 and 48.34 / 36.27).
MEASURED_ENTRY = {DF: 27.32, RA: 28.54}       # 1800 items per track
MEASURED_NONPROC = {DF: 31.10, RA: 32.95}     # the 1530 non-proc items of the same draws


@pytest.mark.parametrize("track", [DF, RA])
def test_release_mix_floors_are_the_measured_reference_plus_1_5(track):
    assert CS.TRACK_ENTRY_FLOOR[track] == round(MEASURED_ENTRY[track] + 1.5, 1)
    assert CS.KEEPALIVE_NONPROC_FLOOR[track] == round(MEASURED_NONPROC[track] + 1.5, 1)
    # the keep-alive non-proc floor lies above the entry floor; the entry floor stays in force as well
    assert CS.KEEPALIVE_NONPROC_FLOOR[track] > CS.TRACK_ENTRY_FLOOR[track]
    # the measured draws: six full release draws of 300 items on the deployed pools, 45 proc and 255 non-proc each
    out = rounds(tier_pool(POOLS[track]), 6, n=S.SAMPLE_SIZE)
    assert S.SAMPLE_SIZE == 300 and all(sum(c.values()) == 300 and c["proc_sealed"] == 45 for c, _, _ in out)
    assert sum(sum(c.values()) - c["proc_sealed"] for c, _, _ in out) == 1530


# ------------------------------------------------------------------------------------------------ scores per tier
def _items(tiers_correct):
    rows, probs = [], []
    for i, (tier, ok) in enumerate(tiers_correct):
        rows.append({"id": f"x{i}", "tier": tier, "question": {"type": "noul"}, "labels": ["no", "yes"], "expected": "yes"})
        probs.append({"yes": 0.9, "no": 0.1} if ok else {"yes": 0.1, "no": 0.9})
    return SC.score_run(rows, probs)[0]


def test_nonproc_scores_leave_out_the_proc_items():
    items = _items([("proc_sealed", True)] * 10 + [("same_family", False)] * 5 + [("new_family", True)] * 5)
    nonproc, by_tier = nonproc_scores(items)
    assert nonproc == pytest.approx(SC.score_items(items[10:]).total)
    assert nonproc < SC.score_items(items).total
    assert set(by_tier) == {"proc_sealed", "same_family", "new_family"} and by_tier["proc_sealed"] > by_tier["same_family"]
    assert nonproc_scores(items[:10]) == (None, {"proc_sealed": round(SC.score_items(items[:10]).total, 4)})


# ------------------------------------------------------------------------------------------------ keep-alive rule
def test_keepalive_needs_a_nonproc_score_above_the_floor():
    st = ST.SpecState()
    subs = [_sub(st, 1, ST.CHAMPION, 60.0, nonproc=60.0), _sub(st, 2, ST.PENDING_CONFIRM, 50.0, nonproc=30.0),
            _sub(st, 3, ST.PENDING_CONFIRM, 50.0, nonproc=None),          # last scored before this release: fail closed
            _sub(st, 4, ST.PENDING_CONFIRM, 50.0, nonproc=32.6),          # at the floor: no
            _sub(st, 5, ST.PENDING_CONFIRM, 50.0, nonproc=float("nan")),
            _sub(st, 6, ST.PENDING_CONFIRM, 20.0, nonproc=80.0)]          # below the whole-draw entry floor: no
    out = compute_keepalive(st, [DF], _snap(subs), {DF: 24.8}, now=NOW)[DF]        # default: the release non-proc floor
    assert out == [(1, 60.0)]
    assert [u for u, _ in compute_keepalive(st, [DF], _snap(subs), {DF: 24.8}, nonproc_floors={DF: 25.0}, now=NOW)[DF]] == [1, 2, 4]
    # a track without a non-proc floor still needs a non-proc score
    assert [u for u, _ in compute_keepalive(st, [DF], _snap(subs), {DF: 24.8}, nonproc_floors={}, now=NOW)[DF]] == [1, 2, 4]


def test_default_nonproc_floor_is_the_release_constant(monkeypatch):
    st = ST.SpecState()
    subs = [_sub(st, 1, ST.CHAMPION, 60.0, nonproc=31.0), _sub(st, 2, ST.CHAMPION, 60.0, nonproc=35.0, track=RA)]
    assert compute_keepalive(st, [DF, RA], _snap(subs), {DF: 0.0, RA: 0.0}, now=NOW) == {DF: [], RA: [(2, 60.0)]}
    monkeypatch.setitem(CS.KEEPALIVE_NONPROC_FLOOR, DF, 30.0)
    monkeypatch.setitem(CS.KEEPALIVE_NONPROC_FLOOR, RA, 36.0)
    assert compute_keepalive(st, [DF, RA], _snap(subs), {DF: 0.0, RA: 0.0}, now=NOW) == {DF: [(1, 60.0)], RA: []}


def test_state_round_trip_and_old_files(tmp_path):
    st = ST.SpecState()
    s = _sub(st, 1, ST.PENDING_CONFIRM, 50.0, nonproc=57.0)
    s.by_tier = {"proc_sealed": 40.0, "same_family": 57.0}
    p = str(tmp_path / "spec.json")
    ST.save(st, p)
    back = ST.load(p).track(DF).submissions[s.key]
    assert back.nonproc == 57.0 and back.by_tier == s.by_tier
    raw = json.load(open(p))
    for sub in raw["tracks"][DF]["submissions"].values():
        del sub["nonproc"], sub["by_tier"]                               # a file of the previous rc6 build
    open(p, "w").write(json.dumps(raw))
    old = ST.load(p)
    assert old.track(DF).submissions[s.key].nonproc is None
    assert compute_keepalive(old, [DF], _snap([s]), {DF: 0.0}, now=NOW)[DF] == []  # waits for its next draw
    assert raw["version"] == ST.SPEC_STATE_VERSION == 1


# ------------------------------------------------------------------------------------------------ evaluator end to end
class TierServer:
    """A model answers correctly iff hash(repo, row id) < skill[repo][tier] (the fixture rule, per tier)."""

    def __init__(self, skills):
        self.skills = skills

    def __call__(self, model_dir, weight_bytes):
        repo = open(os.path.join(model_dir, "skill_key.txt")).read()
        skill = self.skills[repo]

        class B:
            async def decide(self, rows):
                out = []
                for r in rows:
                    h = int(hashlib.sha256(f"{repo}|{r['id']}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
                    labels = [str(x) for x in r["labels"]]
                    gold = str(r["expected"])
                    pick = gold if h < skill[r["tier"]] else next(lab for lab in labels if lab != gold)
                    out.append({lab: (0.85 if lab == pick else 0.15 / (len(labels) - 1)) for lab in labels})
                return out

        @asynccontextmanager
        async def ctx():
            yield B()

        return ctx()


ALL = ("proc_sealed", "same_family", "new_domain", "new_family")


@pytest.mark.usefixtures("rc5_entry_floors")
async def test_proc_only_model_clears_the_rc5_entry_floor_but_gets_no_keepalive(tmp_path):
    """The keep-alive gate on its own: on the rc5 mix (50 % proc) and the rc5 entry floor a model that only learned the
    procedural families clears the entry floor; the non-proc floor still keeps it out of the keep-alive."""
    env = Env(tmp_path, {})
    env.ev.backend_factory = TierServer({
        "a/proc": {"proc_sealed": 1.0, "same_family": 0.3, "new_domain": 0.3, "new_family": 0.3},
        "a/real": dict.fromkeys(ALL, 0.95)})
    env.add(1, "a/proc")
    env.add(2, "a/real")
    commits = {1: "a/proc", 2: "a/real"}
    await env.cycle(commits)
    proc, real = env.sub(1), env.sub(2)
    assert proc.status == real.status == ST.PENDING_CONFIRM               # both passed the first draw (vacant track)
    assert proc.score > CS.TRACK_ENTRY_FLOOR[TRACK] and proc.nonproc < CS.KEEPALIVE_NONPROC_FLOOR[TRACK] < real.nonproc
    first = next(h for h in proc.history if h["kind"] == "first")
    assert first["nonproc"] == proc.nonproc and set(first["by_tier"]) == set(ALL)
    assert first["by_tier"]["proc_sealed"] > first["by_tier"]["same_family"]
    ts = env.ev.state.track(TRACK)
    assert any("non-proc:" in e and proc.key[:12] in e for e in ts.events)
    s = snapshot(1000, [neuron(u, commit(TRACK, r)) for u, r in commits.items()])
    assert [u for u, _ in env.ev.keepalive(env.cfg.active_tracks(env.clock()), s)[TRACK]] == [2]
    # saved with the state, so the remote validator sees the same
    saved = ST.load(env.cfg.state_path).track(TRACK).submissions[proc.key]
    assert saved.nonproc == proc.nonproc and saved.by_tier == proc.by_tier


async def test_release_mix_proc_only_model_does_not_clear_the_entry_floor(tmp_path):
    """1.2.0rc6 (15 % proc; entry floor 28.8 on DF since rc7.post2): learning the procedural families alone does not clear the entry floor."""
    env = Env(tmp_path, {})
    env.ev.backend_factory = TierServer({
        "a/proc": {"proc_sealed": 1.0, "same_family": 0.3, "new_domain": 0.3, "new_family": 0.3},
        "a/real": dict.fromkeys(ALL, 0.95)})
    env.add(1, "a/proc")
    env.add(2, "a/real")
    commits = {1: "a/proc", 2: "a/real"}
    await env.cycle(commits)
    proc, real = env.sub(1), env.sub(2)
    assert proc.status == ST.NOT_BETTER and proc.score < CS.TRACK_ENTRY_FLOOR[TRACK]
    assert real.status == ST.PENDING_CONFIRM and real.nonproc > CS.KEEPALIVE_NONPROC_FLOOR[TRACK]
    s = snapshot(1000, [neuron(u, commit(TRACK, r)) for u, r in commits.items()])
    assert [u for u, _ in env.ev.keepalive(env.cfg.active_tracks(env.clock()), s)[TRACK]] == [2]


async def test_champion_nonproc_follows_its_latest_draw(tmp_path):
    env = Env(tmp_path, {})
    skills = {"a/m1": dict.fromkeys(ALL, 0.95), "a/m2": dict.fromkeys(ALL, 0.5)}
    env.ev.backend_factory = TierServer(skills)
    env.add(1, "a/m1")
    await env.cycle({1: "a/m1"})
    env.clock.advance(13 * 3600)
    await env.cycle({1: "a/m1"})
    champ = env.sub(1)
    assert champ.status == ST.CHAMPION and champ.history[-1]["kind"] == "confirm"
    assert champ.history[-1]["nonproc"] == champ.nonproc
    # served again as the incumbent on a later draw (its answers there are weaker on the non-proc tiers): the champion's
    # non-proc score is rewritten from that draw, and the keep-alive follows the latest one
    skills["a/m1"] = {"proc_sealed": 0.95, "same_family": 0.3, "new_domain": 0.3, "new_family": 0.3}
    env.add(2, "a/m2")
    env.clock.advance(3600)
    await env.cycle({1: "a/m1", 2: "a/m2"}, block=1100)
    assert env.sub(2).last_eval_ts == env.clock() and champ.nonproc < CS.KEEPALIVE_NONPROC_FLOOR[TRACK]
    s = snapshot(1100, [neuron(1, commit(TRACK, "a/m1")), neuron(2, commit(TRACK, "a/m2"))])
    assert env.ev.keepalive(env.cfg.active_tracks(env.clock()), s)[TRACK] == []


@pytest.mark.parametrize("track", [DF, RA])
def test_release_mix_takes_15_percent_proc_every_round(track):
    """Owner decision 16 Oct 2026 (1.2.0rc6): 15 % proc, 85 % LLM-written items, also once the small tiers are used."""
    assert S.TIER_MIX == {"proc_sealed": 0.15, "same_family": 0.425, "new_domain": 0.2125, "new_family": 0.2125}
    pool = tier_pool(POOLS[track])
    for c, _, _ in rounds(pool, 6):
        assert c["proc_sealed"] == 90
