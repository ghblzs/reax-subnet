"""1.2.0rc7: the first champion of a vacant track is decided per cohort (docs/SPECIALIST-TRACKS.md, "First champion of a
track"), and draws a pending entry still needs are never pruned."""

import pytest
from spec_helpers import HOTKEYS
from test_spec_evaluator import TRACK, Env

from reax_subnet.spec import evaluator as E
from reax_subnet.spec import state as ST

# the state machine on the fake models' score scale (conftest.py)
pytestmark = pytest.mark.usefixtures("rc5_entry_floors")

H = 3600


def _kinds(sub):
    return [h["kind"] for h in sub.history if h.get("kind") != "behaviour"]


# ------------------------------------------------------------------------------------------- the live shape
async def _live_shape(tmp_path):
    """Six entries seen in the same block; a budget of 2 an hour gives three first draws one hour apart (groups A, B, C,
    as on the live DF track), and the best entry is in the last group."""
    env = Env(tmp_path, {}, max_evals=2)
    skills = {1: 0.80, 2: 0.79, 3: 0.84, 4: 0.78, 5: 0.86, 6: 0.93}
    commits = {}
    for u, sk in skills.items():
        env.add(u, f"m/u{u}", sk)
        commits[u] = f"m/u{u}"
    for _ in range(3):
        await env.cycle(commits, block=1000)
        env.clock.advance(H)
    ts = env.ev.state.track(TRACK)
    groups = [{env.sub(u).draw1 for u in g} for g in ((1, 2), (3, 4), (5, 6))]
    assert all(len(g) == 1 for g in groups) and len(set().union(*groups)) == 3          # three different first draws
    assert {env.sub(u).first_seen_block for u in skills} == {1000}
    assert all(env.sub(u).status == ST.PENDING_CONFIRM for u in skills)
    return env, commits, ts


async def test_live_shape_crowns_the_playoff_winner_not_the_first_confirmer(tmp_path):
    env, commits, ts = await _live_shape(tmp_path)
    env.clock.t = env.sub(1).confirm_after_ts + 1                     # group A is due
    await env.cycle(commits, block=1100)
    # before 1.2.0rc7, uid 1 was crowned here (the first confirmer on a vacant track)
    assert not ts.champion and env.sub(1).confirm_draw and env.sub(1).status == ST.PENDING_CONFIRM
    assert "waits for its cohort" in " ".join(ts.events[-3:]) and "4 entries still open" in " ".join(ts.events[-3:])
    for _ in range(2):                                                # groups B and C
        env.clock.advance(H)
        await env.cycle(commits, block=1100)
        assert not ts.champion
    assert all(env.sub(u).confirm_draw and _kinds(env.sub(u)) == ["first", "confirm"] for u in commits)
    # the playoff: the budget (2) takes the two best confirmation scores, on one shared draw
    env.clock.advance(H)
    env.server.served.clear()
    out = await env.cycle(commits, block=1100)
    assert sorted(env.server.served) == ["m/u5", "m/u6"]
    assert ts.champion == env.sub(6).key and env.sub(6).status == ST.CHAMPION and out[TRACK][0][0] == 6
    po = [h for h in env.sub(6).history if h["kind"] == "playoff"]
    assert len(po) == 1 and po[0]["field"] == 2 and po[0]["draw"] in ts.draws
    assert env.sub(5).history[-1]["draw"] == po[0]["draw"]           # one shared draw
    # the others keep their challenger path: pending, against the champion with the full margin
    for u in (1, 2, 3, 4, 5):
        s = env.sub(u)
        assert s.status == ST.PENDING_CONFIRM and not s.confirm_draw and s.confirm_score is None
        assert "full margin" in s.reason
    env.clock.advance(H)
    await env.cycle(commits, block=1100)
    # group A against uid 6 (the budget of 2 takes the champion and uid 1): the full margin, not reached
    assert env.sub(1).status == ST.NOT_BETTER and env.sub(1).history[-1]["incumbent"] is not None
    assert ts.champion == env.sub(6).key


async def test_no_crowning_while_an_earlier_or_same_block_entry_is_queued(tmp_path):
    env = Env(tmp_path, {}, max_evals=1)
    env.add(1, "a/one", 0.85)
    env.add(2, "a/two", 0.85)
    commits = {1: "a/one", 2: "a/two"}
    await env.cycle(commits, block=1000)                             # budget 1: uid 2 stays queued
    assert env.sub(1).status == ST.PENDING_CONFIRM and env.sub(2).status == ST.QUEUED
    env.clock.advance(13 * H)                                        # uid 1 confirms first (uid 2 waits in the queue)
    await env.cycle(commits, block=1100)
    assert env.sub(1).confirm_draw and env.sub(2).status == ST.QUEUED
    assert not env.ev.state.track(TRACK).champion


# ---------------------------------------------------------------------------------------- a later cohort
async def test_a_later_cohort_does_not_hold_back_an_earlier_one_and_meets_the_champion(tmp_path):
    env = Env(tmp_path, {})
    env.add(1, "a/early", 0.82)
    env.add(2, "a/late", 0.84)
    env.add(3, "a/later", 0.83)
    await env.cycle({1: "a/early"}, block=1000)
    env.clock.advance(H)
    await env.cycle({1: "a/early", 2: "a/late"}, block=1005)         # uid 2: a later block, pending
    env.clock.advance(11.5 * H)
    commits = {1: "a/early", 2: "a/late", 3: "a/later"}              # uid 3: a later block, still queued at uid 1's turn
    env.ev.cfg = env.cfg
    env.hub.unavailable.add("a/later")
    await env.cycle(commits, block=1010)
    ts = env.ev.state.track(TRACK)
    assert ts.champion == env.sub(1).key                             # block 1000 alone: crowned at its confirmation
    assert env.sub(2).status == ST.PENDING_CONFIRM and env.sub(2).first_incumbent == ""
    env.clock.advance(H)
    await env.cycle(commits, block=1010)
    # uid 2's cohort is later: the normal path, the full margin over the champion crowned meanwhile
    assert env.sub(2).history[-1]["kind"] == "confirm" and env.sub(2).history[-1]["incumbent"] is not None
    assert env.sub(2).status in (ST.NOT_BETTER, ST.CHAMPION)
    assert not any(h["kind"] == "playoff" for u in (1, 2) for h in env.sub(u).history)


async def test_a_confirmed_entry_of_a_later_cohort_returns_to_the_challenger_path(tmp_path):
    env = Env(tmp_path, {}, max_evals=2)
    env.add(1, "a/early", 0.86)
    env.add(2, "a/late", 0.80)
    env.add(3, "a/same", 0.84)
    env.hub.unavailable.add("a/same")                                # uid 3 (block 1000) uploads its repo late
    await env.cycle({1: "a/early", 3: "a/same"}, block=1000)
    env.clock.advance(H)
    commits = {1: "a/early", 2: "a/late", 3: "a/same"}
    await env.cycle(commits, block=1005)                             # uid 2: block 1005, first pass
    env.hub.unavailable.clear()
    env.clock.advance(H)
    await env.cycle(commits, block=1005)                             # uid 3: first pass two hours after uid 1
    assert {env.sub(u).status for u in (1, 2, 3)} == {ST.PENDING_CONFIRM}
    assert env.sub(3).first_seen_block == 1000 and env.sub(2).first_seen_block == 1005
    ts = env.ev.state.track(TRACK)
    for u in (1, 2, 3):                                              # confirmed one after the other
        env.clock.t = env.sub(u).confirm_after_ts + 1
        await env.cycle(commits, block=1100)
        assert env.sub(u).confirm_draw and not ts.champion
    # the block-1000 cohort (uids 1, 3) is complete: its playoff; uid 2 is not in it and returns to the challenger path
    env.clock.advance(H)
    env.server.served.clear()
    await env.cycle(commits, block=1100)
    assert sorted(env.server.served) == ["a/early", "a/same"]
    assert ts.champion in (env.sub(1).key, env.sub(3).key)
    assert env.sub(2).status == ST.PENDING_CONFIRM and not env.sub(2).confirm_draw and "full margin" in env.sub(2).reason
    for _ in range(2):                  # budget 2 = champion + one group an hour: uid 3's group, then uid 2's
        env.clock.advance(H)
        await env.cycle(commits, block=1100)
    assert env.sub(2).history[-1]["kind"] == "confirm" and env.sub(2).history[-1]["incumbent"] is not None


# ------------------------------------------------------------------------------------- who holds a cohort back
async def _one_pending_and(env, other_status, reason, base_model="", wait_kind=""):
    env.add(1, "a/one", 0.85)
    await env.cycle({1: "a/one"}, block=1000)
    ts = env.ev.state.track(TRACK)
    o = ST.Submission(key=f"{HOTKEYS[2]}|x", uid=2, hotkey=HOTKEYS[2], coldkey="c2", track_id=TRACK, repo="a/two",
                      revision="a" * 40, first_seen_block=1000, first_seen_ts=env.clock.t, status=other_status, reason=reason,
                      base_model=base_model, wait_kind=wait_kind)
    ts.submissions[o.key] = o
    return ts, o


async def test_an_infra_deferred_member_holds_back_the_crowning(tmp_path, monkeypatch):
    env = Env(tmp_path, {})
    why = E._no_base(ST.Submission("k", 2, "h", "c", TRACK, "r", "v", 0, 0.0, base_model="Org/Big@" + "c" * 40),
                     "deferred: not enough free disk space for the base on this host")
    ts, o = await _one_pending_and(env, ST.PENDING_VERIFY, why, base_model="Org/Big@" + "c" * 40, wait_kind=ST.WAIT_INFRA)
    monkeypatch.setattr(E.SpecEvaluator, "_sync", lambda self, tdef, snap, now: None)   # keep the hand-made entry
    env.clock.advance(13 * H)
    await env.cycle({1: "a/one"}, block=1100)
    assert env.sub(1).confirm_draw and not ts.champion and o.status == ST.PENDING_VERIFY
    # the base download budget of a cycle is also this host's
    o.reason = E._no_base(o, "deferred to a later cycle (base download budget of this cycle used)")
    env.clock.advance(H)
    await env.cycle({1: "a/one"}, block=1100)
    assert not ts.champion
    # the base is refused for a reason on the miner's side: the cohort is complete
    o.reason = E._no_base(o, "the base repo is gated or private: an operator must place it under REAX_SPEC_BASE_DIR")
    o.wait_kind = ST.WAIT_MINER
    env.clock.advance(H)
    await env.cycle({1: "a/one"}, block=1100)
    assert ts.champion == env.sub(1).key


async def test_a_miner_side_wait_does_not_hold_back_the_crowning(tmp_path):
    env = Env(tmp_path, {})
    env.add(1, "a/one", 0.85)
    env.add(2, "a/private", 0.95)
    env.hub.unavailable.add("a/private")                             # a private (or not yet uploaded) repo
    commits = {1: "a/one", 2: "a/private"}
    await env.cycle(commits, block=1000)
    assert env.sub(2).status == ST.QUEUED and env.sub(2).reason.startswith("waiting for the repo")
    assert env.sub(1).status == ST.PENDING_CONFIRM
    env.clock.advance(13 * H)
    await env.cycle(commits, block=1100)
    assert env.ev.state.track(TRACK).champion == env.sub(1).key       # crowned at its confirmation, as before


async def test_a_never_tried_queued_entry_holds_back_and_a_decided_one_does_not(tmp_path, monkeypatch):
    env = Env(tmp_path, {})
    ts, o = await _one_pending_and(env, ST.QUEUED, "")
    monkeypatch.setattr(E.SpecEvaluator, "_sync", lambda self, tdef, snap, now: None)
    monkeypatch.setattr(E.SpecEvaluator, "_prepare", _prepare_skip(E.SpecEvaluator._prepare, o.key))
    env.clock.advance(13 * H)
    await env.cycle({1: "a/one"}, block=1100)
    assert env.sub(1).confirm_draw and not ts.champion
    o.status, o.reason = ST.NOT_BETTER, "first draw: gain -3.00, margin 24.80"
    env.clock.advance(H)
    await env.cycle({1: "a/one"}, block=1100)
    assert ts.champion == env.sub(1).key


def _prepare_skip(orig, key):
    async def prep(self, tdef, sub, snap, now):
        if sub.key == key:
            return None             # never prepared in this test: it stays queued
        return await orig(self, tdef, sub, snap, now)
    return prep


# ----------------------------------------------------------------------------------------------- the playoff
async def _two_ready(tmp_path, s1, s2):
    env = Env(tmp_path, {})
    env.add(1, "a/one", s1)
    env.add(2, "a/two", s2)
    commits = {1: "a/one", 2: "a/two"}
    await env.cycle(commits, block=1000)
    env.clock.advance(13 * H)
    await env.cycle(commits, block=1100)
    assert env.sub(1).confirm_draw and env.sub(2).confirm_draw
    return env, commits


async def test_playoff_below_the_floor_crowns_nobody(tmp_path, monkeypatch):
    env, commits = await _two_ready(tmp_path, 0.85, 0.86)
    from reax_subnet import competition_split as cs
    monkeypatch.setitem(cs.TRACK_ENTRY_FLOOR, TRACK, 99.0)          # the playoff draw clears no floor
    env.clock.advance(H)
    await env.cycle(commits, block=1100)
    ts = env.ev.state.track(TRACK)
    assert not ts.champion
    assert env.sub(1).status == env.sub(2).status == ST.NOT_BETTER and "playoff" in env.sub(1).reason


async def test_playoff_waits_while_a_member_cannot_be_fetched(tmp_path):
    env, commits = await _two_ready(tmp_path, 0.85, 0.90)
    env.hub.unavailable.add("a/two")                                 # transient: a re-fetch fails once
    env.clock.advance(H)
    env.server.served.clear()
    await env.cycle(commits, block=1100)
    ts = env.ev.state.track(TRACK)
    assert not ts.champion and env.server.served == [] and env.sub(2).attempts == 1
    env.hub.unavailable.clear()
    env.clock.advance(H)
    await env.cycle(commits, block=1100)
    assert ts.champion == env.sub(2).key


async def test_a_track_with_a_champion_releases_waiting_entries(tmp_path):
    env, commits = await _two_ready(tmp_path, 0.85, 0.86)
    ts = env.ev.state.track(TRACK)
    # e.g. a state from before a rollback-and-forward: the track got a champion while both waited for the playoff
    env.sub(1).status, ts.champion = ST.CHAMPION, env.sub(1).key
    env.sub(1).confirm_draw = ""
    assert E.SpecEvaluator._cohort_step(env.ev, ts, env.clock.t) == []
    assert not env.sub(2).confirm_draw and env.sub(2).status == ST.PENDING_CONFIRM


# ------------------------------------------------------------------------------------------- draw retention
def _draw(i):
    return ST.Draw(f"d{i:03d}", float(i), i, i, [f"row-{i}"])


def test_retention_keeps_draws_pending_entries_need_and_the_40_newest_others():
    ts = ST.TrackState(TRACK)
    ts.draws = {d.draw_id: d for d in map(_draw, range(60))}
    pend = ST.Submission("p", 1, "h", "c", TRACK, "r", "v", 0, 0.0, status=ST.PENDING_CONFIRM, draw1="d000",
                         confirm_draw="d005")
    done = ST.Submission("n", 2, "h2", "c", TRACK, "r", "v", 0, 0.0, status=ST.NOT_BETTER, draw1="d001")
    ts.submissions = {"p": pend, "n": done}
    E.SpecEvaluator._prune_draws(ts)
    assert "d000" in ts.draws and "d005" in ts.draws and "d001" not in ts.draws
    others = sorted(k for k in ts.draws if k not in ("d000", "d005"))
    assert len(others) == E.MAX_UNREFERENCED_DRAWS and others[0] == "d020" and others[-1] == "d059"


async def test_a_pending_entry_keeps_its_first_draw_through_many_cycles(tmp_path, monkeypatch):
    # 46 draws of 300 from the 1500-item test pool would retire it (ITEM_MAX_USES); this test is about draw retention
    monkeypatch.setattr("reax_subnet.spec.sampling.ITEM_MAX_USES", 100)
    env = Env(tmp_path, {})
    env.add(1, "a/one", 0.85)
    await env.cycle({1: "a/one"}, block=1000)
    ts = env.ev.state.track(TRACK)
    d1 = env.sub(1).draw1
    for i in range(45):                                              # 45 other first passes before uid 1 is due
        u = 2 + i % 15
        env.add(u, f"a/x{i}", 0.30)
        ts.submissions.pop(next((k for k, s in ts.submissions.items() if s.uid == u), ""), None)
        await env.cycle({1: "a/one", u: f"a/x{i}"}, block=1001 + i)
        env.clock.advance(360)                                       # within the hourly budget
    assert len(ts.draws) >= 41 and d1 in ts.draws
    env.clock.t = env.sub(1).confirm_after_ts + 1
    await env.cycle({1: "a/one"}, block=2000)
    assert ts.champion == env.sub(1).key                             # confirmed on a draw disjoint from its first
    d2 = env.sub(1).history[-1]["draw"]
    assert not set(ts.draws[d2].item_ids) & set(ts.draws[d1].item_ids)


# ------------------------------------------------------------------------------------------ rollback safety
def test_state_with_and_without_the_playoff_fields_loads(tmp_path):
    st = ST.SpecState()
    ts = st.track(TRACK)
    ts.submissions["k"] = ST.Submission("k", 1, "h", "c", TRACK, "r", "v", 5, 0.0, status=ST.PENDING_CONFIRM, draw1="d",
                                        confirm_draw="d2", confirm_score=61.5)
    path = str(tmp_path / "s.json")
    ST.save(st, path)
    back = ST.load(path).track(TRACK).submissions["k"]
    assert back.confirm_draw == "d2" and back.confirm_score == 61.5
    raw = ST._to_json(st)                                            # a file written before 1.2.0rc7
    for f in ("confirm_draw", "confirm_score"):
        del raw["tracks"][TRACK]["submissions"]["k"][f]
    old = ST._from_json(raw).track(TRACK).submissions["k"]
    assert old.confirm_draw == "" and old.confirm_score is None and old.status == ST.PENDING_CONFIRM


def _answers(rows, skill, key):
    """FakeServer's rule: correct iff hash(key, row id) < skill."""
    import hashlib
    out = []
    for r in rows:
        h = int(hashlib.sha256(f"{key}|{r['id']}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        labels = [str(x) for x in r["labels"]]
        gold = str(r["expected"])
        pick = gold if h < skill else next(lab for lab in labels if lab != gold)
        out.append({lab: (0.85 if lab == pick else 0.15 / (len(labels) - 1)) for lab in labels})
    return out


def test_playoff_twin_must_beat_its_original_by_the_full_margin_and_ties_go_to_the_earlier(tmp_path):
    from spec_helpers import synthetic_pool

    from reax_subnet.spec import scoring as SC
    env = Env(tmp_path, {})
    ts = env.ev.state.track(TRACK)
    rows = synthetic_pool(300)
    subs = [ST.Submission(f"k{u}", u, HOTKEYS[u], f"c{u}", TRACK, "r", "v", 1000, 0.0, status=ST.PENDING_CONFIRM,
                          draw1="d", confirm_draw="d2", confirm_score=60.0) for u in (1, 2)]
    ts.submissions = {s.key: s for s in subs}
    a = SC.score_run(rows, _answers(rows, 0.85, "orig"))[0]
    b = SC.score_run(rows, _answers(rows, 0.853, "orig"))[0]         # the same answers, a hair better
    tdef = env.cfg.tracks[0]
    E.SpecEvaluator._decide_playoff(env.ev, ts, tdef, subs, {"k1": a, "k2": b}, "p" * 16, 7, env.clock.t, {"k2": ["k1"]})
    assert subs[1].score > subs[0].score and ts.champion == "k1"          # the twin does not beat uid 1 by the margin
    assert subs[1].status == ST.PENDING_CONFIRM and "full margin" in subs[1].reason
    # equal scores: the earlier commitment (lower uid in the same block)
    ts.champion = ""
    for s in subs:
        s.status, s.confirm_draw, s.champion_since_ts = ST.PENDING_CONFIRM, "d2", None
    E.SpecEvaluator._decide_playoff(env.ev, ts, tdef, subs[::-1], {"k1": a, "k2": a}, "q" * 16, 7, env.clock.t, {})
    assert ts.champion == "k1"


async def test_an_unfetchable_entry_at_the_head_takes_no_slot(tmp_path):
    """1.2.0rc8.post1 (#20923): queued entries whose repo cannot be fetched (401/404) are skipped in the group; the entry
    behind them is evaluated in the same cycle, and the unfetchable ones are fetched again on the next cycle. Before, every
    candidate took a slot: two unfetchable entries at the head of a two-slot cycle evaluated nobody."""
    env = Env(tmp_path, {}, max_evals=2)
    env.add(1, "a/champ", 0.80)
    await env.cycle({1: "a/champ"}, block=1000)
    env.clock.advance(13 * H)
    await env.cycle({1: "a/champ"}, block=1100)
    assert env.sub(1).status == ST.CHAMPION
    env.add(2, "a/private", 0.95)
    env.hub.unavailable.add("a/private")
    env.add(4, "a/gone", 0.95)
    env.hub.unavailable.add("a/gone")
    env.add(3, "a/next", 0.95)
    commits = {1: "a/champ", 2: "a/private", 4: "a/gone", 3: "a/next"}
    env.clock.advance(2 * H)
    env.server.served.clear()
    await env.cycle(commits, block=1200)                     # budget 2 = champion + one challenger
    assert env.sub(2).status == env.sub(4).status == ST.QUEUED and env.sub(2).reason.startswith("waiting for the repo")
    assert env.sub(3).status == ST.PENDING_CONFIRM and "a/next" in env.server.served
    env.hub.fetches.clear()
    env.clock.advance(2 * H)
    await env.cycle(commits, block=1300)
    assert "a/private" in env.hub.fetches and "a/gone" in env.hub.fetches    # re-checked on the next cycle


async def test_two_tracks_share_the_cycle_budget(tmp_path):
    """1.2.0rc8.post1 (#20923): with the same cycle timestamp on both tracks, the first track of the schedule took the
    whole budget and the second got a turn every second cycle only. Now each track with work gets its share."""
    from spec_helpers import commit, neuron, snapshot, synthetic_pool, write_pool

    from reax_subnet.spec import config as C
    RA = "risk_assessment"
    env = Env(tmp_path, {}, max_evals=4)
    env.cfg.tracks = C.DEFAULT_TRACKS[:2]
    write_pool(str(tmp_path / "sealed"), RA, synthetic_pool(1500))
    env.ev = env.make()
    repos = {1: (TRACK, "d/champ", 0.80), 2: (RA, "r/champ", 0.80)}
    for uid, (t, repo, skill) in repos.items():
        env.add(uid, repo, skill, category=t)

    async def cycle(block):
        return await env.ev.evaluate_cycle(snapshot(block, [neuron(u, commit(t, r)) for u, (t, r, _) in repos.items()]))

    await cycle(1000)
    env.clock.advance(13 * H)
    await cycle(1100)
    tsd, tsr = env.ev.state.track(TRACK), env.ev.state.track(RA)
    assert tsd.champion and tsr.champion
    for uid, t in ((3, TRACK), (4, TRACK), (5, TRACK), (6, RA), (7, RA)):
        repos[uid] = (t, f"q/{uid}", 0.95)
        env.add(uid, f"q/{uid}", 0.95, category=t)
    env.clock.advance(2 * H)
    await cycle(1200)                                          # budget 4: two for each track, not four for the first
    first = {s.uid for ts in (tsd, tsr) for s in ts.submissions.values() if s.status == ST.PENDING_CONFIRM}
    assert first & {3, 4, 5} and first & {6, 7}, first


async def test_a_hub_outage_rejects_at_most_the_skip_bound_per_cycle(tmp_path):
    """Review F1 (rc8.post1): every fetch fails during a hub outage; entries past their publish deadline are rejected by
    a failed fetch, so the loop that passes over unusable entries must stop after a few, not reject the whole queue."""
    from reax_subnet.spec import config as C
    env = Env(tmp_path, {}, max_evals=2)
    env.add(1, "a/champ", 0.80)
    await env.cycle({1: "a/champ"}, block=1000)
    env.clock.advance(13 * H)
    await env.cycle({1: "a/champ"}, block=1100)
    commits = {1: "a/champ"}
    for uid in range(2, 12):
        env.add(uid, f"q/{uid}", 0.95)
        env.hub.unavailable.add(f"q/{uid}")
        commits[uid] = f"q/{uid}"
    env.clock.advance(H)
    await env.cycle(commits, block=1200)                     # first seen at block 1200; nothing usable
    env.hub.fetches.clear()
    env.clock.advance(H)
    await env.cycle(commits, block=1200 + C.PUBLISH_DEADLINE_BLOCKS + 1)
    rejected = [u for u in range(2, 12) if env.sub(u).status == ST.REJECTED]
    assert len([f for f in env.hub.fetches if f.startswith("q/")]) <= E.MAX_SKIPS_PER_CYCLE
    assert len(rejected) <= E.MAX_SKIPS_PER_CYCLE


def test_has_work_matches_what_run_track_would_evaluate(tmp_path):
    """Review F2: a due confirmation whose first draw is gone, or an entry waiting for its repo, claims no share."""
    env = Env(tmp_path, {})
    tdef = env.cfg.tracks[0]
    ts = env.ev.state.track(TRACK)
    s = ST.Submission(key="k", uid=2, hotkey=HOTKEYS[2], coldkey="c", track_id=TRACK, repo="r/x", revision="a" * 40,
                      first_seen_block=1, first_seen_ts=env.clock.t, status=ST.PENDING_CONFIRM)
    s.confirm_after_ts, s.draw1 = env.clock.t - 1, "lost"
    ts.submissions[s.key] = s
    assert not env.ev._has_work(tdef, env.clock.t)
    s.status, s.reason = ST.QUEUED, "waiting for the repo: 401"
    assert not env.ev._has_work(tdef, env.clock.t)
    s.reason = ""
    assert env.ev._has_work(tdef, env.clock.t)



async def test_many_unfetchable_entries_at_the_head_never_hold_the_queue(tmp_path):
    """Review R1 (rc8.post1): more unfetchable entries than the skip bound at the head; the entry behind them is still
    evaluated (waiting entries go last once they have been tried)."""
    env = Env(tmp_path, {}, max_evals=2)
    env.add(1, "a/champ", 0.80)
    await env.cycle({1: "a/champ"}, block=1000)
    env.clock.advance(13 * H)
    await env.cycle({1: "a/champ"}, block=1100)
    commits = {1: "a/champ"}
    for uid in range(2, 9):
        env.add(uid, f"p/{uid}", 0.95)
        env.hub.unavailable.add(f"p/{uid}")
        commits[uid] = f"p/{uid}"
    env.add(9, "a/next", 0.95)
    commits[9] = "a/next"
    for _ in range(3):
        env.clock.advance(H)
        await env.cycle(commits, block=1200)
    assert env.sub(9).status == ST.PENDING_CONFIRM
