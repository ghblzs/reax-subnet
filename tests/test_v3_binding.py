"""Eval mechanism v3, stage 3 (1.2.0rc9.post3): X tier BINDING mode and the capped generalisation penalty."""

import hashlib
import json
import logging
import math
from contextlib import asynccontextmanager

import pytest
from spec_helpers import FakeServer, commit, neuron, snapshot, synthetic_pool, write_pool
from test_spec_evaluator import TRACK, Env, crown
from test_v3_xtier import ids, with_x

from reax_subnet.spec import config as C
from reax_subnet.spec import evaluator as E
from reax_subnet.spec import sampling as S
from reax_subnet.spec import scoring as SC
from reax_subnet.spec import state as ST

pytestmark = pytest.mark.usefixtures("rc5_entry_floors")
H = 3600
INF = float("inf")


# ------------------------------------------------------------------------------------------------ flags and config
@pytest.mark.parametrize("value,expect", [("", False), ("0", False), ("false", False), ("off", False), ("no", False),
                                          ("1", True), ("true", True), ("TRUE", True), ("Yes", True), (" on ", True)])
def test_v3_decisions_flag_strict_values(tmp_path, value, expect):
    assert C.from_env({"REAX_SPEC_SEALED_DIR": str(tmp_path), "REAX_SPEC_V3_DECISIONS": value}).v3_decisions is expect
    assert C.from_env({"REAX_SPEC_SEALED_DIR": str(tmp_path)}).v3_decisions is False     # unset: the release default (off)
    with pytest.raises(ValueError):                                                      # unknown value: refuses to start
        C.from_env({"REAX_SPEC_SEALED_DIR": str(tmp_path), "REAX_SPEC_V3_DECISIONS": "junk"})


@pytest.mark.parametrize("value,expect", [("", "off"), ("off", "off"), ("binding", "binding"), (" Binding ", "binding"),
                                          ("shadow", "shadow"), ("scored", "shadow"), ("bogus", "off")])
def test_x_mode_parsing(tmp_path, value, expect):
    assert C.from_env({"REAX_SPEC_SEALED_DIR": str(tmp_path), "REAX_SPEC_X_TIER_MODE": value}).x_tier_mode == expect
    assert "binding" in S.X_TIER_MODES and C.from_env({"REAX_SPEC_SEALED_DIR": str(tmp_path)}).x_tier_mode == "off"


def test_g_med_json_parsing(tmp_path, caplog):
    env = lambda v: C.from_env({"REAX_SPEC_SEALED_DIR": str(tmp_path), "REAX_SPEC_G_MED": v}).g_med   # noqa: E731
    assert env('{"demand_forecasting": 12.3, "risk_assessment": 9}') == {"demand_forecasting": 12.3, "risk_assessment": 9.0}
    assert env("") == {} and env("  ") == {}
    with caplog.at_level(logging.ERROR, logger="reax.spec.config"):
        assert env("{not json") == {} and env("[1, 2]") == {} and env('"x"') == {}
        assert env('{"a": "high", "b": null, "c": true, "d": 3.5, "e": Infinity}') == {"d": 3.5}
    assert sum("REAX_SPEC_G_MED" in r.getMessage() for r in caplog.records) >= 6
    assert C.from_env({"REAX_SPEC_SEALED_DIR": str(tmp_path)}).g_med == {}


@pytest.mark.parametrize("value,expect", [("", 10.0), ("10", 10.0), ("25.5", 25.5), ("0", 0.0), ("none", INF), ("INF", INF), (" None ", INF),
                                          ("-3", 10.0), ("abc", 10.0), ("nan", 10.0)])
def test_penalty_cap_parsing(tmp_path, value, expect):
    assert C.from_env({"REAX_SPEC_SEALED_DIR": str(tmp_path), "REAX_SPEC_GAP_PENALTY_CAP": value}).gap_penalty_cap == expect


def test_defaults_are_off_and_constants():
    cfg = C.SpecConfig("s", "w", "p")
    assert cfg.x_tier_mode == "off" and cfg.g_med == {} and cfg.gap_penalty_cap == 10.0
    assert SC.GAP_WEIGHT == 0.5 and SC.GAP_ALLOWANCE_FLOOR == 8.0 and SC.GAP_PENALTY_CAP == 10.0
    assert S.X_TIERS == {"cross_author", "award_realdata"}


# ------------------------------------------------------------------------------------------------ draw composition
def n_x(rows):
    return sum(1 for r in rows if r["tier"] in S.X_TIERS)


def test_binding_draw_has_exactly_the_share_inside_the_normal_size():
    pool = with_x()
    rows, _ = S.draw(pool, n=300, seed=5, x_mode="binding")
    assert len(rows) == 300 and n_x(rows) == round(0.2 * 300) == 60
    assert ids(rows) == sorted(ids(rows)) and len({r["question"]["type"] for r in rows if r["tier"] in S.X_TIERS}) == 3
    off = S.draw(pool, n=300, seed=5)[0]
    count = lambda rs, t: sum(1 for r in rs if r["tier"] == t)   # noqa: E731
    assert count(rows, "proc_sealed") == count(off, "proc_sealed")
    for share in (0.1, 0.25):
        rows, _ = S.draw(pool, n=300, seed=5, x_mode="binding", x_share=share)
        assert len(rows) == 300 and n_x(rows) == round(300 * share)
    assert ids(S.draw(pool, n=300, seed=5, x_mode="binding")[0]) == ids(S.draw(pool, n=300, seed=5, x_mode="binding")[0])
    assert ids(S.draw(pool, n=300, seed=5, x_mode="binding")[0]) == ids(S.draw(pool, n=300, seed=5, x_mode="scored")[0])


def test_shortage_takes_what_exists_and_fills_from_the_normal_tiers():
    pool = synthetic_pool(1500) + with_x(0, 25)
    rows, _ = S.draw(pool, n=300, seed=3, x_mode="binding")
    assert n_x(rows) == 25 and len(rows) == 300
    assert ids(S.draw(synthetic_pool(1500), n=300, seed=3, x_mode="binding")[0]) == ids(S.draw(synthetic_pool(1500), n=300, seed=3)[0])


def test_disjointness_and_the_reuse_cap_include_the_x_items():
    pool = with_x()
    first = S.draw(pool, n=300, seed=5, x_mode="binding")[0]
    used = {r["id"]: 1 for r in first}
    second = S.draw(pool, n=300, seed=6, x_mode="binding", use_counts=used, exclude=set(ids(first)))[0]
    assert not set(ids(first)) & set(ids(second)) and n_x(second) == 60 and len(second) == 300
    last = {r["id"]: S.ITEM_MAX_USES - 1 for r in pool if r["tier"] in S.X_TIERS}    # one use left: still drawn
    assert n_x(S.draw(pool, n=300, seed=5, x_mode="binding", use_counts=last)[0]) == 60
    retired = {r["id"]: S.ITEM_MAX_USES for r in pool if r["tier"] in S.X_TIERS}
    rows = S.draw(pool, n=300, seed=5, x_mode="binding", use_counts=retired)[0]
    assert n_x(rows) == 0 and len(rows) == 300                                       # all retired: the normal tiers carry the draw
    fresh_first = {r["id"]: 5 for r in pool if r["tier"] in S.X_TIERS and r["id"] < "x-0400"}
    rows = S.draw(pool, n=300, seed=5, x_mode="binding", use_counts=fresh_first)[0]
    assert all(r["id"] >= "x-0400" for r in rows if r["tier"] in S.X_TIERS)           # least used first


def test_award_realdata_tier_counts_as_x_and_x_runway_is_counted_apart():
    pool = with_x(1500, 300)
    for r in pool:
        if r["tier"] == S.X_TIER and r["id"] < "x-0100":
            r["tier"] = "award_realdata"
    assert n_x(S.draw(pool, n=300, seed=5, x_mode="binding")[0]) == 60
    assert not any(r["tier"] in S.X_TIERS for r in S.draw(pool, n=300, seed=5)[0])
    assert S.x_pool_gen(pool) == S.x_pool_gen(with_x(1500, 300))
    assert S.nonproc_draws_left(pool, {}, 300) == S.nonproc_draws_left(synthetic_pool(1500), {}, 300)
    assert S.x_draws_left(pool, {}, 300, 0.2) == 300 * S.ITEM_MAX_USES / 60
    assert S.x_draws_left(pool, {r["id"]: 20 for r in pool}, 300, 0.2) == 300 * 4 / 60
    assert S.x_draws_left(synthetic_pool(100), {}, 300, 0.2) == 0.0
    items = [SC.ItemScore(f"i{i}", "choice", t, True, True, .25, 0., 0., .75) for i, t in enumerate(["same_family", "award_realdata", "cross_author"])]
    gen, x = SC.split_x(items)
    assert len(gen) == 1 and len(x) == 2 and SC.split_x(items, "cross_author")[1][0].item_id == "i2"


def test_score_item_handles_every_question_type_of_the_real_data_items():
    rows = [("noul", ["no", "yes"], "yes"), ("noul", ["no", "yes"], "no"), ("choice", list("abc"), "b"),
            ("choice", list("abcde"), "e"), ("score", ["0", "1", "2", "3"], 2), ("score", list("01234"), 4)]
    for kind, labels, gold in rows:
        row = {"id": "r", "tier": "award_realdata", "question": {"type": kind}, "labels": labels, "expected": gold}
        perfect = {lab: (1.0 if lab == str(gold) else 0.0) for lab in labels}
        it = SC.score_item(row, perfect)
        assert it.valid and it.correct and it.tier == "award_realdata" and it.brier == 0.0
        assert SC.score_item(row, None).valid is False


# ------------------------------------------------------------------------------------------------ penalty math
@pytest.mark.parametrize("gaps,ses,g_med,cap,expect,allowance", [
    ([40.0], [4.0], 10.0, 10.0, 10.0, 8.0),                  # 0.5 x (30 - 8) = 11 -> capped at 10
    ([30.0], [4.0], 10.0, 10.0, 6.0, 8.0),                   # 0.5 x (20 - 8) = 6, below the cap
    ([60.0], [4.0], 10.0, 10.0, 10.0, 8.0),                  # 0.5 x 42 = 21 -> capped at 10
    ([60.0], [4.0], 10.0, INF, 21.0, 8.0),                   # cap lifted
    ([60.0], [4.0], 10.0, None, 21.0, 8.0),
    ([40.0], [10.0], 10.0, 10.0, 5.0, 20.0),                 # A = 2 x SE = 20: 0.5 x (30 - 20)
    ([25.0], [10.0], 10.0, 10.0, 0.0, 20.0),                 # inside the allowance
    ([18.0], [1.0], 10.0, 10.0, 0.0, 8.0),                   # excess exactly the floor allowance
    ([18.4], [1.0], 10.0, 10.0, 0.2, 8.0),
    ([5.0], [4.0], 10.0, 10.0, 0.0, 8.0),                    # below the field median: never a bonus
    ([60.0], [4.0], 10.0, 0.0, 0.0, 8.0),                    # cap 0: no penalty
])
def test_penalty_table(gaps, ses, g_med, cap, expect, allowance):
    pen, d = SC.gap_penalty(gaps, ses, g_med, cap=cap)
    assert pen == pytest.approx(expect) and d["penalty"] == pen and d["allowance"] == pytest.approx(allowance)


def test_allowance_comes_from_one_draw_not_pooled():
    # three draws, each SE 10: rms = 10 (NOT 10 / sqrt(3) = 5.8): A = 20 whatever the number of draws
    for k in (1, 2, 3, 9):
        pen, d = SC.gap_penalty([40.0] * k, [10.0] * k, 10.0, cap=None)
        assert d["se_gap"] == pytest.approx(10.0) and d["allowance"] == pytest.approx(20.0) and pen == pytest.approx(5.0)
    pen, d = SC.gap_penalty([40.0, 40.0], [6.0, 8.0], 10.0)              # rms(6, 8) = 7.07
    assert d["se_gap"] == pytest.approx(math.sqrt((36 + 64) / 2)) and d["allowance"] == pytest.approx(14.142, abs=1e-3)
    assert d["gap"] == 40.0 and pen == pytest.approx(0.5 * (30 - 14.142), abs=1e-3)


def test_no_x_draws_or_no_g_med_means_no_penalty():
    pen, d = SC.gap_penalty([], [], 10.0, cap=10.0)
    assert pen == 0.0 and d["draws"] == 0
    pen, d = SC.gap_penalty([60.0], [4.0], None, cap=10.0)
    assert pen == 0.0 and d["g_med"] is None and d["gap"] == 60.0


def test_sandbagging_generator_items_never_raises_s_eff():
    """S = 0.8 s_gen + 0.2 s_x on a 300-item draw with 60 X items; S_eff = S - P. One point less on s_gen never helps."""
    def s_eff(s_gen, s_x, g_med=10.0, se=4.0, cap=10.0):
        s = 0.8 * s_gen + 0.2 * s_x
        return s - SC.gap_penalty([s_gen - s_x], [se], g_med, cap=cap)[0]
    for cap in (10.0, INF, 3.0):
        for s_x in (5.0, 20.0, 35.0, 50.0):
            for s_gen in range(0, 101, 1):
                a, b = s_eff(float(s_gen), s_x, cap=cap), s_eff(s_gen - 1.0, s_x, cap=cap)
                assert b <= a + 1e-9, (cap, s_x, s_gen)
    below_cap = s_eff(70.0, 20.0, cap=INF) - s_eff(69.0, 20.0, cap=INF)           # inside the penalised zone, below the cap
    assert below_cap == pytest.approx(0.8 - 0.5)
    at_cap = s_eff(95.0, 5.0, cap=10.0) - s_eff(94.0, 5.0, cap=10.0)              # the cap binds: the full 0.8
    assert at_cap == pytest.approx(0.8)
    assert s_eff(70.0, 20.0, cap=INF) - s_eff(70.0, 21.0, cap=INF) == pytest.approx(-0.2 - 0.5)   # a private point is worth 0.7


# ------------------------------------------------------------------------------------------------ evaluator scenarios
class GapServer(FakeServer):
    """skills[repo] = (skill on the generator items, skill on the cross-author items)."""

    def __call__(self, model_dir, weight_bytes):
        repo = open(model_dir + "/skill_key.txt").read()
        self.served.append(repo)
        sk = self.skills[repo]

        class B:
            async def decide(self, rows):
                out = []
                for r in rows:
                    h = int(hashlib.sha256(f"{repo}|{r['id']}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
                    labels = [str(x) for x in r["labels"]]
                    gold = str(r["expected"])
                    wrong = next(lab for lab in labels if lab != gold)
                    pick = gold if h < sk[1 if r.get("tier") in S.X_TIERS else 0] else wrong
                    out.append({lab: (0.85 if lab == pick else 0.15 / (len(labels) - 1)) for lab in labels})
                return out

        @asynccontextmanager
        async def ctx():
            yield B()

        return ctx()


def mk(tmp_path, skills, mode="binding", g_med=5.0, cap=INF, v3=False, pool=None):
    env = Env(tmp_path, {})
    write_pool(str(tmp_path / "sealed"), TRACK, pool or with_x())
    env.server = GapServer(skills)
    env.ev.backend_factory = env.server
    env.ev.cfg.x_tier_mode = mode
    env.ev.cfg.g_med = {} if g_med is None else {TRACK: g_med}
    env.ev.cfg.gap_penalty_cap = cap
    env.ev.cfg.v3_decisions = v3
    return env


CLEAN = (0.70, 0.70)
GAPPY = (0.97, 0.30)     # wins on raw S against CLEAN, with a gap of about 70 points
MILD = (0.90, 0.45)      # a gap of about 50: a penalty of about 15, still above the entry floor
BOTH = {"a/champ": CLEAN, "a/gappy": GAPPY}


async def run_race(env, challenger="a/gappy", champion="a/champ"):
    await crown(env, 1, champion, None)
    env.add(2, challenger, None)
    await env.cycle({1: champion, 2: challenger})
    env.clock.advance(13 * H)
    await env.cycle({1: champion, 2: challenger})
    return env.sub(1), env.sub(2), env.ev.state.track(TRACK)


@pytest.mark.parametrize("v3", [False, True])
async def test_a_big_gap_loses_a_crown_that_raw_s_would_win(tmp_path, v3):
    shadow = mk(tmp_path / "shadow", BOTH, mode="shadow", v3=v3)
    s1, s2, _ = await run_race(shadow)
    assert s2.status == ST.CHAMPION                       # raw: gappy wins (in shadow the X items are not even in S)
    raw = mk(tmp_path / "raw", BOTH, mode="binding", g_med=500.0, v3=v3)      # a G_med far above any gap: P = 0, X items in S
    r1, r2, _ = await run_race(raw)
    assert r2.status == ST.CHAMPION and all(h["gap_binding"]["penalty"] == 0.0 for h in r2.history if "gap_binding" in h)
    env = mk(tmp_path / "bind", BOTH, mode="binding", v3=v3)
    c, ch, ts = await run_race(env)
    assert ch.status == ST.NOT_BETTER and ts.champion == c.key and c.status == ST.CHAMPION
    first = next(h for h in ch.history if h["kind"] == "first")
    gb = first["gap_binding"]
    assert gb["penalty"] > 10 and gb["g_med"] == 5.0 and gb["raw"] > gb["s_eff"] and gb["allowance"] >= 8.0
    assert first["score"] == pytest.approx(gb["s_eff"]) and ch.score == pytest.approx(ch.history[-1]["gap_binding"]["s_eff"])
    assert gb["gap"] > 50 and first["se_gap"] > 0 and {"s_gen", "s_x", "gap", "se_gap"} <= first.keys()
    for k in ("draws", "excess", "uncapped", "cap", "w"):
        assert k in gb
    json.dumps(ST._to_json(env.ev.state), allow_nan=False)           # nothing infinite in the state file


async def test_the_cap_limits_the_penalty_to_ten_points(tmp_path):
    capped = mk(tmp_path / "cap", BOTH, cap=10.0)
    _, ch, _ = await run_race(capped)
    gb = next(h for h in ch.history if h["kind"] == "first")["gap_binding"]
    assert gb["penalty"] == 10.0 and gb["uncapped"] > 10.0 and gb["cap"] == 10.0
    assert gb["raw"] - gb["s_eff"] == pytest.approx(10.0)
    assert ch.status == ST.CHAMPION                       # lost with the cap lifted (above), keeps the crown with the 10-point cap


async def test_a_champion_with_a_big_gap_loses_to_a_clean_challenger_with_lower_raw_s(tmp_path):
    skills = {"a/champ": GAPPY, "a/clean": (0.78, 0.78)}
    shadow = mk(tmp_path / "shadow", skills, mode="shadow")
    _, c2, _ = await run_race(shadow, challenger="a/clean")
    assert c2.status == ST.NOT_BETTER                       # on generator items the gappy champion is better
    env = mk(tmp_path / "bind", skills)
    champ, clean, ts = await run_race(env, challenger="a/clean")
    assert clean.status == ST.CHAMPION and champ.status == ST.FORMER and ts.champion == clean.key
    inc = [h for h in champ.history if h["kind"] == "incumbent"]
    assert inc and inc[-1]["gap_binding"]["penalty"] > 10
    conf = next(h for h in clean.history if h["kind"] == "confirm")
    assert conf["gap_binding"]["raw"] < inc[-1]["gap_binding"]["raw"] and conf["score"] > conf["incumbent"]
    assert champ.score == pytest.approx(inc[-1]["gap_binding"]["s_eff"]) and champ.score < inc[-1]["gap_binding"]["raw"]


async def test_the_floor_check_uses_s_eff(tmp_path):
    skills = {"a/gappy": (0.90, 0.05)}
    free = mk(tmp_path / "free", skills, g_med=500.0)
    free.add(1, "a/gappy", None)
    await free.cycle({1: "a/gappy"})
    assert free.sub(1).status == ST.PENDING_CONFIRM                   # clears the floor (24.8) on raw S
    env = mk(tmp_path / "bind", skills)
    env.add(1, "a/gappy", None)
    await env.cycle({1: "a/gappy"})
    gb = env.sub(1).history[0]["gap_binding"]
    assert gb["raw"] > 24.8 > gb["s_eff"] and env.sub(1).status == ST.NOT_BETTER


@pytest.mark.parametrize("v3", [False, True])
async def test_a_pending_entry_from_before_the_switch_gets_its_penalty_at_confirmation(tmp_path, v3):
    ctl = mk(tmp_path / "ctl", BOTH, mode="off", v3=v3)
    _, c2, _ = await run_race(ctl)
    assert c2.status == ST.CHAMPION                       # without the switch it is crowned
    env = mk(tmp_path / "sw", BOTH, mode="off", v3=v3)
    await crown(env, 1, "a/champ", None)
    env.add(2, "a/gappy", None)
    await env.cycle({1: "a/champ", 2: "a/gappy"})
    assert env.sub(2).status == ST.PENDING_CONFIRM and all("gap_binding" not in h for h in env.sub(2).history)
    assert env.sub(2).decision_rule == ("v3" if v3 else "")
    env.ev.cfg.x_tier_mode = "binding"                    # switched on while it waits
    env.clock.advance(13 * H)
    await env.cycle({1: "a/champ", 2: "a/gappy"})
    sub = env.sub(2)
    conf = next(h for h in sub.history if h["kind"] == "confirm")
    assert conf["gap_binding"]["penalty"] > 10 and conf["gap_binding"]["draws"] == 1       # P from its confirmation draw
    ts = env.ev.state.track(TRACK)
    assert sub.status == ST.NOT_BETTER and ts.champion == env.sub(1).key
    assert sum(1 for i in ts.draws[conf["draw"]].item_ids if i.startswith("x-")) == 60 and len(ts.draws[conf["draw"]].item_ids) == 300


async def test_penalty_uses_all_the_entrys_x_draws_and_a_fresh_entry_has_none(tmp_path):
    env = mk(tmp_path, BOTH)
    _, ch, ts = await run_race(env)
    ev = env.ev
    fresh = ST.Submission("k", 9, "h", "c", TRACK, "r", "v", 1, 0.0)
    assert ev._binding_penalty(fresh, TRACK) == (0.0, SC.gap_penalty([], [], 5.0)[1])
    assert ev._binding_penalty(None, TRACK)[0] == 0.0
    draws = ev._x_draws(ch)
    assert len(draws) >= 1
    pen, det = ev._binding_penalty(ch, TRACK)
    assert det["draws"] == len(draws) and pen == pytest.approx(
        SC.gap_penalty([g for g, _ in draws.values()], [e for _, e in draws.values()], 5.0, cap=INF)[0])
    one, _ = ev._binding_penalty(fresh, TRACK, [("d1", 70.0, 5.0)])
    assert one == pytest.approx(0.5 * (70 - 5 - 10)) and ev._binding_penalty(fresh, TRACK, [("d1", 70.0, None)])[0] == 0.0
    assert ev._binding_penalty(fresh, TRACK, [("d1", 70.0, math.inf)])[0] == 0.0       # unknown noise: no penalty on that draw


async def test_playoff_uses_s_eff(tmp_path):
    skills = {"m/u1": MILD, "m/u2": (0.78, 0.78)}
    results = {}
    for mode in ("shadow", "binding"):
        env = mk(tmp_path / mode, skills, mode=mode)
        commits = {}
        for u in (1, 2):
            env.add(u, f"m/u{u}", None)
            commits[u] = f"m/u{u}"
        await env.cycle(commits, block=1000)
        env.clock.advance(13 * H)
        await env.cycle(commits, block=1100)
        env.clock.advance(600)
        await env.cycle(commits, block=1200)
        ts = env.ev.state.track(TRACK)
        results[mode] = (env.sub(1), env.sub(2), ts)
    u1, u2, ts = results["shadow"]
    assert ts.champion == u1.key                          # generator items only: the gappy model is better
    u1, u2, ts = results["binding"]
    assert ts.champion == u2.key and u2.status == ST.CHAMPION
    po = [h for h in u1.history if h["kind"] == "playoff"]
    assert len(po) == 1 and po[0]["gap_binding"]["penalty"] > 5 and po[0]["score"] == pytest.approx(po[0]["gap_binding"]["s_eff"])
    assert po[0]["gap_binding"]["draws"] == 3             # first, confirmation and playoff draw
    pdraw = ts.draws[po[0]["draw"]]
    assert sum(1 for i in pdraw.item_ids if i.startswith("x-")) == 60 and len(pdraw.item_ids) == 300       # X items in the playoff
    assert {"s_gen", "s_x", "se_gap", "gap"} <= po[0].keys()


async def test_keepalive_reads_s_eff(tmp_path):
    envs = {}
    for name, g_med in (("free", 500.0), ("bind", 5.0)):
        env = mk(tmp_path / name, {"a/mild": MILD}, g_med=g_med)
        await crown(env, 1, "a/mild", None)
        assert env.sub(1).status == ST.CHAMPION
        envs[name] = env
    free, bind = envs["free"].sub(1), envs["bind"].sub(1)
    pen = [h for h in bind.history if "gap_binding" in h][-1]["gap_binding"]["penalty"]
    assert pen > 5 and free.nonproc - bind.nonproc == pytest.approx(pen) and free.score - bind.score == pytest.approx(pen)
    floor = (free.nonproc + bind.nonproc) / 2          # between the two: only the champion without the penalty clears it
    for name, kept in (("free", True), ("bind", False)):
        env = envs[name]
        snap = snapshot(1000, [neuron(1, commit(TRACK, "a/mild"))])
        out = E.compute_keepalive(env.ev.state, [TRACK], snap, floors={TRACK: 0.0}, nonproc_floors={TRACK: floor}, now=env.clock())
        assert bool(out[TRACK]) is kept, name
    out = E.compute_leaders(envs["bind"].ev.state, [TRACK], snapshot(1000, [neuron(1, commit(TRACK, "a/mild"))]))
    assert out[TRACK] == [(1, pytest.approx(bind.score))]          # the displayed score is S_eff


# ------------------------------------------------------------------------------------------------ fail closed and rollback
async def test_missing_g_med_runs_the_track_as_shadow_and_logs_once_per_cycle(tmp_path, caplog):
    env = mk(tmp_path, BOTH, g_med=None)
    with caplog.at_level(logging.WARNING, logger="reax.spec.evaluator"):
        champ, ch, ts = await run_race(env)
    assert ch.status == ST.CHAMPION and not any("gap_binding" in h for s in ts.submissions.values() for h in s.history)
    first = next(h for h in ch.history if h["kind"] == "first")
    assert "gap_shadow" in first and first["gap_shadow"]["penalty"] == 0.0           # reported as in shadow
    d = ts.draws[first["draw"]]
    assert len(d.item_ids) == 360 and sum(i.startswith("x-") for i in d.item_ids) == 60      # X on top of the 300, as in shadow
    warns = [r for r in caplog.records if "without a configured G_med" in r.getMessage()]
    events = [e for e in ts.events if "no G_med configured" in e]
    assert warns and len(events) <= len(warns)
    env.ev._gmed_warned.clear()
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="reax.spec.evaluator"):
        for _ in range(5):
            assert env.ev._x_mode_for(TRACK) == "shadow"
    assert len([r for r in caplog.records if "without a configured G_med" in r.getMessage()]) == 1
    other = mk(tmp_path / "other", BOTH, g_med=None)
    other.ev.cfg.g_med = {"risk_assessment": 9.0}                                     # configured for another track only
    assert other.ev._x_mode_for(TRACK) == "shadow"


async def test_rollback_to_off_or_shadow_draws_300_without_x_and_applies_no_penalty(tmp_path):
    env = mk(tmp_path, BOTH)
    _, _, ts = await run_race(env)
    path = env.ev.cfg.state_path
    loaded = ST.load(path)                                                            # state with gap_binding rows loads
    assert any("gap_binding" in h for s in loaded.track(TRACK).submissions.values() for h in s.history)
    raw = json.load(open(path))
    raw["tracks"][TRACK]["submissions"][next(iter(raw["tracks"][TRACK]["submissions"]))]["history"][0]["future_field"] = {"x": 1}
    ST._from_json(raw)                                                                # unknown fields are tolerated
    for mode in ("off", "shadow"):
        env.ev.cfg.x_tier_mode = mode
        uid = 3 if mode == "off" else 4
        env.add(uid, f"a/new-{mode}", None)
        env.server.skills[f"a/new-{mode}"] = CLEAN
        champion = ts.champion
        await env.cycle({1: "a/champ", 2: "a/gappy", uid: f"a/new-{mode}"})
        sub = env.sub(uid)
        first = next(h for h in sub.history if h["kind"] == "first")
        assert "gap_binding" not in first and ts.champion == champion
        n = len(ts.draws[first["draw"]].item_ids)
        assert n == (300 if mode == "off" else 360)
        assert (sum(i.startswith("x-") for i in ts.draws[first["draw"]].item_ids) == 0) == (mode == "off")


async def test_off_mode_with_the_tier_in_the_pool_is_unchanged_by_the_new_fields(tmp_path):
    a = Env(tmp_path / "a", {})
    b = mk(tmp_path / "b", {"a/champ": CLEAN, "a/gappy": CLEAN}, mode="off")
    b.server = a.server = GapServer({"a/champ": CLEAN, "a/gappy": CLEAN})
    a.ev.backend_factory = b.ev.backend_factory = a.server
    for env in (a, b):
        await crown(env, 1, "a/champ", None)
    assert a.sub(1).score == pytest.approx(b.sub(1).score) and not any("gap_binding" in h for h in b.sub(1).history)


async def test_an_entrys_draws_stay_disjoint_with_x_items_and_x_runway_is_alarmed_apart(tmp_path, caplog):
    env = mk(tmp_path, BOTH, cap=10.0)       # the capped penalty lets the challenger reach its confirmation
    _, ch, ts = await run_race(env)
    first = next(h for h in ch.history if h["kind"] == "first")["draw"]
    conf = next(h for h in ch.history if h["kind"] == "confirm")["draw"]
    a, b = ts.draws[first].item_ids, ts.draws[conf].item_ids
    assert not set(a) & set(b) and len(a) == len(b) == 300
    assert sum(i.startswith("x-") for i in a) == sum(i.startswith("x-") for i in b) == 60
    assert max(ts.used.values()) <= S.ITEM_MAX_USES
    pool = env.ev._pool(TRACK)
    rows = [r for r in pool if r["tier"] != S.X_TIER][:300]
    with caplog.at_level(logging.WARNING, logger="reax.spec.evaluator"):
        env.ev._x_runway(ts, pool, rows, 1.0)                              # a draw without any X item although binding is on
    assert any("cross-author shortage: 0 of 60" in e for e in ts.events)
    assert not any("cross-author refill needed" in e for e in ts.events)     # 500 X items: plenty of runway
    ts.used.update({r["id"]: S.ITEM_MAX_USES - 1 for r in pool if r["tier"] == S.X_TIER})
    with caplog.at_level(logging.WARNING, logger="reax.spec.evaluator"):
        env.ev._x_runway(ts, pool, rows, 2.0)
    assert any("cross-author refill needed" in e for e in ts.events) and any("cross-author refill needed" in r.getMessage() for r in caplog.records)
    assert not any("refill needed: about" in e and "cross-author" not in e for e in ts.events[-2:])    # the LLM-written alarm is separate
