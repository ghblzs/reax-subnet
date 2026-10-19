"""Eval mechanism v3, stage 2 plumbing: the cross-author tier (off by default) and the shadow gap."""

import hashlib
import math

import pytest
from spec_helpers import synthetic_pool, write_pool
from test_spec_evaluator import TRACK, Env, crown

from reax_subnet.spec import config as C
from reax_subnet.spec import sampling as S
from reax_subnet.spec import scoring as SC
from reax_subnet.spec import state as ST

pytestmark = pytest.mark.usefixtures("rc5_entry_floors")
H = 3600


def with_x(n_base=1500, n_x=500):
    rows = synthetic_pool(n_base)
    kinds = ("choice", "noul", "score")
    for i in range(n_x):
        k = kinds[i % 3]
        r = {"id": f"x-{i:04d}", "category": "demand_forecasting", "tier": S.X_TIER, "state": {"i": i}, "question": {"type": k, "instructions": f"x {i}"}}
        if k == "choice":
            r.update(labels=["a", "b", "c", "d"], expected="abcd"[i % 4])
            r["question"]["criteria"] = {"a": "A", "b": "B", "c": "C", "d": "D"}
        elif k == "noul":
            r.update(labels=["no", "yes"], expected="yes" if i % 2 else "no")
            r["question"]["criteria"] = {"true": "yes", "false": "no"}
        else:
            r.update(labels=["0", "1", "2", "3"], expected=i % 4)
            r["question"]["criteria"] = ["l0", "l1", "l2", "l3"]
        rows.append(r)
    return rows


def ids(rows):
    return [r["id"] for r in rows]


def digest(rows):
    return hashlib.sha256(",".join(ids(rows)).encode()).hexdigest()[:16]


# ------------------------------------------------------------------------------------------------ sampling
def test_defaults_are_off():
    assert S.X_TIER_MODE == "off" and S.X_TIER_SHARE == 0.20 and S.X_TIER == "cross_author"
    cfg = C.SpecConfig("s", "w", "p")
    assert cfg.x_tier_mode == "off" and cfg.x_tier_share == 0.20


@pytest.mark.parametrize("seed,use_kind,golden", [(1, "none", "cf6e72e228340a81"), (7, "first700", "8976197b44f02374"),
                                                    (12, "every2nd", "6f9e53ab9abc9335")])
def test_off_draws_the_same_items_as_release_18e2597(seed, use_kind, golden):
    """Golden digests computed with sampling.py of the base commit (before the tier existed), pool without the tier."""
    pool = synthetic_pool(1500)
    use = {"none": {}, "first700": {r["id"]: 1 for r in pool[:700]}, "every2nd": {r["id"]: 3 for r in pool[::2]}}[use_kind]
    assert digest(S.draw(pool, n=300, seed=seed, use_counts=use)[0]) == golden
    for mode in ("off", "shadow", "scored"):     # a pool without the tier: every mode is "off"
        assert digest(S.draw(pool, n=300, seed=seed, use_counts=use, x_mode=mode)[0]) == golden


@pytest.mark.parametrize("mode", ["off", "shadow", "scored"])
def test_a_pool_with_the_tier_and_mode_off_never_draws_it(mode):
    pool = with_x()
    base = S.draw(synthetic_pool(1500), n=300, seed=5)[0]
    off = S.draw(pool, n=300, seed=5)[0]
    assert ids(off) == ids(base) and not any(r["tier"] == S.X_TIER for r in off)
    assert ids(S.draw(pool, n=300, seed=5, x_mode="off")[0]) == ids(base)
    if mode != "off":
        rows = S.draw(pool, n=300, seed=5, x_mode=mode)[0]
        assert any(r["tier"] == S.X_TIER for r in rows)


def test_shadow_adds_the_share_on_top_and_leaves_the_normal_draw_untouched():
    pool = with_x()
    off = S.draw(pool, n=300, seed=5)[0]
    rows, fresh = S.draw(pool, n=300, seed=5, x_mode="shadow")
    x = [r for r in rows if r["tier"] == S.X_TIER]
    assert len(x) == round(300 * 0.20) == 60 and len(rows) == 360
    assert ids([r for r in rows if r["tier"] != S.X_TIER]) == ids(off)
    assert ids(rows) == sorted(ids(rows))
    assert len({r["question"]["type"] for r in x}) == 3
    rows2 = S.draw(pool, n=300, seed=5, x_mode="shadow", x_share=0.1)[0]
    assert len([r for r in rows2 if r["tier"] == S.X_TIER]) == 30


def test_scored_replaces_the_share_of_the_nonproc_tiers_and_never_takes_proc():
    pool = with_x()
    off = S.draw(pool, n=300, seed=5)[0]
    rows, _ = S.draw(pool, n=300, seed=5, x_mode="scored")
    x = [r for r in rows if r["tier"] == S.X_TIER]
    assert len(x) == 60 and len(rows) == 300
    count = lambda rs, t: sum(1 for r in rs if r["tier"] == t)   # noqa: E731
    assert count(rows, "proc_sealed") == count(off, "proc_sealed")                 # proc keeps its own share
    nonproc_off = len(off) - count(off, "proc_sealed")
    assert len(rows) - count(rows, "proc_sealed") - len(x) == pytest.approx(nonproc_off - 60, abs=3)


def test_scored_and_shadow_use_what_exists_and_without_items_are_off():
    few = synthetic_pool(1500) + with_x(0, 10)
    rows, _ = S.draw(few, n=300, seed=3, x_mode="shadow")
    assert sum(1 for r in rows if r["tier"] == S.X_TIER) == 10
    rows, _ = S.draw(few, n=300, seed=3, x_mode="scored")
    assert sum(1 for r in rows if r["tier"] == S.X_TIER) == 10 and len(rows) <= 300
    none = synthetic_pool(1500)
    assert ids(S.draw(none, n=300, seed=3, x_mode="scored")[0]) == ids(S.draw(none, n=300, seed=3)[0])


def test_x_items_respect_exclusion_and_retirement():
    pool = with_x()
    first = S.draw(pool, n=300, seed=5, x_mode="shadow")[0]
    used = {r["id"]: 1 for r in first}
    second = S.draw(pool, n=300, seed=6, x_mode="shadow", use_counts=used, exclude=set(ids(first)))[0]
    assert not set(ids(first)) & set(ids(second))
    retired = {r["id"]: S.ITEM_MAX_USES for r in pool if r["tier"] == S.X_TIER}
    rows = S.draw(pool, n=300, seed=5, x_mode="shadow", use_counts=retired)[0]
    assert not any(r["tier"] == S.X_TIER for r in rows)


def test_refill_count_ignores_the_x_tier():
    base = synthetic_pool(1500)
    assert S.nonproc_draws_left(with_x(), {}, 300) == S.nonproc_draws_left(base, {}, 300)


def test_x_pool_generation_fingerprint():
    assert S.x_pool_gen(synthetic_pool(100)) == ""
    a, b = S.x_pool_gen(with_x(100, 50)), S.x_pool_gen(with_x(100, 51))
    assert a and b and a != b and a == S.x_pool_gen(with_x(300, 50))


def test_env_config(tmp_path):
    base = {"REAX_SPEC_SEALED_DIR": str(tmp_path)}
    cfg = C.from_env({**base, "REAX_SPEC_X_TIER_MODE": "Shadow", "REAX_SPEC_X_TIER_SHARE": "0.1"})
    assert cfg.x_tier_mode == "shadow" and cfg.x_tier_share == 0.1
    assert C.from_env({**base, "REAX_SPEC_X_TIER_MODE": "bogus"}).x_tier_mode == "off"


# ------------------------------------------------------------------------------------------------ gap functions
def test_gap_penalty_formula():
    pen, d = SC.gap_penalty([20.0, 22.0], [2.0, 2.0], g_med=5.0)
    se = math.sqrt((4 + 4) / 2)                                 # 2.0: one draw's SE, not pooled over the draws
    assert d["gap"] == 21.0 and d["se_gap"] == pytest.approx(se) and d["excess"] == 16.0
    assert d["allowance"] == 8.0                                 # max(8, 2 x 2.0)
    assert pen == pytest.approx(0.5 * (16.0 - 8.0)) == d["penalty"]


def test_gap_penalty_allowance_grows_with_the_noise_and_has_a_floor():
    pen, d = SC.gap_penalty([20.0], [8.0], g_med=5.0)           # 2 x SE = 16 > 8
    assert d["allowance"] == 16.0 and pen == 0.0                 # excess 15 < 16
    pen, d = SC.gap_penalty([22.0], [8.0], g_med=5.0)
    assert pen == pytest.approx(0.5 * (17 - 16))
    pen, d = SC.gap_penalty([13.0], [0.1], g_med=5.0)           # excess exactly the floor allowance: no penalty
    assert pen == 0.0 and d["excess"] == 8.0
    assert SC.gap_penalty([13.01], [0.1], 5.0)[0] == pytest.approx(0.5 * 0.01)


def test_gap_penalty_negative_excess_and_weights_and_missing_se():
    assert SC.gap_penalty([-3.0], [1.0], g_med=5.0)[0] == 0.0
    assert SC.gap_penalty([30.0], None, 5.0, w=1.0)[0] == pytest.approx(30 - 5 - 8)
    assert SC.gap_penalty([30.0], 0.0, 5.0, w=0.25, allowance_floor=4.0)[0] == pytest.approx(0.25 * (25 - 4))
    assert SC.gap_penalty([30.0, 20.0], [None, 3.0], 5.0)[1]["se_gap"] == pytest.approx(math.sqrt(9 / 2))


def test_gap_penalty_without_g_med_or_draws_is_report_only():
    pen, d = SC.gap_penalty([30.0], [1.0], g_med=None)
    assert pen == 0.0 and d["gap"] == 30.0 and d["g_med"] is None and "reported only" in d["reason"]
    assert SC.gap_penalty([], [], 5.0)[0] == 0.0


def test_median_gap_needs_five_coldkeys():
    assert SC.median_gap({f"c{i}": float(i) for i in range(4)}) is None
    assert SC.median_gap({f"c{i}": float(i) for i in range(5)}) == 2.0
    assert SC.median_gap({f"c{i}": float(i) for i in range(6)}) == 2.5
    assert SC.median_gap({f"c{i}": 10.0 for i in range(5)}, reference=[0.0, 0.0]) == 10.0
    assert SC.median_gap({f"c{i}": 10.0 for i in range(4)}, reference=[0.0]) is None     # reference models are no coldkeys
    assert SC.median_gap({f"c{i}": 1.0 for i in range(5)}, min_coldkeys=3) == 1.0


def test_gap_se_is_a_bootstrap_of_both_groups():
    from reax_subnet.spec.scoring import ItemScore
    def mk(prefix, n, ok, tier="t"):
        return [ItemScore(f"{prefix}{i}", "choice", tier, True, i % ok == 0, 0.25, 0.0, 0.5, 0.75) for i in range(n)]
    gen, x = mk("g", 240, 2), mk("x", 60, 3, S.X_TIER)
    a, b = SC.gap_se(gen, x, seed=1), SC.gap_se(gen, x, seed=1)
    assert a == b and 0 < a < 20
    assert SC.gap_se(gen, x[:1], seed=1) == math.inf
    g, xx = SC.split_x(gen + x)
    assert len(g) == 240 and len(xx) == 60 and all(it.tier == S.X_TIER for it in xx)


# ------------------------------------------------------------------------------------------------ evaluator
def env_with_x(tmp_path, mode):
    env = Env(tmp_path, {})
    write_pool(str(tmp_path / "sealed"), TRACK, with_x())
    env.ev.cfg.x_tier_mode = mode
    return env


async def run_scenario(env):
    await crown(env, 1, "a/champ", 0.65)
    env.add(2, "a/chal", 0.85)
    await env.cycle({1: "a/champ", 2: "a/chal"})
    env.clock.advance(13 * H)
    await env.cycle({1: "a/champ", 2: "a/chal"})
    return env.sub(1), env.sub(2), env.ev.state.track(TRACK)


async def test_shadow_gap_is_recorded_but_changes_no_decision_or_score(tmp_path):
    off = env_with_x(tmp_path / "off", "off")
    shadow = env_with_x(tmp_path / "shadow", "shadow")
    o1, o2, ots = await run_scenario(off)
    s1, s2, sts = await run_scenario(shadow)
    assert (o1.status, o2.status) == (s1.status, s2.status) == (ST.FORMER, ST.CHAMPION)
    assert s2.score == pytest.approx(o2.score) and s1.score == pytest.approx(o1.score)     # the X items are not in the total
    assert not any("s_x" in h for h in o2.history)
    firsts = [h for h in s2.history if h["kind"] in ("first", "confirm")]
    assert firsts and all({"s_gen", "s_x", "x_gen", "gap", "gap_se", "gap_shadow"} <= h.keys() for h in firsts)
    assert firsts[0]["x_gen"] == S.x_pool_gen(with_x())
    d = firsts[-1]["gap_shadow"]
    assert d["penalty"] == 0.0 and d["g_med"] is None and d["coldkeys"] == 2          # fewer than 5 coldkeys: report only
    # the draws carry the extra items, every model answers all of them, and the champion's own gap is recorded
    assert any(h["kind"] == "incumbent" and "s_x" in h for h in s1.history)
    draw = sts.draws[firsts[0]["draw"]]
    assert sum(1 for i in draw.item_ids if i.startswith("x-")) == 60 and len(draw.item_ids) == 360
    assert any("cross-author gap" in e and "not applied" in e for e in sts.events)


async def test_scored_mode_counts_the_items_in_the_total_and_still_applies_no_penalty(tmp_path):
    env = env_with_x(tmp_path, "scored")
    s1, s2, ts = await run_scenario(env)
    first = next(h for h in s2.history if h["kind"] == "first")
    assert len(ts.draws[first["draw"]].item_ids) == 300 and sum(i.startswith("x-") for i in ts.draws[first["draw"]].item_ids) == 60
    assert "gap_shadow" in first and first["gap_shadow"]["penalty"] == 0.0
    assert s2.status == ST.CHAMPION


async def test_off_mode_pool_with_the_tier_behaves_like_the_pool_without(tmp_path):
    a = Env(tmp_path / "a", {})
    b = env_with_x(tmp_path / "b", "off")
    ra, rb = await run_scenario(a), await run_scenario(b)
    assert [h["draw"] for h in ra[1].history if h["kind"] != "behaviour"] == [h["draw"] for h in rb[1].history if h["kind"] != "behaviour"]
    assert ra[2].draws.keys() == rb[2].draws.keys()
    assert all(ra[2].draws[k].item_ids == rb[2].draws[k].item_ids for k in ra[2].draws)
