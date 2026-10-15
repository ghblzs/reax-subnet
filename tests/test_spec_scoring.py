import math

import pytest
from spec_helpers import synthetic_pool

from reax_subnet.spec import prompt as P
from reax_subnet.spec import scoring as SC


def row(kind, labels, expected, tier="t", rid="x"):
    return {"id": rid, "tier": tier, "labels": labels, "expected": expected, "question": {"type": kind}}


def test_choice_competence_and_brier_by_hand():
    rows = [row("choice", ["a", "b", "c", "d"], "a", rid="1"), row("choice", ["a", "b", "c", "d"], "b", rid="2")]
    probs = [{"a": 0.7, "b": 0.1, "c": 0.1, "d": 0.1}, {"a": 0.4, "b": 0.3, "c": 0.15, "d": 0.15}]
    items, s = SC.score_run(rows, probs)
    # accuracy 0.5, chance 0.25 -> 100 * (0.5 - 0.25) / 0.75
    assert s.primitives["choice"]["competence"] == pytest.approx(33.3333, abs=1e-3)
    b1 = 0.09 + 0.01 * 3
    b2 = 0.16 + 0.49 + 0.0225 * 2
    bu = 0.75 ** 2 + 3 * 0.25 ** 2
    assert s.calibration == pytest.approx(100 * (1 - (b1 + b2) / 2 / bu), abs=1e-6)
    assert s.total == pytest.approx(0.75 * s.competence + 0.25 * s.calibration)
    assert s.failures == 0 and items[0].correct and not items[1].correct


def test_noul_abstain_is_wrong_and_score_partial_credit():
    r = [row("noul", ["no", "yes"], "yes", rid="1"), row("noul", ["no", "yes"], "yes", rid="2")]
    _i, s = SC.score_run(r, [{"yes": 0.9, "no": 0.1}, {"yes": 0.6, "no": 0.4}])
    assert s.primitives["noul"]["accuracy"] == 0.5 and s.primitives["noul"]["competence"] == pytest.approx(0.0)
    r = [row("score", ["0", "1", "2", "3"], 3, rid="1")]
    _i, s = SC.score_run(r, [{"0": 0.0, "1": 0.0, "2": 0.5, "3": 0.5}])
    err = abs(2.5 - 3) / 3
    chance = (3 + 2 + 1 + 0) / 4 / 3
    assert s.primitives["score"]["competence"] == pytest.approx(100 * (1 - err / chance))


def test_invalid_answers_cost_calibration_and_competence():
    r = [row("choice", ["a", "b"], "a", rid="1"), row("choice", ["a", "b"], "a", rid="2")]
    _i, s = SC.score_run(r, [{"a": 1.0, "b": 0.0}, None])
    assert s.failures == 1 and s.primitives["choice"]["accuracy"] == 0.5
    for bad in ({"a": 0.5}, {"a": 2, "b": -1}, {"a": float("nan"), "b": 0.5}, {"a": 0.9, "b": 0.9}, "x"):
        assert SC.clean_probs(bad, ["a", "b"]) is None


def test_competence_clipped_at_zero_and_perfect_is_100():
    r = [row("choice", ["a", "b"], "a", rid=str(i)) for i in range(10)]
    _i, bad = SC.score_run(r, [{"a": 0.0, "b": 1.0}] * 10)
    _i, good = SC.score_run(r, [{"a": 1.0, "b": 0.0}] * 10)
    assert bad.competence == 0.0 and bad.calibration == 0.0
    assert good.competence == 100.0 and good.calibration == 100.0 and good.total == 100.0


def test_uniform_answers_score_zero_skill():
    r = [row("choice", ["a", "b", "c", "d"], "a", rid=str(i)) for i in range(8)]
    _i, s = SC.score_run(r, [{k: 0.25 for k in "abcd"}] * 8)
    assert s.calibration == pytest.approx(0.0, abs=1e-9)


def test_paired_se_zero_for_identical_and_positive_for_different():
    pool = synthetic_pool(300)
    a = [SC.score_item(r, {lab: (0.85 if lab == str(r["expected"]) else 0.05) for lab in map(str, r["labels"])}) for r in pool]
    assert SC.paired_se(a, a, seed=1) == 0.0
    b = [SC.score_item(r, {lab: 1 / len(r["labels"]) for lab in map(str, r["labels"])}) for r in pool]
    se = SC.paired_se(a, b, seed=1)
    assert 0 < se < 10
    assert SC.paired_se(a, b, seed=1) == se                # deterministic for a seed
    with pytest.raises(ValueError):
        SC.paired_se(a, b[:-1], seed=1)


def test_by_tier():
    pool = synthetic_pool(60)
    items = [SC.score_item(r, {lab: 1 / len(r["labels"]) for lab in map(str, r["labels"])}) for r in pool]
    assert set(SC.by_tier(items)) == {"proc_sealed", "same_family", "new_domain", "new_family"}


def test_prompt_render_and_readout():
    q = {"type": "choice", "instructions": "pick", "criteria": {"x": "first", "y": {"k": 1}}}
    r = P.render({"s": 1}, q)
    assert r.letters == ("A", "B") and r.keys == ("x", "y")
    assert r.user.startswith("SHARED STATE (JSON string):\n{") and "A: first" in r.user and 'B: {"k": 1}' in r.user
    assert r.user.endswith("Answer with one option letter only.")
    p = P.answer_probs({"question": q}, r, [2.0, 0.0])
    assert sum(p.values()) == pytest.approx(1) and p["x"] == pytest.approx(1 / (1 + math.exp(-1.0)))   # T = 2
    noul = {"type": "noul", "instructions": "?", "criteria": {"true": "T", "false": "F"}}
    rn = P.render({}, noul)
    assert rn.keys == ("false", "true") and "A: F" in rn.user
    assert P.answer_probs({"question": noul}, rn, [0.0, 2.0])["yes"] > 0.5
    with pytest.raises(P.PromptError):
        P.render({}, {"type": "choice", "criteria": {str(i): "o" for i in range(27)}})
    with pytest.raises(P.PromptError):
        P.render({}, {"type": "score", "criteria": "abc"})
