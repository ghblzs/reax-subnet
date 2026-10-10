import math

import pytest

from reax_subnet.scoring import (
    ScoringParams,
    compute_weights,
    ema_update,
    item_passes,
    round_score,
    speed_factor,
    total_variation,
)

P = ScoringParams()


def test_total_variation():
    assert total_variation([1, 0], [0, 1]) == 1.0
    assert total_variation([0.5, 0.5], [0.5, 0.5]) == 0.0
    with pytest.raises(ValueError):
        total_variation([1], [0.5, 0.5])


def test_item_pass_threshold():
    assert item_passes([0.6, 0.4], [0.6 + P.tau - 1e-9, 0.4 - P.tau + 1e-9], P)
    assert not item_passes([0.6, 0.4], [0.6 + P.tau + 1e-6, 0.4 - P.tau - 1e-6], P)


def test_speed_factor():
    assert speed_factor(0.0, P) == 1.0
    assert speed_factor(P.latency_slow_s, P) == 0.0
    assert speed_factor(math.inf, P) == 0.0
    assert 0 < speed_factor((P.latency_fast_s + P.latency_slow_s) / 2, P) < 1


REF = {f"i{k}": (0.7, 0.2, 0.1) for k in range(8)}


def test_round_score_perfect_fast():
    score, fid = round_score(dict(REF), REF, 0.1, P)
    assert fid == 1.0 and score == 1.0


def test_round_score_none_and_partial():
    assert round_score(None, REF, 0.1, P) == (0.0, 0.0)
    answers = dict(REF)
    for k in range(3):  # 5/8 correct < 0.8
        answers[f"i{k}"] = (0.1, 0.2, 0.7)
    assert round_score(answers, REF, 0.1, P)[0] == 0.0
    answers = dict(REF)
    answers["i0"] = (0.1, 0.2, 0.7)  # 7/8 correct
    score, fid = round_score(answers, REF, 0.1, P)
    assert fid == 7 / 8 and score == pytest.approx(7 / 8)


def test_round_score_missing_items_and_length_mismatch():
    answers = {k: v for k, v in REF.items() if k != "i0"}
    answers["i1"] = (0.5, 0.5)
    _, fid = round_score(answers, REF, 0.1, P)
    assert fid == 6 / 8


def test_slow_miner_loses_at_most_latency_weight():
    fast, _ = round_score(dict(REF), REF, 0.0, P)
    slow, _ = round_score(dict(REF), REF, P.latency_slow_s, P)
    assert fast - slow == pytest.approx(P.latency_weight)


def test_ema():
    assert ema_update(0.0, 1.0, 0.1) == pytest.approx(0.1)


def test_weights_proportional_and_threshold():
    w = compute_weights({1: 1.0, 2: 0.5, 3: 0.29, 0: 0.9}, owner_uid=0, params=P)
    assert set(w) == {1, 2}
    assert w[1] == pytest.approx(2 / 3) and w[2] == pytest.approx(1 / 3)


def test_weights_burn_fraction():
    p = ScoringParams(burn_fraction=0.25)
    w = compute_weights({1: 1.0, 2: 1.0}, owner_uid=0, params=p)
    assert w == pytest.approx({0: 0.25, 1: 0.375, 2: 0.375})
    assert math.fsum(w.values()) == pytest.approx(1.0)


def test_weights_nobody_qualifies_goes_to_owner():
    assert compute_weights({1: 0.1}, owner_uid=0, params=P) == {0: 1.0}
    assert compute_weights({1: 0.1}, owner_uid=None, params=P) == {}


def test_weights_without_owner_ignore_burn():
    p = ScoringParams(burn_fraction=0.5)
    assert compute_weights({1: 1.0}, owner_uid=None, params=p) == {1: 1.0}


def test_weights_ignore_nan():
    assert compute_weights({1: float("nan"), 2: 1.0}, owner_uid=None, params=P) == {2: 1.0}


def test_deterministic():
    ema = {uid: (uid * 7919 % 101) / 100 for uid in range(256)}
    assert compute_weights(ema, 0, P) == compute_weights(dict(reversed(list(ema.items()))), 0, P)


@pytest.mark.parametrize(
    "kwargs",
    [{"tau": 0}, {"tau": 1}, {"latency_fast_s": 5, "latency_slow_s": 5}, {"ema_alpha": 0}, {"burn_fraction": 1.5}],
)
def test_params_validated(kwargs):
    with pytest.raises(ValueError):
        ScoringParams(**kwargs)


def test_full_burn_is_owner_only():
    assert compute_weights({1: 1.0, 2: 0.9}, owner_uid=0, params=ScoringParams(burn_fraction=1.0)) == {0: 1.0}


def test_full_burn_without_owner_sets_nothing():
    assert compute_weights({1: 1.0}, owner_uid=None, params=ScoringParams(burn_fraction=1.0)) == {}


def test_softmax_temperature():
    from reax_subnet.backends import softmax

    hot = softmax([4.0, 0.0])
    warm = softmax([4.0, 0.0], 4.0)
    assert hot[0] > 0.98 and abs(warm[0] - 0.7311) < 1e-3
