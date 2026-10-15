"""Engine competition (unreleased; docs/ENGINE-COMPETITION.md): pure rules, config gating, real measurements of the
synthetic engines in scripts/localnet/fake_engine.py, and the validator hook."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import sys

import aiohttp
import pytest

from reax_subnet.chain import Neuron
from reax_subnet.config import ENGINE_REFERENCE_IMAGE, ConfigError, load_validator
from reax_subnet.engcli import submit_main
from reax_subnet.engcomp import (
    Champion, EvalResult, RewardParams, Submission, TrackParams, TrackState, check_correctness, commitment_text,
    competition_weights, merge_weights, needs_remeasurement, observe, parse_engine_commitment, pending, speed_score,
    update_champion,
)
from reax_subnet.engrunner import DevRunner, evaluate
from reax_subnet.engvalidator import EngineCompetition
from reax_subnet.manifest import commitment_text as s1_commitment
from reax_subnet.manifest import load_manifest
from reax_subnet.s1backends import S1HashBackend
from reax_subnet.s1validator import S1Validator

D1, D2, D3, D4 = (hashlib.sha256(x.encode()).hexdigest() for x in "abcd")
FAKE = os.path.join(os.path.dirname(__file__), "..", "scripts", "localnet", "fake_engine.py")
P = TrackParams("s1-fast", pool="s1-fast", temperature=2.07)
P1 = dataclasses.replace(P, confirmations=1)  # title rules without the second measurement


# ------------------------------------------------------------------------------------------------ commitments
def test_commitment_roundtrip_and_limits():
    text = commitment_text("s1-fast", "ghcr.io/alice/plumb-fast", D1)
    assert text == f"reax-eng/1:s1-fast:ghcr.io/alice/plumb-fast@sha256:{D1}"
    assert len(text.encode()) <= 128
    assert parse_engine_commitment(text) == ("s1-fast", "ghcr.io/alice/plumb-fast", D1)
    assert parse_engine_commitment(f"reax-eng/1:s1-fast:localhost:5000/eng@sha256:{D1}")[1] == "localhost:5000/eng"
    for bad in (
        f"reax-eng/1:s1-fast:ghcr.io/alice/plumb-fast:latest",  # a tag, not a digest
        f"reax-eng/1:s1-fast:ghcr.io/alice/plumb@sha256:{D1[:-1]}",  # short digest
        f"reax-eng/2:s1-fast:ghcr.io/alice/plumb@sha256:{D1}",  # other protocol
        f"reax-s1/1:s1-fast:ghcr.io/alice/plumb@sha256:{D1}",
        f"reax-eng/1:s1-fast:ghcr.io/../plumb@sha256:{D1}",
        f"reax-eng/1:s1-fast:Ghcr.io/alice/plumb@sha256:{D1}",  # registries are lower case
        f"reax-eng/1:s1-fast:{'a' * 60}@sha256:{D1}",  # over 128 bytes
        f"reax-eng/1:s1-fast:ghcr.io/alice/plumb fast@sha256:{D1}",
        None, 12,
    ):
        assert parse_engine_commitment(bad) is None, bad
    with pytest.raises(ValueError):
        commitment_text("s1-fast", "x" * 60, D1)
    # a serving miner's commitment is not an engine submission and vice versa
    assert parse_engine_commitment(s1_commitment("s1-fast", load_manifest("s1-fast").manifest_id)) is None


# ------------------------------------------------------------------------------------------------ correctness
def test_correctness_gates():
    ref = {"a": (0.7, 0.2, 0.1), "b": (0.5, 0.5)}
    assert check_correctness(dict(ref), ref, P).ok
    assert check_correctness(dict(ref), ref, P).mean_distance == 0.0
    missing = check_correctness({"a": ref["a"]}, ref, P)
    assert not missing.ok and "missing" in missing.reason
    wrong_len = check_correctness({"a": (0.7, 0.3), "b": ref["b"]}, ref, P)
    assert not wrong_len.ok
    shifted = {"a": (0.6, 0.3, 0.1), "b": ref["b"]}  # one logit moved by ~0.9 -> mean ~0.45 > 0.10
    r = check_correctness(shifted, ref, P)
    assert not r.ok and r.reason.startswith("mean distance")
    # top choice rate: with 100 identical items and two flipped near-ties the rate is 98 % < 99 %
    many = {f"i{i}": (0.5004, 0.4996) for i in range(100)}
    flipped = dict(many)
    flipped["i0"] = flipped["i1"] = (0.4996, 0.5004)
    r = check_correctness(flipped, many, P)
    assert not r.ok and "top choice" in r.reason
    assert check_correctness(flipped, many, P, top_agreement=False).ok


def test_speed_score_is_median_of_paired_ratios():
    assert speed_score([20, 22, 100], [10, 10, 10]) == pytest.approx(2.2)
    assert speed_score([10], [0]) == 0.0


# ------------------------------------------------------------------------------------------------ champion rule
def _ok(digest, score, block=0):
    return EvalResult(digest=digest, track="s1-fast", ok=True, score=score, block=block)


def test_order_uses_the_on_chain_commitment_block_and_binding_disables_first_committer_rule():
    st = TrackState()
    squat = Submission(2, "hk-squat", "s1-fast", "ghcr.io/s/e", D1, block=500)
    owner = Submission(9, "hk-owner", "s1-fast", "ghcr.io/o/e", D1, block=400)
    # without image binding the earlier ON-CHAIN commitment owns the digest, whatever the scan order
    assert observe(st, [squat, owner], block=900) == [owner]
    # with binding both are eligible here; the runner then rejects the one the image is not bound to
    st2 = TrackState()
    assert set(observe(st2, [squat, owner], block=900, first_committer_only=False)) == {squat, owner}
    st2.not_owner.add(f"{D1}:hk-squat")
    assert observe(st2, [squat, owner], block=901, first_committer_only=False) == [owner]


def test_copy_of_a_digest_counts_only_for_the_first_committer():
    st = TrackState()
    a = Submission(5, "hk-a", "s1-fast", "ghcr.io/a/e", D1)
    observe(st, [a], block=100)
    copy = Submission(6, "hk-b", "s1-fast", "ghcr.io/b/e", D1)  # same digest, other hotkey, later
    eligible = observe(st, [a, copy], block=200)
    assert eligible == [a]
    # even when the first committer leaves, the digest stays theirs
    assert observe(st, [copy], block=300) == []


def test_champion_needs_min_gain_and_margin():
    st = TrackState()
    a = Submission(5, "hk-a", "s1-fast", "ghcr.io/a/e", D1)
    b = Submission(6, "hk-b", "s1-fast", "ghcr.io/b/e", D2)
    c = Submission(7, "hk-c", "s1-fast", "ghcr.io/c/e", D3)
    subs = observe(st, [a, b, c], block=10)
    st.results[D1] = _ok(D1, 1.02)  # below min_gain (3 %)
    assert update_champion(st, subs, P1, block=20) is None
    st.results[D1] = _ok(D1, 1.50)
    ch = update_champion(st, subs, P1, block=30)
    assert (ch.digest, ch.uid, ch.crowned_block) == (D1, 5, 30)
    st.results[D2] = _ok(D2, 1.53)  # +2 % over the champion: not enough
    assert update_champion(st, subs, P1, block=40).digest == D1
    st.results[D2] = _ok(D2, 1.56)  # +4 %
    ch = update_champion(st, subs, P1, block=50)
    assert (ch.digest, ch.crowned_block) == (D2, 50)
    # a rejected re-measurement vacates the title; the next best valid submission takes it
    st.results[D2] = EvalResult(digest=D2, track="s1-fast", ok=False, reason="determinism")
    ch = update_champion(st, subs, P1, block=60)
    assert ch.digest == D1
    # the champion withdraws its commitment
    ch = update_champion(st, [b, c], P1, block=70)
    assert ch is None
    assert [h["event"] for h in st.history] == ["crowned", "crowned", "vacated", "crowned", "vacated"]


def test_tie_goes_to_the_earlier_commitment():
    st = TrackState()
    late = Submission(3, "hk-late", "s1-fast", "ghcr.io/l/e", D2)
    early = Submission(9, "hk-early", "s1-fast", "ghcr.io/e/e", D1)
    observe(st, [early], block=10)
    subs = observe(st, [late, early], block=20)
    st.results[D1] = _ok(D1, 1.4)
    st.results[D2] = _ok(D2, 1.4)
    assert update_champion(st, subs, P1, block=30).hotkey == "hk-early"


def test_title_changes_only_after_a_confirming_second_measurement():
    P = dataclasses.replace(globals()["P"], confirm_gap_blocks=50)
    st = TrackState()
    a = Submission(5, "hk-a", "s1-fast", "ghcr.io/a/e", D1)
    b = Submission(6, "hk-b", "s1-fast", "ghcr.io/b/e", D2)
    subs = observe(st, [a, b], block=10)
    st.results[D1] = _ok(D1, 1.50, block=100)
    assert update_champion(st, subs, P, block=100) is None and st.challenger["digest"] == D1
    assert needs_remeasurement(st, subs) == [a]
    # the same measurement seen again does not count twice, nor does one too soon after (e.g. after a restart)
    assert update_champion(st, subs, P, block=101) is None
    st.results[D1] = _ok(D1, 1.45, block=120)
    assert update_champion(st, subs, P, block=120) is None and len(st.challenger["measurements"]) == 1
    st.results[D1] = _ok(D1, 1.40, block=200)  # second draw, lower (winner's curse)
    ch = update_champion(st, subs, P, block=200)
    assert ch.digest == D1 and ch.score == pytest.approx(1.40) and st.challenger is None
    # a challenger that clears the margin once but not on the second draw never takes the title
    st.results[D2] = _ok(D2, 1.50, block=300)
    assert update_champion(st, subs, P, block=300).digest == D1 and st.challenger["digest"] == D2
    assert set(s.digest for s in needs_remeasurement(st, subs)) == {D1, D2}  # champion re-measured too
    st.results[D2] = _ok(D2, 1.41, block=400)  # +0.7 %: below the margin now
    assert update_champion(st, subs, P, block=400).digest == D1 and st.challenger is None


def test_pending_oldest_first_and_limited():
    st = TrackState()
    subs = [Submission(i, f"hk{i}", "s1-fast", "ghcr.io/x/e", d) for i, d in enumerate((D3, D1, D2))]
    observe(st, [subs[1]], block=5)
    eligible = observe(st, subs, block=9)
    assert [s.digest for s in pending(st, eligible, 2)] == [D1, D3]
    st.results[D1] = _ok(D1, 1.0)
    assert [s.digest for s in pending(st, eligible, 5)] == [D3, D2]


def test_weights_winner_take_most_with_decay():
    reward = RewardParams(share=0.2, champion_start=1.0, champion_floor=0.5, decay_blocks=1000, runner_ups=2)
    default = RewardParams()
    assert (default.share, default.champion_start, default.champion_floor, default.runner_ups) == (0.2, 1.0, 0.5, 0)
    st = TrackState()
    subs = [Submission(i, f"hk{i}", "s1-fast", "ghcr.io/x/e", d) for i, d in zip((1, 2, 3, 4), (D1, D2, D3, D4))]
    observe(st, subs, block=0)
    st.results = {D1: _ok(D1, 2.0), D2: _ok(D2, 1.5), D3: _ok(D3, 1.2), D4: _ok(D4, 1.01)}  # D4 below min gain
    st.champion = Champion(D1, "hk1", 1, 2.0, crowned_block=0)
    assert competition_weights(st, subs, P, reward, block=0) == {1: 1.0, 2: 0.0, 3: 0.0}
    half = competition_weights(st, subs, P, reward, block=500)
    assert half[1] == pytest.approx(0.75) and half[2] == pytest.approx(0.25 * 2 / 3) and 4 not in half
    late = competition_weights(st, subs, P, reward, block=10_000)
    assert late[1] == pytest.approx(0.5) and math.fsum(late.values()) == pytest.approx(1.0)
    # alone, the champion's decayed part stays with the serving pools
    assert competition_weights(st, subs[:1], P, reward, block=10_000) == {1: 0.5}
    assert competition_weights(st, subs, P, default, block=50_400) == {1: pytest.approx(0.75)}
    base = {10: 0.6, 11: 0.4}
    merged = merge_weights(base, {1: 1.0}, 0.2)
    assert merged == pytest.approx({1: 0.2, 10: 0.48, 11: 0.32})
    assert merge_weights(base, {}, 0.2) == base  # no champion: serving weights unchanged
    assert merge_weights({}, {1: 1.0}, 0.2) == {}  # nothing set at all stays nothing (rc3 rule)
    # the owner's burn fraction is kept; the share comes out of the miner part only
    burned = merge_weights({0: 0.2, 10: 0.48, 11: 0.32}, {1: 1.0}, 0.2, owner_uid=0, burn_fraction=0.2)
    assert burned == pytest.approx({0: 0.2, 1: 0.16, 10: 0.384, 11: 0.256})
    # a decayed champion (0.5) takes half of its share; the rest stays with the serving miners
    assert merge_weights(base, {1: 0.5}, 0.2) == pytest.approx({1: 0.1, 10: 0.54, 11: 0.36})
    st.champion = None
    assert competition_weights(st, subs, P, reward, block=0) == {}


def test_state_roundtrip():
    st = TrackState()
    observe(st, [Submission(1, "hk1", "s1-fast", "ghcr.io/x/e", D1)], block=3)
    st.results[D1] = EvalResult(digest=D1, track="s1-fast", ok=False, reason="x")  # infinite distances
    st.champion = Champion(D1, "hk1", 1, 1.5, 7)
    back = TrackState.from_json(json.loads(json.dumps(st.to_json())))
    assert back.results[D1].mean_distance == math.inf and back.first_seen == st.first_seen
    assert back.champion == st.champion


# ------------------------------------------------------------------------------------------------ config gating
BASE_ENV = {"REAX_NETUID": "9", "REAX_ALLOW_DEV_MODELS": "1", "REAX_S1_FAST_RUNTIME_URL": "hash",
            "REAX_FEATURE_S1_PRO_POOL": "0"}  # one fake runtime; the pro pool is on by default since 1.2


def _env(**over):
    env = {**BASE_ENV, "REAX_NETWORK": "ws://127.0.0.1:9951", "REAX_FEATURE_ENGINE_COMPETITION": "1",
           "REAX_ENGINE_RUNNER": "dev", "REAX_ENGINE_DEV_MAP": "/tmp/x.json",
           "REAX_ENGINE_REFERENCE": "ref"}
    env.update(over)
    return {k: v for k, v in env.items() if v is not None}


def test_engine_is_fixed_on_finney_and_a_flag_elsewhere(tmp_path):
    assert load_validator(_env()).engine is not None  # local chain, flag on
    assert load_validator(_env(REAX_NETWORK="test")).engine is not None
    assert load_validator(_env(REAX_FEATURE_ENGINE_COMPETITION=None)).engine is None  # dev networks: the flag decides
    assert not load_validator(_env()).engine.release_mode
    # finney: the reference engine is the release constant (the published 1.2.0rc1 s1-fast runtime); with the host side
    # configured the competition is fixed on, without it the validator serves only.
    fixed = {"REAX_NETUID": "92", "REAX_ENGINE_WEIGHTS_DIR": str(tmp_path), "REAX_SPEC_SEALED_DIR": str(tmp_path)}
    published = f"ghcr.io/reaxlabs/reax-s1-fast-runtime@sha256:{D1}"
    for network in ("finney", "wss://entrypoint-finney.opentensor.ai:443"):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("reax_subnet.config.ENGINE_REFERENCE_IMAGE", published)
            cfg = load_validator({**fixed, "REAX_NETWORK": network})
            assert cfg.engine is not None and cfg.engine.release_mode and cfg.engine.reference == published
            assert cfg.spec_sealed_dir == str(tmp_path)  # on without any flag
            with pytest.raises(ConfigError, match="fixed by the release"):  # no kill switch on finney
                load_validator({**fixed, "REAX_NETWORK": network, "REAX_FEATURE_ENGINE_COMPETITION": "0"})
            with pytest.raises(ConfigError, match="release constant"):  # the reference image is not an operator choice
                load_validator({**fixed, "REAX_NETWORK": network, "REAX_ENGINE_REFERENCE": published})
        assert load_validator({**fixed, "REAX_NETWORK": network}).engine.reference == ENGINE_REFERENCE_IMAGE
        no_weights = {k: v for k, v in fixed.items() if k != "REAX_ENGINE_WEIGHTS_DIR"}
        assert load_validator({**no_weights, "REAX_NETWORK": network}).engine is None  # host side missing: serving only
    assert load_validator({"REAX_NETUID": "92", "REAX_MECHANISM": "mpm1"}).engine is None  # s1 only


def test_engine_config_checks(tmp_path):
    with pytest.raises(ConfigError, match="REAX_ALLOW_DEV_MODELS"):
        load_validator(_env(REAX_NETWORK="test", REAX_ALLOW_DEV_MODELS=None, REAX_S1_FAST_RUNTIME_URL=None))
    docker = dict(REAX_ENGINE_RUNNER="docker", REAX_ENGINE_DEV_MAP=None)
    with pytest.raises(ConfigError, match="pinned by digest"):
        load_validator(_env(**docker, REAX_ENGINE_REFERENCE="ghcr.io/x/ref:latest"))
    with pytest.raises(ConfigError, match="WEIGHTS_DIR"):
        load_validator(_env(**docker, REAX_ENGINE_REFERENCE=f"ghcr.io/x/ref@sha256:{D1}"))
    cfg = load_validator(_env(**docker, REAX_ENGINE_REFERENCE=f"ghcr.io/x/ref@sha256:{D1}",
                              REAX_ENGINE_WEIGHTS_DIR=str(tmp_path)))
    assert cfg.engine.runner == "docker" and cfg.engine.interval_s == 6 * 3600


def test_submit_tool_needs_the_flag_off_mainnet(monkeypatch, capsys):
    monkeypatch.delenv("REAX_FEATURE_ENGINE_COMPETITION", raising=False)
    monkeypatch.setenv("REAX_NETWORK", "ws://127.0.0.1:9951")
    with pytest.raises(SystemExit, match="REAX_FEATURE_ENGINE_COMPETITION=1"):
        submit_main([f"ghcr.io/a/e@sha256:{D1}"])
    monkeypatch.setenv("REAX_FEATURE_ENGINE_COMPETITION", "1")
    submit_main([f"ghcr.io/a/e@sha256:{D1}", "--dry-run"])
    assert capsys.readouterr().out.strip() == f"reax-eng/1:s1-fast:ghcr.io/a/e@sha256:{D1}"
    monkeypatch.delenv("REAX_FEATURE_ENGINE_COMPETITION")
    monkeypatch.setenv("REAX_NETWORK", "finney")  # open on mainnet, no flag
    submit_main([f"ghcr.io/a/e@sha256:{D1}", "--dry-run"])
    assert capsys.readouterr().out.strip() == f"reax-eng/1:s1-fast:ghcr.io/a/e@sha256:{D1}"
    with pytest.raises(SystemExit, match="digest"):
        submit_main(["ghcr.io/a/e:latest", "--dry-run"])


# ------------------------------------------------------------------------------------------------ measurements
FAST_P = dataclasses.replace(P, timed_items=48, latency_items=8, probe_pairs=8, determinism_items=8,
                             start_timeout_s=30, min_top_agreement=0.95, confirmations=1,
                             reference_drift=0.9)  # timing noise of the fake engines on a busy CI box is not what these tests check
VARIANTS = ("reference", "fast", "cache", "broken", "jitter")


def ref_of(variant: str) -> str:
    return f"localhost/reax-eng-{variant}@sha256:" + hashlib.sha256(variant.encode()).hexdigest()


@pytest.fixture
def dev_runner(tmp_path):
    mapping = {ref_of(v): [sys.executable, FAKE, "--variant", v, "--delay", "0.01", "--jitter", "0.4"] for v in VARIANTS}
    return DevRunner(str(tmp_path / "eng"), mapping)


async def test_measurements_rank_and_reject(dev_runner):
    out = {}
    for v in ("fast", "cache", "broken", "jitter"):
        out[v] = await evaluate(dev_runner, ref_of("reference"), ref_of(v), "s1-fast", FAST_P)
    assert out["fast"].ok and 1.6 < out["fast"].score < 2.4, out["fast"]
    assert out["fast"].mean_distance == 0.0 and out["fast"].determinism_max_distance == 0.0
    assert not out["cache"].ok and out["cache"].reason.startswith("anti-cache probe")
    assert not out["broken"].ok and out["broken"].reason.startswith("correctness")
    assert not out["jitter"].ok  # caught by the determinism re-run or by the distance gates


async def test_unknown_image_and_dead_reference(dev_runner):
    res = await evaluate(dev_runner, ref_of("reference"), ref_of("nope"), "s1-fast", FAST_P)
    assert not res.ok and res.reason.startswith("start")
    from reax_subnet.engrunner import ReferenceError_

    with pytest.raises(ReferenceError_):
        await evaluate(dev_runner, ref_of("nope"), ref_of("fast"), "s1-fast", FAST_P)


# ------------------------------------------------------------------------------------------------ validator hook
async def test_validator_pays_the_champion_its_share(tmp_path, dev_runner, make_wallet, fake_chain_cls):
    from aiohttp import web

    from reax_subnet.miner import S1Miner

    vwallet = make_wallet("//Validator")
    owner = make_wallet("//Owner")
    miner_w = make_wallet("//Miner/serve")
    m = S1Miner(miner_w, S1HashBackend("s1-fast"), max_concurrent=4, per_caller_concurrency=4)
    m.set_validators(frozenset({vwallet.hotkey.ss58_address}))
    runner = web.AppRunner(m.app())
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    fast_ref, cache_ref = ref_of("fast"), ref_of("cache")
    neurons = [
        Neuron(0, owner.hotkey.ss58_address, "c0", False, 0.0, None, None),
        Neuron(1, vwallet.hotkey.ss58_address, "c1", True, 1e6, None, None),
        Neuron(2, miner_w.hotkey.ss58_address, "c2", False, 0.0, "127.0.0.1", port,
               commitment=s1_commitment("s1-fast", load_manifest("s1-fast").manifest_id)),
        Neuron(3, "hk-fast", "c3", False, 0.0, None, None,
               commitment="reax-eng/1:s1-fast:" + fast_ref),
        Neuron(4, "hk-cache", "c4", False, 0.0, None, None,
               commitment="reax-eng/1:s1-fast:" + cache_ref),
        Neuron(5, "hk-copy", "c5", False, 0.0, None, None,
               commitment="reax-eng/1:s1-fast:" + fast_ref.replace("localhost/", "ghcr.io/")),
    ]
    chain = fake_chain_cls(neurons=neurons, owner_hotkey=owner.hotkey.ss58_address)
    env = {**BASE_ENV, "REAX_NETWORK": "ws://127.0.0.1:9951", "REAX_ALLOW_PRIVATE_IPS": "1",
           "REAX_STATE_PATH": str(tmp_path / "state.json"), "REAX_WARMUP_ROUNDS": "0", "REAX_EMA_ALPHA": "0.5"}
    cfg = load_validator(env)
    dev_runner.mapping[fast_ref.replace("localhost/", "ghcr.io/")] = dev_runner.mapping[fast_ref]
    comp = EngineCompetition(dev_runner, {"s1-fast": ref_of("reference")}, str(tmp_path / "eng.json"),
                             tracks={"s1-fast": FAST_P}, reward=RewardParams(share=0.2), interval_s=3600,
                             start_utc=None)
    val = S1Validator(cfg, vwallet, chain, {"s1-fast": S1HashBackend("s1-fast")}, engine=comp)
    plain = S1Validator(cfg, vwallet, chain, {"s1-fast": S1HashBackend("s1-fast")})
    async with aiohttp.ClientSession(trust_env=False) as session:
        for _ in range(3):
            summary = await val.run_round(session)
    await runner.cleanup()
    report = {r["uid"]: r for r in summary.get("engine_competition", {}).get("s1-fast", [])} or None
    st = comp.states["s1-fast"]
    assert st.champion is not None and st.champion.uid == 3, st.history
    assert st.results[cache_ref.rsplit(":", 1)[1]].ok is False
    assert report is None  # the cycle ran in round 1 only (interval 1 h)
    uids, weights, _ = chain.set_calls[-1]
    w = dict(zip(uids, weights))
    assert w[3] == pytest.approx(0.2, rel=1e-3) and w[2] == pytest.approx(0.8, rel=1e-3)  # 50/50: engine 0.20; the
    # unclaimed track share and the rest stay with serving; decay is a few blocks old
    assert 4 not in w and 5 not in w  # cheater and copier earn nothing
    # the same state without the competition: identical serving weights, no competitor
    plain.state = val.state
    assert plain.compute_weights(chain.snapshot()) == {2: pytest.approx(1.0)}
    # state survives a restart
    again = EngineCompetition(dev_runner, {"s1-fast": ref_of("reference")}, str(tmp_path / "eng.json"),
                              tracks={"s1-fast": FAST_P}, reward=RewardParams(share=0.2), start_utc=None)
    assert again.states["s1-fast"].champion.digest == st.champion.digest


# ------------------------------------------------------------------------------------------------ review fixes
def test_registry_allowlist_and_socket_symlink(tmp_path):
    from reax_subnet.engrunner import DockerRunner, EngineError, UdsSidecarBackend, registry_of

    assert registry_of("ghcr.io/a/b") == "ghcr.io" and registry_of("alice/eng") == "docker.io"
    assert registry_of("localhost:5000/x") == "localhost:5000" and registry_of("eng") == "docker.io"
    runner = DockerRunner(str(tmp_path / "w"), str(tmp_path))
    import asyncio as _a

    for ref in (f"10.0.0.5:5000/x@sha256:{D1}", f"localhost:5000/x@sha256:{D1}", f"evil.example/x@sha256:{D1}"):
        with pytest.raises(EngineError, match="not allowed"):
            _a.run(runner._ensure_image(ref))
    with pytest.raises(EngineError, match="digest"):
        _a.run(runner._ensure_image("ghcr.io/a/b:latest"))
    import socket as _s

    real = tmp_path / "real.sock"
    srv = _s.socket(_s.AF_UNIX)
    srv.bind(str(real))
    link = tmp_path / "link.sock"
    link.symlink_to(real)
    assert UdsSidecarBackend("s1-fast", str(real), "t", 1, 1).socket_ok()
    assert not UdsSidecarBackend("s1-fast", str(link), "t", 1, 1).socket_ok()
    srv.close()


class _FlakyRunner:
    binds_hotkey = True

    def __init__(self, owner):
        self.owner = owner
        self.pulls = 0

    async def owner_of(self, ref):
        self.pulls += 1
        from reax_subnet.engrunner import InfraError

        if self.owner is None:
            raise InfraError("registry rate limit")
        return self.owner

    async def start(self, *a, **k):  # pragma: no cover - never reached in these tests
        raise AssertionError("should not start")


def _snap_with(commitment, hotkey="hk-a"):
    from reax_subnet.chain import Snapshot

    return Snapshot(block=1000, neurons=(Neuron(1, "hk-v", "c", True, 1e6, None, None),
                                         Neuron(2, hotkey, "c", False, 0.0, None, None, commitment=commitment)),
                    owner_hotkey=None)


async def test_infra_failures_are_retried_then_recorded(tmp_path):
    comp = EngineCompetition(_FlakyRunner(None), {"s1-fast": "ghcr.io/r/e@sha256:" + D4}, str(tmp_path / "s.json"),
                             tracks={"s1-fast": P}, interval_s=0)
    snap = _snap_with(f"reax-eng/1:s1-fast:ghcr.io/a/e@sha256:{D1}")
    for i in range(P.max_infra_attempts):
        await comp.run_cycle(snap, {1})
        st = comp.states["s1-fast"]
        if i < P.max_infra_attempts - 1:
            assert D1 not in st.results and st.attempts[D1] == i + 1
    assert st.results[D1].ok is False and st.results[D1].reason.startswith("inconclusive 3 times")


async def test_image_bound_to_another_hotkey_is_ignored(tmp_path):
    runner = _FlakyRunner("hk-owner")
    comp = EngineCompetition(runner, {"s1-fast": "ghcr.io/r/e@sha256:" + D4}, str(tmp_path / "s.json"),
                             tracks={"s1-fast": P}, interval_s=0)
    snap = _snap_with(f"reax-eng/1:s1-fast:ghcr.io/a/e@sha256:{D1}", hotkey="hk-squatter")
    await comp.run_cycle(snap, {1})
    await comp.run_cycle(snap, {1})
    st = comp.states["s1-fast"]
    assert f"{D1}:hk-squatter" in st.not_owner and D1 not in st.results and runner.pulls == 1


async def test_evaluation_budget_fails_a_hanging_engine(dev_runner, tmp_path):
    tight = dataclasses.replace(FAST_P, eval_budget_s=3.0)
    dev_runner.mapping[ref_of("hang")] = [sys.executable, "-c", "import time; time.sleep(600)"]
    comp = EngineCompetition(dev_runner, {"s1-fast": ref_of("reference")}, str(tmp_path / "s.json"),
                             tracks={"s1-fast": tight}, interval_s=0)
    digest = ref_of("hang").rsplit(":", 1)[1]
    await comp.run_cycle(_snap_with("reax-eng/1:s1-fast:" + ref_of("hang")), {1})
    res = comp.states["s1-fast"].results[digest]
    assert not res.ok and "budget" in res.reason
    import glob as _g

    assert not _g.glob(str(tmp_path / "eng" / "eng-*")), "session directories are removed after each engine"
