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
from reax_subnet.config import ConfigError, load_validator
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
    st = TrackState()
    a = Submission(5, "hk-a", "s1-fast", "ghcr.io/a/e", D1)
    b = Submission(6, "hk-b", "s1-fast", "ghcr.io/b/e", D2)
    subs = observe(st, [a, b], block=10)
    st.results[D1] = _ok(D1, 1.50, block=100)
    assert update_champion(st, subs, P, block=100) is None and st.challenger["digest"] == D1
    assert needs_remeasurement(st, subs) == [a]
    # the same measurement seen again does not count twice
    assert update_champion(st, subs, P, block=101) is None
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
    assert (default.champion_start, default.runner_ups) == (0.75, 2)  # 75 / 16.7 / 8.3 at crowning
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
    # alone, the champion keeps the whole competition share
    assert competition_weights(st, subs[:1], P, reward, block=10_000) == {1: 1.0}
    base = {10: 0.6, 11: 0.4}
    merged = merge_weights(base, {1: 1.0}, 0.2)
    assert merged == pytest.approx({1: 0.2, 10: 0.48, 11: 0.32})
    assert merge_weights(base, {}, 0.2) == base  # no champion: serving weights unchanged
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
BASE_ENV = {"REAX_NETUID": "9", "REAX_ALLOW_DEV_MODELS": "1", "REAX_S1_FAST_RUNTIME_URL": "hash"}


def _env(**over):
    env = {**BASE_ENV, "REAX_NETWORK": "ws://127.0.0.1:9951", "REAX_FEATURE_ENGINE_COMPETITION": "1",
           "REAX_ALLOW_UNRELEASED_FEATURES": "1", "REAX_ENGINE_RUNNER": "dev", "REAX_ENGINE_DEV_MAP": "/tmp/x.json",
           "REAX_ENGINE_REFERENCE": "ref"}
    env.update(over)
    return {k: v for k, v in env.items() if v is not None}


def test_flag_is_refused_outside_local_and_test_chains():
    assert load_validator(_env()).engine is not None
    assert load_validator(_env(REAX_NETWORK="test")).engine is not None
    for network in ("finney", "wss://entrypoint-finney.opentensor.ai:443", "ws://10.0.0.5:9944",
                    "ws://127.0.0.1.evil.example:9944", "ws://user@127.0.0.1:9944"):
        env = _env(REAX_NETWORK=network, REAX_ALLOW_DEV_MODELS=None, REAX_S1_FAST_RUNTIME_URL=None)
        with pytest.raises(ConfigError, match="not available in this release"):
            load_validator(env)
    with pytest.raises(ConfigError, match="not available in this release"):
        load_validator(_env(REAX_ALLOW_UNRELEASED_FEATURES=None))
    assert load_validator(_env(REAX_FEATURE_ENGINE_COMPETITION=None)).engine is None


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


def test_submit_tool_refuses_mainnet(monkeypatch, capsys):
    monkeypatch.setenv("REAX_NETWORK", "finney")
    monkeypatch.setenv("REAX_FEATURE_ENGINE_COMPETITION", "1")
    monkeypatch.setenv("REAX_ALLOW_UNRELEASED_FEATURES", "1")
    with pytest.raises(SystemExit, match="not released"):
        submit_main([f"ghcr.io/a/e@sha256:{D1}"])
    monkeypatch.setenv("REAX_NETWORK", "ws://127.0.0.1:9951")
    submit_main([f"ghcr.io/a/e@sha256:{D1}", "--dry-run"])
    assert capsys.readouterr().out.strip() == f"reax-eng/1:s1-fast:ghcr.io/a/e@sha256:{D1}"
    with pytest.raises(SystemExit, match="digest"):
        submit_main(["ghcr.io/a/e:latest", "--dry-run"])


# ------------------------------------------------------------------------------------------------ measurements
FAST_P = dataclasses.replace(P, timed_items=48, latency_items=8, probe_pairs=8, determinism_items=8,
                             start_timeout_s=30, min_top_agreement=0.95, confirmations=1)
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
                             tracks={"s1-fast": FAST_P}, reward=RewardParams(share=0.2), interval_s=3600)
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
    assert w[3] == pytest.approx(0.2) and w[2] == pytest.approx(0.8)
    assert 4 not in w and 5 not in w  # cheater and copier earn nothing
    # the same state without the competition: identical serving weights, no competitor
    plain.state = val.state
    assert plain.compute_weights(chain.snapshot()) == {2: pytest.approx(1.0)}
    # state survives a restart
    again = EngineCompetition(dev_runner, {"s1-fast": ref_of("reference")}, str(tmp_path / "eng.json"),
                              tracks={"s1-fast": FAST_P}, reward=RewardParams(share=0.2))
    assert again.states["s1-fast"].champion.digest == st.champion.digest
