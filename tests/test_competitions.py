"""1.2 release candidate: active 50/50 split, specialist evaluator wired to the weights, one GPU at a time,
serving-only fallback, entry floors (docs/COMPETITIONS.md)."""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta

import pytest

from reax_subnet import competition_split as CS
from reax_subnet.chain import Neuron, Snapshot
from reax_subnet.competition_split import (
    COMPETITIONS_START_UTC, SPLIT_PROFILE, TRACK_ENTRY_FLOOR, TRACK_SCHEDULE, active_tracks, combine_weights,
)
from reax_subnet.competitions import CompetitionDriver, SpecialistAdapter
from reax_subnet.config import ConfigError, load_validator
from reax_subnet.engcomp import Champion, EvalResult
from reax_subnet.engvalidator import EngineCompetition
from reax_subnet.spec import config as SC

D = "ab" * 32
SERVING = {0: 0.0, 10: 0.5, 11: 0.3, 12: 0.2}


def total(w):
    return sum(w.values())


def _snap(block=900, owner_hotkey="owner"):
    neurons = [Neuron(0, "owner", "c0", False, 0.0, None, None)]
    neurons += [Neuron(u, f"hk{u}", f"c{u}", False, 0.0, None, None) for u in (7, 8, 9, 10, 11, 12)]
    neurons.append(Neuron(3, "hk-eng", "c3", False, 0.0, None, None,
                          commitment=f"reax-eng/1:s1-fast:ghcr.io/a/e@sha256:{D}", commitment_block=5))
    return Snapshot(block=block, neurons=tuple(neurons), owner_hotkey=owner_hotkey)


# ------------------------------------------------------------------------------------------------ A: the split
def test_50_50_is_the_active_profile_and_80_20_stays_ready():
    assert SPLIT_PROFILE == "50/50"
    p = CS.profile_for()
    assert (p.engine, p.tracks, p.inference) == (0.20, (0.15, 0.15), pytest.approx(0.50))
    old = CS.profile_for("80/20")
    assert (old.engine, old.tracks, old.inference) == (0.10, (0.10,), pytest.approx(0.80))


def test_the_first_pair_runs_from_16_oct_to_30_oct():
    assert COMPETITIONS_START_UTC.isoformat() == "2026-10-16T12:00:00+00:00"
    assert active_tracks(COMPETITIONS_START_UTC - timedelta(seconds=1)) == []
    assert active_tracks(COMPETITIONS_START_UTC) == ["demand_forecasting", "risk_assessment"]
    end = COMPETITIONS_START_UTC + timedelta(days=14)
    assert end.isoformat() == "2026-10-30T12:00:00+00:00"
    assert active_tracks(end - timedelta(seconds=1)) == ["demand_forecasting", "risk_assessment"]
    assert active_tracks(end) == []  # TODO in competition_split.py: the next pair


def test_the_specialist_evaluator_uses_the_same_schedule():
    spec = {t.track_id: (t.start_ts, t.days) for t in SC.DEFAULT_TRACKS}
    rel = {s.track_id: (s.start.timestamp(), round((s.end - s.start).days)) for s in TRACK_SCHEDULE}
    assert spec == rel and set(spec) == {"demand_forecasting", "risk_assessment"}
    assert SC.ROTATION_DAYS == CS.ROTATION_DAYS


def test_50_50_pays_everything_it_claims_and_the_rest_stays_in_inference():
    tr = {"demand_forecasting": [(20, 5.0), (21, 4.0), (22, 3.0)], "risk_assessment": [(30, 2.0)]}
    out = combine_weights(SERVING, 3, 1.0, tr)
    assert total(out) == pytest.approx(1.0)
    assert out[3] == pytest.approx(0.20)
    assert out[20] == pytest.approx(0.15 * 0.75) and out[21] == pytest.approx(0.15 * 0.17)
    assert out[22] == pytest.approx(0.15 * 0.08) and out[30] == pytest.approx(0.15 * 0.75)
    # unclaimed risk ranks 2-3 (0.15 * 0.25) go back to the inference part, pro rata; the owner (uid 0) gets nothing
    inference = out[10] + out[11] + out[12]
    assert inference == pytest.approx(1 - 0.20 - 0.15 * 0.75 - 0.15 * 0.17 - 0.15 * 0.08 - 0.15 * 0.75)
    assert out[10] / out[11] == pytest.approx(0.5 / 0.3)
    assert 0 not in out or out[0] == 0


def test_no_specialist_results_and_no_champion_means_inference_only():
    assert combine_weights(SERVING, None, 0.0, {}) == {10: 0.5, 11: 0.3, 12: 0.2}


# ------------------------------------------------------------------------------------------------ E: entry floor
def test_entry_floor_constants_start_without_floors():
    assert TRACK_ENTRY_FLOOR == {"demand_forecasting": None, "risk_assessment": None}


def test_a_track_pays_nothing_at_or_below_its_floor():
    tr = {"demand_forecasting": [(20, 5.0), (21, 3.0), (22, 1.0)]}
    floors = {"demand_forecasting": 3.0, "risk_assessment": None}
    out = combine_weights(SERVING, None, 0.0, tr, floors=floors)
    assert out[20] == pytest.approx(0.15 * 0.75)
    assert 21 not in out and 22 not in out  # at the floor (3.0) and below: nothing
    assert total(out) == pytest.approx(1.0)
    only_below = combine_weights(SERVING, None, 0.0, {"demand_forecasting": [(21, 3.0)]}, floors=floors)
    assert only_below == {10: pytest.approx(0.5), 11: pytest.approx(0.3), 12: pytest.approx(0.2)}
    # rank shares are by rank among those above the floor: the first one above is rank 1
    out = combine_weights(SERVING, None, 0.0, {"demand_forecasting": [(20, 4.0), (21, 3.0)]}, floors=floors)
    assert out[20] == pytest.approx(0.15 * 0.75) and 21 not in out
    # None = no floor: any positive score ranks
    assert 21 in combine_weights(SERVING, None, 0.0, tr, floors={"demand_forecasting": None})
    # the release default applies without the argument
    assert combine_weights(SERVING, None, 0.0, tr) == combine_weights(SERVING, None, 0.0, tr, floors=TRACK_ENTRY_FLOOR)


# ------------------------------------------------------------------------------------------------ B: the wiring
class FakeSpecEvaluator:
    """Stands in for reax_subnet.spec.evaluator.SpecEvaluator: same surface the adapter uses."""

    def __init__(self, results):
        self.results, self.cycles, self.clock = results, 0, lambda: COMPETITIONS_START_UTC.timestamp() + 3600
        self.cfg = SC.SpecConfig(sealed_dir="/x", work_dir="/x", state_path="/x")

    async def evaluate_cycle(self, snapshot):
        self.cycles += 1
        return {t: list(r) for t, r in self.results.items()}

    def leaders(self, active, snapshot):
        return {t.track_id: list(self.results.get(t.track_id, [])) for t in active}


def _engine(tmp_path, now, adapter=None, champion=True):
    comp = EngineCompetition(object(), {"s1-fast": "ghcr.io/r/ref@sha256:" + D}, str(tmp_path / "s.json"),
                             specialist=adapter, utcnow=lambda: now[0])
    if champion:
        st = comp.states["s1-fast"]
        st.first_seen[D] = (5, "hk-eng")
        st.results[D] = EvalResult(digest=D, track="s1-fast", ok=True, score=1.5, block=900)
        st.champion = Champion(D, "hk-eng", 3, 1.5, 900)
    return comp


async def test_fake_evaluator_results_reach_the_weights_end_to_end(tmp_path):
    now = [COMPETITIONS_START_UTC + timedelta(hours=1)]
    fake = FakeSpecEvaluator({"demand_forecasting": [(7, 0.9), (8, 0.8), (9, 0.7)], "risk_assessment": [(10, 0.5)]})
    adapter = SpecialistAdapter(fake, free_mib=_free(10**6))
    comp = _engine(tmp_path, now, adapter)
    snap = _snap()
    base = {0: 0.0, 11: 0.6, 12: 0.4}
    assert comp.weights(base, snap, {0}) [3] == pytest.approx(0.20)  # nothing from the specialists yet: only the engine
    assert 7 not in comp.weights(base, snap, {0})
    assert await adapter.run_cycle(snap) and fake.cycles == 1
    out = comp.weights(base, snap, {0})
    assert total(out) == pytest.approx(1.0)
    assert out[3] == pytest.approx(0.20)
    assert out[7] == pytest.approx(0.15 * 0.75) and out[8] == pytest.approx(0.15 * 0.17)
    assert out[9] == pytest.approx(0.15 * 0.08)
    assert out[10] == pytest.approx(0.15 * 0.75 + 0)  # risk_assessment rank 1 (uid 10 has no serving weight here)
    assert out[11] / out[12] == pytest.approx(0.6 / 0.4)
    # after the window the tracks are not asked and pay nothing
    now[0] = COMPETITIONS_START_UTC + timedelta(days=14, seconds=1)
    out = comp.weights(base, snap, {0})
    assert out[3] == pytest.approx(0.20) and 7 not in out


async def test_restart_restores_the_leaders_from_the_saved_state(tmp_path):
    fake = FakeSpecEvaluator({"demand_forecasting": [(7, 0.9)]})
    adapter = SpecialistAdapter(fake, free_mib=_free(10**6))
    assert adapter.track_results(["demand_forecasting"], COMPETITIONS_START_UTC) == {}
    adapter.refresh(_snap())  # no GPU, no cycle
    assert adapter.track_results(["demand_forecasting", "risk_assessment"], COMPETITIONS_START_UTC) == {
        "demand_forecasting": [(7, 0.9)]}
    assert fake.cycles == 0


def _free(mib):
    async def free(gpu):
        return mib
    return free


# ------------------------------------------------------------------------------------------------ C: one GPU
class _Probe:
    """Counts how many GPU jobs run at the same moment."""

    def __init__(self):
        self.running = self.peak = 0
        self.order = []

    async def work(self, name, seconds=0.05):
        self.running += 1
        self.peak = max(self.peak, self.running)
        self.order.append(f"{name}:start")
        await asyncio.sleep(seconds)
        self.order.append(f"{name}:end")
        self.running -= 1


class _Engine:
    start_utc = None

    def __init__(self, probe, gate=None):
        self.probe, self.gate, self.cycles = probe, gate, 0

    def weights(self, base, snap, exclude, burn_fraction=0.0):
        return dict(base)

    def is_engine_commitment(self, commitment):
        return False

    def due(self):
        return True

    async def run_cycle(self, snap, exclude, between=None):
        self.cycles += 1
        if self.gate is not None:
            await self.gate.wait()
        await self.probe.work("engine")
        return {"s1-fast": []}


class _Spec(SpecialistAdapter):
    def __init__(self, probe, **kw):
        super().__init__(FakeSpecEvaluator({}), **kw)
        self.probe = probe
        self.runs = 0

    async def run_cycle(self, snap):
        self.runs += 1
        await self.probe.work("spec")
        return True


async def test_engine_and_specialist_never_overlap_on_the_gpu():
    probe = _Probe()
    driver = CompetitionDriver(_Engine(probe), _Spec(probe, interval_s=0))
    await asyncio.gather(driver.run(_snap(), {0}), driver.run(_snap(), {0}))  # two overlapping requests
    assert probe.peak == 1
    assert len(probe.order) == 8  # two engine and two spec jobs, one after the other
    for i in range(0, 8, 2):
        assert probe.order[i].endswith(":start") and probe.order[i + 1].endswith(":end")


async def test_specialist_cycle_is_skipped_cleanly_when_memory_is_short(caplog):
    fake = FakeSpecEvaluator({"demand_forecasting": [(7, 0.9)]})
    adapter = SpecialistAdapter(fake, free_mib=_free(10_000))  # < 25 GB
    with caplog.at_level(logging.WARNING):
        assert await adapter.run_cycle(_snap()) is False
    assert fake.cycles == 0 and adapter.track_results(["demand_forecasting"], COMPETITIONS_START_UTC) == {}
    assert "skipped" in caplog.text and not adapter.due()  # retried after the interval, nothing recorded
    ok = SpecialistAdapter(fake, free_mib=_free(26 * 1024))
    assert await ok.run_cycle(_snap()) is True and fake.cycles == 1
    unknown = SpecialistAdapter(fake, free_mib=_free(None))  # nvidia-smi unreadable: the check is skipped
    assert await unknown.run_cycle(_snap()) is True


async def test_spec_waits_for_the_engine_cycle_in_the_background_and_rounds_are_not_blocked(tmp_path):
    probe, gate = _Probe(), asyncio.Event()
    engine = _Engine(probe, gate)
    spec = _Spec(probe, interval_s=0)
    driver = CompetitionDriver(engine, spec)
    task = driver.kick(_snap(), {0})
    assert task is not None and driver.running
    await asyncio.sleep(0.02)
    assert driver.kick(_snap(), {0}) is None  # one cycle at a time; a round just returns
    assert spec.runs == 0  # waiting behind the engine measurement
    gate.set()
    await task
    assert engine.cycles == 1 and spec.runs == 1 and probe.peak == 1 and not driver.running


async def test_a_failing_cycle_never_raises_into_the_round():
    class Boom(_Engine):
        async def run_cycle(self, snap, exclude, between=None):
            raise RuntimeError("boom")

    driver = CompetitionDriver(Boom(_Probe()))
    task = driver.kick(_snap(), {0})
    assert await task == {}


async def test_serving_rounds_continue_during_a_long_competition_cycle(tmp_path, make_wallet, fake_chain_cls):
    import aiohttp

    from reax_subnet.config import load_validator
    from reax_subnet.manifest import commitment_text, load_manifest
    from reax_subnet.s1backends import S1HashBackend
    from reax_subnet.s1validator import SHARES_FAST_ONLY, S1Validator

    owner, vwallet = make_wallet("//Owner"), make_wallet("//Validator")
    neurons = [Neuron(0, owner.hotkey.ss58_address, "c0", False, 0.0, None, None),
               Neuron(1, vwallet.hotkey.ss58_address, "c1", True, 5000.0, None, None)] + [
        Neuron(2 + i, make_wallet(f"//M/{i}").hotkey.ss58_address, f"c{2 + i}", False, 0.0, None, None,
               commitment=commitment_text("s1-fast", load_manifest("s1-fast").manifest_id)) for i in range(3)]
    chain = fake_chain_cls(neurons=neurons, owner_hotkey=owner.hotkey.ss58_address)
    cfg = load_validator({"REAX_NETUID": "9", "REAX_ALLOW_DEV_MODELS": "1", "REAX_NETWORK": "ws://127.0.0.1:9951",
                          "REAX_ALLOW_PRIVATE_IPS": "1", "REAX_STATE_PATH": str(tmp_path / "state.json"),
                          "REAX_WARMUP_ROUNDS": "0", "REAX_FEATURE_S1_PRO_POOL": "0"})
    probe, gate = _Probe(), asyncio.Event()
    engine = _Engine(probe, gate)
    val = S1Validator(cfg, vwallet, chain, {"s1-fast": S1HashBackend("s1-fast")}, shares=SHARES_FAST_ONLY,
                      engine=engine, competitions=CompetitionDriver(engine))
    async with aiohttp.ClientSession(trust_env=False) as session:
        for _ in range(3):  # three rounds while the measurement is still "running"
            await asyncio.wait_for(val.run_round(session), 20)
    assert val.state.rounds == 3 and val.competitions.running and engine.cycles == 1
    gate.set()
    await val.competitions._task
    assert not val.competitions.running


# ------------------------------------------------------------------------------------------------ D: serving-only
def test_validator_without_competition_config_on_finney_serves_only(caplog, tmp_path):
    with caplog.at_level(logging.WARNING, logger="reax.config"):
        cfg = load_validator({"REAX_NETUID": "92"})
    assert cfg.engine is None and cfg.spec_sealed_dir == ""
    warnings = [r for r in caplog.records if "serving-only" in r.getMessage()]
    assert len(warnings) == 1
    for name in ("REAX_ENGINE_WEIGHTS_DIR", "REAX_SPEC_SEALED_DIR", "reference engine image"):
        assert name in warnings[0].getMessage()
    assert "100%" in warnings[0].getMessage()


def test_each_missing_piece_alone_keeps_a_finney_validator_serving_only(monkeypatch, tmp_path, caplog):
    published = f"ghcr.io/reaxlabs/reax-s1-fast-runtime@sha256:{D}"
    monkeypatch.setattr("reax_subnet.config.ENGINE_REFERENCE_IMAGE", published)
    both = {"REAX_NETUID": "92", "REAX_ENGINE_WEIGHTS_DIR": str(tmp_path), "REAX_SPEC_SEALED_DIR": str(tmp_path)}
    assert load_validator(both).engine is not None
    for drop in ("REAX_ENGINE_WEIGHTS_DIR", "REAX_SPEC_SEALED_DIR"):
        env = {k: v for k, v in both.items() if k != drop}
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="reax.config"):
            assert load_validator(env).engine is None
        assert drop in caplog.text


def test_the_reference_engine_is_a_release_constant_with_an_env_override_on_dev_networks_only(monkeypatch, tmp_path):
    from reax_subnet import config

    assert config.ENGINE_REFERENCE_IMAGE.startswith("ghcr.io/reaxlabs/reax-s1-fast-runtime@sha256:")
    published = f"ghcr.io/reaxlabs/reax-s1-fast-runtime@sha256:{D}"
    monkeypatch.setattr(config, "ENGINE_REFERENCE_IMAGE", published)
    host = {"REAX_ENGINE_WEIGHTS_DIR": str(tmp_path), "REAX_SPEC_SEALED_DIR": str(tmp_path)}
    assert load_validator({"REAX_NETUID": "92", **host}).engine.reference == published
    dev = {"REAX_NETUID": "9", "REAX_NETWORK": "ws://127.0.0.1:9951", "REAX_ALLOW_DEV_MODELS": "1",
           "REAX_FEATURE_ENGINE_COMPETITION": "1", **host}
    assert load_validator(dev).engine.reference == published  # dev networks default to the constant too
    other = f"ghcr.io/x/ref@sha256:{'cd' * 32}"
    assert load_validator({**dev, "REAX_ENGINE_REFERENCE": other}).engine.reference == other
    with pytest.raises(ConfigError, match="release constant"):
        load_validator({"REAX_NETUID": "92", "REAX_ENGINE_REFERENCE": other, **host})


def test_a_dev_network_does_not_need_sealed_pools(tmp_path):
    dev = {"REAX_NETUID": "9", "REAX_NETWORK": "ws://127.0.0.1:9951", "REAX_ALLOW_DEV_MODELS": "1",
           "REAX_FEATURE_ENGINE_COMPETITION": "1", "REAX_ENGINE_WEIGHTS_DIR": str(tmp_path),
           "REAX_ENGINE_REFERENCE": f"ghcr.io/x/ref@sha256:{D}"}
    cfg = load_validator(dev)
    assert cfg.engine is not None and cfg.spec_sealed_dir == ""
