"""REAX_COMPETITIONS_MODE=remote and reax-competition-worker (docs/COMPETITIONS.md, "Remote mode")."""

import asyncio
import dataclasses
import json
import logging
import os
import sys
import time
from datetime import timedelta

import pytest

from reax_subnet import competition_split as CS
from reax_subnet import engrunner as ER
from reax_subnet import remote as RM
from reax_subnet import worker as W
from reax_subnet.chain import Neuron, Snapshot
from reax_subnet.competition_split import COMPETITIONS_START_UTC
from reax_subnet.competitions import CompetitionDriver, SpecialistAdapter
from reax_subnet.config import (ENGINE_REFERENCE_IMAGE, ENGINE_STATE_FILE, REMOTE_MAX_AGE_S, SPEC_STATE_FILE, ConfigError,
                                load_validator)
from reax_subnet.engcomp import Champion, EvalResult
from reax_subnet.engvalidator import EngineCompetition
from reax_subnet.spec import config as SC
from reax_subnet.spec import state as ST
from reax_subnet.spec.commitment import SpecCommitment
from reax_subnet.spec.evaluator import SpecEvaluator
from test_engcomp import FAKE, FAST_P, ref_of


@pytest.fixture(autouse=True)
def _release_floors_off_unless_floor_test(request, monkeypatch):
    """The split/payout tests use small synthetic scores; the release entry floors (24.8 / 33.7) are tested in the floor tests."""
    if any(k in request.node.name for k in ("floor_constants", "at_or_below_its_floor")):
        return
    from reax_subnet import competition_split as _cs
    for _k in list(_cs.TRACK_ENTRY_FLOOR):
        monkeypatch.setitem(_cs.TRACK_ENTRY_FLOOR, _k, None)


D = "ab" * 32
HOUR = 3600
NOW0 = COMPETITIONS_START_UTC + timedelta(hours=1)
T_SPEC = "demand_forecasting"
REV = "c" * 40
SERVING = {0: 0.0, 11: 0.6, 12: 0.4}


def snap(*, eng_hotkey="hk-eng", eng_digest=D, specs=(), block=900):
    neurons = [Neuron(0, "owner", "c0", False, 0.0, None, None), Neuron(11, "hk11", "c11", False, 0.0, None, None),
               Neuron(12, "hk12", "c12", False, 0.0, None, None),
               Neuron(3, eng_hotkey, "c3", False, 0.0, None, None,
                      commitment=f"reax-eng/1:s1-fast:ghcr.io/a/e@sha256:{eng_digest}", commitment_block=5)]
    for uid, repo in specs:
        neurons.append(Neuron(uid, f"hk-spec{uid}", f"c{uid}", False, 0.0, None, None, commitment=SpecCommitment(T_SPEC, repo, REV).text()))
    return Snapshot(block=block, neurons=tuple(neurons), owner_hotkey="owner")


class Clock:
    def __init__(self, t=None):
        self.t = time.time() if t is None else t

    def __call__(self):
        return self.t


class _R:
    binds_hotkey = True


def write_engine_state(path, *, champion=True, saved_at=None, not_owner=(), binds=True):
    comp = EngineCompetition(_R(), {"s1-fast": "ghcr.io/r/ref@sha256:" + D}, str(path), utcnow=lambda: NOW0)
    if champion:
        st = comp.states["s1-fast"]
        st.first_seen[D] = (5, "hk-eng")
        st.results[D] = EvalResult(digest=D, track="s1-fast", ok=True, score=1.5, block=900)
        st.champion = Champion(D, "hk-eng", 3, 1.5, 900)
        st.not_owner |= set(not_owner)
    comp.runner.binds_hotkey = binds
    comp.save()
    if saved_at is not None:
        raw = json.load(open(path))
        raw["saved_at"] = saved_at
        json.dump(raw, open(path, "w"))
    return comp


def stamp(path, ts):
    """Pretend the worker saved the file at ``ts`` (tests must not depend on the time of day they run)."""
    raw = json.load(open(path))
    raw["saved_at"] = int(ts)
    json.dump(raw, open(path, "w"))


def remote_engine(path, clock=None, **kw):
    now = [NOW0]
    comp = RM.RemoteEngineCompetition(str(path), utcnow=lambda: now[0], wall=clock or Clock(), **kw)
    comp._now = now
    return comp


# ------------------------------------------------------------------------------------------------ read_state
def test_read_state_rejects_missing_stale_future_and_garbage(tmp_path):
    p = tmp_path / "s.json"
    now = 1_000_000.0
    assert RM.read_state(str(p), now=now)[1] == "file missing"
    p.write_text("{nope")
    assert RM.read_state(str(p), now=now)[1].startswith("unreadable")
    p.write_text("[1]")
    assert RM.read_state(str(p), now=now)[1] == "not a JSON object"
    p.write_text(json.dumps({"saved_at": now - REMOTE_MAX_AGE_S - 1}))
    assert RM.read_state(str(p), now=now)[1].startswith("stale")
    p.write_text(json.dumps({"saved_at": now - REMOTE_MAX_AGE_S + 60}))
    assert RM.read_state(str(p), now=now)[0] is not None
    p.write_text(json.dumps({"saved_at": now + 3 * HOUR}))
    assert "future" in RM.read_state(str(p), now=now)[1]
    p.write_text(json.dumps({"version": 1}))                     # no saved_at: the file's own mtime counts
    assert RM.read_state(str(p), now=time.time())[0] is not None
    assert RM.read_state(str(p), now=time.time() + 30 * HOUR)[1].startswith("stale")


# ------------------------------------------------------------------------------------------------ engine, remote
def test_remote_pays_exactly_like_local(tmp_path):
    path = tmp_path / ENGINE_STATE_FILE
    local = write_engine_state(path)
    remote = remote_engine(path)
    s = snap()
    want = local.weights(SERVING, s, {0}, 0.0)
    got = remote.weights(SERVING, s, {0}, 0.0)
    assert got == want and got[3] == pytest.approx(0.20, rel=1e-3)
    assert sum(got.values()) == pytest.approx(1.0)


def test_remote_runs_no_cycle_and_never_writes(tmp_path):
    path = tmp_path / ENGINE_STATE_FILE
    write_engine_state(path)
    before = (os.stat(path).st_mtime_ns, open(path).read())
    remote = remote_engine(path)
    assert remote.due() is False
    assert asyncio.run(remote.run_cycle(snap(), {0})) == {}
    remote.save()
    remote.weights(SERVING, snap(), {0})
    assert (os.stat(path).st_mtime_ns, open(path).read()) == before
    assert sorted(os.listdir(tmp_path)) == [ENGINE_STATE_FILE]


def test_a_corrupt_file_is_left_alone_pays_inference_and_warns_once(tmp_path, caplog):
    path = tmp_path / ENGINE_STATE_FILE
    path.write_text("{ not json")
    remote = remote_engine(path)
    with caplog.at_level(logging.WARNING, logger="reax.remote"):
        for _ in range(4):
            assert remote.weights(SERVING, snap(), {0}) == pytest.approx(SERVING)
    assert path.read_text() == "{ not json" and not os.path.exists(str(path) + ".bad")
    assert len([r for r in caplog.records if "unusable" in r.getMessage()]) == 1
    write_engine_state(path)                                               # the worker recovers
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="reax.remote"):
        assert remote.weights(SERVING, snap(), {0})[3] == pytest.approx(0.20, rel=1e-3)
    assert "fresh again" in caplog.text
    path.write_text("{ not json")                                          # a second outage warns again
    with caplog.at_level(logging.WARNING, logger="reax.remote"):
        remote.weights(SERVING, snap(), {0})
    assert len([r for r in caplog.records if "unusable" in r.getMessage()]) == 1


def test_missing_stale_wrong_version_and_invalid_content_all_pay_inference(tmp_path, caplog):
    path = tmp_path / ENGINE_STATE_FILE
    clock = Clock()
    remote = remote_engine(path, clock)
    with caplog.at_level(logging.WARNING, logger="reax.remote"):
        assert remote.weights(SERVING, snap(), {0}) == pytest.approx(SERVING)          # missing
        write_engine_state(path)
        assert remote.weights(SERVING, snap(), {0})[3] > 0.19                          # fresh
        clock.t += REMOTE_MAX_AGE_S + 10                                               # the worker died
        assert remote.weights(SERVING, snap(), {0}) == pytest.approx(SERVING)
        raw = json.load(open(path))
        raw.update(saved_at=clock.t, version=99)
        json.dump(raw, open(path, "w"))
        assert remote.weights(SERVING, snap(), {0}) == pytest.approx(SERVING)
        raw.update(version=1, tracks={"s1-fast": {"champion": {"digest": 1}}})
        json.dump(raw, open(path, "w"))
        assert remote.weights(SERVING, snap(), {0}) == pytest.approx(SERVING)
    msgs = [r.getMessage() for r in caplog.records if "unusable" in r.getMessage()]
    assert len(msgs) == 2 and "file missing" in msgs[0] and "stale" in msgs[1]       # one per outage; the bad states after the stale one are the same outage


def test_fresh_file_just_inside_the_26_hour_limit_pays(tmp_path):
    path = tmp_path / ENGINE_STATE_FILE
    clock = Clock()
    write_engine_state(path, saved_at=int(clock.t) - REMOTE_MAX_AGE_S + 120)
    assert remote_engine(path, clock).weights(SERVING, snap(), {0})[3] > 0.19
    write_engine_state(path, saved_at=int(clock.t) - REMOTE_MAX_AGE_S - 120)
    assert remote_engine(path, clock).weights(SERVING, snap(), {0}) == pytest.approx(SERVING)


def test_champion_must_still_hold_its_uid_and_commitment(tmp_path):
    path = tmp_path / ENGINE_STATE_FILE
    write_engine_state(path)
    remote = remote_engine(path)
    assert remote.weights(SERVING, snap(), {0})[3] > 0.19
    # deregistered / hotkey replaced on that uid: the title belongs to the hotkey, not the uid
    assert 3 not in remote.weights(SERVING, snap(eng_hotkey="hk-new"), {0})
    # committed to another image now
    assert remote.weights(SERVING, snap(eng_digest="cd" * 32), {0}) == pytest.approx(SERVING)
    # moved to another uid: follows the hotkey
    moved = Snapshot(block=900, neurons=tuple(dataclasses.replace(n, uid=40) if n.uid == 3 else n for n in snap().neurons),
                     owner_hotkey="owner")
    assert remote.weights(SERVING, moved, {0})[40] > 0.19
    # the validator itself is never paid
    assert 3 not in remote.weights(SERVING, snap(), {0, 3})


def test_an_image_bound_to_another_hotkey_earns_nothing_in_remote_mode(tmp_path):
    path = tmp_path / ENGINE_STATE_FILE
    write_engine_state(path, not_owner=[f"{D}:hk-eng"])
    assert 3 not in remote_engine(path).weights(SERVING, snap(), {0})


def test_before_the_start_instant_the_weights_are_the_serving_weights(tmp_path):
    path = tmp_path / ENGINE_STATE_FILE
    write_engine_state(path)
    remote = remote_engine(path)
    remote._now[0] = COMPETITIONS_START_UTC - timedelta(seconds=1)
    assert remote.weights(SERVING, snap(), {0}) == SERVING


# ------------------------------------------------------------------------------------------------ specialists, remote
def spec_state(path, *, saved=True, score=(30.0, 25.0, 20.0)):
    st = ST.SpecState()
    ts = st.track(T_SPEC)
    for i, (repo, sc) in enumerate(zip(("a/m1", "a/m2", "a/m3"), score)):
        c = SpecCommitment(T_SPEC, repo, REV).text()
        key = f"hk-spec{20 + i}|{c}"
        sub = ST.Submission(key=key, uid=20 + i, hotkey=f"hk-spec{20 + i}", coldkey="c", track_id=T_SPEC, repo=repo, revision=REV,
                            first_seen_block=1, first_seen_ts=1.0, status=ST.CHAMPION if i == 0 else ST.FORMER, score=sc,
                            champion_since_ts=100.0 - i)
        ts.submissions[key] = sub
        if i == 0:
            ts.champion = key
    ST.save(st, str(path))
    return st


SPECS = [(20, "a/m1"), (21, "a/m2"), (22, "a/m3")]


def test_remote_spec_leaders_equal_the_local_evaluators(tmp_path):
    path = tmp_path / SPEC_STATE_FILE
    st = spec_state(path)
    stamp(path, NOW0.timestamp())
    cfg = SC.SpecConfig(sealed_dir=str(tmp_path), work_dir=str(tmp_path / "w"), state_path=str(path))
    local = SpecEvaluator(cfg)
    ev = RM.RemoteSpecEvaluator(str(path), clock=lambda: NOW0.timestamp())
    s = snap(specs=SPECS)
    active = cfg.active_tracks(NOW0.timestamp())
    assert [t.track_id for t in active] == ["demand_forecasting", "risk_assessment"]
    assert ev.leaders(active, s) == local.leaders(active, s)
    assert ev.leaders(active, s)[T_SPEC] == [(20, 30.0), (21, 25.0), (22, 20.0)]
    # one of them re-committed to another repo: it is dropped, the others stay
    s2 = snap(specs=[(20, "a/m1"), (21, "a/other"), (22, "a/m3")])
    assert ev.leaders(active, s2)[T_SPEC] == [(20, 30.0), (22, 20.0)]
    assert ev.leaders(active, snap(specs=[]))[T_SPEC] == []              # all deregistered
    assert ev.leaders([], s) == {}


def test_remote_tracks_pay_75_17_8_with_floors_and_stale_goes_to_inference(tmp_path, caplog):
    sp = tmp_path / SPEC_STATE_FILE
    spec_state(sp)
    stamp(sp, NOW0.timestamp())
    clock = Clock(NOW0.timestamp())
    ev = RM.RemoteSpecEvaluator(str(sp), clock=clock)
    adapter = RM.RemoteSpecAdapter(ev, free_mib=None)
    assert adapter.due() is False and asyncio.run(adapter.run_cycle(snap())) is False
    ep = tmp_path / ENGINE_STATE_FILE
    write_engine_state(ep, champion=False)
    stamp(ep, NOW0.timestamp())
    comp = RM.RemoteEngineCompetition(str(ep), specialist=adapter, utcnow=lambda: NOW0, wall=clock)
    s = snap(specs=SPECS)
    adapter.refresh(s)
    w = comp.weights(SERVING, s, {0})
    assert w[20] == pytest.approx(0.15 * 0.75) and w[21] == pytest.approx(0.15 * 0.17) and w[22] == pytest.approx(0.15 * 0.08)
    assert sum(w.values()) == pytest.approx(1.0)
    # a floor (release constant) is applied exactly as in local mode
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(CS.TRACK_ENTRY_FLOOR, T_SPEC, 22.0)
        assert 22 not in comp.weights(SERVING, s, {0}) and comp.weights(SERVING, s, {0})[21] > 0
    # the worker stops: after 26 h nothing is paid and there is exactly one warning
    clock.t += REMOTE_MAX_AGE_S + 60
    with caplog.at_level(logging.WARNING, logger="reax.remote"):
        for _ in range(3):
            adapter.refresh(s)
            assert comp.weights(SERVING, s, {0}) == pytest.approx(SERVING)
    assert len([r for r in caplog.records if "specialist tracks" in r.getMessage()]) == 1


def test_the_two_state_files_fail_independently(tmp_path):
    sp, ep = tmp_path / SPEC_STATE_FILE, tmp_path / ENGINE_STATE_FILE
    spec_state(sp)
    write_engine_state(ep)
    stamp(sp, NOW0.timestamp())
    stamp(ep, NOW0.timestamp())
    adapter = RM.RemoteSpecAdapter(RM.RemoteSpecEvaluator(str(sp), clock=lambda: NOW0.timestamp()), free_mib=None)
    comp = RM.RemoteEngineCompetition(str(ep), specialist=adapter, utcnow=lambda: NOW0, wall=lambda: NOW0.timestamp())
    s = snap(specs=SPECS)
    adapter.refresh(s)
    both = comp.weights(SERVING, s, {0})
    assert both[3] > 0.19 and both[20] > 0.1
    sp.write_text("garbage")
    adapter.refresh(s)
    only_engine = comp.weights(SERVING, s, {0})
    assert only_engine[3] == pytest.approx(0.20, rel=1e-3) and 20 not in only_engine


# ------------------------------------------------------------------------------------------------ validator config
FINNEY = {"REAX_NETUID": "92", "REAX_NETWORK": "finney"}


def test_competitions_mode_default_and_values(tmp_path):
    assert load_validator({"REAX_NETUID": "92"}).competitions_mode == "local"
    with pytest.raises(ConfigError, match="REAX_COMPETITIONS_MODE"):
        load_validator({**FINNEY, "REAX_COMPETITIONS_MODE": "maybe"})
    cfg = load_validator({**FINNEY, "REAX_COMPETITIONS_MODE": "off"})
    assert cfg.engine is None and cfg.spec_sealed_dir == ""


def test_off_on_finney_warns_and_refuses_the_dev_flag(caplog):
    with caplog.at_level(logging.WARNING, logger="reax.config"):
        load_validator({**FINNEY, "REAX_COMPETITIONS_MODE": "off"})
    assert "100% inference" in caplog.text or "100%% inference" in caplog.text
    with pytest.raises(ConfigError, match="fixed by the release"):
        load_validator({**FINNEY, "REAX_COMPETITIONS_MODE": "off", "REAX_FEATURE_ENGINE_COMPETITION": "0"})


def test_remote_needs_only_the_state_dir_and_no_docker_data(tmp_path):
    env = {**FINNEY, "REAX_COMPETITIONS_MODE": "remote"}
    with pytest.raises(ConfigError, match="REAX_COMPETITIONS_STATE_DIR"):
        load_validator(env)
    with pytest.raises(ConfigError, match="absolute"):
        load_validator({**env, "REAX_COMPETITIONS_STATE_DIR": "relative/dir"})
    cfg = load_validator({**env, "REAX_COMPETITIONS_STATE_DIR": str(tmp_path)})     # no weights dir, no sealed dir, no docker
    assert cfg.competitions_mode == "remote" and cfg.competitions_state_dir == str(tmp_path)
    e = cfg.engine
    assert e.runner == "remote" and e.release_mode and e.state_path == str(tmp_path / ENGINE_STATE_FILE)
    assert e.reference == ENGINE_REFERENCE_IMAGE and cfg.spec_sealed_dir == "" and e.workdir == ""
    with pytest.raises(ConfigError, match="fixed by the release"):
        load_validator({**env, "REAX_COMPETITIONS_STATE_DIR": str(tmp_path), "REAX_FEATURE_ENGINE_COMPETITION": "1"})
    with pytest.raises(ConfigError, match="REAX_MECHANISM=s1"):
        load_validator({**env, "REAX_COMPETITIONS_STATE_DIR": str(tmp_path), "REAX_MECHANISM": "mpm1"})


def test_remote_ignores_worker_variables_with_a_warning(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="reax.config"):
        cfg = load_validator({**FINNEY, "REAX_COMPETITIONS_MODE": "remote", "REAX_COMPETITIONS_STATE_DIR": str(tmp_path),
                              "REAX_ENGINE_WEIGHTS_DIR": "/nowhere", "REAX_SPEC_SEALED_DIR": "/nowhere"})
    assert cfg.engine.runner == "remote" and "ignored by the validator" in caplog.text


def test_remote_on_a_development_network_needs_no_flag(tmp_path):
    cfg = load_validator({"REAX_NETUID": "9", "REAX_NETWORK": "ws://127.0.0.1:9951", "REAX_COMPETITIONS_MODE": "remote",
                          "REAX_COMPETITIONS_STATE_DIR": str(tmp_path)})
    assert cfg.engine.runner == "remote" and not cfg.engine.release_mode


def test_the_validator_in_remote_mode_builds_no_docker_runner(tmp_path, monkeypatch):
    from reax_subnet import cli

    def boom(*a, **k):
        raise AssertionError("a remote validator must not construct a docker runner")

    monkeypatch.setattr(ER, "DockerRunner", boom)
    monkeypatch.setattr(ER, "DevRunner", boom)
    cfg = load_validator({**FINNEY, "REAX_COMPETITIONS_MODE": "remote", "REAX_COMPETITIONS_STATE_DIR": str(tmp_path)})
    engine, driver = cli._engine_competition(cfg)
    assert isinstance(engine, RM.RemoteEngineCompetition) and isinstance(driver.spec, RM.RemoteSpecAdapter)
    assert engine.state_path == str(tmp_path / ENGINE_STATE_FILE) and driver.spec.evaluator.cfg.state_path == str(tmp_path / SPEC_STATE_FILE)
    assert not driver.due() and driver.kick(snap(), set()) is None            # no cycle is ever started
    assert engine.start_utc == COMPETITIONS_START_UTC


# ------------------------------------------------------------------------------------------------ worker
def worker_env(tmp_path, **over):
    env = {"REAX_NETUID": "9", "REAX_NETWORK": "ws://127.0.0.1:9951", "REAX_ALLOW_DEV_MODELS": "1",
           "REAX_COMPETITIONS_STATE_DIR": str(tmp_path / "state"), "REAX_ENGINE_WORKDIR": str(tmp_path / "eng"),
           "REAX_SPEC_WORK_DIR": str(tmp_path / "specwork"), "REAX_ENGINE_RUNNER": "dev", "REAX_ENGINE_DEV_MAP": str(tmp_path / "map.json"),
           "REAX_ENGINE_REFERENCE": ref_of("reference"), "REAX_SPEC_SEALED_DIR": str(tmp_path / "sealed"),
           "REAX_ENGINE_EVAL_INTERVAL_S": "30"}
    env.update(over)
    return {k: v for k, v in env.items() if v is not None}


def test_worker_config_needs_no_wallet_or_token_and_demands_its_parts(tmp_path):
    (tmp_path / "sealed").mkdir()
    wc = W.load_worker(worker_env(tmp_path))
    assert wc.run_engine and wc.run_spec and wc.engine.state_path == str(tmp_path / "state" / ENGINE_STATE_FILE)
    assert wc.spec_sealed_dir == str(tmp_path / "sealed")
    only_spec = W.load_worker(worker_env(tmp_path, REAX_ENGINE_RUNNER=None, REAX_ENGINE_DEV_MAP=None, REAX_ENGINE_REFERENCE=None), engine=False)
    assert not only_spec.run_engine and only_spec.run_spec                      # no engine weights, no reference needed
    only_engine = W.load_worker(worker_env(tmp_path, REAX_SPEC_SEALED_DIR=None), spec=False)
    assert only_engine.run_engine and not only_engine.run_spec and only_engine.spec_sealed_dir == ""
    with pytest.raises(ConfigError, match="REAX_SPEC_SEALED_DIR"):
        W.load_worker(worker_env(tmp_path, REAX_SPEC_SEALED_DIR=None))
    with pytest.raises(ConfigError, match="REAX_COMPETITIONS_STATE_DIR"):
        W.load_worker(worker_env(tmp_path, REAX_COMPETITIONS_STATE_DIR=None))
    with pytest.raises(ConfigError, match="nothing to do"):
        W.load_worker(worker_env(tmp_path), engine=False, spec=False)
    docker = {"REAX_ENGINE_RUNNER": "docker", "REAX_ENGINE_DEV_MAP": None}
    with pytest.raises(ConfigError, match="WEIGHTS_DIR"):
        W.load_worker({**worker_env(tmp_path, **docker), "REAX_ENGINE_REFERENCE": f"ghcr.io/x/r@sha256:{D}"})


def test_worker_private_directories_must_not_be_shown_to_the_validator(tmp_path):
    (tmp_path / "sealed").mkdir()
    for name in ("REAX_ENGINE_WORKDIR", "REAX_SPEC_WORK_DIR"):
        with pytest.raises(ConfigError, match="must not be inside"):
            W.load_worker(worker_env(tmp_path, **{name: str(tmp_path / "state" / "x")}))
    inner = tmp_path / "state" / "sealed"
    inner.mkdir(parents=True)
    with pytest.raises(ConfigError, match="REAX_SPEC_SEALED_DIR"):
        W.load_worker(worker_env(tmp_path, REAX_SPEC_SEALED_DIR=str(inner)))


def test_worker_on_finney_uses_the_release_reference_and_never_a_wallet(tmp_path):
    (tmp_path / "w").mkdir()
    (tmp_path / "sealed").mkdir()
    env = {"REAX_NETUID": "92", "REAX_NETWORK": "finney", "REAX_COMPETITIONS_STATE_DIR": str(tmp_path / "state"),
           "REAX_ENGINE_WEIGHTS_DIR": str(tmp_path / "w"), "REAX_SPEC_SEALED_DIR": str(tmp_path / "sealed"),
           "REAX_ENGINE_WORKDIR": str(tmp_path / "e"), "REAX_SPEC_WORK_DIR": str(tmp_path / "sw")}
    wc = W.load_worker(env)
    assert wc.engine.reference == ENGINE_REFERENCE_IMAGE and wc.engine.release_mode and wc.common.runtime_token_file == ""
    assert not any(k in env for k in ("REAX_WALLET_NAME", "REAX_WALLET_PATH", "REAX_RUNTIME_TOKEN_FILE"))


def test_touch_state_refreshes_saved_at_atomically(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"saved_at": 5, "version": 1, "keep": [1, 2]}))
    assert W.touch_state(str(p), now=1234)
    assert json.load(open(p)) == {"saved_at": 1234, "version": 1, "keep": [1, 2]}
    assert oct(os.stat(p).st_mode & 0o777) == "0o644" and os.listdir(tmp_path) == ["x.json"]
    assert not W.touch_state(str(tmp_path / "missing.json"))
    p.write_text("garbage")
    assert not W.touch_state(str(p))


class FakeChain:
    def __init__(self, snapshot, fail_first=0):
        self.snapshot_value, self.calls, self.fail_first = snapshot, 0, fail_first

    def snapshot(self):
        self.calls += 1
        if self.calls <= self.fail_first:
            raise RuntimeError("rpc down")
        return self.snapshot_value


@pytest.fixture
def dev_map(tmp_path):
    mapping = {ref_of(v): [sys.executable, FAKE, "--variant", v, "--delay", "0.01", "--jitter", "0.0"] for v in ("reference", "fast")}
    json.dump(mapping, open(tmp_path / "map.json", "w"))
    return mapping


async def test_worker_runs_engine_and_spec_cycles_and_writes_both_files(tmp_path, dev_map):
    (tmp_path / "sealed").mkdir()
    env = worker_env(tmp_path, REAX_ENGINE_DEV_PARAMS=json.dumps({"track": {
        "timed_items": 48, "latency_items": 8, "probe_pairs": 8, "determinism_items": 8, "start_timeout_s": 30,
        "min_top_agreement": 0.95, "confirmations": 1, "reference_drift": 0.9}}))
    wc = W.load_worker(env)
    fast = ref_of("fast")
    s = snap(eng_digest=fast.rsplit(":", 1)[1])
    s = Snapshot(block=900, neurons=tuple(dataclasses.replace(n, commitment="reax-eng/1:s1-fast:" + fast) if n.uid == 3 else n
                                          for n in s.neurons), owner_hotkey="owner")
    # the spec evaluator is replaced by a fake that only records that it was driven (its docker/GPU path is tested elsewhere)
    calls = []

    class FakeSpec:
        cfg = SC.SpecConfig(sealed_dir="", work_dir="", state_path="")
        clock = staticmethod(lambda: NOW0.timestamp())
        state = ST.SpecState()

        async def evaluate_cycle(self, snapshot):
            calls.append("spec")
            self.state.track(T_SPEC).events.append([1, "cycle"])
            return {}

        def leaders(self, active, snapshot):
            return {}

    adapter = SpecialistAdapter(FakeSpec(), free_mib=_free(10 ** 6))
    engine, _a, _e = W.build_local_stack(wc.engine, "", os.environ, run_engine=True)
    engine.specialist = adapter
    engine.start_utc = None
    adapter.evaluator.cfg = dataclasses.replace(SC.SpecConfig(sealed_dir="", work_dir="", state_path=""), tracks=SC.DEFAULT_TRACKS)
    await W.run_worker(wc, FakeChain(s), once=True, stack=(engine, adapter, adapter.evaluator))
    ep, sp = tmp_path / "state" / ENGINE_STATE_FILE, tmp_path / "state" / SPEC_STATE_FILE
    assert ep.exists() and sp.exists() and calls == ["spec"]
    raw = json.load(open(ep))
    assert time.time() - raw["saved_at"] < 120 and raw["binds_hotkey"] is False
    assert raw["tracks"]["s1-fast"]["champion"]["uid"] == 3                        # the fast engine won
    assert oct(os.stat(ep).st_mode & 0o777) == "0o644"
    # what the worker wrote is exactly what a remote validator pays from
    remote = RM.RemoteEngineCompetition(str(ep), start_utc=None, tracks={"s1-fast": FAST_P})
    assert remote.weights(SERVING, s, {0})[3] > 0.19


async def test_a_spec_only_worker_never_writes_the_engine_file(tmp_path, dev_map):
    (tmp_path / "sealed").mkdir()
    wc = W.load_worker(worker_env(tmp_path, REAX_ENGINE_RUNNER=None, REAX_ENGINE_DEV_MAP=None, REAX_ENGINE_REFERENCE=None), engine=False)
    engine, adapter, ev = W.build_local_stack(wc.engine, wc.spec_sealed_dir, os.environ | {"REAX_SPEC_WORK_DIR": str(tmp_path / "specwork")},
                                              run_engine=False, state_dir=wc.state_dir)
    assert isinstance(engine, W.IdleEngine) and not engine.due()
    await W.run_worker(wc, FakeChain(snap()), once=True, stack=(engine, adapter, ev))
    assert os.listdir(tmp_path / "state") == [SPEC_STATE_FILE]


async def test_chain_failure_is_survived_and_the_heartbeat_keeps_files_fresh(tmp_path, dev_map):
    (tmp_path / "sealed").mkdir()
    wc = dataclasses.replace(W.load_worker(worker_env(tmp_path)), poll_s=1, heartbeat_s=1)
    engine, adapter, ev = W.build_local_stack(wc.engine, "", os.environ, run_engine=True)
    chain = FakeChain(snap(), fail_first=1)
    task = asyncio.ensure_future(W.run_worker(wc, chain, stack=(engine, None, None)))
    ep = tmp_path / "state" / ENGINE_STATE_FILE
    for _ in range(60):
        await asyncio.sleep(0.1)
        if ep.exists():
            break
    first = json.load(open(ep))["saved_at"]
    old = first - 100
    raw = json.load(open(ep))
    raw["saved_at"] = old
    json.dump(raw, open(ep, "w"))
    await asyncio.sleep(1.6)                                                       # one heartbeat later
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert json.load(open(ep))["saved_at"] >= first and chain.calls >= 1


def _free(mib):
    async def f(gpu):
        return mib
    return f


def _compose(*parts):
    yaml = pytest.importorskip("yaml")
    root = os.path.join(os.path.dirname(__file__), "..")
    with open(os.path.join(root, *parts)) as fh:
        return yaml.safe_load(fh)["services"]


def _text(svc):
    return json.dumps(svc)


def test_validator_compose_keeps_the_socket_and_sealed_data_out_of_the_validator():
    svcs = _compose("deploy", "validator-1.2", "compose.yml")
    v, w = svcs["validator"], svcs["competition-worker"]
    assert v["environment"]["REAX_COMPETITIONS_MODE"] == "remote" and v["build"]["target"] == "validator"
    vt = _text(v)
    for forbidden in ("docker.sock", "group_add", "REAX_SPEC_SEALED_DIR", "REAX_ENGINE_WEIGHTS_DIR", "REAX_COMPETITION_DIR}:", "sealed"):
        assert forbidden not in vt, forbidden
    state_mount = [m for m in v["volumes"] if m.endswith(":/competitions-state:ro")]
    assert len(state_mount) == 1 and "/state:/competitions-state:ro" in state_mount[0]
    assert v["environment"]["REAX_COMPETITIONS_STATE_DIR"] == "/competitions-state"
    # the worker: socket, sealed dir read-only, work dir, state dir; no wallets, no tokens, not the host network
    wt = _text(w)
    assert any(m.startswith("/var/run/docker.sock:") for m in w["volumes"])
    assert any(m.endswith(":ro") and "REAX_SPEC_SEALED_DIR" in m for m in w["volumes"])
    for forbidden in ("wallets", "runtime-token", "REAX_RUNTIME_TOKEN_FILE", "router", "secrets"):
        assert forbidden not in wt, forbidden
    assert w.get("network_mode") is None and w["build"]["target"] == "competition-worker"
    assert w["environment"]["REAX_COMPETITIONS_STATE_DIR"].endswith("/state") and w["read_only"] is True
    assert "ALL" in w["cap_drop"]
    assert not any("docker.sock" in m for k, s in svcs.items() if k != "competition-worker" for m in s.get("volumes", []))


def test_keyless_box_compose_has_no_secrets():
    w = _compose("deploy", "competition-worker", "compose.yml")["competition-worker"]
    wt = _text(w)
    for forbidden in ("wallets", "runtime-token", "REAX_WALLET", "REAX_RUNTIME_TOKEN_FILE", "router"):
        assert forbidden not in wt, forbidden
    assert w["command"] == ["--engine", "--spec"] and w["build"]["target"] == "competition-worker"
    assert any(m.startswith("/var/run/docker.sock:") for m in w["volumes"])


def test_dockerfile_has_the_worker_target_and_the_validator_target_has_no_docker_cli():
    root = os.path.join(os.path.dirname(__file__), "..")
    text = open(os.path.join(root, "Dockerfile")).read()
    stages = {}
    cur = None
    for line in text.splitlines():
        if line.startswith("FROM "):
            cur = line.split(" AS ")[-1].strip()
            stages[cur] = []
        elif cur and not line.lstrip().startswith("#"):
            stages[cur].append(line)
    assert "docker:27-cli" in "\n".join(stages["competition-worker"]) and 'ENTRYPOINT ["reax-competition-worker"]' in stages["competition-worker"]
    assert "docker:27-cli" not in "\n".join(stages["validator"])
    pyproject = open(os.path.join(root, "pyproject.toml")).read()
    assert 'reax-competition-worker = "reax_subnet.worker:main"' in pyproject
