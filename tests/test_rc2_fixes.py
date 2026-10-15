"""Fixes for the review findings on 1.2.0rc2 (H1-H5, M1, M2, M4, M6, M7, L1, L2)."""

import dataclasses
import hashlib
import json
import logging
import os
import sys

import pytest
from test_engine_hardening import FakeDocker, manifest
from test_remote_competitions import (D, ENGINE_STATE_FILE, NOW0, SERVING, SPEC_STATE_FILE, T_SPEC, remote_engine, snap, spec_state,
                                      stamp, write_engine_state)
from test_spec_evaluator import env  # noqa: F401 - the fixture

from reax_subnet import engrunner as ER
from reax_subnet import remote as RM
from reax_subnet import worker as W
from reax_subnet.config import load_validator
from reax_subnet.engcomp import Champion, TrackParams, TrackState
from reax_subnet.engrunner import DockerRunner
from reax_subnet.spec import fetch as F
from reax_subnet.spec import rules as R
from reax_subnet.spec import state as ST
from reax_subnet.spec import templates as TPL
from reax_subnet.spec import vllm as V
from reax_subnet.spec.vllm import HostUnavailable, ModelUnavailable

D1 = hashlib.sha256(b"a").hexdigest()
REF = f"ghcr.io/a/e@sha256:{D1}"
P = TrackParams("s1-fast", pool="s1-fast", temperature=2.07)
DATA = os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture
def runner(tmp_path):
    return DockerRunner(str(tmp_path / "w"), str(tmp_path))


# ------------------------------------------------------------------------------------------------ H1
async def test_h1_missing_nvidia_smi_skips_the_preflight_instead_of_crashing(runner, monkeypatch, caplog):
    async def no_smi(argv, t):
        if argv[0] == "nvidia-smi":
            raise FileNotFoundError(2, "No such file or directory", "nvidia-smi")
        return 0, "", ""

    monkeypatch.setattr(ER, "_run", no_smi)
    with caplog.at_level(logging.WARNING, logger="reax.engines"):
        await runner.preflight()
    assert "nvidia-smi" in caplog.text


async def test_h1_and_m6_missing_nvidia_smi_means_no_gpu_busy_verdict(monkeypatch):
    async def run(argv, t):
        if argv[0] == "nvidia-smi":
            raise FileNotFoundError("nvidia-smi")
        return 0, "PID\n4242\n", ""

    monkeypatch.setattr(ER, "_run", run)
    assert await ER.candidate_gpu_busy("c", "0") is False


def test_h1_worker_service_sees_the_gpu_for_nvidia_smi_only():
    yaml = pytest.importorskip("yaml")
    root = os.path.join(os.path.dirname(__file__), "..", "deploy")
    for f in ("validator-1.2/compose.yml", "competition-worker/compose.yml"):
        svc = yaml.safe_load(open(os.path.join(root, f)))["services"]["competition-worker"]
        dev = svc["deploy"]["resources"]["reservations"]["devices"][0]
        assert dev["driver"] == "nvidia" and dev["capabilities"] == ["utility"]          # no compute capability
        assert svc["environment"]["HOME"] == "/tmp"                                      # L2


# ------------------------------------------------------------------------------------------------ H2
def test_h2_vllm_is_started_through_a_umask_wrapper():
    cmd = V.build_docker_command(V.ServeSpec("n", "/m", "/s", 0.3, template="qwen.jinja", aux_dir="/a"))
    i = cmd.index(V.VLLM_IMAGE)
    assert cmd[cmd.index("--entrypoint") + 1] == "sh" and cmd.index("--entrypoint") < i
    assert cmd[i + 1:i + 4] == ["-c", 'umask 000; exec vllm serve "$@"', "--"]
    assert cmd[i + 4] == "/model"                                                         # the vllm arguments follow unchanged
    assert "--uds" in cmd


async def test_h2_unconnectable_socket_is_a_host_problem_and_fails_fast(tmp_path, monkeypatch):
    b = V.VllmDockerBackend(str(tmp_path), 1, gpu_total_gb=96.0)

    class Http:
        def get(self, url):
            raise PermissionError(13, "Permission denied")

    async def run(*cmd, timeout=60):
        return 0, "true"

    monkeypatch.setattr(V, "_run", run)
    b._http = Http()
    with pytest.raises(HostUnavailable, match="not connectable"):
        await b._wait_healthy()


async def _flaky(env, bad, exc):
    real = env.ev.backend_factory

    def factory(model_dir, wb):
        if any(r in open(os.path.join(model_dir, "skill_key.txt")).read() for r in bad):
            raise exc
        return real(model_dir, wb)

    env.ev.backend_factory = factory


async def test_h2_host_failures_never_count_toward_rejection(env):
    env.add(1, "a/m1", 0.9)
    await _flaky(env, ["a/m1"], HostUnavailable("socket not connectable"))
    for _ in range(6):
        await env.cycle({1: "a/m1"})
    assert env.sub(1).attempts == 0 and env.sub(1).status == ST.QUEUED


async def test_h2_model_failures_still_reject_after_max_attempts(env):
    from reax_subnet.spec import config as C
    env.add(1, "a/m1", 0.9)
    await _flaky(env, ["a/m1"], ModelUnavailable("out of memory"))
    for _ in range(C.MAX_ATTEMPTS):
        await env.cycle({1: "a/m1"})
    assert env.sub(1).status == ST.REJECTED and "repeated attempts" in env.sub(1).reason


async def test_h2_a_group_where_every_model_fails_is_a_host_problem(env):
    env.add(1, "a/m1", 0.9)
    env.add(2, "a/m2", 0.8)
    await _flaky(env, ["a/m1", "a/m2"], ModelUnavailable("container exited"))
    for _ in range(5):
        await env.cycle({1: "a/m1", 2: "a/m2"})
    assert env.sub(1).attempts == 0 and env.sub(2).attempts == 0
    assert env.sub(1).status != ST.REJECTED and env.sub(2).status != ST.REJECTED


async def test_h2_one_failing_model_among_working_ones_still_counts(env):
    env.add(1, "a/m1", 0.9)
    env.add(2, "a/m2", 0.8)
    await _flaky(env, ["a/m2"], ModelUnavailable("container exited"))
    await env.cycle({1: "a/m1", 2: "a/m2"})
    assert env.sub(2).attempts == 1 and env.sub(1).attempts == 0


# ------------------------------------------------------------------------------------------------ H3
def test_h3_oci_manifest_and_attestation_entries():
    doc = json.load(open(os.path.join(DATA, "docker_manifest_inspect_oci_alpine.json")))     # real `docker manifest inspect -v alpine:3.20`
    assert all("OCIManifest" in e and "SchemaV2Manifest" not in e for e in doc)
    assert ER.manifest_compressed_size(doc) == 612 + 3630321                                  # linux/amd64; attestations skipped
    assert ER.manifest_compressed_size(doc[0]) == 612 + 3630321                              # a single OCI entry
    only_attestation = [e for e in doc if e["Descriptor"]["platform"]["os"] == "unknown"]
    with pytest.raises(ValueError):
        ER.manifest_compressed_size(only_attestation)


def test_h3_attestation_sizes_never_count():
    att = {"Descriptor": {"platform": {"os": "unknown", "architecture": "unknown"}},
           "OCIManifest": {"config": {"size": 10 ** 12}, "layers": [{"size": 1}]}}
    assert ER.manifest_compressed_size([manifest([5]), att]) == 1005
    index = {"mediaType": "application/vnd.oci.image.index.v1+json", "manifests": [{}]}
    with pytest.raises(ValueError):
        ER.manifest_compressed_size(index)


async def test_h3_oci_image_is_pulled_not_marked_failed(runner, monkeypatch):
    doc = json.load(open(os.path.join(DATA, "docker_manifest_inspect_oci_alpine.json")))
    d = FakeDocker(doc)
    monkeypatch.setattr(ER, "_run", d)
    assert await runner.owner_of(REF) == "hk-a" and len(d.cmds("docker", "pull")) == 1


# ------------------------------------------------------------------------------------------------ H4
def test_h4_drift_is_one_sided():
    ER._check_drift(100.0, 100.0, P)
    ER._check_drift(150.0, 100.0, P)                                       # faster than the baseline: harmless
    ER._check_drift(88.5, 100.0, P)
    for tp in (87.0, 50.0):
        with pytest.raises(ER.ReferenceError_, match="below its solo baseline"):
            ER._check_drift(tp, 100.0, P)
    ER._check_latency_drift(0.010, 0.010, P)
    ER._check_latency_drift(0.005, 0.010, P)                               # a faster reference is harmless
    ER._check_latency_drift(0.0111, 0.010, P)
    with pytest.raises(ER.ReferenceError_, match="above its solo baseline"):
        ER._check_latency_drift(0.0113, 0.010, P)
    ER._check_latency_drift(1.0, None, P)


async def test_h4_a_slowed_reference_does_not_raise_the_score(tmp_path, monkeypatch):
    from test_engcomp import FAKE, FAST_P, ref_of
    mapping = {ref_of(v): [sys.executable, FAKE, "--variant", v, "--delay", "0.01", "--jitter", "0.0"] for v in ("reference", "fast")}
    runner = ER.DevRunner(str(tmp_path / "eng"), mapping)
    p = dataclasses.replace(FAST_P, reference_drift=0.9)

    async def base(ref, params):
        return 1e6                                                          # the reference was "much faster" alone

    # with the real solo baseline the score is the plain ratio ...
    honest = await ER.evaluate(runner, ref_of("reference"), ref_of("fast"), "s1-fast", p)
    assert honest.ok
    # ... and with a huge baseline (the reference runs "slowed" relative to it) the denominator is the baseline, so the
    # candidate gains nothing from a slowed reference: its score collapses instead of inflating
    monkeypatch.setattr(ER, "reference_baseline", base)
    monkeypatch.setattr(ER, "_check_drift", lambda *a: None)
    monkeypatch.setattr(ER, "_check_latency_drift", lambda *a: None)
    slowed = await ER.evaluate(runner, ref_of("reference"), ref_of("fast"), "s1-fast", p)
    assert slowed.score < honest.score and slowed.score < 0.01


# ------------------------------------------------------------------------------------------------ H5
FINNEY = {"REAX_NETUID": "92", "REAX_NETWORK": "finney"}


def test_h5_finney_never_runs_competitions_in_process_unless_asked_explicitly(tmp_path, caplog):
    w, s = tmp_path / "w", tmp_path / "s"
    w.mkdir(), s.mkdir()
    rc1_env = {**FINNEY, "REAX_ENGINE_WEIGHTS_DIR": str(w), "REAX_SPEC_SEALED_DIR": str(s)}      # a 1.2.0rc1 compose
    with caplog.at_level(logging.ERROR, logger="reax.config"):
        cfg = load_validator(rc1_env)
    assert cfg.engine is None and cfg.spec_sealed_dir == "" and cfg.competitions_mode == "off"
    assert "REAX_COMPETITIONS_MODE=local" in caplog.text
    cfg = load_validator({**rc1_env, "REAX_COMPETITIONS_MODE": "local"})                            # explicit: as before
    assert cfg.competitions_mode == "local" and cfg.engine is not None and cfg.engine.runner == "docker"
    assert load_validator(FINNEY).engine is None                                                    # nothing configured: serving-only
    half = load_validator({**FINNEY, "REAX_ENGINE_WEIGHTS_DIR": str(w)})                           # incomplete local setup: serving-only
    assert half.engine is None
    assert load_validator({**FINNEY, "REAX_COMPETITIONS_MODE": "remote", "REAX_COMPETITIONS_STATE_DIR": str(tmp_path)}).engine.runner == "remote"


def test_h5_development_networks_keep_local_as_the_default():
    assert load_validator({"REAX_NETUID": "92", "REAX_NETWORK": "ws://127.0.0.1:9944"}).competitions_mode == "local"


# ------------------------------------------------------------------------------------------------ M1 / L1
def test_l1_non_finite_or_huge_saved_at_is_rejected(tmp_path):
    p = tmp_path / "s.json"
    for text in ('{"saved_at": NaN}', '{"saved_at": Infinity}', '{"saved_at": -Infinity}', '{"saved_at": 1' + "0" * 400 + "}"):
        p.write_text(text)
        raw, why = RM.read_state(str(p), now=1e6)
        assert raw is None and ("finite" in why or "range" in why), text


def test_m1_type_corrupt_engine_state_pays_inference_and_never_raises(tmp_path):
    path = tmp_path / ENGINE_STATE_FILE
    for champ in ({"digest": D, "hotkey": "hk-eng", "uid": 3, "score": 1.5, "crowned_block": "x"},
                  {"digest": D, "hotkey": "hk-eng", "uid": 3, "score": 1.5, "crowned_block": None},
                  {"digest": D, "hotkey": "hk-eng", "uid": "3", "score": 1.5, "crowned_block": 900},
                  {"digest": D, "hotkey": ["hk"], "uid": 3, "score": 1.5, "crowned_block": 900},
                  {"digest": D, "hotkey": "hk-eng", "uid": 3, "score": "abc", "crowned_block": 900},
                  {"digest": D, "hotkey": "hk-eng", "uid": 3, "score": float("nan"), "crowned_block": 900},
                  {"digest": D, "hotkey": "hk-eng", "uid": True, "score": 1.5, "crowned_block": 900},
                  {"digest": D}, [1], "champion"):
        write_engine_state(path)
        raw = json.load(open(path))
        raw["tracks"]["s1-fast"]["champion"] = champ
        json.dump(raw, open(path, "w"), allow_nan=True)
        assert remote_engine(path).weights(SERVING, snap(), {0}) == pytest.approx(SERVING), champ


def test_m1_champion_from_json_accepts_a_good_record_and_ints_as_scores():
    c = Champion.from_json({"digest": D, "hotkey": "hk", "uid": 3, "score": 2, "crowned_block": 900})
    assert c == Champion(D, "hk", 3, 2.0, 900)
    with pytest.raises(ValueError):
        TrackState.from_json({"challenger": [1]})


def test_m1_weights_never_raises_whatever_the_payment_code_does(tmp_path, monkeypatch, caplog):
    path = tmp_path / ENGINE_STATE_FILE
    write_engine_state(path)
    remote = remote_engine(path)
    from reax_subnet.engvalidator import EngineCompetition
    real = EngineCompetition.weights
    calls = []

    def boom(self, base, snap_, exclude, burn=0.0):
        calls.append(1)
        if len(calls) == 1:
            raise TypeError("garbage in the state")
        return real(self, base, snap_, exclude, burn)

    monkeypatch.setattr(EngineCompetition, "weights", boom)
    with caplog.at_level(logging.WARNING, logger="reax.remote"):
        out = remote.weights(SERVING, snap(), {0})
    assert out == pytest.approx(SERVING) and "unusable" in caplog.text
    monkeypatch.setattr(EngineCompetition, "weights", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    assert remote.weights(SERVING, snap(), {0}) == dict(SERVING)


def test_m1_saved_at_overflow_in_a_state_file_pays_inference(tmp_path):
    path = tmp_path / ENGINE_STATE_FILE
    write_engine_state(path)
    text = open(path).read().replace('"saved_at":', '"saved_at": 1' + "0" * 400 + ', "x":', 1)
    open(path, "w").write(text)
    assert remote_engine(path).weights(SERVING, snap(), {0}) == pytest.approx(SERVING)


# ------------------------------------------------------------------------------------------------ M2
def test_m2_corrupt_spec_state_stops_paying_the_old_leaders(tmp_path, caplog):
    path = tmp_path / SPEC_STATE_FILE
    spec_state(path)
    stamp(path, NOW0.timestamp())
    ev = RM.RemoteSpecEvaluator(str(path), clock=lambda: NOW0.timestamp())
    s = snap(specs=[(20, "a/m1"), (21, "a/m2"), (22, "a/m3")])
    active = ev.cfg.active_tracks(NOW0.timestamp())
    assert ev.leaders(active, s)[T_SPEC]
    from reax_subnet.remote import RemoteSpecAdapter
    ad = RemoteSpecAdapter(ev)
    ad.refresh(s)
    assert ad.track_results([T_SPEC], NOW0)
    raw = json.load(open(path))
    sub = next(iter(raw["tracks"][T_SPEC]["submissions"].values()))
    sub["score"] = "abc"
    json.dump(raw, open(path, "w"))
    with caplog.at_level(logging.WARNING, logger="reax.remote"):
        assert ev.leaders(active, s) == {t.track_id: [] for t in active}
    assert "invalid content" in caplog.text
    ad.refresh(s)
    assert not ad.track_results([T_SPEC], NOW0)                            # the stale leaders are gone


# ------------------------------------------------------------------------------------------------ M4
async def test_m4_only_images_pulled_in_this_session_are_removed(runner, monkeypatch):
    d = FakeDocker(manifest([1]), present=True)                              # already on the host: not ours
    monkeypatch.setattr(ER, "_run", d)
    assert await runner.owner_of(REF) == "hk-a"
    await runner.discard(REF)
    assert d.cmds("docker", "rmi") == []
    d = FakeDocker(manifest([1]))                                            # pulled now: ours
    monkeypatch.setattr(ER, "_run", d)
    await runner.owner_of(REF)
    await runner.discard(REF)
    await runner.discard(REF)                                                # once only
    assert len(d.cmds("docker", "rmi", REF)) == 1


async def test_m4_configured_images_are_never_removed(tmp_path, monkeypatch):
    r = DockerRunner(str(tmp_path / "w"), str(tmp_path), protected_images=[REF])
    d = FakeDocker(manifest([1]), run_code=125)
    monkeypatch.setattr(ER, "_run", d)
    with pytest.raises(ER.InfraError):
        await r.start(REF, "s1-fast", P)                                     # pulled (not on the host) and the start failed
    assert d.cmds("docker", "rmi") == []


async def test_m4_the_stop_of_an_engine_removes_only_what_the_runner_pulled(runner, monkeypatch):
    d = FakeDocker(manifest([1]), present=True)
    monkeypatch.setattr(ER, "_run", d)
    eng = await runner.start(REF, "s1-fast", P)
    await eng.stop()
    assert d.cmds("docker", "rmi") == []
    d = FakeDocker(manifest([1]))
    monkeypatch.setattr(ER, "_run", d)
    eng = await runner.start(REF, "s1-fast", P)
    await eng.stop()
    assert len(d.cmds("docker", "rmi", REF)) == 1


def test_m4_the_worker_protects_the_reference_and_the_vllm_image(tmp_path):
    from reax_subnet.config import _engine, load_common
    wd = tmp_path / "w"
    sd = tmp_path / "s"
    wd.mkdir(), sd.mkdir()
    env = {**FINNEY, "REAX_ENGINE_WEIGHTS_DIR": str(wd), "REAX_SPEC_SEALED_DIR": str(sd), "REAX_ENGINE_WORKDIR": str(tmp_path / "e"),
           "REAX_ENGINE_STATE_PATH": str(tmp_path / "e.json")}
    e, _ = _engine(env, load_common(env))
    engine, _, _ = W.build_local_stack(e, "", env)
    assert engine.runner.protected_images == frozenset({e.reference, V.VLLM_IMAGE})


# ------------------------------------------------------------------------------------------------ M7
def tokens_of(*names):
    return F.tokenizer_tokens({"added_tokens_decoder": {str(i): {"content": n} for i, n in enumerate(names)}}, None)


def test_m7_tokenizer_must_define_the_markers_of_the_chosen_template():
    qwen = {"architectures": ["Qwen3ForCausalLM"]}
    llama = {"architectures": ["LlamaForCausalLM"]}
    gemma = {"architectures": ["Gemma3ForCausalLM"]}
    assert R.check_tokenizer(qwen, tokens_of("<|im_start|>", "<|im_end|>")) == []
    assert R.check_tokenizer(gemma, tokens_of("<start_of_turn>", "<end_of_turn>", "<bos>")) == []
    assert R.check_tokenizer(llama, tokens_of("<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>")) == []
    (msg,) = R.check_tokenizer(llama, tokens_of("<|im_start|>", "<|im_end|>"))                     # a ChatML fine-tune of Llama
    assert "<|start_header_id|>" in msg and "<|eot_id|>" in msg and "chat" in msg
    assert R.check_tokenizer(qwen, tokens_of("<start_of_turn>"))
    assert R.check_tokenizer(qwen, frozenset())                                                  # nothing defined at all
    assert R.check_tokenizer(qwen, None) == []                                                   # not collected (unit fixtures)
    assert set(TPL.TEMPLATE_MARKERS) == set(TPL.TEMPLATE_FOR_ARCH.values())


def test_m7_tokenizer_tokens_come_from_added_tokens_and_the_vocabulary():
    t = F.tokenizer_tokens({"added_tokens_decoder": {"1": {"content": "<a>"}, "2": "junk"}},
                           {"added_tokens": [{"content": "<b>"}, 5], "model": {"vocab": {"<c>": 0, "d": 1}}})
    assert t == {"<a>", "<b>", "<c>", "d"}
    assert F.tokenizer_tokens(None, None) == frozenset()
    assert F.tokenizer_tokens({"added_tokens_decoder": []}, {"added_tokens": "x", "model": {"vocab": []}}) == frozenset()


def test_m7_a_model_without_the_markers_is_refused_before_any_gpu_run(tmp_path):
    from spec_helpers import write_model, make_weights
    p = str(tmp_path / "m")
    write_model(p, make_weights(1), arch="LlamaForCausalLM", hotkey="hk", category="demand_forecasting")
    ins = F.inspect_dir(p, category="demand_forecasting", hotkey="hk")
    assert any("<|start_header_id|>" in v for v in ins.violations)


def test_m7_templates_survive_a_tokenizer_without_bos_and_gemma4_is_not_accepted():
    jinja2 = pytest.importorskip("jinja2")
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    envj = ImmutableSandboxedEnvironment()
    for name in ("gemma3.jinja", "llama3.jinja"):
        tpl = envj.from_string(open(TPL.template_path(name)).read())
        out = tpl.render(messages=[{"role": "user", "content": "hi"}], add_generation_prompt=True, bos_token=None)
        assert "None" not in out and out
    assert not any(a.startswith("Gemma4") for a in R.ARCH_ALLOWLIST)


def test_large_tokenizer_json_is_read_for_the_marker_rule(tmp_path):
    """Real checkpoints (Qwen3.5 vocabulary) ship a ~20 MB tokenizer.json: above the 16 MiB config cap, below the tokenizer cap."""
    import json as _json
    from reax_subnet.spec import rules as R
    big = {"added_tokens": [{"id": 1, "content": "<|im_start|>"}, {"id": 2, "content": "<|im_end|>"}],
           "model": {"vocab": {f"t{i}": i for i in range(1_200_000)}}}
    p = tmp_path / "tokenizer.json"
    p.write_text(_json.dumps(big))
    assert R.MAX_JSON_BYTES < p.stat().st_size < R.MAX_TOKENIZER_JSON_BYTES
    assert R.read_json_file(str(p), R.MAX_TOKENIZER_JSON_BYTES)["added_tokens"][0]["content"] == "<|im_start|>"
