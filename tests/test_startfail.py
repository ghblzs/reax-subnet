"""Worker rc9.post1: round-based start failures, infrastructure attribution and rollback."""
import os
import importlib.util
import sys
from pathlib import Path

import httpx
import pytest
from test_spec_evaluator import Env, TRACK

from reax_subnet.spec import config as C, evaluator as E, fetch as F, state as ST, vllm as V

pytestmark = pytest.mark.usefixtures("rc5_entry_floors")


async def champion(tmp_path):
    env = Env(tmp_path, {}, sample=300)
    env.add(1, "a/champ", 0.95)
    await env.cycle({1: "a/champ"})
    env.clock.advance(13 * 3600)
    await env.cycle({1: "a/champ"})
    assert env.sub(1).status == ST.CHAMPION
    env.add(2, "a/next", 0.3)
    return env


def scripted(env, failures):
    """Raise the next scripted exception for the incumbent; preserve real rows for retry assertions."""
    calls = []
    real = env.server

    def factory(model_dir, wb):
        repo = Path(model_dir, "skill_key.txt").read_text()
        if repo == "a/champ" and failures:
            exc = failures.pop(0)
            if exc:
                calls.append((repo, exc))
                raise exc
        ctx = real(model_dir, wb)

        class Wrapped:
            async def __aenter__(self):
                backend = await ctx.__aenter__()
                decide = backend.decide

                async def record(rows):
                    calls.append((repo, rows))
                    return await decide(rows)
                backend.decide = record
                return backend

            async def __aexit__(self, *args):
                return await ctx.__aexit__(*args)
        return Wrapped()
    env.ev.backend_factory = factory
    return calls


def test_window_does_not_reset_on_success_and_is_bounded():
    sub = ST.Submission("s", 1, "h", "c", TRACK, "a/m", "r", 1, 0)
    ts = ST.TrackState(TRACK, submissions={sub.key: sub})
    for i, outcome in enumerate(["model", "ok", "model", "ok"]):
        E.SpecEvaluator._serve_round(ts, sub, outcome, i)
    assert sub.attempts == 2 and sub.status == ST.QUEUED
    assert sub.reason == "start-up failures: 2 of the last 6 rounds"
    for i in range(20):
        E.SpecEvaluator._serve_round(ts, sub, "ok", i + 4)
    assert sub.attempts == 0 and sub.reason == "" and len(sub.serve_log) == 12


async def test_alternating_champion_fail_ok_loses_crown_at_three_of_six(tmp_path):
    env = await champion(tmp_path)
    sub = env.sub(1)
    for i, fail in enumerate([True, False, True, False, True]):
        if i:
            # Keep scheduling a comparison, as a fresh challenger would, after each weak challenge loses.
            env.sub(2).status, env.sub(2).reason = ST.QUEUED, ""
        scripted(env, [V.ModelUnavailable("invalid tokenizer")] * 2 if fail else [])
        env.clock.advance(3601)
        await env.cycle({1: "a/champ", 2: "a/next"})
        assert sub.attempts == sum([True, False, True, False, True][:i+1])
        if i < 4:
            assert env.ev.state.track(TRACK).champion == sub.key
    assert sub.status == ST.REJECTED and not env.ev.state.track(TRACK).champion
    assert [entry[1] for entry in sub.serve_log[-5:]] == ["model", "ok", "model", "ok", "model"]


async def test_immediate_retry_success_is_one_ok_round_same_draw(tmp_path, monkeypatch):
    env = await champion(tmp_path)
    original = env.ev._serve
    rows_seen = []

    async def serve(p, rows, now):
        if p.sub.uid == 1:
            rows_seen.append(rows)
        return await original(p, rows, now)
    monkeypatch.setattr(env.ev, "_serve", serve)
    calls = scripted(env, [V.ModelUnavailable("container exited"), None])
    n = len(env.sub(1).serve_log)
    env.clock.advance(3601)
    await env.cycle({1: "a/champ", 2: "a/next"})
    assert len(rows_seen) == 2 and rows_seen[0] is rows_seen[1]
    assert len(env.sub(1).serve_log) == n + 1 and env.sub(1).serve_log[-1][1] == "ok"
    assert env.sub(1).attempts == 0 and env.sub(1).reason == ""
    assert [x[0] for x in calls] == ["a/champ", "a/champ", "a/next"]
    assert any("start-up retry on draw" in e for e in env.ev.state.track(TRACK).events)


async def test_double_failure_abandons_then_remeasures_after_restart_without_challengers(tmp_path):
    env = await champion(tmp_path)
    scripted(env, [V.ModelUnavailable("invalid config")] * 2)
    env.clock.advance(3601)
    await env.cycle({1: "a/champ", 2: "a/next"})
    assert env.sub(2).status == ST.QUEUED and not env.sub(2).draw1
    assert env.sub(1).serve_retry_pending and env.sub(1).attempts == 1
    env.ev = env.make()
    env.clock.advance(3601)
    env.server.served.clear()
    await env.cycle({1: "a/champ"})
    assert env.server.served == ["a/champ"]
    assert not env.sub(1).serve_retry_pending and env.sub(1).attempts == 1
    assert env.sub(1).reason == "start-up failures: 1 of the last 6 rounds"


@pytest.mark.parametrize("exc", [V.HostUnavailable("docker daemon unavailable"),
                                 V.HostUnavailable("GPU not free before start"), OSError("disk full")])
async def test_champion_infra_suspension_bound_and_recovery(tmp_path, exc):
    env = await champion(tmp_path)
    since = env.sub(1).last_serve_ok_ts
    for _ in range(3):
        scripted(env, [exc, exc])
        env.clock.advance(24 * 3600)
        await env.cycle({1: "a/champ", 2: "a/next"})
    assert env.sub(1).attempts == 0 and env.sub(1).status == ST.CHAMPION
    assert all(x[1] == "infra" for x in env.sub(1).serve_log[-3:])
    assert env.sub(1).pay_suspended_since_ts == since + C.CHAMPION_LIVENESS_S
    env.ev = env.make()  # persisted suspension survives restart and clears on a successful round
    assert env.sub(1).pay_suspended_since_ts is not None
    env.clock.advance(3601)
    await env.cycle({1: "a/champ", 2: "a/next"})
    assert env.sub(1).pay_suspended_since_ts is None and env.sub(1).champion_rounds_without_ok == 0
    assert env.sub(1).status == ST.CHAMPION and env.sub(1).attempts == 0


async def test_whole_group_failure_overrides_model_attribution(tmp_path):
    env = await champion(tmp_path)

    def fail(*args):
        raise V.ModelUnavailable("container exited during start-up")
    env.ev.backend_factory = fail
    for _ in range(4):
        env.clock.advance(3601)
        await env.cycle({1: "a/champ", 2: "a/next"})
    assert env.sub(1).status == ST.CHAMPION and env.sub(1).attempts == env.sub(2).attempts == 0
    assert env.sub(1).serve_log[-1][1] == env.sub(2).serve_log[-1][1] == "infra"


async def test_fetch_infra_past_deadline_never_expires_or_counts(tmp_path):
    env = Env(tmp_path, {}, sample=20)
    env.add(1, "a/m", 0.9)

    async def fetch(*args, **kwargs):
        raise F.FetchError("infra", "HF 503 / network timeout")
    env.hub.fetch = fetch
    for block in [1000, 1000 + C.PUBLISH_DEADLINE_BLOCKS + 1, 1000 + 2*C.PUBLISH_DEADLINE_BLOCKS]:
        env.clock.advance(3601)
        await env.cycle({1: "a/m"}, block=block)
    assert env.sub(1).status == ST.QUEUED and env.sub(1).attempts == 0
    assert env.sub(1).wait_kind == ST.WAIT_INFRA and env.sub(1).serve_log == []


@pytest.mark.parametrize("status,kind", [(401, "unavailable"), (403, "unavailable"), (404, "unavailable"), (429, "infra"), (500, "infra"), (503, "infra")])
def test_hf_listing_error_classification(monkeypatch, status, kind):
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    def fail(*args, **kwargs):
        raise HfHubHTTPError("hub failed", response=httpx.Response(status, request=httpx.Request("GET", "https://huggingface.co/a")))
    monkeypatch.setattr(HfApi, "model_info", fail)
    with pytest.raises(F.FetchError) as exc:
        F.HfHub().listing("a/m", "a"*40)
    assert exc.value.kind == kind


@pytest.mark.parametrize("reply,raises", [((0, "98000"), False), ((0, "1000"), True), ((124, "timeout"), True)])
async def test_gpu_preflight_is_bounded_and_classifies_busy_gpu(monkeypatch, reply, raises):
    clock = [0.0]
    async def sleep(seconds):
        clock[0] += seconds
    monkeypatch.setattr(V.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(V.asyncio, "sleep", sleep)
    async def run(*cmd, timeout):
        assert cmd[0] == "nvidia-smi" and 0 < timeout <= 10
        return reply
    monkeypatch.setattr(V, "_run", run)
    if raises:
        with pytest.raises(V.HostUnavailable):
            await V.check_gpu_free("0", 30)
    else:
        await V.check_gpu_free("0", 30)
    assert clock[0] == (60 if raises else 0)


def test_real_rc8post2_loader_reads_new_state_and_new_loader_reads_old():
    # Verbatim git blob 1b10c35:src/reax_subnet/spec/state.py, kept as the actual rollback dataclass path.
    path = Path(__file__).parent / "data/rollback/rc8post2_state.py"
    spec = importlib.util.spec_from_file_location("rc8post2_state", path)
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    sub = ST.Submission("s", 1, "h", "c", TRACK, "a/m", "r", 1, 0, status=ST.CHAMPION,
                        serve_log=[[1, "model"], [2, "ok"]], attempts=1, serve_retry_pending=True,
                        unavailable_fetches=2, last_serve_ok_ts=2, champion_rounds_without_ok=3,
                        pay_suspended_since_ts=72*3600)
    new = ST.SpecState(tracks={TRACK: ST.TrackState(TRACK, submissions={"s": sub}, champion="s")})
    rolled = old.from_json(ST._to_json(new))
    assert rolled.tracks[TRACK].champion == "s" and rolled.tracks[TRACK].submissions["s"].attempts == 1
    assert not hasattr(rolled.tracks[TRACK].submissions["s"], "serve_log")
    assert not hasattr(rolled.tracks[TRACK].submissions["s"], "pay_suspended_since_ts")
    rolled.tracks[TRACK].submissions["s"].attempts = 99  # unknown old consecutive counter must not cause false dethroning
    upgraded = ST.from_json(old._to_json(rolled))
    restored = upgraded.tracks[TRACK].submissions["s"]
    assert restored.serve_log == [] and restored.attempts == 0 and not restored.serve_retry_pending
    assert restored.pay_suspended_since_ts is None and restored.unavailable_fetches == 0
    assert restored.status == ST.CHAMPION and upgraded.tracks[TRACK].champion == "s"


@pytest.mark.parametrize("stage", ["inspect", "logs"])
async def test_docker_daemon_failure_during_startup_is_infrastructure(tmp_path, monkeypatch, stage):
    backend = V.VllmDockerBackend(str(tmp_path), 1)
    backend._http = object()

    async def run(*cmd, timeout):
        return (1, "Cannot connect to the Docker daemon") if cmd[1] == stage else (0, "false")
    monkeypatch.setattr(V, "_run", run)
    with pytest.raises(V.HostUnavailable, match="docker .* failed"):
        await backend._wait_healthy()


@pytest.mark.parametrize("code,kind", [(41, "unavailable"), (42, "infra"), (1, "infra")])
async def test_download_errors_preserve_child_attribution(tmp_path, monkeypatch, code, kind):
    class Proc:
        returncode = code

        async def communicate(self):
            return b"", b"HF request failed"

    async def spawn(*args, **kwargs):
        return Proc()
    monkeypatch.setattr(F.asyncio, "create_subprocess_exec", spawn)
    with pytest.raises(F.FetchError) as exc:
        await F.HfHub().download("a/m", "a" * 40, str(tmp_path / "download"))
    assert exc.value.kind == kind
    assert not (tmp_path / "download").exists()


async def test_download_timeout_is_infrastructure_and_child_is_reaped(tmp_path, monkeypatch):
    class Proc:
        killed = waited = False

        async def communicate(self):
            raise F.asyncio.TimeoutError()

        def kill(self):
            self.killed = True

        async def wait(self):
            self.waited = True
    proc = Proc()

    async def spawn(*args, **kwargs):
        return proc
    monkeypatch.setattr(F.asyncio, "create_subprocess_exec", spawn)
    with pytest.raises(F.FetchError) as exc:
        await F.HfHub().download("a/m", "a" * 40, str(tmp_path / "download"))
    assert exc.value.kind == "infra" and proc.killed and proc.waited
    assert not (tmp_path / "download").exists()


async def test_challenger_uses_same_window_across_successful_first_draw(tmp_path):
    env = await champion(tmp_path)
    env.server.skills.update({"a/champ": 0.55, "a/next": 0.99})
    broken = True

    def factory(model_dir, wb):
        if broken and Path(model_dir, "skill_key.txt").read_text() == "a/next":
            raise V.ModelUnavailable("invalid tokenizer")
        return env.server(model_dir, wb)
    env.ev.backend_factory = factory
    for i, fail in enumerate([True, False, True, True]):
        broken = fail
        env.clock.advance(13 * 3600)
        await env.cycle({1: "a/champ", 2: "a/next"})
        assert env.sub(2).attempts == [1, 1, 2, 3][i]
    assert env.sub(2).status == ST.REJECTED
    assert [e[1] for e in env.sub(2).serve_log] == ["model", "ok", "model", "model"]
    assert env.sub(1).status == ST.CHAMPION


@pytest.mark.parametrize("infra_first", [True, False])
async def test_mixed_champion_retry_is_inconclusive_if_either_failure_is_infra(tmp_path, infra_first):
    env = await champion(tmp_path)
    failures = [V.HostUnavailable("docker daemon"), V.ModelUnavailable("container exited")]
    scripted(env, failures if infra_first else failures[::-1])
    env.clock.advance(3601)
    await env.cycle({1: "a/champ", 2: "a/next"})
    assert env.sub(1).serve_log[-1][1] == "infra" and env.sub(1).attempts == 0
    assert env.sub(1).status == ST.CHAMPION


@pytest.mark.parametrize("infra", [False, True])
async def test_extension_failure_is_recorded_and_rejected_entry_cannot_be_recrowned(tmp_path, monkeypatch, infra):
    from test_items_playoff import CLOSE, _to_playoff
    monkeypatch.setattr(E, "PLAYOFF_MIN_GAP_SE", 100.0)
    env = Env(tmp_path, {})
    orig = env.ev._serve
    n = 0

    async def serve(p, rows, now):
        nonlocal n
        if p.sub.uid == 3:
            n += 1
            if n == 4:
                p.sub.serve_log = [[now-3, "model"], [now-2, "model"], [now-1, "ok"]]
                p.sub.reason = "could not be served: container exited"
                p.infra_failure = infra
                return None
        return await orig(p, rows, now)
    monkeypatch.setattr(env.ev, "_serve", serve)
    await _to_playoff(env, CLOSE)
    assert env.sub(3).serve_log[-1][1] == ("infra" if infra else "model")
    if infra:
        assert env.sub(3).attempts == 2 and env.sub(3).status == ST.CHAMPION
    else:
        assert env.sub(3).attempts == 3 and env.sub(3).status == ST.REJECTED
        assert env.ev.state.track(TRACK).champion == env.sub(2).key


async def test_playoff_fetch_infrastructure_is_bounded_without_strikes(tmp_path):
    from test_cohort import H, _two_ready
    env, commits = await _two_ready(tmp_path, 0.85, 0.90)
    real = env.hub.fetch

    async def fetch(commitment, *args, **kwargs):
        if commitment.repo == "a/two":
            raise F.FetchError("infra", "HF 503")
        return await real(commitment, *args, **kwargs)
    env.hub.fetch = fetch
    first_draw = env.sub(2).draw1
    for _ in range(C.PLAYOFF_MAX_ABORTS - 1):
        env.clock.advance(H)
        env.server.served.clear()
        await env.cycle(commits, block=1100)
        assert not env.ev.state.track(TRACK).champion and env.server.served == []
        assert env.sub(2).attempts == 0
    env.clock.advance(H)
    await env.cycle(commits, block=1100)
    assert env.ev.state.track(TRACK).champion == env.sub(1).key
    assert env.sub(2).draw1 == first_draw and not env.sub(2).confirm_draw
    assert env.sub(2).attempts == 0
    env.hub.fetch = real
    env.server.skills["a/one"] = 0.55  # full challenger margin is required after the bound crowned a champion
    env.clock.advance(H)
    await env.cycle(commits, block=1100)
    assert env.ev.state.track(TRACK).champion == env.sub(2).key


@pytest.mark.parametrize("first", [V.ModelUnavailable("invalid config"), V.HostUnavailable("docker failed")])
async def test_retry_preflight_retains_first_attempt_verdict(tmp_path, first):
    env = await champion(tmp_path)
    scripted(env, [first, V.GPUPreflightUnavailable("GPU not free before start")])
    env.clock.advance(3601)
    await env.cycle({1: "a/champ", 2: "a/next"})
    assert env.sub(1).serve_log[-1][1] == ("infra" if isinstance(first, V.HostUnavailable) else "model")
    assert "GPU not free" not in env.sub(1).reason


async def test_failed_champion_serves_only_one_host_probe(tmp_path):
    env = await champion(tmp_path)
    env.add(3, "a/three", 0.4)
    env.add(4, "a/four", 0.5)
    scripted(env, [V.HostUnavailable("host timeout")] * 2)
    before = len(env.sub(1).serve_log)
    env.server.served.clear()
    env.clock.advance(3601)
    await env.cycle({1: "a/champ", 2: "a/next", 3: "a/three", 4: "a/four"})
    assert len(env.server.served) == 1
    assert len(env.ev.state.eval_times) == 3  # two champion attempts, one probe
    assert len(env.sub(1).serve_log) == before + 1
    assert sum(len(env.sub(uid).serve_log) for uid in (2, 3, 4)) == 1
    assert all(env.sub(uid).status == ST.QUEUED and not env.sub(uid).draw1 for uid in (2, 3, 4))


@pytest.mark.parametrize("failure,expected", [
    (V.ModelUnavailable("container exited during start-up"), "infra"),
    (V.ModelUnavailable("container exited: RuntimeError: CUDA driver initialization failed"), "infra"),
    (V.ModelUnavailable("invalid config"), "model"),
    (V.ModelUnavailable("invalid tokenizer"), "model"),
    (V.ModelUnavailable("unsupported architecture"), "model"),
    (V.ModelUnavailable("CUDA out of memory."), "infra"),  # no evidence of a free GPU
    (V.ModelBudgetOOM("CUDA out of memory."), "model"),
])
async def test_lone_champion_requires_model_specific_evidence(tmp_path, failure, expected):
    env = await champion(tmp_path)
    env.sub(1).serve_retry_pending = True
    scripted(env, [failure, failure])
    env.clock.advance(3601)
    await env.cycle({1: "a/champ"})
    assert env.sub(1).serve_log[-1][1] == expected
    assert env.sub(1).attempts == (expected == "model")


async def test_gpu_preflight_polls_until_memory_is_released(monkeypatch):
    clock, calls = [0.0], []
    async def run(*cmd, timeout):
        calls.append(clock[0])
        return 0, "98000" if len(calls) == 3 else "1000"
    async def sleep(seconds):
        clock[0] += seconds
    monkeypatch.setattr(V.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(V.asyncio, "sleep", sleep)
    monkeypatch.setattr(V, "_run", run)
    await V.check_gpu_free("0", 30)
    assert calls == [0, 2, 4]


def test_gpu_memory_phrase_in_echoed_config_is_not_a_host_error():
    text = "ValueError: Free memory on device cuda:0 (2/96 GiB) on startup is less than desired GPU memory utilization (0.2, 20 GiB)"
    assert V.host_lacks_gpu_memory(text)
    assert V.host_lacks_gpu_memory("(EngineCore_DP0 pid=123) " + text)
    assert not V.host_lacks_gpu_memory("config={'description': '" + text + "'}")
    assert not V.host_lacks_gpu_memory('  "description": "' + text + '"')
    assert not V.host_lacks_gpu_memory("ValueError: config text Free memory on device cuda:0\nless than desired GPU memory utilization")


async def test_unavailable_refetches_are_spaced_and_infra_never_counts(tmp_path):
    env = await champion(tmp_path)
    sub = env.sub(1)
    import shutil
    from spec_helpers import snapshot, neuron, commit
    shutil.rmtree(env.ev._model_dir(sub))
    ts = env.ev.state.track(TRACK)
    snap = snapshot(1000, [neuron(1, commit(TRACK, "a/champ"))])
    outcomes = ["unavailable", "infra", "unavailable", "unavailable"]
    times = []
    async def fetch(*args, **kwargs):
        times.append(env.clock())
        raise F.FetchError(outcomes.pop(0), "repo error")
    env.hub.fetch = fetch
    for i in range(4):
        await env.ev._prepare(env.cfg.tracks[0], sub, snap, env.clock())
        count = sub.unavailable_fetches
        assert count == [1, 1, 2, 3][i] and sub.attempts == 0
        if i < 3:
            env.clock.advance(C.RETRY_UNAVAILABLE_S - 1)
            await env.ev._prepare(env.cfg.tracks[0], sub, snap, env.clock())
            assert sub.unavailable_fetches == count and len(times) == i + 1
            env.clock.advance(1)
    assert all(b - a >= C.RETRY_UNAVAILABLE_S for a, b in zip(times, times[1:]))
    assert sub.status == ST.REJECTED and not ts.champion and not sub.serve_retry_pending


def test_liveness_needs_three_attempts_and_crown_clears_retry_state():
    sub = ST.Submission("s", 1, "h", "c", TRACK, "a/m", "r", 1, 0, status=ST.CHAMPION,
                        score=90, champion_since_ts=0, last_serve_ok_ts=0)
    ts = ST.TrackState(TRACK, champion="s", submissions={"s": sub})
    for i in range(2):
        E.SpecEvaluator._serve_round(ts, sub, "infra", C.CHAMPION_LIVENESS_S + i)
        assert sub.pay_suspended_since_ts is None
    E.SpecEvaluator._serve_round(ts, sub, "infra", C.CHAMPION_LIVENESS_S + 2)
    assert sub.pay_suspended_since_ts == C.CHAMPION_LIVENESS_S + 2
    assert sub.attempts == 0 and ts.champion == "s"
    sub.serve_retry_pending = True
    ev = object.__new__(E.SpecEvaluator)
    ev._crown(ts, sub, C.CHAMPION_LIVENESS_S + 3, "test")
    assert not sub.serve_retry_pending and sub.pay_suspended_since_ts is None


@pytest.mark.parametrize("preflight_ok", [True, False])
async def test_vllm_oom_is_specific_only_after_free_gpu_preflight(tmp_path, monkeypatch, preflight_ok):
    backend = V.VllmDockerBackend(str(tmp_path), 1)
    backend._http = object()
    backend._gpu_preflight_ok = preflight_ok
    async def run(*cmd, timeout):
        return (0, "false") if cmd[1] == "inspect" else (0, "torch.OutOfMemoryError: CUDA out of memory.")
    monkeypatch.setattr(V, "_run", run)
    with pytest.raises(V.ModelUnavailable) as exc:
        await backend._wait_healthy()
    assert isinstance(exc.value, V.ModelBudgetOOM) == preflight_ok


async def test_rc9_loader_drops_suspension_signal_and_still_ranks_champion(tmp_path):
    # rc9 (18e2597) has exactly the rc8.post2 state/remote/compute_leaders code. This demonstrates the
    # required validator follow-up, rather than silently pretending a new worker field changes old payouts.
    from spec_helpers import snapshot, neuron, commit
    path = Path(__file__).parent / "data/rollback/rc8post2_state.py"
    spec = importlib.util.spec_from_file_location("rc9_legacy_state", path)
    legacy = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = legacy
    spec.loader.exec_module(legacy)
    env = await champion(tmp_path)
    sub = env.sub(1)
    sub.pay_suspended_since_ts = env.clock()
    raw = ST._to_json(env.ev.state)
    parsed = legacy.from_json(raw)
    assert not hasattr(parsed.track(TRACK).submissions[sub.key], "pay_suspended_since_ts")
    snap = snapshot(1000, [neuron(1, commit(TRACK, "a/champ"))])
    assert E.compute_leaders(parsed, [TRACK], snap)[TRACK][0][0] == 1


# ----------------------------------------------------------------- critic FX1: real vLLM log
def test_host_gpu_memory_detected_in_a_real_vllm_start_up_log():
    from reax_subnet.spec import vllm as V
    log = open(os.path.join(os.path.dirname(__file__), "data", "vllm", "host-gpu-memory.log")).read()
    lines = log.splitlines()
    assert not V.host_lacks_gpu_memory("\n".join(lines[-20:]))         # the error line sits above the traceback
    assert V.host_lacks_gpu_memory("\n".join(lines[-V.STARTUP_LOG_TAIL:]))
    line = "ValueError: Free memory on device cuda:0 (24.27/94.86 GiB) on startup is less than desired GPU memory utilization (0.261, 24.76 GiB)."
    assert V.host_lacks_gpu_memory("\x1b[1;36m(EngineCore pid=1)\x1b[0;0m " + line)
    assert V.host_lacks_gpu_memory("[rank0]: " + line)
    assert not V.host_lacks_gpu_memory("config: {'note': '" + line + "'}")
