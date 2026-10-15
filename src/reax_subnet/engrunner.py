"""Start competition engines and measure them against the reference engine (docs/ENGINE-COMPETITION.md).

Engine contract: the image serves the runtime sidecar contract (``POST /internal/v1/decide`` with the
``X-S1M-Internal-Token`` header, ``GET /health``) on the unix socket named by ``RUNTIME_UDS`` (mode 0666), reads the
token from ``S1M_INTERNAL_TOKEN_FILE`` and the pinned weights from ``PLUMB_MODEL_DIR`` (read-only). It runs with no
network, a read-only root filesystem and no capabilities; ``/tmp`` is the only writable path. A submitted image
carries the label ``ai.reax.engine.hotkey=<ss58 of the committing hotkey>``, so a digest cannot be claimed by
anybody else.

Two runners:

* ``DockerRunner`` (real validators): ``docker pull <image>@sha256:<digest>`` from an allowed registry, then
  ``docker run --network none`` with one GPU, CPU/memory/pids/log caps, the weights mounted read-only and a fresh
  socket directory per engine. The idle engine is paused (``docker pause``) while the other one is measured;
* ``DevRunner`` (local test chains and unit tests only): a JSON map from image reference to a local command.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import shutil
import signal
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Protocol, Sequence

from .engcomp import EvalResult, TrackParams, check_correctness, max_distance, median, speed_score
from .s1backends import SidecarBackend
from .s1contract import Decision, parse_decision
from .s1items import S1Item, generate_s1_items

log = logging.getLogger("reax.engines")

SOCKET_NAME = "engine.sock"
HOTKEY_LABEL = "ai.reax.engine.hotkey"
DEFAULT_REGISTRIES = ("ghcr.io", "docker.io", "quay.io")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def clean(text: str, limit: int = 200) -> str:
    """Engine-influenced text for logs and state: no control characters, bounded length."""
    return _CONTROL.sub(" ", text)[-limit:]


class EngineError(RuntimeError):
    """The engine under test failed (it loses this evaluation)."""


class InfraError(RuntimeError):
    """Pull/start failed for a reason that may be ours (registry rate limit, docker hiccup): inconclusive."""


ENGINE_GPU_MIB = 12_500  # one s1-fast engine (Plumb-4B, 12.3 GB measured on the validator card)


class ReferenceError_(RuntimeError):
    """The reference engine failed before the candidate ran: the evaluation is inconclusive and is not recorded."""


class ReferenceFailedWithCandidate(ReferenceError_):
    """The reference failed while the candidate was loaded (the candidate may have starved it)."""


class UdsSidecarBackend(SidecarBackend):
    """The sidecar client over a unix socket."""

    def __init__(self, model: str, socket_path: str, token: str, max_inflight: int, timeout_s: float):
        super().__init__(model, "http://engine", token, max_inflight=max_inflight, timeout_s=timeout_s)
        self.socket_path = socket_path

    def socket_ok(self) -> bool:
        """The path is a real socket, not a symlink to some other socket on the host."""
        try:
            return stat.S_ISSOCK(os.lstat(self.socket_path).st_mode)
        except OSError:
            return False

    async def _client(self):
        import aiohttp

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout), trust_env=False,
                connector=aiohttp.UnixConnector(path=self.socket_path, limit=256),
            )
        return self._session


@dataclass
class Engine:
    ref: str
    client: UdsSidecarBackend
    stop: Callable[[], Awaitable[None]]
    pause: Callable[[], Awaitable[None]]
    resume: Callable[[], Awaitable[None]]


class Runner(Protocol):
    binds_hotkey: bool

    async def preflight(self) -> None:
        """Both engines (reference + candidate) are loaded at once. On a shared 96 GB GPU the serving runtimes stay
        resident, so check the free memory first: too little = inconclusive (cycle skipped), never a failed candidate.
        If nvidia-smi cannot be read the check is skipped."""
        code, out, _ = await _run(["nvidia-smi", f"--id={self.gpu}", "--query-gpu=memory.free",
                                   "--format=csv,noheader,nounits"], 30)
        try:
            free_mib = int(out.strip().splitlines()[0])
        except (ValueError, IndexError):
            return
        need = 2 * ENGINE_GPU_MIB
        if code == 0 and free_mib < need:
            raise InfraError(f"GPU {self.gpu} has {free_mib} MiB free, an evaluation needs about {need} MiB "
                             "(two s1-fast engines next to the serving runtimes)")

    async def owner_of(self, ref: str) -> Optional[str]: ...

    async def start(self, ref: str, model: str, params: TrackParams, keep_image: bool = False) -> Engine: ...


async def _run(argv: Sequence[str], timeout_s: float) -> tuple[int, str, str]:
    """A subprocess without blocking the event loop; output capped at 64 KiB per stream."""
    proc = await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout_s)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return -1, "", f"timeout after {timeout_s:.0f} s"
    return proc.returncode or 0, out[-65536:].decode(errors="replace"), err[-65536:].decode(errors="replace")


def _session_dir(root: str) -> tuple[str, str]:
    os.makedirs(root, mode=0o700, exist_ok=True)
    os.chmod(root, 0o700)
    d = tempfile.mkdtemp(prefix="eng-", dir=root)
    io = os.path.join(d, "io")
    os.mkdir(io)
    os.chmod(io, 0o777)  # the engine runs as its own uid and must create its socket here
    token = secrets.token_hex(24)
    with open(os.path.join(io, "token"), "w", encoding="ascii") as fh:
        fh.write(token)
    os.chmod(os.path.join(io, "token"), 0o644)
    return d, token


def _keep_log(root: str, d: str, name: str, text: str) -> None:
    logs = os.path.join(root, "logs")
    os.makedirs(logs, exist_ok=True)
    with open(os.path.join(logs, f"{name}.log"), "w") as fh:
        fh.write(text[-100_000:])
    shutil.rmtree(d, ignore_errors=True)


def registry_of(image: str) -> str:
    first, _, rest = image.partition("/")
    if rest and ("." in first or ":" in first or first == "localhost"):
        return first
    return "docker.io"


class DockerRunner:
    binds_hotkey = True

    def __init__(self, workdir: str, weights_dir: str, gpu: str = "0", memory: str = "64g", cpus: str = "8",
                 user: str = "10002:10002", docker_runtime: str = "", pull: bool = True,
                 registries: Sequence[str] = DEFAULT_REGISTRIES, pull_timeout_s: float = 1800.0):
        self.workdir = workdir
        self.weights_dir = os.path.abspath(weights_dir)
        self.gpu = gpu
        self.memory = memory
        self.cpus = cpus
        self.user = user
        self.docker_runtime = docker_runtime
        self.pull = pull
        self.registries = tuple(registries)
        self.pull_timeout_s = pull_timeout_s

    async def _ensure_image(self, ref: str) -> None:
        if "@sha256:" not in ref:
            raise EngineError("engine images are run by digest only")
        if registry_of(ref.split("@", 1)[0]) not in self.registries:
            raise EngineError(f"registry {registry_of(ref.split('@', 1)[0])!r} is not allowed")
        code, _, _ = await _run(["docker", "image", "inspect", "--format", "{{.Id}}", ref], 60)
        if code == 0 or not self.pull:
            return
        code, _, err = await _run(["docker", "pull", "-q", ref], self.pull_timeout_s)
        if code != 0:
            raise InfraError(f"docker pull failed: {clean(err)}")

    async def owner_of(self, ref: str) -> Optional[str]:
        """The hotkey the image is bound to (its label), or "" if none."""
        await self._ensure_image(ref)
        code, out, err = await _run(["docker", "image", "inspect", "--format", "{{json .Config.Labels}}", ref], 60)
        if code != 0:
            raise InfraError(f"docker inspect failed: {clean(err)}")
        try:
            labels = json.loads(out) or {}
        except ValueError:
            labels = {}
        value = labels.get(HOTKEY_LABEL, "") if isinstance(labels, dict) else ""
        return value if isinstance(value, str) else ""

    async def start(self, ref: str, model: str, params: TrackParams, keep_image: bool = False) -> Engine:
        await self._ensure_image(ref)
        d, token = _session_dir(self.workdir)
        name = "reax-eng-" + secrets.token_hex(6)
        cmd = ["docker", "run", "-d", "--name", name, "--network", "none", "--read-only",
               "--tmpfs", "/tmp:rw,exec,size=16g", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
               "--pids-limit", "1024", "--ulimit", "nofile=4096:4096", "--memory", self.memory,
               "--memory-swap", self.memory, "--cpus", self.cpus, "--shm-size", "2g", "--user", self.user,
               "--log-driver", "local", "--log-opt", "max-size=10m", "--log-opt", "max-file=2",
               "--gpus", f"device={self.gpu}",
               "-v", f"{self.weights_dir}:/model:ro", "-v", f"{os.path.join(d, 'io')}:/run/reax",
               "-e", "RUNTIME_UDS=/run/reax/" + SOCKET_NAME, "-e", "S1M_INTERNAL_TOKEN_FILE=/run/reax/token",
               "-e", "PLUMB_MODEL_DIR=/model", "-e", "HF_HUB_OFFLINE=1", "-e", "OPENAI_API_KEY="]
        if self.docker_runtime:
            cmd += ["--runtime", self.docker_runtime]
        cmd.append(ref)
        code, _, err = await _run(cmd, 300)
        if code != 0:
            await _run(["docker", "rm", "-fv", name], 120)
            shutil.rmtree(d, ignore_errors=True)
            raise InfraError(f"docker run failed: {clean(err)}")

        async def stop() -> None:
            text = ""
            try:
                _, out, err = await _run(["docker", "logs", "--tail", "200", name], 60)
                text = out + err
            finally:
                await _run(["docker", "rm", "-fv", name], 120)
                _keep_log(self.workdir, d, name, text)
                if not keep_image:
                    await _run(["docker", "rmi", ref], 120)

        async def pause() -> None:
            await _run(["docker", "pause", name], 60)

        async def resume() -> None:
            await _run(["docker", "unpause", name], 60)

        client = UdsSidecarBackend(model, os.path.join(d, "io", SOCKET_NAME), token, params.concurrency,
                                   params.request_timeout_s)
        return Engine(ref, client, stop, pause, resume)


class DevRunner:
    """Local test chains only: ``{"<image>@sha256:<digest>": ["python", "fake_engine.py", "--variant", "fast"]}``.

    No hotkey binding (copy protection falls back to the first committer of a digest)."""

    binds_hotkey = False

    def __init__(self, workdir: str, mapping: dict[str, list[str]]):
        self.workdir = workdir
        self.mapping = mapping

    @classmethod
    def from_file(cls, workdir: str, path: str) -> "DevRunner":
        with open(path, encoding="utf-8") as fh:
            return cls(workdir, json.load(fh))

    async def owner_of(self, ref: str) -> Optional[str]:
        return None

    async def start(self, ref: str, model: str, params: TrackParams, keep_image: bool = False) -> Engine:
        argv = self.mapping.get(ref)
        if not argv:
            raise EngineError("image not available to the dev runner")
        d, token = _session_dir(self.workdir)
        sock = os.path.join(d, "io", SOCKET_NAME)
        env = {**os.environ, "RUNTIME_UDS": sock, "S1M_INTERNAL_TOKEN_FILE": os.path.join(d, "io", "token"),
               "OPENAI_API_KEY": ""}
        log_path = os.path.join(d, "engine.log")
        log_fh = open(log_path, "w")
        proc = subprocess.Popen(argv, env=env, stdout=log_fh, stderr=subprocess.STDOUT)

        async def stop() -> None:
            if proc.poll() is None:
                proc.send_signal(signal.SIGCONT)
                proc.terminate()
                try:
                    await asyncio.to_thread(proc.wait, 10)
                except subprocess.TimeoutExpired:
                    proc.kill()
            log_fh.close()
            with open(log_path, errors="replace") as fh:
                text = fh.read()
            _keep_log(self.workdir, d, os.path.basename(d), text)

        async def pause() -> None:
            if proc.poll() is None:
                proc.send_signal(signal.SIGSTOP)

        async def resume() -> None:
            if proc.poll() is None:
                proc.send_signal(signal.SIGCONT)

        client = UdsSidecarBackend(model, sock, token, params.concurrency, params.request_timeout_s)
        return Engine(ref, client, stop, pause, resume)


# ------------------------------------------------------------------------------------------------ items
def _perturb_state(state: Any, nonce: str) -> Any:
    """A secret record id makes every item unpredictable, so no answer table can cover it."""
    if isinstance(state, dict):
        return {"ref": nonce, **state}
    if isinstance(state, list):
        return {"ref": nonce, "entries": state}
    return f"[ref {nonce}] {state}"


def fresh_decisions(n: int, model: str) -> list[Decision]:
    out = []
    for it in generate_s1_items(n):
        it = S1Item(it.item_id, _perturb_state(it.state, secrets.token_hex(4)), it.kind, it.instructions, it.criteria)
        out.append(parse_decision(it.decision(), model))
    return out


def probe_decisions(pairs: int, model: str) -> tuple[list[Decision], list[Decision]]:
    """Base items, then variants that reuse a base item's state with another item's question, or its question with
    another item's state. An engine that caches by state (or by question) answers the variant from its cache."""
    a = generate_s1_items(pairs)
    b = generate_s1_items(pairs)
    base, variants = [], []
    for i, (x, y) in enumerate(zip(a, b)):
        nonce = secrets.token_hex(4)
        state = _perturb_state(x.state, nonce)
        base.append(parse_decision(S1Item(x.item_id, state, x.kind, x.instructions, x.criteria).decision(), model))
        if i % 2 == 0:  # same state, other question
            v = S1Item(secrets.token_hex(16), state, y.kind, y.instructions, y.criteria)
        else:  # same question, other state
            v = S1Item(secrets.token_hex(16), _perturb_state(y.state, nonce), x.kind, x.instructions, x.criteria)
        variants.append(parse_decision(v.decision(), model))
    return base, variants


# ------------------------------------------------------------------------------------------------ measuring
Answers = dict[str, Optional[tuple[float, ...]]]


async def _answers(client: UdsSidecarBackend, decisions: Sequence[Decision]) -> Answers:
    results = await client.decide_each(decisions, retries=0)
    out: Answers = {}
    for d, a in zip(decisions, results):
        if isinstance(a, BaseException):
            out[d.ident] = None
            continue
        try:
            out[d.ident] = tuple(float(a["probabilities"][k]) for k in d.question.keys)
        except (KeyError, TypeError, ValueError):
            out[d.ident] = None
    return out


async def timed_run(client: UdsSidecarBackend, decisions: Sequence[Decision]) -> tuple[float, Answers]:
    """Decisions per second over the whole batch (validator clock), with every answer."""
    started = time.monotonic()
    answers = await _answers(client, decisions)
    elapsed = time.monotonic() - started
    return len(decisions) / elapsed if elapsed > 0 else 0.0, answers


async def latencies(client: UdsSidecarBackend, decisions: Sequence[Decision]) -> tuple[list[float], Answers]:
    lat, out = [], {}
    for d in decisions:
        started = time.monotonic()
        out.update(await _answers(client, [d]))
        lat.append(time.monotonic() - started)
    return lat, out


PAUSE_DRAIN_S = 0.5
PROBE_GRACE_S = 120.0  # after /health is up, the authenticated test decision must succeed within this time


async def wait_ready(engine: Engine, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    healthy_since = None
    while time.monotonic() < deadline:
        if engine.client.socket_ok() and await engine.client.ready():
            if await engine.client.probe():
                return True
            healthy_since = healthy_since or time.monotonic()
            if time.monotonic() - healthy_since > PROBE_GRACE_S:
                return False
        await asyncio.sleep(1.0)
    return False


def _complete(answers: Answers) -> bool:
    return all(v is not None for v in answers.values())


class _Solo:
    """Measure one engine while the other one is paused, so neither can disturb the other's timing."""

    def __init__(self, active: Engine, idle: Optional[Engine], pause: bool = True):
        self.active, self.idle = active, (idle if pause else None)

    async def __aenter__(self):
        if not self.active.client.socket_ok():
            raise EngineError("engine socket replaced")
        if self.idle is not None:
            await self.idle.pause()
            await asyncio.sleep(PAUSE_DRAIN_S)  # GPU work the idle engine queued before the pause drains first
        return self.active.client

    async def __aexit__(self, *exc):
        if self.idle is not None:
            await self.idle.resume()


async def evaluate(runner: Runner, reference_ref: str, candidate_ref: str, track: str, params: TrackParams,
                   block: int = 0, digest: str = "") -> EvalResult:
    """Measure one candidate against the reference in one session (both engines loaded on the same GPU; the idle
    one is paused during every measurement).

    Raises ReferenceError_ when the reference engine fails before the candidate starts (inconclusive),
    ReferenceFailedWithCandidate when it fails while the candidate is loaded, and InfraError when the candidate
    image cannot be pulled or started for infrastructure reasons. Wrap it in ``asyncio.wait_for`` with
    ``params.eval_budget_s``.
    """
    model = params.pool
    res = EvalResult(digest=digest or candidate_ref.rsplit(":", 1)[-1], track=track, ok=False, block=block)
    ref = cand = None
    try:
        try:
            ref = await runner.start(reference_ref, model, params, keep_image=True)
        except (EngineError, InfraError) as exc:
            raise ReferenceError_(f"reference engine did not start: {exc}") from None
        if not await wait_ready(ref, params.start_timeout_s):
            raise ReferenceError_("reference engine not ready")
        try:
            cand = await runner.start(candidate_ref, model, params)
        except EngineError as exc:
            res.reason = clean(f"start: {exc}")
            return res
        if not await wait_ready(cand, params.start_timeout_s):
            res.reason = "not ready within the start timeout (or failed the authenticated test decision)"
            return res
        try:
            return await _measure(ref, cand, params, res)
        except ReferenceError_ as exc:
            raise ReferenceFailedWithCandidate(str(exc)) from None
    finally:
        for eng in (cand, ref):
            if eng is None:
                continue
            try:
                await eng.resume()
                await eng.client.close()
            except Exception:  # noqa: BLE001 - cleanup of one engine must not skip the other
                pass
            try:
                await eng.stop()
            except Exception as exc:  # noqa: BLE001
                log.warning("could not stop engine %s: %r", eng.ref[:60], exc)


async def _measure(ref: Engine, cand: Engine, params: TrackParams, res: EvalResult) -> EvalResult:
    model = params.pool
    warm = fresh_decisions(min(32, params.timed_items), model)
    async with _Solo(ref, cand, params.pause_idle) as c:
        await _answers(c, warm)
    async with _Solo(cand, ref, params.pause_idle) as c:
        await _answers(c, warm)

    ref_tp, cand_tp, ref_ans, cand_ans = [], [], {}, {}
    last_items: list[Decision] = []
    for run in range(params.runs):
        items = fresh_decisions(params.timed_items, model)
        order = [(ref, cand), (cand, ref)]
        if run % 2:
            order.reverse()  # ABBA: neither engine always runs first
        for eng, other in order:
            async with _Solo(eng, other, params.pause_idle) as c:
                tp, ans = await timed_run(c, items)
            if eng is ref:
                if not _complete(ans):
                    raise ReferenceError_("reference engine failed on timed items")
                ref_tp.append(tp)
                ref_ans.update(ans)
            else:
                cand_tp.append(tp)
                cand_ans.update(ans)
                if not _complete(ans):  # stop early: no need to sit through more timeouts
                    missing = sum(v is None for v in ans.values())
                    res.reason = f"{missing} of {len(ans)} timed items failed"
                    return res
        last_items = items
    res.items = len(ref_ans)
    res.throughput = median(cand_tp)
    res.ref_throughput = median(ref_tp)
    corr = check_correctness(cand_ans, ref_ans, params)  # type: ignore[arg-type]
    res.mean_distance, res.max_distance, res.top_agreement = corr.mean_distance, corr.max_distance, corr.top_agreement
    if not corr.ok:
        res.reason = f"correctness: {corr.reason}"
        return res

    base, variants = probe_decisions(params.probe_pairs, model)
    async with _Solo(ref, cand, params.pause_idle) as c:
        p_ref = await _answers(c, base + variants)
    if not _complete(p_ref):
        raise ReferenceError_("reference engine failed on probe items")
    async with _Solo(cand, ref, params.pause_idle) as c:
        _, p_base = await latencies(c, base)  # one at a time: a cache sees every base item first
        _, p_var = await latencies(c, variants)
    probe = check_correctness({**p_base, **p_var}, p_ref, params, top_agreement=False)  # type: ignore[arg-type]
    res.probe_max_distance = probe.max_distance
    if not probe.ok:
        res.reason = f"anti-cache probe: {probe.reason}"
        return res

    again = last_items[: params.determinism_items]
    async with _Solo(cand, ref, params.pause_idle) as c:
        _, rep = await latencies(c, again)
    res.determinism_max_distance = max_distance(rep, {d.ident: cand_ans[d.ident] for d in again},  # type: ignore[misc]
                                                params.temperature)
    if res.determinism_max_distance > params.determinism_tau:
        res.reason = (f"determinism: a repeated item moved by {res.determinism_max_distance:.3f} "
                      f"> {params.determinism_tau}")
        return res

    lat_items = fresh_decisions(params.latency_items, model)
    async with _Solo(ref, cand, params.pause_idle) as c:
        ref_lat, ref_l_ans = await latencies(c, lat_items)
    async with _Solo(cand, ref, params.pause_idle) as c:
        cand_lat, cand_l_ans = await latencies(c, lat_items)
    if not _complete(ref_l_ans):
        raise ReferenceError_("reference engine failed on latency items")
    lat_corr = check_correctness(cand_l_ans, ref_l_ans, params, top_agreement=False)  # type: ignore[arg-type]
    if not lat_corr.ok:
        res.reason = f"correctness at concurrency 1: {lat_corr.reason}"
        return res
    res.latency_p50_s, res.ref_latency_p50_s = median(cand_lat), median(ref_lat)
    if res.latency_p50_s > params.max_latency_ratio * res.ref_latency_p50_s:
        res.reason = (f"latency p50 {res.latency_p50_s * 1000:.1f} ms > {params.max_latency_ratio} x reference "
                      f"{res.ref_latency_p50_s * 1000:.1f} ms")
        return res
    res.score = speed_score(cand_tp, ref_tp)
    res.ok = True
    return res


async def reference_alone_ok(runner: Runner, reference_ref: str, params: TrackParams) -> bool:
    """After a reference failure with a candidate loaded: does the reference work on its own?"""
    eng = None
    try:
        eng = await runner.start(reference_ref, params.pool, params, keep_image=True)
        if not await wait_ready(eng, params.start_timeout_s):
            return False
        return _complete(await _answers(eng.client, fresh_decisions(16, params.pool)))
    except Exception:  # noqa: BLE001
        return False
    finally:
        if eng is not None:
            try:
                await eng.client.close()
            finally:
                await eng.stop()
