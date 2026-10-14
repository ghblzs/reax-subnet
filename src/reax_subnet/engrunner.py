"""Start competition engines and measure them against the reference engine (docs/ENGINE-COMPETITION.md).

Engine contract: the image serves the runtime sidecar contract (``POST /internal/v1/decide`` with the
``X-S1M-Internal-Token`` header, ``GET /health``) on the unix socket named by ``RUNTIME_UDS`` (mode 0666), reads the
token from ``S1M_INTERNAL_TOKEN_FILE`` and the pinned weights from ``PLUMB_MODEL_DIR`` (read-only). It runs with no
network, a read-only root filesystem and no capabilities; ``/tmp`` is the only writable path.

Two runners:

* ``DockerRunner`` (real validators): ``docker pull <image>@sha256:<digest>`` then ``docker run --network none``
  with one GPU, the weights mounted read-only and a fresh socket directory per engine;
* ``DevRunner`` (local test chains and unit tests only): a JSON map from image reference to a local command.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, Sequence

from .engcomp import EvalResult, TrackParams, check_correctness, max_distance, median, speed_score
from .s1backends import SidecarBackend
from .s1contract import Decision, parse_decision
from .s1items import S1Item, generate_s1_items

log = logging.getLogger("reax.engines")

SOCKET_NAME = "engine.sock"


class EngineError(RuntimeError):
    """The engine under test failed (it loses this evaluation)."""


class ReferenceError_(RuntimeError):
    """The reference engine failed: the evaluation is inconclusive and is not recorded."""


class UdsSidecarBackend(SidecarBackend):
    """The sidecar client over a unix socket."""

    def __init__(self, model: str, socket_path: str, token: str, max_inflight: int, timeout_s: float):
        super().__init__(model, "http://engine", token, max_inflight=max_inflight, timeout_s=timeout_s)
        self.socket_path = socket_path

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
    stop: Callable[[], None]


class Runner(Protocol):
    def start(self, ref: str, model: str, params: TrackParams) -> Engine: ...


def _session_dir(root: str) -> tuple[str, str]:
    os.makedirs(root, mode=0o700, exist_ok=True)
    d = tempfile.mkdtemp(prefix="eng-", dir=root)
    io = os.path.join(d, "io")
    os.mkdir(io)
    os.chmod(io, 0o777)  # the engine runs as its own uid and must create its socket here
    token = secrets.token_hex(24)
    with open(os.path.join(io, "token"), "w", encoding="ascii") as fh:
        fh.write(token)
    os.chmod(os.path.join(io, "token"), 0o644)
    return d, token


class DockerRunner:
    def __init__(self, workdir: str, weights_dir: str, gpu: str = "0", memory: str = "64g",
                 user: str = "10002:10002", docker_runtime: str = "", pull: bool = True):
        self.workdir = workdir
        self.weights_dir = os.path.abspath(weights_dir)
        self.gpu = gpu
        self.memory = memory
        self.user = user
        self.docker_runtime = docker_runtime
        self.pull = pull

    def start(self, ref: str, model: str, params: TrackParams) -> Engine:
        if "@sha256:" not in ref:
            raise EngineError("engine images are run by digest only")
        if self.pull:
            r = subprocess.run(["docker", "pull", "-q", ref], capture_output=True, text=True, timeout=3600)
            if r.returncode != 0:
                raise EngineError(f"docker pull failed: {r.stderr.strip()[-200:]}")
        d, token = _session_dir(self.workdir)
        name = "reax-eng-" + secrets.token_hex(6)
        cmd = ["docker", "run", "-d", "--name", name, "--network", "none", "--read-only",
               "--tmpfs", "/tmp:rw,exec,size=16g", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
               "--pids-limit", "2048", "--memory", self.memory, "--shm-size", "2g", "--user", self.user,
               "--gpus", f"device={self.gpu}",
               "-v", f"{self.weights_dir}:/model:ro", "-v", f"{os.path.join(d, 'io')}:/run/reax",
               "-e", "RUNTIME_UDS=/run/reax/" + SOCKET_NAME, "-e", "S1M_INTERNAL_TOKEN_FILE=/run/reax/token",
               "-e", "PLUMB_MODEL_DIR=/model", "-e", "HF_HUB_OFFLINE=1", "-e", "OPENAI_API_KEY="]
        if self.docker_runtime:
            cmd += ["--runtime", self.docker_runtime]
        cmd.append(ref)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            shutil.rmtree(d, ignore_errors=True)
            raise EngineError(f"docker run failed: {r.stderr.strip()[-200:]}")

        def stop() -> None:
            try:
                logs = subprocess.run(["docker", "logs", "--tail", "200", name], capture_output=True, text=True,
                                      timeout=60)
                with open(os.path.join(d, "engine.log"), "w") as fh:
                    fh.write(logs.stdout[-100_000:] + logs.stderr[-100_000:])
            except Exception:  # noqa: BLE001 - logs are best effort
                pass
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=120)

        client = UdsSidecarBackend(model, os.path.join(d, "io", SOCKET_NAME), token, params.concurrency,
                                   params.request_timeout_s)
        return Engine(ref, client, stop)


class DevRunner:
    """Local test chains only: ``{"<image>@sha256:<digest>": ["python", "fake_engine.py", "--variant", "fast"]}``."""

    def __init__(self, workdir: str, mapping: dict[str, list[str]]):
        self.workdir = workdir
        self.mapping = mapping

    @classmethod
    def from_file(cls, workdir: str, path: str) -> "DevRunner":
        with open(path, encoding="utf-8") as fh:
            return cls(workdir, json.load(fh))

    def start(self, ref: str, model: str, params: TrackParams) -> Engine:
        argv = self.mapping.get(ref)
        if not argv:
            raise EngineError("image not available to the dev runner")
        d, token = _session_dir(self.workdir)
        sock = os.path.join(d, "io", SOCKET_NAME)
        env = {**os.environ, "RUNTIME_UDS": sock, "S1M_INTERNAL_TOKEN_FILE": os.path.join(d, "io", "token"),
               "OPENAI_API_KEY": ""}
        log_fh = open(os.path.join(d, "engine.log"), "w")
        proc = subprocess.Popen(argv, env=env, stdout=log_fh, stderr=subprocess.STDOUT)

        def stop() -> None:
            proc.terminate()
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()
            log_fh.close()

        client = UdsSidecarBackend(model, sock, token, params.concurrency, params.request_timeout_s)
        return Engine(ref, client, stop)


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


PROBE_GRACE_S = 120.0  # after /health is up, the authenticated test decision must succeed within this time


async def wait_ready(engine: Engine, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    healthy_since = None
    while time.monotonic() < deadline:
        if os.path.exists(engine.client.socket_path) and await engine.client.ready():
            if await engine.client.probe():
                return True
            healthy_since = healthy_since or time.monotonic()
            if time.monotonic() - healthy_since > PROBE_GRACE_S:
                return False
        await asyncio.sleep(1.0)
    return False


def _complete(answers: Answers) -> bool:
    return all(v is not None for v in answers.values())


async def evaluate(runner: Runner, reference_ref: str, candidate_ref: str, track: str, params: TrackParams,
                   block: int = 0, digest: str = "") -> EvalResult:
    """Measure one candidate against the reference in one session (both engines loaded on the same GPU).

    Raises ReferenceError_ when the reference engine itself fails (nothing is recorded then).
    """
    model = params.pool
    res = EvalResult(digest=digest or candidate_ref.rsplit(":", 1)[-1], track=track, ok=False, block=block)
    ref = cand = None
    try:
        try:
            ref = runner.start(reference_ref, model, params)
        except EngineError as exc:
            raise ReferenceError_(f"reference engine did not start: {exc}") from None
        if not await wait_ready(ref, params.start_timeout_s):
            raise ReferenceError_("reference engine not ready")
        try:
            cand = runner.start(candidate_ref, model, params)
        except EngineError as exc:
            res.reason = f"start: {exc}"
            return res
        if not await wait_ready(cand, params.start_timeout_s):
            res.reason = "not ready within the start timeout (or failed the authenticated test decision)"
            return res
        warm = fresh_decisions(min(32, params.timed_items), model)
        await _answers(ref.client, warm)
        await _answers(cand.client, warm)

        ref_tp, cand_tp, ref_ans, cand_ans = [], [], {}, {}
        last_items: list[Decision] = []
        for run in range(params.runs):
            items = fresh_decisions(params.timed_items, model)
            order = [(ref, ref_tp, ref_ans), (cand, cand_tp, cand_ans)]
            if run % 2:
                order.reverse()  # ABBA: neither engine always runs first
            for eng, tps, store in order:
                tp, ans = await timed_run(eng.client, items)
                if eng is ref and not _complete(ans):
                    raise ReferenceError_("reference engine failed on timed items")
                tps.append(tp)
                store.update(ans)
            last_items = items
        res.items = len(ref_ans)
        res.throughput = median(cand_tp)
        res.ref_throughput = median(ref_tp)
        if not _complete(cand_ans):
            missing = sum(v is None for v in cand_ans.values())
            res.reason = f"{missing} of {len(cand_ans)} timed items failed"
            return res
        corr = check_correctness(cand_ans, ref_ans, params)  # type: ignore[arg-type]
        res.mean_distance, res.max_distance, res.top_agreement = corr.mean_distance, corr.max_distance, corr.top_agreement
        if not corr.ok:
            res.reason = f"correctness: {corr.reason}"
            return res

        base, variants = probe_decisions(params.probe_pairs, model)
        p_ref = await _answers(ref.client, base + variants)
        if not _complete(p_ref):
            raise ReferenceError_("reference engine failed on probe items")
        _, p_base = await latencies(cand.client, base)  # one at a time: a cache sees every base item first
        _, p_var = await latencies(cand.client, variants)
        probe = check_correctness({**p_base, **p_var}, p_ref, params, top_agreement=False)  # type: ignore[arg-type]
        res.probe_max_distance = probe.max_distance
        if not probe.ok:
            res.reason = f"anti-cache probe: {probe.reason}"
            return res

        again = last_items[: params.determinism_items]
        _, rep = await latencies(cand.client, again)
        res.determinism_max_distance = max_distance(rep, {d.ident: cand_ans[d.ident] for d in again},  # type: ignore[misc]
                                                    params.temperature)
        if res.determinism_max_distance > params.determinism_tau:
            res.reason = (f"determinism: a repeated item moved by {res.determinism_max_distance:.3f} "
                          f"> {params.determinism_tau}")
            return res

        lat_items = fresh_decisions(params.latency_items, model)
        ref_lat, ref_l_ans = await latencies(ref.client, lat_items)
        cand_lat, cand_l_ans = await latencies(cand.client, lat_items)
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
    finally:
        for eng in (cand, ref):
            if eng is not None:
                try:
                    await eng.client.close()
                finally:
                    eng.stop()

