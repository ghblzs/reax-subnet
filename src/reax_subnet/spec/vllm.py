"""Serve one candidate model in the pinned vLLM image and read the option-letter logits.

The container has no network (``--network none``), a read-only root filesystem, no capabilities, the model mounted
read-only and a GPU memory cap computed from the model card; vLLM listens on a unix socket in a private directory
under the host-visible work directory. It runs as its own uid (``SPEC_USER``) with CPU/memory/swap/ulimit/log caps and
a fixed chat template per architecture (``templates.py``); the submission's own template is never rendered.
``trust_remote_code`` is never passed. The container is always removed, also on errors and timeouts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import secrets
import shutil
import string
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import aiohttp

from . import prompt as P
from . import templates as TPL

log = logging.getLogger("reax.spec.vllm")

# Same pinned image as runtimes/s1-pro/Dockerfile (vLLM 0.30.0).
VLLM_IMAGE = "vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90"
GPU_TOTAL_GB = 96.0       # fallback when nvidia-smi cannot be read; normally the card size is detected
GPU_BUDGET_GB = 25.0      # the candidate must fit next to the s1-fast and s1-pro reference runtimes
# GPU memory the vLLM process uses OUTSIDE its --gpu-memory-utilization share: the CUDA context (created before vLLM
# takes its start-up snapshot, so never counted) and allocations made after the memory profile (kernel warm-up,
# first batches). Measured 1.22 GiB (0.62 + 0.61) for Qwen3.5-9B bf16 on an RTX PRO 6000 Blackwell (CC mode); 1.5 GiB
# leaves a margin. Without it a model capped at the budget takes budget + 1.2 GiB and dies with CUDA OOM mid-scoring.
GPU_OVERHEAD_GB = 1.5
MAX_MODEL_LEN = 8192
MAX_NUM_SEQS = 32
STARTUP_TIMEOUT_S = 900
REQUEST_TIMEOUT_S = 120
GIB = 1024 ** 3
# sitecustomize.py that switches vLLM's UVA off (confidential-compute GPUs: UVA views of pinned host memory go stale).
# Enabled with REAX_VLLM_DISABLE_UVA=1; the file is copied into the run directory and mounted read-only.
CCFIX_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ccfix", "sitecustomize.py")
UVA_ENV = "REAX_VLLM_DISABLE_UVA"
SPEC_USER = "10003:10003"     # the vLLM container's uid:gid: not the validator's, so an escape does not own the hotkey file
MAX_RESPONSE_BYTES = 1 << 20  # a vLLM answer is a few KB; anything larger is a hostile or broken container
AUX_MOUNT = "/aux"            # read-only: the fixed chat template (and the UVA fix)
SOCK_MOUNT = "/run/vllm"


class Backend(Protocol):
    """What the evaluator needs from a served model: answer probabilities for rows (None = failed item)."""

    async def decide(self, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, float] | None]: ...


class ModelUnavailable(RuntimeError):
    """The model could not be served (container failed to start, out of memory, ...)."""


class HostUnavailable(ModelUnavailable):
    """The model could not be served for a reason that is not the model's: docker refused to run, the socket cannot be
    connected, the container was never healthy. Inconclusive: it does not count toward ``MAX_ATTEMPTS``."""


class GPUPreflightUnavailable(HostUnavailable):
    """The GPU preflight failed before the model was started."""


class ModelBudgetOOM(ModelUnavailable):
    """Model OOM after a successful free-GPU preflight, within the configured budget."""


def model_specific_failure(detail: str) -> bool:
    """Conservative evidence for a lone champion; generic container exits and CUDA faults are inconclusive."""
    detail = detail.lower()
    if any(x in detail for x in ("cuda driver", "driver initialization", "cuda initialization", "nccl", "disk", "i/o error")):
        return False
    return any(x in detail for x in ("invalid config", "invalid tokenizer", "invalid weights", "unsupported architecture",
                                    "no fixed chat template for architecture", "not a single token for this tokenizer",
                                    "safetensorerror", "error while deserializing header", "size mismatch for",
                                    "weights ("))


# vLLM refuses to start when another process holds part of the GPU: "Free memory on device cuda:0 (24.27/94.86 GiB) on
# startup is less than desired GPU memory utilization (0.261, 24.76 GiB)". The share is set from the weights, so this is
# the host (memory held by someone else), not the model.
_HOST_GPU_MEMORY = re.compile(
    r"^(?:\([^\n)]{1,100}\)\s+)?(?:\[rank\d+\]:\s+)?(?:ERROR\s+[^\n]{0,160}?\]\s+)?\s*ValueError: "
    r"Free memory on device [^\n]{0,120}?less than desired GPU memory utilization[^\n]*$", re.M)


STARTUP_LOG_TAIL = 200   # lines read when a container exits during start-up (classification only)
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def host_lacks_gpu_memory(logs: str) -> bool:
    return bool(_HOST_GPU_MEMORY.search(_ANSI.sub("", logs or "")))


def gpu_memory_utilization(weight_bytes: int, *, gpu_total_gb: float = GPU_TOTAL_GB, budget_gb: float = GPU_BUDGET_GB) -> float:
    """Fraction of the GPU vLLM may take: weights + 15 % + a fixed 6 GB for activations and KV cache, capped so that this
    share plus ``GPU_OVERHEAD_GB`` (what the process uses outside the share) stays within the budget.

    Raises ModelUnavailable if the weights alone do not fit the budget."""
    need_gb = weight_bytes * 1.15 / GIB + 6.0
    if weight_bytes * 1.15 / GIB + 2.0 > budget_gb:
        raise ModelUnavailable(f"weights ({weight_bytes / GIB:.1f} GiB) do not fit the {budget_gb:.0f} GB budget")
    if budget_gb > gpu_total_gb:
        budget_gb = gpu_total_gb
    # Round DOWN so the instance never exceeds the budget.
    return math.floor(min(need_gb, budget_gb - GPU_OVERHEAD_GB) / gpu_total_gb * 1000) / 1000


def detect_gpu_total_gb(device: str = "0") -> float:
    """Memory of the card in GiB, from nvidia-smi; falls back to GPU_TOTAL_GB if it cannot be read."""
    import subprocess

    try:
        out = subprocess.run(["nvidia-smi", "-i", str(device), "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20, check=True).stdout.strip().splitlines()[0]
        return float(out) / 1024.0
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        log.warning("could not read the GPU memory size from nvidia-smi; assuming %.0f GiB", GPU_TOTAL_GB)
        return GPU_TOTAL_GB


@dataclass(frozen=True)
class ServeSpec:
    name: str
    model_dir: str
    sock_dir: str
    gpu_util: float
    gpu_device: str = "0"
    image: str = VLLM_IMAGE
    disable_uva: bool = False
    eager: bool = True     # skip torch.compile and CUDA-graph capture: one token per item, start-up dominates the wall time
    aux_dir: str = ""      # read-only mount at /aux: holds ``template`` (and sitecustomize.py with disable_uva)
    template: str = ""     # file name inside aux_dir, passed as --chat-template
    user: str = ""         # uid:gid of the container; "" = the caller's own (the miner CLI)
    runtime: str = ""      # docker --runtime (e.g. runsc), REAX_SPEC_DOCKER_RUNTIME
    cpus: str = "8"


def build_docker_command(s: ServeSpec) -> list[str]:
    uid = os.getuid() if hasattr(os, "getuid") else 1000
    gid = os.getgid() if hasattr(os, "getgid") else 1000
    user = s.user or f"{uid}:{gid}"
    uva = ["-e", f"PYTHONPATH={AUX_MOUNT}", "-e", f"{UVA_ENV}=1"] if s.disable_uva else []
    aux = ["-v", f"{s.aux_dir}:{AUX_MOUNT}:ro"] if s.aux_dir else []
    tpl = ["--chat-template", f"{AUX_MOUNT}/{s.template}", "--chat-template-content-format", "string"] if s.template else []
    return [
        "docker", "run", "-d", "--name", s.name,
        "--gpus", f"device={s.gpu_device}",
        *(["--runtime", s.runtime] if s.runtime else []),
        "--network", "none", "--read-only", "--ipc", "private",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--pids-limit", "1024",
        "--memory", "48g", "--memory-swap", "48g", "--cpus", s.cpus, "--ulimit", "nofile=4096:4096",
        "--log-driver", "local", "--log-opt", "max-size=10m", "--log-opt", "max-file=2",
        "--user", user,
        # exec: Docker mounts tmpfs noexec by default, and Triton loads the shared objects it compiles into HOME/.triton.
        "--tmpfs", "/tmp:rw,exec,nosuid,nodev,size=8g,mode=1777",
        "-e", "HOME=/tmp", "-e", "XDG_CACHE_HOME=/tmp/.cache", "-e", "VLLM_CACHE_ROOT=/tmp/vllm",
        "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1", "-e", "VLLM_NO_USAGE_STATS=1", "-e", "DO_NOT_TRACK=1",
        "-e", "OPENAI_API_KEY=",
        # vLLM binds the --uds socket itself (mode 0755 under the default umask): umask 000 lets the worker uid connect.
        "--entrypoint", "sh",
        "-v", f"{s.model_dir}:/model:ro",
        "-v", f"{s.sock_dir}:{SOCK_MOUNT}:rw",
        *aux,
        *uva,
        s.image,
        "-c", 'umask 000; exec vllm serve "$@"', "--",
        "/model", "--served-model-name", "cand",
        "--uds", f"{SOCK_MOUNT}/vllm.sock",
        "--max-model-len", str(MAX_MODEL_LEN), "--max-num-seqs", str(MAX_NUM_SEQS),
        "--gpu-memory-utilization", f"{s.gpu_util}",
        "--logprobs-mode", "processed_logits", "--max-logprobs", "64",
        "--generation-config", "vllm", "--no-enable-prefix-caching",
        *tpl,
        *(["--enforce-eager"] if s.eager else []),
    ]


def architecture_of(model_dir: str) -> str:
    """The single architecture named by config.json (read as data, size-capped); "" if unreadable."""
    from .rules import read_json_file

    try:
        archs = read_json_file(os.path.join(model_dir, "config.json")).get("architectures")
        return archs[0] if isinstance(archs, list) and len(archs) == 1 and isinstance(archs[0], str) else ""
    except (OSError, ValueError, AttributeError):
        return ""


def prepare_run_dir(work_dir: str | None, *, template: str, disable_uva: bool) -> tuple[str, str, str]:
    """``(run_dir, sock_dir, aux_dir)``. ``run_dir`` is private (0700) and lives under ``<work_dir>/sock`` so the docker
    daemon, which resolves ``-v`` sources on the HOST, finds it where the validator sees it (the work directory is
    mounted at the same path); without a work directory (the miner CLI on a plain host) the system temp dir is used.
    ``sock_dir`` is mounted read-write (0777: the container uid creates the socket there, nothing else is in it),
    ``aux_dir`` read-only with the fixed chat template (copied from the package) and, if asked for, the UVA fix."""
    if work_dir:
        root = os.path.join(os.path.abspath(work_dir), "sock")
        os.makedirs(root, mode=0o700, exist_ok=True)
        run_dir = tempfile.mkdtemp(prefix="sock-", dir=root)
    else:
        run_dir = tempfile.mkdtemp(prefix="reax-spec-sock-")
    os.chmod(run_dir, 0o700)
    sock_dir, aux_dir = os.path.join(run_dir, "s"), os.path.join(run_dir, "a")
    os.mkdir(sock_dir)
    os.chmod(sock_dir, 0o777)
    os.mkdir(aux_dir)
    shutil.copyfile(TPL.template_path(template), os.path.join(aux_dir, template))
    if disable_uva:
        shutil.copyfile(CCFIX_FILE, os.path.join(aux_dir, "sitecustomize.py"))
    for name in os.listdir(aux_dir):
        os.chmod(os.path.join(aux_dir, name), 0o644)
    os.chmod(aux_dir, 0o755)
    return run_dir, sock_dir, aux_dir


async def _run(*cmd: str, timeout: float = 60) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, "timeout"
    return proc.returncode or 0, out.decode("utf-8", "replace")[-2000:]


async def check_gpu_free(device: str, required_gib: float) -> None:
    """Wait up to 60 seconds for memory released by a previous container. Never start on a failed probe."""
    deadline = time.monotonic() + 60
    detail = "GPU free-memory probe failed"
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise GPUPreflightUnavailable(detail)
        try:
            code, out = await _run("nvidia-smi", "-i", str(device), "--query-gpu=memory.free",
                                   "--format=csv,noheader,nounits", timeout=min(10, remaining))
            free = float(out.strip().splitlines()[0]) / 1024 if code == 0 else -1
            if free >= required_gib:
                return
            detail = f"GPU not free before start ({free:.1f} GiB free, {required_gib:.1f} GiB needed)"
        except (OSError, ValueError, IndexError) as exc:
            detail = f"GPU free-memory probe failed: {exc}"
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise GPUPreflightUnavailable(detail)
        await asyncio.sleep(min(2, remaining))


async def _read_json(r: aiohttp.ClientResponse, limit: int = MAX_RESPONSE_BYTES) -> Any:
    """The JSON body of a vLLM response, at most ``limit`` bytes (like SidecarBackend): a compromised container must
    not be able to exhaust the validator's memory."""
    chunks, size = [], 0
    async for chunk in r.content.iter_chunked(64 * 1024):
        size += len(chunk)
        if size > limit:
            raise ModelUnavailable("vLLM response too large")
        chunks.append(chunk)
    return json.loads(b"".join(chunks))


class VllmDockerBackend:
    """Async context manager: ``async with VllmDockerBackend(dir, weight_bytes) as b: await b.decide(rows)``."""

    def __init__(self, model_dir: str, weight_bytes: int, *, gpu_device: str = "0", gpu_total_gb: float | None = None,
                 budget_gb: float = GPU_BUDGET_GB, image: str = VLLM_IMAGE, concurrency: int = MAX_NUM_SEQS,
                 disable_uva: bool | None = None, work_dir: str | None = None, user: str = "", runtime: str = "",
                 cpus: str = "8"):
        if not gpu_total_gb:
            gpu_total_gb = detect_gpu_total_gb(gpu_device)
        self.disable_uva = os.environ.get(UVA_ENV) == "1" if disable_uva is None else disable_uva
        self.util = gpu_memory_utilization(weight_bytes, gpu_total_gb=gpu_total_gb, budget_gb=budget_gb)
        self.required_gib = self.util * gpu_total_gb + GPU_OVERHEAD_GB
        self.model_dir = os.path.abspath(model_dir)
        self.gpu_device, self.image, self.concurrency = gpu_device, image, concurrency
        self.name = "reax-spec-" + secrets.token_hex(6)
        self.work_dir, self.user, self.runtime, self.cpus = work_dir, user, runtime, cpus
        self.run_dir = self.sock_dir = ""
        self._http: aiohttp.ClientSession | None = None
        self._letter_ids: dict[str, int] = {}
        self._gpu_preflight_ok = False

    async def __aenter__(self) -> "VllmDockerBackend":
        arch = architecture_of(self.model_dir)
        template = TPL.template_name(arch)
        if template is None:
            raise ModelUnavailable(f"no fixed chat template for architecture {arch!r}")
        await check_gpu_free(self.gpu_device, self.required_gib)
        self._gpu_preflight_ok = True
        self.run_dir, self.sock_dir, aux_dir = prepare_run_dir(self.work_dir, template=template, disable_uva=self.disable_uva)
        try:
            spec = ServeSpec(self.name, self.model_dir, self.sock_dir, self.util, self.gpu_device, self.image, self.disable_uva,
                             aux_dir=aux_dir, template=template, user=self.user, runtime=self.runtime, cpus=self.cpus)
            code, out = await _run(*build_docker_command(spec), timeout=120)
            if code != 0:
                raise HostUnavailable(f"docker run failed: {out[-300:]}")
            self._http = aiohttp.ClientSession(connector=aiohttp.UnixConnector(path=os.path.join(self.sock_dir, "vllm.sock")),
                                               timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S))
            await self._wait_healthy()
            await self._load_letter_ids()
        except BaseException:
            await self.__aexit__(None, None, None)
            raise
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._http is not None:
            await self._http.close()
            self._http = None
        if self.name:
            await _run("docker", "rm", "-f", self.name, timeout=60)
        if self.run_dir:
            shutil.rmtree(self.run_dir, ignore_errors=True)
            self.run_dir = self.sock_dir = ""

    async def _wait_healthy(self) -> None:
        assert self._http is not None
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while time.monotonic() < deadline:
            code, state = await _run("docker", "inspect", "-f", "{{.State.Running}}", self.name, timeout=20)
            if code != 0:
                raise HostUnavailable(f"docker inspect failed: {state[-300:]}")
            if state.strip() != "true":
                code, logs = await _run("docker", "logs", "--tail", str(STARTUP_LOG_TAIL), self.name, timeout=20)
                if code != 0:
                    raise HostUnavailable(f"docker logs failed: {logs[-300:]}")
                host_gpu = host_lacks_gpu_memory(logs)   # the vLLM error line sits ~40 lines above the end (critic FX1)
                logs = "\n".join(logs.splitlines()[-20:])
                if host_gpu:
                    raise HostUnavailable(f"container exited during start-up, the GPU has too little free memory: {logs[-400:]}")
                if self._gpu_preflight_ok and re.search(r"(?:torch\.OutOfMemoryError: CUDA out of memory|CUDA out of memory\.)", logs):
                    raise ModelBudgetOOM(f"container exited during start-up: {logs[-400:]}")
                raise ModelUnavailable(f"container exited during start-up: {logs[-400:]}")
            try:
                async with self._http.get("http://vllm/health") as r:
                    if r.status == 200:
                        return
            except PermissionError as exc:  # the socket exists but this uid may not connect: waiting will not help
                raise HostUnavailable(f"vLLM socket is not connectable by this user ({exc})") from None
            except (aiohttp.ClientError, OSError, asyncio.TimeoutError) as exc:
                if isinstance(getattr(exc, "os_error", None), PermissionError):
                    raise HostUnavailable(f"vLLM socket is not connectable by this user ({exc.os_error})") from None
            await asyncio.sleep(3)
        raise HostUnavailable("vLLM not healthy before the start-up deadline")

    async def _load_letter_ids(self) -> None:
        assert self._http is not None
        for letter in string.ascii_uppercase:
            async with self._http.post("http://vllm/tokenize", json={"model": "cand", "prompt": letter, "add_special_tokens": False}) as r:
                tokens = (await _read_json(r)).get("tokens", [])
            if len(tokens) != 1:
                raise ModelUnavailable(f"option letter {letter!r} is not a single token for this tokenizer")
            self._letter_ids[letter] = tokens[0]

    async def _one(self, row: Mapping[str, Any]) -> dict[str, float] | None:
        assert self._http is not None
        try:
            rendered = P.render(row["state"], row["question"])
            ids = [self._letter_ids[lab] for lab in rendered.letters]
            body = {"model": "cand", "messages": [{"role": "system", "content": rendered.system},
                                                  {"role": "user", "content": rendered.user}],
                    "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": len(ids),
                    "allowed_token_ids": ids, "return_tokens_as_token_ids": True,
                    "chat_template_kwargs": {"enable_thinking": False}}
            async with self._http.post("http://vllm/v1/chat/completions", json=body) as r:
                if r.status != 200:
                    return None
                j = await _read_json(r)
            top = j["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            got = {int(e["token"][9:]): float(e["logprob"]) for e in top if str(e["token"]).startswith("token_id:")}
            return P.answer_probs(row, rendered, [got[i] for i in ids])
        except (P.PromptError, KeyError, ValueError, TypeError, IndexError, AttributeError, aiohttp.ClientError, OSError,
                asyncio.TimeoutError):
            return None

    async def decide(self, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, float] | None]:
        sem = asyncio.Semaphore(self.concurrency)

        async def go(r: Mapping[str, Any]) -> dict[str, float] | None:
            async with sem:
                return await self._one(r)

        return list(await asyncio.gather(*(go(r) for r in rows)))
