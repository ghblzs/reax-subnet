"""Serve one candidate model in the pinned vLLM image and read the option-letter logits.

The container has no network (``--network none``), a read-only root filesystem, no capabilities, the model mounted
read-only and a GPU memory cap computed from the model card; vLLM listens on a unix socket in a private directory.
``trust_remote_code`` is never passed. The container is always removed, also on errors and timeouts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import shutil
import string
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import aiohttp

from . import prompt as P

log = logging.getLogger("reax.spec.vllm")

# Same pinned image as runtimes/s1-pro/Dockerfile (vLLM 0.30.0).
VLLM_IMAGE = "vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90"
GPU_TOTAL_GB = 96.0
GPU_BUDGET_GB = 25.0      # measured: s1-fast 12.3 + s1-pro 58.7 + candidate <= 25 = ~96 GB card
MAX_MODEL_LEN = 8192
MAX_NUM_SEQS = 32
STARTUP_TIMEOUT_S = 900
REQUEST_TIMEOUT_S = 120
GIB = 1024 ** 3


class Backend(Protocol):
    """What the evaluator needs from a served model: answer probabilities for rows (None = failed item)."""

    async def decide(self, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, float] | None]: ...


class ModelUnavailable(RuntimeError):
    """The model could not be served (container failed to start, out of memory, ...)."""


def gpu_memory_utilization(weight_bytes: int, *, gpu_total_gb: float = GPU_TOTAL_GB, budget_gb: float = GPU_BUDGET_GB) -> float:
    """Fraction of the GPU vLLM may take: weights + 15 % + a fixed 6 GB for activations and KV cache, capped at the budget.

    Raises ModelUnavailable if the weights alone do not fit the budget."""
    need_gb = weight_bytes * 1.15 / GIB + 6.0
    if weight_bytes * 1.15 / GIB + 2.0 > budget_gb:
        raise ModelUnavailable(f"weights ({weight_bytes / GIB:.1f} GiB) do not fit the {budget_gb:.0f} GB budget")
    return round(min(need_gb, budget_gb) / gpu_total_gb, 3)


@dataclass(frozen=True)
class ServeSpec:
    name: str
    model_dir: str
    sock_dir: str
    gpu_util: float
    gpu_device: str = "0"
    image: str = VLLM_IMAGE


CCFIX_MOUNT = "/opt/reax-ccfix"


def ccfix_docker_args(env: Mapping[str, str] | None = None) -> list[str]:
    """Docker args for confidential-compute GPUs: mount the UVA-off ``sitecustomize.py`` and enable it.

    Empty unless ``REAX_VLLM_DISABLE_UVA=1``. The directory is ``REAX_CCFIX_DIR`` (a HOST path, it is passed to docker -v)
    or ``runtimes/vllm-ccfix`` of a source checkout."""
    env = os.environ if env is None else env
    if env.get("REAX_VLLM_DISABLE_UVA") != "1":
        return []
    d = env.get("REAX_CCFIX_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "runtimes", "vllm-ccfix")
    d = os.path.abspath(d)
    if not os.path.isfile(os.path.join(d, "sitecustomize.py")):
        raise ModelUnavailable(f"REAX_VLLM_DISABLE_UVA=1 but {d}/sitecustomize.py not found (set REAX_CCFIX_DIR)")
    return ["-v", f"{d}:{CCFIX_MOUNT}:ro", "-e", f"PYTHONPATH={CCFIX_MOUNT}", "-e", "REAX_VLLM_DISABLE_UVA=1"]


def build_docker_command(s: ServeSpec) -> list[str]:
    uid = os.getuid() if hasattr(os, "getuid") else 1000
    gid = os.getgid() if hasattr(os, "getgid") else 1000
    return [
        "docker", "run", "-d", "--name", s.name,
        "--gpus", f"device={s.gpu_device}",
        "--network", "none", "--read-only", "--ipc", "private",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--pids-limit", "1024", "--memory", "48g",
        "--user", f"{uid}:{gid}",
        "--tmpfs", "/tmp:rw,size=8g,mode=1777",
        "-e", "HOME=/tmp", "-e", "XDG_CACHE_HOME=/tmp/.cache", "-e", "VLLM_CACHE_ROOT=/tmp/vllm",
        "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1", "-e", "VLLM_NO_USAGE_STATS=1", "-e", "DO_NOT_TRACK=1",
        "-e", "OPENAI_API_KEY=",
        "-v", f"{s.model_dir}:/model:ro",
        "-v", f"{s.sock_dir}:/run/vllm:rw",
        *ccfix_docker_args(),
        s.image,
        "/model", "--served-model-name", "cand",
        "--uds", "/run/vllm/vllm.sock",
        "--max-model-len", str(MAX_MODEL_LEN), "--max-num-seqs", str(MAX_NUM_SEQS),
        "--gpu-memory-utilization", f"{s.gpu_util}",
        "--logprobs-mode", "processed_logits", "--max-logprobs", "64",
        "--generation-config", "vllm", "--no-enable-prefix-caching",
    ]


async def _run(*cmd: str, timeout: float = 60) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, "timeout"
    return proc.returncode or 0, out.decode("utf-8", "replace")[-2000:]


class VllmDockerBackend:
    """Async context manager: ``async with VllmDockerBackend(dir, weight_bytes) as b: await b.decide(rows)``."""

    def __init__(self, model_dir: str, weight_bytes: int, *, gpu_device: str = "0", gpu_total_gb: float = GPU_TOTAL_GB,
                 budget_gb: float = GPU_BUDGET_GB, image: str = VLLM_IMAGE, concurrency: int = MAX_NUM_SEQS):
        self.util = gpu_memory_utilization(weight_bytes, gpu_total_gb=gpu_total_gb, budget_gb=budget_gb)
        self.model_dir = os.path.abspath(model_dir)
        self.gpu_device, self.image, self.concurrency = gpu_device, image, concurrency
        self.name = "reax-spec-" + secrets.token_hex(6)
        self.sock_dir = ""
        self._http: aiohttp.ClientSession | None = None
        self._letter_ids: dict[str, int] = {}

    async def __aenter__(self) -> "VllmDockerBackend":
        self.sock_dir = tempfile.mkdtemp(prefix="reax-spec-sock-")
        os.chmod(self.sock_dir, 0o700)
        try:
            spec = ServeSpec(self.name, self.model_dir, self.sock_dir, self.util, self.gpu_device, self.image)
            code, out = await _run(*build_docker_command(spec), timeout=120)
            if code != 0:
                raise ModelUnavailable(f"docker run failed: {out[-300:]}")
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
        if self.sock_dir:
            shutil.rmtree(self.sock_dir, ignore_errors=True)
            self.sock_dir = ""

    async def _wait_healthy(self) -> None:
        assert self._http is not None
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while time.monotonic() < deadline:
            code, state = await _run("docker", "inspect", "-f", "{{.State.Running}}", self.name, timeout=20)
            if code != 0 or state.strip() != "true":
                _c, logs = await _run("docker", "logs", "--tail", "20", self.name, timeout=20)
                raise ModelUnavailable(f"container exited during start-up: {logs[-400:]}")
            try:
                async with self._http.get("http://vllm/health") as r:
                    if r.status == 200:
                        return
            except (aiohttp.ClientError, OSError, asyncio.TimeoutError):
                pass
            await asyncio.sleep(3)
        raise ModelUnavailable("vLLM not healthy before the start-up deadline")

    async def _load_letter_ids(self) -> None:
        assert self._http is not None
        for letter in string.ascii_uppercase:
            async with self._http.post("http://vllm/tokenize", json={"model": "cand", "prompt": letter, "add_special_tokens": False}) as r:
                tokens = (await r.json()).get("tokens", [])
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
                j = await r.json()
            top = j["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            got = {int(e["token"][9:]): float(e["logprob"]) for e in top if str(e["token"]).startswith("token_id:")}
            return P.answer_probs(row, rendered, [got[i] for i in ids])
        except (P.PromptError, KeyError, ValueError, TypeError, IndexError, aiohttp.ClientError, OSError, asyncio.TimeoutError):
            return None

    async def decide(self, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, float] | None]:
        sem = asyncio.Semaphore(self.concurrency)

        async def go(r: Mapping[str, Any]) -> dict[str, float] | None:
            async with sem:
                return await self._one(r)

        return list(await asyncio.gather(*(go(r) for r in rows)))
