"""Configuration of the specialist evaluator (environment variables ``REAX_SPEC_*``, see docs/SPECIALIST-TRACKS.md)."""

from __future__ import annotations

import calendar
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Mapping

from ..competition_split import ROTATION_DAYS, TRACK_SCHEDULE
from . import machine, rules, sampling, vllm
from .commitment import _valid_repo

MAX_EVALS_PER_HOUR = 6            # served-model evaluations per hour, all tracks together (incumbent runs count)
MAX_ATTEMPTS = 3
PUBLISH_DEADLINE_BLOCKS = 7200    # the repo must be public within ~24 h of the commitment (docs/SPECIALIST-TRACKS.md)
RETRY_UNAVAILABLE_S = 600
MAX_OPEN_PER_COLDKEY_TRACK = 2
MAX_NEW_PER_COLDKEY_WEEK = 4
MIN_FREE_GB = 60                  # free space on the work directory's filesystem before a submission is downloaded
MAX_LEADERS = 3                   # champion + two former champions (the split layer decides the shares)
BASE_RETRY_CYCLES = 6             # a declared base that could not be fetched is tried again after this many cycles
BASE_BUDGET_GB = 40               # base model downloads per cycle (spec/bases.py)
BASE_MIN_FREE_GB = 50             # free space kept on the base directory's filesystem after a base download


@dataclass(frozen=True)
class TrackDef:
    track_id: str
    category: str
    start_ts: float
    days: int = ROTATION_DAYS

    def active(self, now: float) -> bool:
        return self.start_ts <= now < self.start_ts + self.days * 86400


def _ts(text: str) -> float:
    return float(calendar.timegm(time.strptime(text, "%Y-%m-%dT%H:%M:%SZ")))


def _from_schedule() -> tuple[TrackDef, ...]:
    """The release schedule (competition_split.TRACK_SCHEDULE) is the one source of truth; the sealed category of a
    track is its id. REAX_SPEC_TRACKS_FILE (development only) replaces it."""
    return tuple(TrackDef(t.track_id, t.track_id, t.start.timestamp(), round((t.end - t.start).total_seconds() / 86400))
                 for t in TRACK_SCHEDULE)


DEFAULT_TRACKS = _from_schedule()


@dataclass
class SpecConfig:
    sealed_dir: str
    work_dir: str
    state_path: str
    tracks: tuple[TrackDef, ...] = DEFAULT_TRACKS
    rules: rules.TrackRules = field(default_factory=rules.TrackRules)
    sample_size: int = sampling.SAMPLE_SIZE
    max_evals_per_hour: int = MAX_EVALS_PER_HOUR
    confirm_delay_s: float = machine.CONFIRM_DELAY_S
    entry_floor: float = 0.0
    gpu_device: str = "0"
    gpu_total_gb: float = 0.0          # 0 = read the card size from nvidia-smi
    gpu_budget_gb: float = vllm.GPU_BUDGET_GB
    image: str = vllm.VLLM_IMAGE
    base_dir: str = ""
    user: str = vllm.SPEC_USER         # uid:gid of the vLLM container (not the validator's uid)
    docker_runtime: str = ""           # e.g. runsc (gVisor); same idea as REAX_ENGINE_DOCKER_RUNTIME
    cpus: str = "8"
    min_free_bytes: int = MIN_FREE_GB * 1024 ** 3
    base_fetch: bool = True            # download a missing declared base into base_dir (spec/bases.py)
    base_retry_cycles: int = BASE_RETRY_CYCLES
    base_budget_bytes: int = BASE_BUDGET_GB * 1024 ** 3
    base_min_free_bytes: int = BASE_MIN_FREE_GB * 1024 ** 3
    base_preseed: tuple[str, ...] = ()  # "<hf id>@<revision>" fetched in the background when the evaluator starts

    def active_tracks(self, now: float) -> list[TrackDef]:
        return [t for t in self.tracks if t.active(now)]


def load_tracks(path: str) -> tuple[TrackDef, ...]:
    """JSON list of {"track_id", "category", "start": "YYYY-MM-DDTHH:MM:SSZ", "days": 14}."""
    with open(path) as fh:
        raw = json.load(fh)
    return tuple(TrackDef(str(t["track_id"]), str(t["category"]), _ts(t["start"]), int(t.get("days", ROTATION_DAYS))) for t in raw)


def from_env(env: Mapping[str, str] = os.environ, state_dir: str = "") -> SpecConfig:
    sealed = env.get("REAX_SPEC_SEALED_DIR", "")
    if not sealed:
        raise ValueError("REAX_SPEC_SEALED_DIR (host path of the sealed pool, mounted read-only) is required")
    base = state_dir or env.get("REAX_SPEC_STATE_DIR", "") or os.path.expanduser("~/.reax")
    tracks = load_tracks(env["REAX_SPEC_TRACKS_FILE"]) if env.get("REAX_SPEC_TRACKS_FILE") else DEFAULT_TRACKS
    return SpecConfig(
        sealed_dir=sealed,
        work_dir=env.get("REAX_SPEC_WORK_DIR", os.path.join(base, "spec-work")),
        state_path=env.get("REAX_SPEC_STATE", os.path.join(base, "spec-state.json")),
        tracks=tracks,
        sample_size=int(env.get("REAX_SPEC_SAMPLE_SIZE", sampling.SAMPLE_SIZE)),
        max_evals_per_hour=int(env.get("REAX_SPEC_MAX_EVALS_PER_HOUR", MAX_EVALS_PER_HOUR)),
        gpu_device=env.get("REAX_SPEC_GPU_DEVICE", "0"),
        gpu_total_gb=float(env.get("REAX_SPEC_GPU_TOTAL_GB", 0)),
        gpu_budget_gb=float(env.get("REAX_SPEC_GPU_BUDGET_GB", vllm.GPU_BUDGET_GB)),
        image=env.get("REAX_SPEC_VLLM_IMAGE", vllm.VLLM_IMAGE),
        base_dir=env.get("REAX_SPEC_BASE_DIR", ""),
        user=_user(env.get("REAX_SPEC_UID", vllm.SPEC_USER)),
        docker_runtime=env.get("REAX_SPEC_DOCKER_RUNTIME", "").strip(),
        cpus=env.get("REAX_SPEC_CPUS", "8").strip() or "8",
        min_free_bytes=int(float(env.get("REAX_SPEC_MIN_FREE_GB", MIN_FREE_GB)) * 1024 ** 3),
        base_fetch=env.get("REAX_SPEC_BASE_FETCH", "1").strip().lower() not in ("0", "false", "no", "off"),
        base_retry_cycles=max(1, int(env.get("REAX_SPEC_BASE_RETRY_CYCLES", BASE_RETRY_CYCLES))),
        base_budget_bytes=int(float(env.get("REAX_SPEC_BASE_BUDGET_GB", BASE_BUDGET_GB)) * 1024 ** 3),
        base_min_free_bytes=int(float(env.get("REAX_SPEC_BASE_MIN_FREE_GB", BASE_MIN_FREE_GB)) * 1024 ** 3),
        base_preseed=tuple(x.strip() for x in env.get("REAX_SPEC_BASE_PRESEED", "").split(",") if x.strip()),
    )


def _user(value: str) -> str:
    """``uid:gid`` of the vLLM container; it must not be root."""
    value = value.strip()
    parts = value.split(":")
    if len(parts) != 2 or not all(p.isdigit() for p in parts) or int(parts[0]) == 0:
        raise ValueError(f"REAX_SPEC_UID must be <uid>:<gid> with a non-root uid, got {value!r}")
    return value


_REVISION_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}")


def base_dir_name(base_model: str) -> str | None:
    """``<hf id>@<revision>`` -> its directory name ``<hf id with / -> __>@<revision>``; None if it is not a well-formed
    name (it comes from a miner's manifest: no path separators, no ``..``, nothing that starts with a dot)."""
    repo, sep, rev = base_model.partition("@")
    if not sep or not _valid_repo(repo) or not _REVISION_NAME.fullmatch(rev) or ".." in rev:
        return None
    return repo.replace("/", "__", 1) + "@" + rev


def dir_base_provider(base_dir: str):
    """Base models prepared by the operator (or fetched by spec/bases.py) as ``<base_dir>/<hf id with / -> __>@<revision>/``."""
    def provider(base_model: str) -> str | None:
        name = base_dir_name(base_model) if base_dir else None
        if name is None:
            return None
        p = os.path.join(base_dir, name)
        return p if os.path.isdir(p) else None
    return provider
