"""Configuration of the specialist evaluator (environment variables ``REAX_SPEC_*``, see docs/SPECIALIST-TRACKS.md)."""

from __future__ import annotations

import calendar
import json
import os
import time
from dataclasses import dataclass, field
from typing import Mapping

from ..competition_split import ROTATION_DAYS, TRACK_SCHEDULE
from . import machine, rules, sampling, vllm

MAX_EVALS_PER_HOUR = 6            # served-model evaluations per hour, all tracks together (incumbent runs count)
MAX_ATTEMPTS = 3
PUBLISH_DEADLINE_BLOCKS = 7200    # the repo must be public within ~24 h of the commitment (DESIGN.md section 3)
RETRY_UNAVAILABLE_S = 600
MAX_OPEN_PER_COLDKEY_TRACK = 2
MAX_NEW_PER_COLDKEY_WEEK = 4
MAX_LEADERS = 3                   # champion + two former champions (the split layer decides the shares)


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
    gpu_total_gb: float = vllm.GPU_TOTAL_GB
    gpu_budget_gb: float = vllm.GPU_BUDGET_GB
    image: str = vllm.VLLM_IMAGE
    base_dir: str = ""

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
        gpu_total_gb=float(env.get("REAX_SPEC_GPU_TOTAL_GB", vllm.GPU_TOTAL_GB)),
        gpu_budget_gb=float(env.get("REAX_SPEC_GPU_BUDGET_GB", vllm.GPU_BUDGET_GB)),
        image=env.get("REAX_SPEC_VLLM_IMAGE", vllm.VLLM_IMAGE),
        base_dir=env.get("REAX_SPEC_BASE_DIR", ""),
    )


def dir_base_provider(base_dir: str):
    """Base models prepared by the operator as ``<base_dir>/<hf id with / -> __>@<revision>/``."""
    def provider(base_model: str) -> str | None:
        if not base_dir:
            return None
        p = os.path.join(base_dir, base_model.replace("/", "__", 1))
        return p if os.path.isdir(p) else None
    return provider
