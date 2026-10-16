"""``reax-competition-worker``: runs ONLY the competition cycles (docs/COMPETITIONS.md, "Remote mode").

* Reads the chain read-only (no wallet, no hotkey, no weights, no runtime or router tokens).
* Runs the engine competition (``--engine``) and/or the specialist tracks (``--spec``; both when neither is given) with
  docker, on one GPU under one lock, with the same preflights and anti-cheat as a validator in local mode.
* Writes ``engine-competition.json`` and ``spec-state.json`` atomically into ``REAX_COMPETITIONS_STATE_DIR``; a heartbeat
  refreshes their ``saved_at`` while a cycle runs, so the validator (REAX_COMPETITIONS_MODE=remote) can tell a live worker
  from a dead one (26 h limit).

It runs unchanged (a) as a container on the validator host WITHOUT wallets or tokens mounted, and (b) on a separate
keyless GPU box. Whoever reads its state files pays; this process never sets weights.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Optional

from .competition_split import COMPETITIONS_START_UTC, ENGINE_HELD, SPLIT_PROFILE, release_engine_start
from .competitions import CompetitionDriver, SpecialistAdapter
from .config import ENGINE_STATE_FILE, SPEC_STATE_FILE, CommonConfig, ConfigError, EngineConfig, _bool, _engine, _int, load_common
from .engcomp import RewardParams
from .engvalidator import TRACKS, EngineCompetition

log = logging.getLogger("reax.worker")


# ------------------------------------------------------------------------------------------------ building the stack
def engine_params(e: EngineConfig):
    """Track and reward parameters: the release constants; development networks may override them (JSON)."""
    tracks, reward = dict(TRACKS), RewardParams()
    if e.dev_params:
        over = json.loads(e.dev_params)
        tracks = {t: dataclasses.replace(p, **over.get("track", {})) for t, p in tracks.items()}
        reward = dataclasses.replace(reward, **over.get("reward", {}))
    return tracks, reward


def build_specialist(sealed_dir: str, e: EngineConfig, env: Mapping[str, str], *, state_path: str = "") -> SpecialistAdapter:
    """The specialist-track evaluator (reax_subnet.spec) behind the adapter the share layer reads. Its schedule is
    competition_split.TRACK_SCHEDULE; REAX_SPEC_TRACKS_FILE is a development-network override only."""
    from .spec import config as spec_config
    from .spec.evaluator import SpecEvaluator

    senv = dict(env)
    senv["REAX_SPEC_SEALED_DIR"] = sealed_dir
    senv.setdefault("REAX_SPEC_GPU_DEVICE", e.gpu)  # one GPU for both
    if state_path:
        senv["REAX_SPEC_STATE"] = state_path
    if e.release_mode and senv.pop("REAX_SPEC_TRACKS_FILE", None):
        log.warning("REAX_SPEC_TRACKS_FILE ignored on finney: the track schedule is a release constant")
    spec_cfg = spec_config.from_env(senv)
    return SpecialistAdapter(SpecEvaluator(spec_cfg), gpu=spec_cfg.gpu_device)


def build_local_stack(e: EngineConfig, sealed_dir: str, env: Mapping[str, str], *, run_engine: bool = True,
                      state_dir: str = "", award_gate: Any = None):
    """``(engine, adapter, spec_evaluator)`` for a process that runs the cycles itself (a validator in local mode, or the
    worker). ``engine`` is an ``IdleEngine`` and its file untouched when ``run_engine`` is False (``--spec`` only).
    On finney the engine track starts at ``release_engine_start()``: while it is held no engine cycle ever runs.
    ``award_gate``: the validator's owner-approval gate (award.py); the worker pays nothing and passes none."""
    from .engrunner import DevRunner, DockerRunner
    from .spec.vllm import VLLM_IMAGE

    start = COMPETITIONS_START_UTC if e.release_mode else None
    engine_start = release_engine_start() if e.release_mode else None
    adapter = build_specialist(sealed_dir, e, env, state_path=os.path.join(state_dir, SPEC_STATE_FILE) if state_dir else "") \
        if sealed_dir else None
    if not run_engine:
        return IdleEngine(start), adapter, adapter.evaluator if adapter else None
    if e.runner == "dev":
        runner = DevRunner.from_file(e.workdir, e.dev_map)
    else:
        runner = DockerRunner(e.workdir, e.weights_dir, gpu=e.gpu, docker_runtime=e.docker_runtime,
                              registries=e.registries, memory=e.memory, cpus=e.cpus, nvml_check=e.nvml_check,
                              protected_images=(e.reference, env.get("REAX_SPEC_VLLM_IMAGE", VLLM_IMAGE)))
    tracks, reward = engine_params(e)
    log.info("engine competition: %s runner, reference %s, split profile %s, starts %s", e.runner, e.reference,
             SPLIT_PROFILE, "NEVER: the engine track is HELD (no engine cycle runs)" if engine_start == ENGINE_HELD
             else (engine_start or start).isoformat() if (engine_start or start) else "immediately (development network)")
    engine = EngineCompetition(runner, {"s1-fast": e.reference}, e.state_path, tracks=tracks, reward=reward,
                               interval_s=e.interval_s, max_evals=e.max_evals, start_utc=start, specialist=adapter,
                               engine_start_utc=engine_start, award_gate=award_gate)
    return engine, adapter, adapter.evaluator if adapter else None


class IdleEngine:
    """Placeholder engine of a ``--spec`` only worker: never due, so the driver only runs the specialist cycles."""

    def __init__(self, start_utc: Optional[datetime]):
        self.start_utc = start_utc

    def due(self) -> bool:
        return False

    async def run_cycle(self, snap: Any, exclude: set[int], between: Any = None) -> dict[str, list[dict]]:
        return {}


# ------------------------------------------------------------------------------------------------ state files
def touch_state(path: str, now: Optional[float] = None) -> bool:
    """Refresh ``saved_at`` of an existing state file atomically, without re-serialising live objects (a cycle may be
    changing them in an executor thread). False if the file is missing or unreadable."""
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, ValueError):
        return False
    if not isinstance(raw, dict):
        return False
    raw["saved_at"] = int(time.time() if now is None else now)
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=".touch-", dir=directory)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(raw, fh, indent=1, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o644)  # the validator container reads it with its own uid
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    return True


# ------------------------------------------------------------------------------------------------ config
@dataclass(frozen=True)
class WorkerConfig:
    common: CommonConfig
    engine: EngineConfig            # also describes the GPU / release mode when only --spec runs
    spec_sealed_dir: str            # "" = no specialist tracks
    state_dir: str
    run_engine: bool
    run_spec: bool
    poll_s: int = 30
    heartbeat_s: int = 300


def _inside(child: str, parent: str) -> bool:
    child, parent = os.path.abspath(child), os.path.abspath(parent)
    return os.path.commonpath([child, parent]) == parent


def load_worker(env: Mapping[str, str] = os.environ, *, engine: bool = True, spec: bool = True) -> WorkerConfig:
    if not (engine or spec):
        raise ConfigError("nothing to do: ask for --engine and/or --spec")
    common = load_common(env)
    state_dir = os.path.expanduser((env.get("REAX_COMPETITIONS_STATE_DIR") or "").strip())
    if not state_dir or not os.path.isabs(state_dir):
        raise ConfigError("REAX_COMPETITIONS_STATE_DIR (absolute path of the directory the validator reads) is required")
    cfg, sealed = _engine(env, common, want_engine=engine, want_spec=spec, strict=True)
    assert cfg is not None
    cfg = dataclasses.replace(cfg, state_path=os.path.join(state_dir, ENGINE_STATE_FILE))
    spec_work = os.path.expanduser(env.get("REAX_SPEC_WORK_DIR") or "~/.reax/spec-work")
    private = [("REAX_ENGINE_WORKDIR", cfg.workdir), ("REAX_SPEC_WORK_DIR", spec_work), ("REAX_ENGINE_WEIGHTS_DIR", cfg.weights_dir),
               ("REAX_SPEC_SEALED_DIR", sealed)]
    for name, path in private:
        if path and _inside(path, state_dir):
            raise ConfigError(f"{name} ({path}) must not be inside REAX_COMPETITIONS_STATE_DIR: that directory is shown to "
                              "the validator")
    return WorkerConfig(common=common, engine=cfg, spec_sealed_dir=sealed if spec else "", state_dir=state_dir,
                        run_engine=engine, run_spec=spec, poll_s=_int(env, "REAX_WORKER_POLL_S", 30, 1, 3600),
                        heartbeat_s=_int(env, "REAX_WORKER_HEARTBEAT_S", 300, 1, 3600))


# ------------------------------------------------------------------------------------------------ run
async def run_worker(wc: WorkerConfig, chain: Any, env: Mapping[str, str] = os.environ, *, once: bool = False,
                     stack: Optional[tuple] = None) -> None:
    """Poll, and run what is due. ``once``: one pass over everything that is due, then return."""
    os.makedirs(wc.state_dir, mode=0o755, exist_ok=True)
    os.chmod(wc.state_dir, 0o755)
    engine, adapter, spec_eval = stack or build_local_stack(wc.engine, wc.spec_sealed_dir, env, run_engine=wc.run_engine,
                                                            state_dir=wc.state_dir)
    spec_path = os.path.join(wc.state_dir, SPEC_STATE_FILE)
    engine_path = os.path.join(wc.state_dir, ENGINE_STATE_FILE)

    def write_initial() -> None:
        if wc.run_engine:
            engine.save()
            os.chmod(engine_path, 0o644)
        if spec_eval is not None:
            from .spec import state as ST

            ST.save(spec_eval.state, spec_path)
            os.chmod(spec_path, 0o644)

    write_initial()

    def heartbeat() -> None:
        for p in ([engine_path] if wc.run_engine else []) + ([spec_path] if spec_eval is not None else []):
            touch_state(p)

    async def beat() -> None:
        while True:
            await asyncio.sleep(wc.heartbeat_s)
            heartbeat()

    async def after_step() -> None:
        heartbeat()

    driver = CompetitionDriver(engine, adapter, background=False)
    log.info("competition worker: engine=%s spec=%s, state dir %s, network %s netuid %d", wc.run_engine,
             spec_eval is not None, wc.state_dir, wc.common.network, wc.common.netuid)
    hb = asyncio.ensure_future(beat())
    try:
        while True:
            if driver.due():
                try:
                    snap = await asyncio.to_thread(chain.snapshot)
                    exclude = {snap.owner_uid} if snap.owner_uid is not None else set()
                    await driver.run(snap, exclude, between=after_step)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception("competition cycle failed; retrying at the next poll")
                heartbeat()
            if once:
                return
            await asyncio.sleep(wc.poll_s)
    finally:
        hb.cancel()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="reax-competition-worker", description=__doc__.split("\n\n")[0])
    ap.add_argument("--engine", action="store_true", help="run the engine competition")
    ap.add_argument("--spec", action="store_true", help="run the specialist tracks")
    ap.add_argument("--once", action="store_true", help="run what is due once and exit (smoke test)")
    return ap


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    both = not (args.engine or args.spec)
    try:
        wc = load_worker(engine=args.engine or both, spec=args.spec or both)
    except ConfigError as exc:
        sys.exit(f"configuration error: {exc}")
    from .cli import setup_logging

    setup_logging(wc.common)
    from .chain import SdkChain

    chain = SdkChain(wc.common.network, wc.common.netuid, None, commitments=True)  # read-only: no wallet
    try:
        asyncio.run(run_worker(wc, chain, once=args.once))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
