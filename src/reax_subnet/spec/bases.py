"""Declared base models for the copy check: fetched by the evaluator itself, so honest submissions do not wait for an operator.

A submission names its base as ``<hf id>@<revision>`` (``reax-specialist.json``). When that base is not under
``REAX_SPEC_BASE_DIR``, ``BaseFetcher.ensure`` downloads it into ``<base dir>/<hf id with / -> __>@<revision>/`` the same
way a submission is downloaded (spec/fetch.py: pinned 40-hex revision, public repos only, no token, only the allowed file
patterns, safetensors only, size caps, allowlisted architecture, no ``auto_map``). The base is only read, never served,
so its licence is not restricted.

* Atomic: the download goes to ``<base dir>/.tmp-...`` and is renamed into place only after it passed the checks.
  ``config.dir_base_provider`` never resolves a name that starts with a dot, so nobody sees a half-written base.
* A base that cannot be fetched (gated, private, not found, oversize, refused by a rule) stays missing with a reason;
  the submission stays pending verification and the fetch is tried again after ``retry_cycles`` cycles.
* Per cycle at most ``budget_bytes`` are downloaded, and a download must leave ``min_free_bytes`` free on the base
  directory's filesystem; otherwise the base is deferred to a later cycle (no attempt is counted).
* ``preseed`` fetches the operator's list in the background (not counted against the cycle budget).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import time
import uuid
from typing import Callable, Sequence

from . import config as C
from . import fetch as F
from . import rules as R
from . import tensors as T

log = logging.getLogger("reax.spec")
TMP_PREFIX = ".tmp-"
STALE_TMP_S = 2 * F.DOWNLOAD_TIMEOUT_S + 600   # a temp dir older than this belongs to a dead process
_PINNED = re.compile(r"[0-9a-f]{40}")


def inspect_base(path: str, rules: R.TrackRules) -> list[str]:
    """The rules of a downloaded base (``rules.check_base_facts``), read as data. Returns the violations."""
    files = F.list_dir(path)
    problems = [f"{n}: symlink" for n, s in files.items() if s < 0]
    files = {n: s for n, s in files.items() if s >= 0}
    config = None
    if "config.json" in files:
        try:
            v = R.read_json_file(os.path.join(path, "config.json"))
            config = v if isinstance(v, dict) else None
        except (ValueError, OSError) as exc:
            problems.append(f"config.json: {exc}")
    params = wbytes = None
    try:
        tensors = T.read_model(path)
        params, wbytes = T.param_count(tensors), T.weight_bytes(tensors)
    except T.SafetensorsError as exc:
        problems.append(f"safetensors: {exc}")
    facts = R.RepoFacts(files=files, config=config, params=params, weight_bytes=wbytes, problems=problems)
    return R.check_base_facts(facts, rules)


class BaseFetcher:
    def __init__(self, base_dir: str, hub: F.Hub | None = None, *, rules: R.TrackRules = R.TrackRules(),
                 retry_cycles: int = C.BASE_RETRY_CYCLES, budget_bytes: int = C.BASE_BUDGET_GB * 1024 ** 3,
                 min_free_bytes: int = C.BASE_MIN_FREE_GB * 1024 ** 3,
                 free_bytes: Callable[[str], int] | None = None, clock: Callable[[], float] = time.time):
        self.base_dir = base_dir
        self.hub = hub or F.HfHub()
        self.rules = rules
        self.retry_cycles = max(1, retry_cycles)
        self.budget_bytes = budget_bytes
        self.min_free_bytes = min_free_bytes
        self._free_bytes = free_bytes or (lambda path: shutil.disk_usage(path).free)
        self.clock = clock
        self.provider = C.dir_base_provider(base_dir)
        self.cycle = 0
        self.spent = 0                                   # bytes downloaded this cycle
        self._failed: dict[str, tuple[int, str, bool]] = {}   # base -> (cycle of the last failed attempt, reason, host's)
        self._reasons: dict[str, str] = {}               # base -> why it is not here (failed, deferred, in progress)
        self._host_waits: set[str] = set()               # bases missing for this host's reason (``waits_here``)
        self._inflight: set[str] = set()
        self._preseed: asyncio.Task | None = None
        os.makedirs(base_dir, exist_ok=True)
        self._remove_stale_tmp()

    @classmethod
    def from_config(cls, cfg: C.SpecConfig, hub: F.Hub | None = None, **kw) -> "BaseFetcher":
        return cls(cfg.base_dir, hub, rules=cfg.rules, retry_cycles=cfg.base_retry_cycles, budget_bytes=cfg.base_budget_bytes,
                   min_free_bytes=cfg.base_min_free_bytes, **kw)

    # ------------------------------------------------------------------ public

    def new_cycle(self) -> None:
        self.cycle += 1
        self.spent = 0

    def reason(self, base_model: str) -> str:
        """Why the last ``ensure`` of this base did not produce it ("" = never tried)."""
        return self._reasons.get(base_model, "")

    def waits_here(self, base_model: str) -> bool:
        """True: the last ``ensure`` missed this base for this host's reason (deferred by the cycle's download budget or
        the disk, in progress, not storable), not for the base's own (not on the hub, gated, refused, not pinned). A
        structured signal: the reason text can echo the miner-chosen base id (1.2.0rc7 review F1)."""
        return base_model in self._host_waits

    async def ensure(self, base_model: str, *, budgeted: bool = True) -> str | None:
        """The local directory of the base, downloading it first if needed and allowed now; None = not here (``reason``)."""
        path = self.provider(base_model)
        if path is not None:
            self._reasons.pop(base_model, None)
            self._host_waits.discard(base_model)
            return path
        name = C.base_dir_name(base_model)
        repo, _, rev = base_model.partition("@")
        if name is None or not _PINNED.fullmatch(rev):
            return self._miss(base_model, "not fetched automatically: base_model must be '<public hf id>@<40-hex commit>'")
        if base_model in self._inflight:
            return self._miss(base_model, "the download is in progress", host=True)
        failed = self._failed.get(base_model)
        if failed is not None and self.cycle - failed[0] < self.retry_cycles:
            return self._miss(base_model, failed[1], host=failed[2])
        self._inflight.add(base_model)
        try:
            return await self._fetch(base_model, repo, rev, name, budgeted)
        finally:
            self._inflight.discard(base_model)

    def start_preseed(self, bases: Sequence[str]) -> asyncio.Task | None:
        """Fetch ``bases`` in the background, once per process (never blocks the caller's loop)."""
        if self._preseed is None and bases:
            self._preseed = asyncio.get_running_loop().create_task(self._run_preseed(list(bases)))
        return self._preseed

    # --------------------------------------------------------------- internals

    def _miss(self, base_model: str, why: str, *, host: bool = False) -> None:
        self._reasons[base_model] = why
        if host:
            self._host_waits.add(base_model)
        else:
            self._host_waits.discard(base_model)
        return None

    def _fail(self, base_model: str, why: str, *, host: bool = False) -> None:
        self._failed[base_model] = (self.cycle, why, host)
        log.warning("base model %s could not be fetched: %s (next try in %d cycles)", base_model, why, self.retry_cycles)
        return self._miss(base_model, why, host=host)

    async def _fetch(self, base_model: str, repo: str, rev: str, name: str, budgeted: bool) -> str | None:
        loop = asyncio.get_running_loop()
        try:
            listing = await loop.run_in_executor(None, self.hub.listing, repo, rev)
        except F.FetchError as exc:
            return self._fail(base_model, f"base not available on the hub: {exc}"[:300])
        if listing.sha != rev:
            return self._fail(base_model, "the hub revision of the base does not match the declared revision")
        if listing.gated or listing.private:
            return self._fail(base_model, "the base repo is gated or private: an operator must place it under REAX_SPEC_BASE_DIR")
        bad = R.check_base_listing(listing.files, self.rules)
        if bad:
            return self._fail(base_model, ("base refused: " + "; ".join(bad))[:300])
        size = sum(s for n, s in listing.files.items() if R._is_downloaded(n))
        if budgeted and self.spent + size > self.budget_bytes:
            log.info("base model %s (%.1f GB) deferred: the base download budget of this cycle is used (%.1f of %.1f GB)",
                     base_model, size / 1e9, self.spent / 1e9, self.budget_bytes / 1e9)
            return self._miss(base_model, "deferred to a later cycle (base download budget of this cycle used)", host=True)
        free = self._free_bytes(self.base_dir)
        if free - size < self.min_free_bytes:
            log.warning("base model %s (%.1f GB) skipped: %.0f GB free on the base directory's filesystem, %.0f GB must stay free",
                        base_model, size / 1e9, free / 1e9, self.min_free_bytes / 1e9)
            return self._miss(base_model, "deferred: not enough free disk space for the base on this host", host=True)
        if budgeted:
            self.spent += size
        tmp = os.path.join(self.base_dir, f"{TMP_PREFIX}{name}-{uuid.uuid4().hex[:12]}")
        try:
            await self.hub.download(repo, rev, tmp)
            violations = await loop.run_in_executor(None, inspect_base, tmp, self.rules)
            if violations:
                return self._fail(base_model, ("base refused: " + "; ".join(violations))[:300])
            final = os.path.join(self.base_dir, name)
            try:
                os.rename(tmp, final)
            except OSError:
                if not os.path.isdir(final):     # another process renamed the same base into place first: theirs is used
                    raise
        except F.FetchError as exc:
            return self._fail(base_model, f"base download failed: {exc}"[:300])
        except OSError as exc:
            return self._fail(base_model, f"base could not be stored: {exc}"[:300], host=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        self._failed.pop(base_model, None)
        self._reasons.pop(base_model, None)
        self._host_waits.discard(base_model)
        log.info("base model %s fetched (%.1f GB)", base_model, size / 1e9)
        return final

    async def _run_preseed(self, bases: list[str]) -> None:
        for b in bases:
            try:
                if await self.ensure(b, budgeted=False) is None:
                    log.warning("preseed: base model %s not fetched: %s", b, self.reason(b))
            except Exception:  # noqa: BLE001 - the preseed must never take the worker down
                log.exception("preseed: base model %s failed", b)

    def _remove_stale_tmp(self) -> None:
        now = self.clock()
        for n in os.listdir(self.base_dir):
            p = os.path.join(self.base_dir, n)
            try:
                if n.startswith(TMP_PREFIX) and now - os.path.getmtime(p) > STALE_TMP_S:
                    shutil.rmtree(p, ignore_errors=True)
            except OSError:
                pass
