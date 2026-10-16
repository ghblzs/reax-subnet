"""Capacity probe shards: the probe window split over forked worker processes (1.2.0rc5).

On mainnet the probe's single event loop sat at 100 % of one core for the whole window (~800 requests/s, ~1.2-1.6 ms
of validator CPU per request: signing plus the aiohttp client), so fast miners were clipped at what was left of one
core. With shards each worker process runs the window for a share of the UIDs on its own core:

* Every shard is a ``fork`` of the validator taken after the requests are prepared. It inherits the already loaded
  wallet object and the prepared request bodies; no key file is written and no key material crosses a pipe. A shard
  signs its own requests, so one UID's requests (and nonces) all come from one process.
* All shards open their windows at one common moment (a ``time.monotonic`` deadline; CLOCK_MONOTONIC is system-wide
  on Linux) and the send-skew guard is applied to the first sends of all shards together, so siblings in different
  shards are still loaded within the skew limit (measured on the live host: split/solo median 1.000, max 1.052).
  UIDs of one coldkey go to one shard only while that fits a shard's fair share of the load; a bigger group is split
  (1.2.0rc7, review H1: the live metagraph has a 33-UID coldkey, and the rc5-rc6 public-IP key let sybil UIDs that
  declare competitors' IPs glue those competitors onto one saturated shard).
* A shard sends back only what the window produced (per UID: answered request index, raw response, response headers,
  timings); signatures and answers are checked in the parent as before. A shard that fails, times out or sends
  nothing drops the whole probe, like a lagging one: never partial credit for the UIDs of the other shards.
"""

from __future__ import annotations

import asyncio
import gc
import io
import math
import os
import pickle
import resource
import select
import signal
import struct
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Sequence

from multidict import CIMultiDict

# A shard covers about this many probe lanes (12 UIDs x 16 lanes). At the mainnet cost (~1.4 ms per request) and every
# UID at its pool cap that is about half a core per shard.
PROBE_LANES_PER_SHARD = 192
# Cores left to the validator itself (its event loop, reference runtime threads) when sizing the shards.
PROBE_SHARD_SPARE_CORES = 2
# A shard has this long beyond the warm-up bound to fork and report ready.
PROBE_SHARD_START_SLACK_S = 3.0
# The common window opens this long after the last shard reported ready (the "go" message reaches every shard).
PROBE_SHARD_GO_DELAY_S = 0.05
# A shard's result must arrive this long after its longest window plus latency bound.
PROBE_SHARD_RESULT_SLACK_S = 15.0

_READY, _OK, _ERROR = b"R", b"K", b"E"
_HEADER = struct.Struct("!cQ")  # status, payload length


@dataclass
class ProbeWindow:
    """What one probe window produced (in-process or in one shard)."""

    # (uid, [(request index, raw response, headers)], window_s, decisions answered in the window, first send,
    #  requests answered in the window)
    results: list[tuple[int, list[tuple[int, bytes, Any]], float, int, Optional[float], int]]
    lag_p95: float = 0.0
    lag_samples: int = 0
    cpu_share: float = 0.0  # CPU of the loop's thread during the window / window length (1.0 = one full core)
    started: float = 0.0
    trace: Optional[dict[int, list[tuple[int, float, float, bool]]]] = None
    failed: list[str] = field(default_factory=list)  # shards that failed: the probe is dropped
    refused: dict[int, int] = field(default_factory=dict)  # uid -> lanes the miner turned away with 429/503


def shard_count(lanes_total: int, uids: int, env: Optional[dict] = None) -> int:
    """How many shard processes a probe with ``lanes_total`` lanes uses: ``REAX_PROBE_SHARDS`` (1 = one process,
    the 1.2.0rc4 behaviour; unset or "auto" = sized from the lane count and the cores), never more than the UIDs.
    Shards need ``os.fork`` with a system-wide monotonic clock: Linux only, everywhere else 1."""
    env = os.environ if env is None else env
    if not hasattr(os, "fork") or not sys.platform.startswith("linux"):
        return 1
    raw = str(env.get("REAX_PROBE_SHARDS", "")).strip().lower()
    if raw and raw != "auto":
        try:
            wanted = int(raw)
        except ValueError:
            wanted = 0
        if wanted >= 1:
            return max(1, min(wanted, uids))
    try:
        cores = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cores = os.cpu_count() or 1
    auto = math.ceil(lanes_total / PROBE_LANES_PER_SHARD)
    return max(1, min(auto, cores - PROBE_SHARD_SPARE_CORES, uids))


def sibling_groups(targets: Sequence[Any]) -> list[list[Any]]:
    """Targets of one coldkey, in target order. Axon IPs are not used: they are declared on chain by anyone, so a
    sybil UID declaring a competitor's IP could pull that competitor into its group (review H1)."""
    groups: dict[str, list[Any]] = {}
    out: list[list[Any]] = []
    for n in targets:
        if n.coldkey:
            if n.coldkey not in groups:
                groups[n.coldkey] = []
                out.append(groups[n.coldkey])
            groups[n.coldkey].append(n)
        else:
            out.append([n])
    return out


def _split(group: list[Any], weight: Callable[[Any], float], limit: float) -> list[list[Any]]:
    """``group`` in target-order pieces of at most ``limit`` (a single heavier UID is a piece of its own)."""
    pieces: list[list[Any]] = [[]]
    load = 0.0
    for n in group:
        w = weight(n)
        if pieces[-1] and load + w > limit:
            pieces.append([])
            load = 0.0
        pieces[-1].append(n)
        load += w
    return pieces


def plan_shards(targets: Sequence[Any], weight: Callable[[Any], float], shards: int) -> list[list[Any]]:
    """Coldkey groups packed onto ``shards`` shards by ``weight`` (the validator passes each UID's prepared
    requests/s; largest piece first onto the lightest shard); every shard keeps the targets' order. A group heavier
    than the average shard load is split into pieces of at most that load first, so no group can saturate a shard.
    Empty shards are left out."""
    position = {id(n): i for i, n in enumerate(targets)}
    load = [0.0] * max(1, shards)
    fair = math.fsum(weight(n) for n in targets) / len(load)
    pieces: list[list[Any]] = []
    for g in sibling_groups(targets):
        pieces.extend(_split(g, weight, fair) if math.fsum(weight(n) for n in g) > fair else [g])
    member: dict[int, int] = {}
    for g in sorted(pieces, key=lambda g: (-math.fsum(weight(n) for n in g), position[id(g[0])])):
        s = min(range(len(load)), key=lambda k: (load[k], k))
        load[s] += math.fsum(weight(n) for n in g)
        for n in g:
            member[id(n)] = s
    out: list[list[Any]] = [[] for _ in load]
    for n in targets:
        out[member[id(n)]].append(n)
    return [s for s in out if s]


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]


def _child(window: Callable[[Callable[[], None], Callable[[], Awaitable[None]]], Awaitable[ProbeWindow]],
           to_parent: int, from_parent: int, hard_limit_s: float, inherited: Sequence[int]) -> None:
    """Runs in the forked shard; never returns (``os._exit``: no atexit hooks, no flush of the parent's buffers, no
    finalizers of inherited objects such as the parent's chain or HTTP connections)."""
    code = 1
    try:
        for fd in inherited:  # the other shards' pipes
            try:
                os.close(fd)
            except OSError:
                pass
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM):
            signal.signal(sig, signal.SIG_DFL)
        signal.alarm(max(1, math.ceil(hard_limit_s)))  # a shard that outlives the probe dies (default SIGALRM)
        try:
            signal.set_wakeup_fd(-1)  # the parent loop's self-pipe (Python < 3.12 keeps it after a fork)
        except ValueError:
            pass
        _no_core_dumps()
        asyncio.events._set_running_loop(None)  # we were forked from inside the parent's running loop
        gc.disable()  # short-lived: leave the parent's objects (and their pages) alone

        def ready() -> None:
            _write_all(to_parent, _READY)

        async def go() -> None:
            # Wait (bounded) for the parent's common start time, then sleep until it.
            readable, _, _ = select.select([from_parent], [], [], PROBE_SHARD_START_SLACK_S + 5.0)
            raw = os.read(from_parent, 8) if readable else b""
            if len(raw) != 8:
                raise RuntimeError("no start time from the validator")
            (at,) = struct.unpack("!d", raw)
            await asyncio.sleep(max(0.0, at - time.monotonic()))

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(window(ready, go))
        # Plain values only (see _ResultUnpickler): header names are multidict istr, made str here.
        result.results = [(uid, [(i, raw, [(str(k), str(v)) for k, v in headers.items()]) for i, raw, headers in done],
                           *rest) for uid, done, *rest in result.results]
        payload = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
        _write_all(to_parent, _HEADER.pack(_OK, len(payload)) + payload)
        code = 0
    except BaseException as exc:  # noqa: BLE001 - report anything, then exit
        try:
            msg = f"{type(exc).__name__}: {exc}"[:500].encode("utf-8", "replace")
            _write_all(to_parent, _HEADER.pack(_ERROR, len(msg)) + msg)
        except BaseException:  # noqa: BLE001
            pass
    finally:
        os._exit(code)


def _no_core_dumps() -> None:
    """A shard that crashes must not leave a core file of the validator's memory (the hotkey is in it) wherever the
    host's core_pattern points, nor be readable by ptrace from another process of the same user (review L3)."""
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ValueError, OSError):
        pass
    try:
        import ctypes

        ctypes.CDLL(None, use_errno=True).prctl(4, 0, 0, 0, 0)  # PR_SET_DUMPABLE, 0
    except (OSError, AttributeError):
        pass


class _ResultUnpickler(pickle.Unpickler):
    """Only ProbeWindow (plus the builtin values it holds: tuples, lists, dicts, bytes, str, numbers, None)."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) == (ProbeWindow.__module__, ProbeWindow.__qualname__):
            return ProbeWindow
        raise pickle.UnpicklingError(f"shard result may not contain {module}.{name}")


class _Shard:
    def __init__(self, pid: int, reader: asyncio.StreamReader, transport: asyncio.BaseTransport, to_child: int,
                 read_fd: int):
        self.pid, self.reader, self.transport, self.to_child, self.read_fd = pid, reader, transport, to_child, read_fd
        self.reaped = False


async def run_shards(windows: Sequence[Callable[[Callable[[], None], Callable[[], Awaitable[None]]],
                                                 Awaitable[ProbeWindow]]],
                     ready_timeout_s: float, result_timeout_s: float) -> list[Optional[ProbeWindow]] | list[str]:
    """Forks one process per window coroutine factory, starts their windows at one common moment and returns their
    ``ProbeWindow``s in order; on any failure a list of reasons instead (the probe is dropped)."""
    loop = asyncio.get_running_loop()
    shards: list[_Shard] = []
    failures: list[str] = []
    hard_limit = ready_timeout_s + result_timeout_s + PROBE_SHARD_GO_DELAY_S + 10.0
    try:
        for i, window in enumerate(windows):
            c2p_r, c2p_w = os.pipe()
            p2c_r, p2c_w = os.pipe()
            with warnings.catch_warnings():
                # Python 3.12 warns about fork() in a process with threads. The shard only uses its own event loop,
                # its own sockets and the inherited wallet object (no lock another thread could hold).
                warnings.simplefilter("ignore", DeprecationWarning)
                pid = os.fork()
            if pid == 0:  # the shard
                os.close(c2p_r)
                os.close(p2c_w)
                _child(window, c2p_w, p2c_r, hard_limit, [fd for t in shards for fd in (t.read_fd, t.to_child)])
            os.close(c2p_w)
            os.close(p2c_r)
            reader = asyncio.StreamReader(limit=2 ** 20, loop=loop)
            pipe = os.fdopen(c2p_r, "rb", buffering=0)
            try:
                transport, _ = await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader, loop=loop),
                                                            pipe)
            except BaseException:  # cancelled (the probe's time budget) before this shard was registered
                pipe.close()
                os.close(p2c_w)
                _reap(_Shard(pid, reader, None, p2c_w, c2p_r))  # type: ignore[arg-type]
                raise
            shards.append(_Shard(pid, reader, transport, p2c_w, c2p_r))

        async def ready(i: int, s: _Shard) -> None:
            head = await s.reader.readexactly(1)
            if head != _READY:
                raise RuntimeError(await _error_text(head, s.reader))

        outcome = await asyncio.gather(*(asyncio.wait_for(ready(i, s), ready_timeout_s)
                                         for i, s in enumerate(shards)), return_exceptions=True)
        failures = [f"shard {i} did not start ({_why(r)})" for i, r in enumerate(outcome) if isinstance(r, BaseException)]
        if failures:
            return failures
        at = struct.pack("!d", time.monotonic() + PROBE_SHARD_GO_DELAY_S)
        for i, s in enumerate(shards):
            try:
                _write_all(s.to_child, at)
            except OSError as exc:  # it exited after its ready message (review L1)
                return [f"shard {i} exited before the window opened ({type(exc).__name__})"]

        async def result(s: _Shard) -> ProbeWindow:
            status, size = _HEADER.unpack(await s.reader.readexactly(_HEADER.size))
            payload = await s.reader.readexactly(size)
            if status != _OK:
                raise RuntimeError(payload.decode("utf-8", "replace"))
            # The payload comes from our own forked process (same code, same user), not from a miner: the miners'
            # bytes inside it are plain bytes/str values and are verified in the parent like in-process results.
            # Defence in depth: the unpickler builds nothing but ProbeWindow and builtin containers.
            got = _ResultUnpickler(io.BytesIO(payload)).load()
            if not isinstance(got, ProbeWindow):
                raise RuntimeError("malformed shard result")
            got.results = [(uid, [(i, raw, CIMultiDict(headers)) for i, raw, headers in done], *rest)
                           for uid, done, *rest in got.results]
            return got

        outcome = await asyncio.gather(*(asyncio.wait_for(result(s), result_timeout_s) for s in shards),
                                       return_exceptions=True)
        failures = [f"shard {i} failed ({_why(r)})" for i, r in enumerate(outcome) if isinstance(r, BaseException)]
        if failures:
            return failures
        return list(outcome)  # type: ignore[arg-type]
    finally:
        for s in shards:
            s.transport.close()
            try:
                os.close(s.to_child)
            except OSError:
                pass
            _reap(s)


async def _error_text(head: bytes, reader: asyncio.StreamReader) -> str:
    if head != _ERROR[:1]:
        return f"unexpected message {head!r}"
    rest = await reader.readexactly(_HEADER.size - 1)
    _status, size = _HEADER.unpack(head + rest)
    return (await reader.readexactly(size)).decode("utf-8", "replace")


def _why(exc: BaseException) -> str:
    if isinstance(exc, asyncio.TimeoutError):
        return "timeout"
    if isinstance(exc, asyncio.IncompleteReadError):
        return "exited without a result"
    return str(exc) or type(exc).__name__


def _reap(s: _Shard) -> None:
    """Collect the shard's exit status; one that is still running (timeout, cancelled probe) is killed first."""
    if s.reaped:
        return
    try:
        pid, _status = os.waitpid(s.pid, os.WNOHANG)
        if pid == 0:
            os.kill(s.pid, signal.SIGKILL)
            os.waitpid(s.pid, 0)
    except (ChildProcessError, ProcessLookupError):
        pass
    s.reaped = True
