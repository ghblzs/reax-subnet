"""The specialist-track evaluator the validator runs: ``await SpecEvaluator(cfg).evaluate_cycle(snapshot)``.

Per cycle and active track: read the commitments, check new submissions (rules, tensor hashes, near-duplicates) without
GPU time, then run one group of at most the hourly budget of models on one fresh shared draw: the incumbent plus the
challengers in commitment order (first pass), or the incumbent plus the challengers whose 12-hour wait is over
(confirmation, on a disjoint draw). State is saved after every group. The result maps each active track to its ranked
leaders ``[(uid, score), ...]`` in rank order (champion first, see ``compute_leaders``); turning that into weights is the job of the split layer.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import time
from typing import Any, AsyncContextManager, Awaitable, Callable, Mapping, Sequence

from ..chain import Snapshot
from ..competition_split import TRACK_ENTRY_FLOOR, RankedLeaders
from . import config as C
from . import fetch as F
from . import machine as M
from . import sampling as S
from . import scoring as SC
from . import state as ST
from . import tensors as T
from .commitment import SpecCommitment, parse_commitment
from .vllm import Backend, HostUnavailable, ModelUnavailable, VllmDockerBackend

log = logging.getLogger("reax.spec")
SERVE_TIMEOUT_S = 3600
MARKER = ".reax-verified"

BackendFactory = Callable[[str, int], AsyncContextManager[Backend]]


async def score_model(backend: Backend, rows: Sequence[Mapping[str, Any]]) -> tuple[list[SC.ItemScore], SC.Score]:
    """Shared by the validator and the miner CLI: serve all rows, score the answers."""
    probs = await backend.decide(rows)
    return SC.score_run(rows, probs)


def compute_leaders(state: ST.SpecState, track_ids: Sequence[str], snapshot: Snapshot) -> dict[str, list[tuple[int, float]]]:
    """Champion first, then former champions (most recently crowned first), only submissions their hotkey still
    commits to. This order IS the rank (docs/SPECIALIST-TRACKS.md, "Ranks"): the result is a ``RankedLeaders``, which the
    share layer pays in this order. The scores come from different draws and are reported, never compared.

    Pure function of the saved state and the chain snapshot (no docker, no GPU): the validator in remote mode calls it on
    the state file the competition worker wrote, so the hotkey/uid and commitment are re-verified on every call."""
    live = {n.hotkey: (n.uid, n.commitment) for n in snapshot.neurons}
    out: dict[str, list[tuple[int, float]]] = {}
    for track_id in track_ids:
        ts = state.track(track_id)
        ranked: list[ST.Submission] = []
        champ = ts.submissions.get(ts.champion)
        if champ is not None:
            ranked.append(champ)
        former = [s for s in ts.submissions.values() if s.status == ST.FORMER]
        ranked += sorted(former, key=lambda s: -(s.champion_since_ts or 0))
        res = RankedLeaders()
        for s in ranked:
            uid_c = live.get(s.hotkey)
            if uid_c is None or s.score is None:
                continue
            c = parse_commitment(uid_c[1])
            if c is None or f"{s.hotkey}|{c.text()}" != s.key:
                continue
            res.append((uid_c[0], float(s.score)))
            if len(res) >= C.MAX_LEADERS:
                break
        out[track_id] = res
    return out


class _Prepared:
    def __init__(self, sub: ST.Submission, path: str, weight_bytes: int):
        self.sub, self.path, self.weight_bytes = sub, path, weight_bytes
        self.infra_failure = False  # the last _serve failed for a host reason (docker, socket), not the model's


class SpecEvaluator:
    def __init__(self, cfg: C.SpecConfig, *, fetcher: F.Fetcher | None = None, backend_factory: BackendFactory | None = None,
                 base_provider: T.BaseProvider | None = None, clock: Callable[[], float] = time.time,
                 seed_source: Callable[[str, int], int] | None = None,
                 free_bytes: Callable[[str], int] | None = None):
        self.cfg = cfg
        self.fetcher = fetcher or F.HfFetcher()
        self.backend_factory = backend_factory or self._docker_backend
        self.base_provider = base_provider or C.dir_base_provider(cfg.base_dir)
        self.clock = clock
        self._free_bytes = free_bytes or (lambda path: shutil.disk_usage(path).free)
        self.seed_source = seed_source or (lambda track, block: int.from_bytes(
            hashlib.sha256(f"{track}|{block}|{os.urandom(16).hex()}".encode()).digest()[:8], "big"))
        self.state = ST.load(cfg.state_path)
        self._lock = asyncio.Lock()
        self._pools: dict[str, list[dict]] = {}
        os.makedirs(os.path.join(cfg.work_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(cfg.work_dir, "sketches"), exist_ok=True)

    # ------------------------------------------------------------------ public

    async def evaluate_cycle(self, snapshot: Snapshot) -> dict[str, list[tuple[int, float]]]:
        async with self._lock:
            now = self.clock()
            active = self.cfg.active_tracks(now)
            self._pools.clear()
            for tdef in active:
                self._sync(tdef, snapshot, now)
            ST.save(self.state, self.cfg.state_path)
            for tdef in active:
                try:
                    await self._run_track(tdef, snapshot, now)
                except Exception:  # noqa: BLE001 - one failing track must not stop the others
                    log.exception("track %s: evaluation failed", tdef.track_id)
                ST.save(self.state, self.cfg.state_path)
            self._cleanup(active)
            return self.leaders(active, snapshot)

    def leaders(self, active: Sequence[C.TrackDef], snapshot: Snapshot) -> dict[str, list[tuple[int, float]]]:
        return compute_leaders(self.state, [t.track_id for t in active], snapshot)

    # --------------------------------------------------------------- chain sync

    def _sync(self, tdef: C.TrackDef, snap: Snapshot, now: float) -> None:
        ts = self.state.track(tdef.track_id)
        live: dict[str, tuple[Any, SpecCommitment]] = {}
        for n in snap.neurons:
            c = parse_commitment(n.commitment)
            if c is not None and c.track_id == tdef.track_id:
                live[f"{n.hotkey}|{c.text()}"] = (n, c)
        for key, (n, c) in live.items():
            sub = ts.submissions.get(key)
            if sub is not None:
                sub.uid, sub.coldkey = n.uid, n.coldkey
                continue
            sub = ST.Submission(key=key, uid=n.uid, hotkey=n.hotkey, coldkey=n.coldkey, track_id=tdef.track_id, repo=c.repo,
                                revision=c.revision, first_seen_block=snap.block, first_seen_ts=now)
            ts.submissions[key] = sub
            ts.log(now, f"new submission uid {n.uid} {c.repo}@{c.revision[:8]}")
            why = self._budget_violation(sub, now)
            if why:
                sub.status, sub.reason = ST.REJECTED, why
        for key, sub in ts.submissions.items():
            if key in live or sub.status in (ST.WITHDRAWN, ST.REJECTED, ST.NOT_BETTER):
                continue
            if sub.status == ST.FORMER:
                continue    # stays listed; leaders() checks that the hotkey still holds it
            if ts.champion == key:
                ts.champion = ""
            sub.status, sub.reason = ST.WITHDRAWN, "commitment changed or hotkey left"
            ts.log(now, f"withdrawn {key[:12]}")
        if ts.champion and ts.submissions[ts.champion].status != ST.CHAMPION:
            ts.champion = ""

    def _budget_violation(self, sub: ST.Submission, now: float) -> str:
        """docs/SPECIALIST-TRACKS.md: at most 2 open submissions per coldkey and track, 4 new per coldkey per 7 days."""
        ts = self.state.track(sub.track_id)
        open_ = [s for s in ts.submissions.values()
                 if s is not sub and s.coldkey == sub.coldkey and s.status in (ST.QUEUED, ST.PENDING_CONFIRM)]
        if len(open_) >= C.MAX_OPEN_PER_COLDKEY_TRACK:
            return f"coldkey already has {C.MAX_OPEN_PER_COLDKEY_TRACK} open submissions in this track"
        week = [s for t in self.state.tracks.values() for s in t.submissions.values()
                if s is not sub and s.coldkey == sub.coldkey and s.first_seen_ts > now - 7 * 86400 and s.status != ST.REJECTED]
        if len(week) >= C.MAX_NEW_PER_COLDKEY_WEEK:
            return f"coldkey already made {C.MAX_NEW_PER_COLDKEY_WEEK} submissions in 7 days"
        return ""

    # ------------------------------------------------------------------- helpers

    def _model_dir(self, sub: ST.Submission) -> str:
        return os.path.join(self.cfg.work_dir, "models", hashlib.sha256(sub.key.encode()).hexdigest()[:20])

    def _sketch_path(self, sub: ST.Submission) -> str:
        return os.path.join(self.cfg.work_dir, "sketches", hashlib.sha256(sub.key.encode()).hexdigest()[:20] + ".npz")

    def _docker_backend(self, model_dir: str, weight_bytes: int) -> AsyncContextManager[Backend]:
        return VllmDockerBackend(model_dir, weight_bytes, gpu_device=self.cfg.gpu_device, gpu_total_gb=self.cfg.gpu_total_gb,
                                 budget_gb=self.cfg.gpu_budget_gb, image=self.cfg.image, work_dir=self.cfg.work_dir,
                                 user=self.cfg.user, runtime=self.cfg.docker_runtime, cpus=self.cfg.cpus)

    def _pool(self, category: str) -> list[dict]:
        if category not in self._pools:
            self._pools[category] = S.load_pool(self.cfg.sealed_dir, category)
        return self._pools[category]

    def _entry_floor(self, tdef: C.TrackDef) -> float:
        """The release constant ``competition_split.TRACK_ENTRY_FLOOR`` (None = none) or the configured floor, if higher."""
        return max(self.cfg.entry_floor, TRACK_ENTRY_FLOOR.get(tdef.track_id) or 0.0)

    def _budget(self, now: float) -> int:
        self.state.eval_times = [t for t in self.state.eval_times if t > now - 3600]
        return max(0, self.cfg.max_evals_per_hour - len(self.state.eval_times))

    def _cleanup(self, active: Sequence[C.TrackDef]) -> None:
        keep = set()
        for ts in self.state.tracks.values():
            for s in ts.submissions.values():
                if s.status in (ST.CHAMPION, ST.PENDING_CONFIRM):
                    keep.add(self._model_dir(s))
        root = os.path.join(self.cfg.work_dir, "models")
        for name in os.listdir(root):
            p = os.path.join(root, name)
            if p not in keep:
                shutil.rmtree(p, ignore_errors=True)

    # ----------------------------------------------------- submission preparation

    async def _prepare(self, tdef: C.TrackDef, sub: ST.Submission, snap: Snapshot, now: float) -> _Prepared | None:
        """Fetch and check one submission. None = not usable now (the status says why)."""
        ts = self.state.track(tdef.track_id)
        dest = self._model_dir(sub)
        commitment = SpecCommitment(sub.track_id, sub.repo, sub.revision)
        kept = self._kept(sub, dest)
        if kept is not None:
            return kept
        sub.last_eval_ts = now
        try:
            ins = await self.fetcher.fetch(commitment, dest, category=tdef.category, hotkey=sub.hotkey, rules=self.cfg.rules)
        except F.FetchError as exc:
            if exc.kind == "rejected":
                sub.status, sub.reason = ST.REJECTED, str(exc)[:300]
                ts.log(now, f"rejected {sub.key[:12]}: {sub.reason[:120]}")
            elif snap.block > sub.first_seen_block + C.PUBLISH_DEADLINE_BLOCKS:
                sub.status, sub.reason = ST.REJECTED, f"not available within {C.PUBLISH_DEADLINE_BLOCKS} blocks: {exc}"[:300]
                ts.log(now, f"expired {sub.key[:12]}")
            else:
                sub.reason = f"waiting for the repo: {exc}"[:300]
            return None
        assert ins.hashes is not None
        with open(os.path.join(dest, MARKER), "w") as fh:
            json.dump({"hash": ins.hashes.named, "weight_bytes": ins.hashes.weight_bytes, "files": ins.facts.files}, fh)
        sub.hash_named, sub.hash_unnamed = ins.hashes.named, ins.hashes.unnamed
        sub.base_model = str((ins.facts.manifest or {}).get("base_model", ""))
        # reads about half of the candidate and of its base from disk: never on the event loop
        why = await asyncio.get_running_loop().run_in_executor(None, self._copy_check, ts, sub, dest)
        if why:
            sub.status, sub.reason = ST.REJECTED, why
            ts.log(now, f"rejected {sub.key[:12]}: {why[:120]}")
            shutil.rmtree(dest, ignore_errors=True)
            return None
        return _Prepared(sub, dest, ins.hashes.weight_bytes)

    @staticmethod
    def _kept(sub: ST.Submission, dest: str) -> "_Prepared | None":
        """A model directory kept from an earlier group (champion, pending confirmation): reuse it if untouched."""
        try:
            with open(os.path.join(dest, MARKER)) as fh:
                m = json.load(fh)
            files = {k: v for k, v in F.list_dir(dest).items() if k != MARKER}
            if m["hash"] == sub.hash_named and m["files"] == files:
                return _Prepared(sub, dest, int(m["weight_bytes"]))
        except (OSError, ValueError, KeyError):
            pass
        shutil.rmtree(dest, ignore_errors=True)
        return None

    @staticmethod
    def _earlier(a: ST.Submission, b: ST.Submission) -> bool:
        return (a.first_seen_block, a.uid, a.key) < (b.first_seen_block, b.uid, b.key)

    def _copy_check(self, ts: ST.TrackState, sub: ST.Submission, model_dir: str) -> str:
        """Exact tensor hashes (re-sharding and renaming proof), then the near-duplicate check. Returns a rejection reason."""
        for h in (sub.hash_named, sub.hash_unnamed):
            owner_key = ts.hashes.get(h)
            owner = ts.submissions.get(owner_key) if owner_key else None
            if owner is None or owner is sub:
                continue
            if self._earlier(owner, sub) or owner.status in (ST.CHAMPION, ST.FORMER):
                return f"identical tensors to an earlier commitment (uid {owner.uid})"
            owner.status, owner.reason = ST.REJECTED, f"identical tensors to an earlier commitment (uid {sub.uid})"
            if ts.champion == owner.key:
                ts.champion = ""
        ts.hashes[sub.hash_named] = sub.key
        ts.hashes[sub.hash_unnamed] = sub.key
        base_dir = self.base_provider(sub.base_model) if sub.base_model else None
        if base_dir is None:
            return ""      # no base weights on this host: only exact hashes were compared (flagged in the state)
        sketch = T.delta_sketch(model_dir, base_dir)
        sub.delta_checked = True
        if sketch and all(float((v @ v)) == 0.0 for v in sketch.values()):
            return "weights are identical to the declared base model"
        for other in sorted(ts.submissions.values(), key=lambda s: (s.first_seen_block, s.uid, s.key)):
            if other is sub or other.base_model != sub.base_model or other.status in (ST.REJECTED, ST.WITHDRAWN):
                continue
            if not self._earlier(other, sub) or not os.path.exists(self._sketch_path(other)):
                continue
            stats = T.compare_sketches(T.load_sketch(self._sketch_path(other)), sketch)
            if stats.derived():
                sub.derived_from = other.key
                ts.log(self.clock(), f"{sub.key[:12]} is derived from {other.key[:12]} (cos {stats.cosine:.3f}, rel {stats.relative_l2:.3f})")
                break
        T.save_sketch(sketch, self._sketch_path(sub))
        return ""

    # ------------------------------------------------------------------ one group

    async def _serve(self, p: _Prepared, rows: list[dict], now: float) -> list[SC.ItemScore] | None:
        self.state.eval_times.append(now)
        try:
            async with self.backend_factory(p.path, p.weight_bytes) as backend:
                probs = await asyncio.wait_for(backend.decide(rows), SERVE_TIMEOUT_S)
        except (ModelUnavailable, asyncio.TimeoutError, OSError) as exc:
            p.infra_failure = isinstance(exc, (HostUnavailable, OSError))
            log.warning("serving %s failed%s: %s", p.sub.key[:12], " (host problem, not counted)" if p.infra_failure else "", exc)
            p.sub.reason = f"could not be served: {exc}"[:300]
            return None
        items, _ = SC.score_run(rows, probs)
        return items

    async def _run_track(self, tdef: C.TrackDef, snap: Snapshot, now: float) -> None:
        ts = self.state.track(tdef.track_id)
        budget = self._budget(now)
        if budget < 1:
            return
        due = [s for s in ts.ordered(ST.PENDING_CONFIRM) if s.confirm_after_ts is not None and now >= s.confirm_after_ts]
        confirm_mode = bool(due)
        cands = due if confirm_mode else [
            s for s in ts.ordered(ST.QUEUED)
            if s.last_eval_ts is None or not s.reason.startswith("waiting") or now - s.last_eval_ts >= C.RETRY_UNAVAILABLE_S]
        if not cands:
            return
        free = self._free_bytes(self.cfg.work_dir)
        if free < self.cfg.min_free_bytes:
            # inconclusive, nothing recorded and nobody rejected: the next cycle tries again
            log.warning("track %s: cycle skipped, %.0f GB free on the work directory's filesystem, %.0f GB needed before a "
                        "download", tdef.track_id, free / 1e9, self.cfg.min_free_bytes / 1e9)
            return
        champ = ts.submissions.get(ts.champion) if ts.champion else None
        # download + rule + copy checks for as many candidates as the budget could take (no GPU time spent here)
        prepared: list[_Prepared] = []
        for s in cands[: budget]:
            p = await self._prepare(tdef, s, snap, now)
            if p is not None:
                prepared.append(p)
        if not prepared:
            return
        # models: the incumbent, parents of derived challengers (head-to-head), then the challengers
        ref: list[_Prepared] = []
        if champ is not None:
            cp = await self._prepare(tdef, champ, snap, now)
            if cp is None:
                champ.attempts += 1
                if champ.status == ST.REJECTED or champ.attempts >= C.MAX_ATTEMPTS:
                    ts.log(now, f"champion {champ.key[:12]} unusable: {champ.reason[:100]}")
                    champ.status, ts.champion = ST.REJECTED, ""
                return
            ref.append(cp)
        parents: dict[str, _Prepared] = {}
        for p in prepared:
            pk = p.sub.derived_from
            par = ts.submissions.get(pk) if pk else None
            if par is not None and par is not champ and pk not in parents:
                pp = await self._prepare(tdef, par, snap, now)
                if pp is not None:
                    parents[pk] = pp
        room = budget - len(ref) - len(parents)
        if room < 1:
            return
        prepared = prepared[:room]
        # one fresh draw shared by every model of the group
        exclude: set[str] = set()
        if confirm_mode:
            for p in prepared:
                d = ts.draws.get(p.sub.draw1)
                if d:
                    exclude |= set(d.item_ids)
        seed = self.seed_source(tdef.track_id, snap.block)
        rows, fresh = S.draw(self._pool(tdef.category), n=self.cfg.sample_size, seed=seed, use_counts=ts.used, exclude=exclude)
        draw_id = hashlib.sha256(f"{tdef.track_id}|{seed}".encode()).hexdigest()[:16]
        ts.draws[draw_id] = ST.Draw(draw_id, now, snap.block, seed, [r["id"] for r in rows], fresh)
        for r in rows:
            ts.used[r["id"]] = ts.used.get(r["id"], 0) + 1
        for old in sorted(ts.draws.values(), key=lambda d: d.ts)[:-40]:
            ts.draws.pop(old.draw_id, None)
        if not fresh:
            ts.log(now, "sealed pool exhausted: some items were reused; refill the pool")
        ST.save(self.state, self.cfg.state_path)
        # serve one model at a time
        results: dict[str, list[SC.ItemScore]] = {}
        group = ref + list(parents.values()) + prepared
        failed: list[_Prepared] = []
        for p in group:
            items = await self._serve(p, rows, self.clock())
            if items is None:
                failed.append(p)
                if p.sub is champ:
                    if not p.infra_failure:
                        p.sub.attempts += 1
                    ts.log(now, "incumbent could not be served; group abandoned")
                    return
                continue
            p.sub.attempts = 0
            results[p.sub.key] = items
        # Only failures that are the model's own count toward MAX_ATTEMPTS. If every model of a group of several failed,
        # it is the host (docker, driver, GPU), not the models: nothing is counted and the next cycle tries again.
        host_problem = len(group) > 1 and len(failed) == len(group)
        if host_problem:
            log.warning("track %s: none of the %d models could be served; treated as a host problem, no attempt counted",
                        tdef.track_id, len(group))
            ts.log(now, "no model of the group could be served: host problem, nothing counted")
            return
        for p in failed:
            if p.infra_failure:
                continue
            p.sub.attempts += 1
            if p.sub.attempts >= C.MAX_ATTEMPTS:
                p.sub.status = ST.REJECTED
                p.sub.reason = "could not be served after repeated attempts: " + p.sub.reason
        incumbent_items = results.get(champ.key) if champ else None
        if champ is not None and incumbent_items is None:
            return
        self._decide(ts, tdef, [p.sub for p in prepared if p.sub.key in results], results, champ, confirm_mode, draw_id, seed, now)

    # ----------------------------------------------------- decisions (pure state updates)

    @staticmethod
    def _first_incumbent(ch: ST.Submission) -> str | None:
        """Key of the incumbent the first pass was measured against ("" = vacant track). For a state written before the
        field existed, the first-pass history still tells whether the track was vacant; otherwise None (unknown)."""
        if ch.first_incumbent is not None:
            return ch.first_incumbent
        first = next((h for h in ch.history if h.get("kind") == "first" and h.get("draw") == ch.draw1), None)
        return "" if first is not None and first.get("incumbent") is None else None

    def _decide(self, ts: ST.TrackState, tdef: C.TrackDef, challengers: list[ST.Submission], results: dict[str, list[SC.ItemScore]],
                champ: ST.Submission | None, confirm_mode: bool, draw_id: str, seed: int, now: float) -> None:
        total = {k: SC.score_items(v).total for k, v in results.items()}
        if champ is not None:
            champ.score, champ.last_eval_ts = total[champ.key], now
        for pk in {c.derived_from for c in challengers if c.derived_from}:
            if pk in results and pk in ts.submissions and (champ is None or pk != champ.key):
                ts.submissions[pk].score = total[pk]
        current = champ
        promoted = False
        for ch in challengers:
            ch.score, ch.last_eval_ts = total[ch.key], now
            inc_items = results.get(current.key) if current else None
            se = SC.paired_se(results[ch.key], inc_items, seed) if inc_items is not None else 0.0
            inc_total = total[current.key] if current else None
            parent_ok, parent_gain = True, None
            par = ts.submissions.get(ch.derived_from) if ch.derived_from else None
            if par is not None and par.key in results and par.key != (current.key if current else ""):
                pse = SC.paired_se(results[ch.key], results[par.key], seed)
                parent_gain = total[ch.key] - total[par.key]
                parent_ok = parent_gain >= M.required_margin(pse)
            if not confirm_mode:
                fp = M.first_pass(ch.score, inc_total, se, entry_floor=self._entry_floor(tdef))
                ch.history.append({"draw": draw_id, "kind": "first", "score": ch.score, "incumbent": inc_total, "gain": fp.gain, "delta": fp.delta})
                if fp.passed and parent_ok:
                    ch.status, ch.reason = ST.PENDING_CONFIRM, ""
                    ch.delta, ch.first_gain, ch.draw1 = fp.delta, fp.gain, draw_id
                    ch.first_incumbent = current.key if current else ""
                    ch.confirm_after_ts = now + self.cfg.confirm_delay_s
                    ts.log(now, f"{ch.key[:12]} passed the first draw ({fp.gain:+.2f} vs margin {fp.delta:.2f})")
                else:
                    ch.status = ST.NOT_BETTER
                    ch.reason = f"first draw: gain {fp.gain:+.2f}, margin {fp.delta:.2f}" + ("" if parent_ok else f"; does not beat its parent ({parent_gain:+.2f})")
                    ts.log(now, f"{ch.key[:12]} {ch.reason}")
                continue
            first_inc = self._first_incumbent(ch)
            new_inc = current is not None and first_inc is not None and first_inc != current.key
            ok = M.confirm(ch.score, inc_total, ch.delta or M.MARGIN_FLOOR, entry_floor=self._entry_floor(tdef),
                           promoted_this_round=promoted, new_incumbent=new_inc, se=se) and parent_ok
            ch.history.append({"draw": draw_id, "kind": "confirm", "score": ch.score, "incumbent": inc_total})
            if ok:
                if current is not None:
                    current.status = ST.FORMER
                ch.status, ch.reason, ch.champion_since_ts = ST.CHAMPION, "", now
                ts.champion, current, promoted = ch.key, ch, True
                ts.log(now, f"{ch.key[:12]} is the new champion ({ch.score:.2f})")
            else:
                ch.status = ST.NOT_BETTER
                ch.reason = "confirmation draw: gain too small" if inc_total is not None else "confirmation draw: below the entry floor"
                ts.log(now, f"{ch.key[:12]} {ch.reason}")
