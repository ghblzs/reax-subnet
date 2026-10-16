"""The specialist-track evaluator the validator runs: ``await SpecEvaluator(cfg).evaluate_cycle(snapshot)``.

Per cycle and active track: read the commitments, check new submissions (rules, tensor hashes, near-duplicates) without
GPU time, then run one group of at most the hourly budget of models on one fresh shared draw: the incumbent plus the
challengers in commitment order (first pass), or the incumbent plus the challengers whose 12-hour wait is over
(confirmation, on a disjoint draw). On a vacant track the first champion is decided per cohort (1.2.0rc7): entries that
passed their confirmation wait until every entry first seen in the same block or earlier is decided, then meet on one
shared draw (the playoff). State is saved after every group. The result maps each active track to its ranked
leaders ``[(uid, score), ...]`` in rank order (champion first, see ``compute_leaders``); turning that into weights is the job of the split layer.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import shutil
import time
from dataclasses import dataclass
from typing import Any, AsyncContextManager, Awaitable, Callable, Mapping, Sequence

from ..chain import Snapshot
from ..competition_split import KEEPALIVE_LEGACY_FLOOR, KEEPALIVE_NONPROC_FLOOR, TRACK_ENTRY_FLOOR, RankedLeaders
from . import bases as B
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
RENEWAL_WAIT = "renewal waits for disk: earlier sketches must be renewed before the near-duplicate check"
MAX_UNREFERENCED_DRAWS = 40    # draws kept in the state besides those a pending entry still needs (``_prune_draws``)
# A PENDING_VERIFY entry whose wait is this host's (``ST.WAIT_INFRA``: base download deferred by the per-cycle budget or
# the disk, in progress, not storable; sketch renewal waiting for disk) holds back its cohort's first crowning. A wait on
# the miner's side (private or missing repo, base not on the hub, weights not comparable) does not. No open entry holds
# a cohort back longer than COHORT_MAX_WAIT_S after its first passed confirmation (a base larger than the cycle's download
# budget is deferred on every cycle; a model that never starts on this host fails every playoff).
COHORT_MAX_WAIT_S = 12 * 3600

BackendFactory = Callable[[str, int], AsyncContextManager[Backend]]

# Behavioural near-duplicate check (docs/SPECIALIST-TRACKS.md, "Near-duplicates"). On the shared draw, a candidate whose
# answers are those of an earlier submission served on the same draw is derived from it, whatever its weights look
# like (permuted hidden units, rescaled layers, noise on the sampled positions). Calibration, in Jensen-Shannon bits:
# serving the same weights twice moves the logits by numerical noise only (BF16 logits hold ~2^-5 near |logit| 8; eager
# vs CUDA graphs moved the competence by ~0.2 points on 513 items), i.e. a logit noise sd <= 0.05 and a mean JSD
# <= ~5e-4 bits (tests/test_copycheck2.py measures it on the fixture). Two independent fine-tunes differ in their own
# logit shifts; an independent part of sd >= 0.3 already gives >= ~0.012 bits. The threshold sits a factor 10 above the
# noise ceiling. The rank rule catches a copy whose probabilities were rescaled (a temperature on lm_head): the order of
# the labels stays the same on (almost) every item, which two independent models do not reach. It counts only items with
# at least 3 labels: on a yes/no item the order is just the argmax, which two independent fine-tunes of one task share
# on nearly every item (review rc6 F5). It also needs the JSD below TWIN_ORDER_JSD_BITS, far above what a temperature
# near 1 moves. The rule: twin = items >= TWIN_MIN_ITEMS and (jsd < TWIN_JSD_BITS or (order items >= TWIN_MIN_ORDER_ITEMS
# and same order >= TWIN_SAME_ORDER and jsd < TWIN_ORDER_JSD_BITS)).
TWIN_JSD_BITS = 0.005          # mean JSD below this: the same behaviour
TWIN_SAME_ORDER = 0.99         # or: the labels in the same order (equal within TWIN_TIE) on at least this share of the items
TWIN_ORDER_JSD_BITS = 0.05     # ... with at least 3 labels, and a mean JSD below this
TWIN_TIE = 1e-6
TWIN_MIN_ITEMS = 50            # items valid for both models needed for any verdict
TWIN_MIN_ORDER_ITEMS = 20      # items with at least 3 labels needed for the order rule


@dataclass(frozen=True)
class BehaviourStats:
    items: int            # items with a valid answer from both models
    jsd_bits: float       # mean Jensen-Shannon divergence of the option distributions (0 = identical, 1 = disjoint)
    mean_abs: float       # mean absolute probability difference per option
    same_order: float     # share of the items with at least 3 labels that have the labels in the same order
    order_items: int = 0  # items with at least 3 labels (the base of ``same_order``)

    def twin(self) -> bool:
        if self.items < TWIN_MIN_ITEMS:
            return False
        return self.jsd_bits < TWIN_JSD_BITS or (self.order_items >= TWIN_MIN_ORDER_ITEMS
                                                 and self.same_order >= TWIN_SAME_ORDER and self.jsd_bits < TWIN_ORDER_JSD_BITS)

    def as_dict(self) -> dict[str, float]:
        return {"items": self.items, "jsd": round(self.jsd_bits, 6), "abs": round(self.mean_abs, 6), "order": round(self.same_order, 4),
                "order_items": self.order_items}


def _order(p: Sequence[float]) -> tuple[int, ...]:
    return tuple(0 if abs(a - b) < TWIN_TIE else (1 if a > b else -1) for i, a in enumerate(p) for b in p[i + 1:])


def behaviour_stats(rows: Sequence[Mapping[str, Any]], a: Sequence[Any], b: Sequence[Any]) -> BehaviourStats:
    """Per-item option distributions of two models on the same rows, compared."""
    jsd = mad = 0.0
    same = n = n_order = 0
    for r, pa, pb in zip(rows, a, b):
        labels = [str(x) for x in r["labels"]]
        ca, cb = SC.clean_probs(pa, labels), SC.clean_probs(pb, labels)
        if ca is None or cb is None:
            continue
        va, vb = [ca[lab] for lab in labels], [cb[lab] for lab in labels]
        j = 0.0
        for x, y in zip(va, vb):
            m = (x + y) / 2
            j += (x * math.log2(x / m) if x > 0 else 0.0) + (y * math.log2(y / m) if y > 0 else 0.0)
        jsd += max(0.0, j / 2)
        mad += sum(abs(x - y) for x, y in zip(va, vb)) / len(labels)
        if len(labels) >= 3:
            same += _order(va) == _order(vb)
            n_order += 1
        n += 1
    if n == 0:
        return BehaviourStats(0, 1.0, 1.0, 0.0)
    return BehaviourStats(n, jsd / n, mad / n, same / n_order if n_order else 0.0, n_order)


def nonproc_scores(items: Sequence[SC.ItemScore]) -> tuple[float | None, dict[str, float]]:
    """The score on the draw's non-proc items (tier != "proc_sealed"; None if it has none) and the score per tier."""
    rest = [it for it in items if it.tier != "proc_sealed"]
    nonproc = SC.score_items(rest).total if rest else None
    return nonproc, {t: round(sc.total, 4) for t, sc in SC.by_tier(items).items()}


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
        # a former champion crowned by a release whose copy check could pass without the base is not ranked
        former = [s for s in ts.submissions.values() if s.status == ST.FORMER and s.delta_checked]
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


# Keep-alive (docs/SPECIALIST-TRACKS.md, "Keep-alive"): the statuses that can hold a slot, in slot priority order (a
# coldkey with more qualifying entries than slots keeps the champion first). A keep-alive bridges a decision, it is not
# a standing payment: an entry pending confirmation holds it for at most KEEPALIVE_HOURS after its first pass, a new
# champion (waiting for the owner's approval of its award) for at most KEEPALIVE_HOURS after it was crowned. Decided
# entries (former champions, not better, rejected) and QUEUED, PENDING_VERIFY or WITHDRAWN entries never qualify. An entry
# must also score above ``KEEPALIVE_NONPROC_FLOOR`` on the non-proc items of its latest draw (``Submission.nonproc``), so
# learning the public procedural families alone does not earn it. An entry last scored before 1.2.0rc6 has no such score
# (``nonproc`` None and no ``by_tier``); it qualifies only with a total score above ``KEEPALIVE_LEGACY_FLOOR`` (1.2.0rc7),
# which a rules-only model cannot reach under the rc5 mix.
KEEPALIVE_STATUSES = (ST.CHAMPION, ST.PENDING_CONFIRM)
KEEPALIVE_HOURS = 24.0
KEEPALIVE_FUTURE_S = 3600.0    # a window start further in the future than this gives no slot


def _keepalive_since(s: ST.Submission) -> float | None:
    """When the entry's keep-alive window started: crowning for a champion, the first pass for a pending one. The first
    pass is read from ``confirm_after_ts`` (set once, at the first pass, to first pass + ``CONFIRM_DELAY_S``), not from
    ``last_eval_ts``, which a re-fetch at every confirmation attempt moves (an entry whose confirmation keeps failing must
    not keep its keep-alive)."""
    if s.status == ST.CHAMPION:
        t = s.champion_since_ts
    else:
        c = s.confirm_after_ts
        t = c - M.CONFIRM_DELAY_S if isinstance(c, (int, float)) else None
    return float(t) if isinstance(t, (int, float)) and math.isfinite(t) else None


def compute_keepalive(state: ST.SpecState, track_ids: Sequence[str], snapshot: Snapshot,
                      floors: Mapping[str, float | None] | None = None,
                      now: float | None = None,
                      nonproc_floors: Mapping[str, float | None] | None = None,
                      legacy_floors: Mapping[str, float | None] | None = None) -> dict[str, list[tuple[int, float]]]:
    """Per track, the keep-alive entries ``[(uid, score), ...]`` the share layer splits ``KEEPALIVE_SHARE`` among.

    An entry is a submission that passed the copy checks (exact hashes and the near-duplicate check against its base:
    ``delta_checked``, not pending verification, not rejected), whose hotkey still holds its uid and still commits to
    it, and whose latest score on a shared draw is above the track's entry floor (``floors``, default the release
    ``TRACK_ENTRY_FLOOR``), in one of the ``KEEPALIVE_STATUSES``: pending confirmation (first pass taken) for at most
    ``KEEPALIVE_HOURS`` after that first pass, or champion for at most ``KEEPALIVE_HOURS`` after its crowning. A
    missing, non-finite or far-future window start gives nothing (fail closed). Former champions, not-better, queued,
    pending-verification, withdrawn, rejected, never-scored and below-floor entries get nothing. The same draw's score on
    its non-proc items must be above ``nonproc_floors`` (default the release ``KEEPALIVE_NONPROC_FLOOR``); without a
    non-proc score and without per-tier scores (last scored before 1.2.0rc6) its total score must instead be above
    ``legacy_floors`` (default the release ``KEEPALIVE_LEGACY_FLOOR``; a track without one: no slot). Whether a champion's
    award is approved is not visible here: ``combine_weights`` drops an approved champion's slot. At most
    ``MAX_OPEN_PER_COLDKEY_TRACK`` entries per coldkey and track (by status priority, then commitment order). Pure,
    like ``compute_leaders``."""
    now = time.time() if now is None else now
    live = {n.hotkey: n for n in snapshot.neurons}
    out: dict[str, list[tuple[int, float]]] = {}
    for track_id in track_ids:
        floor = floors.get(track_id) if floors is not None else TRACK_ENTRY_FLOOR.get(track_id)
        np_floor = (nonproc_floors if nonproc_floors is not None else KEEPALIVE_NONPROC_FLOOR).get(track_id)
        legacy_floor = (legacy_floors if legacy_floors is not None else KEEPALIVE_LEGACY_FLOOR).get(track_id)
        ts = state.track(track_id)
        cands = []
        for s in ts.submissions.values():
            if s.status not in KEEPALIVE_STATUSES or not s.delta_checked:
                continue
            since = _keepalive_since(s)
            # fail closed on a timestamp more than an hour in the future (a clock that jumped): it would never expire
            if since is None or now - since > KEEPALIVE_HOURS * 3600.0 or since > now + KEEPALIVE_FUTURE_S:
                continue
            score = s.score
            if not isinstance(score, (int, float)) or not math.isfinite(score) or score <= 0 \
                    or (floor is not None and score <= floor):
                continue
            nonproc = s.nonproc
            if nonproc is None and not s.by_tier:
                # scored before 1.2.0rc6: no non-proc score; the total must be out of a rules-only model's reach
                if legacy_floor is None or score <= legacy_floor:
                    continue
            elif not isinstance(nonproc, (int, float)) or not math.isfinite(nonproc) \
                    or (np_floor is not None and nonproc <= np_floor):
                continue
            n = live.get(s.hotkey)
            c = parse_commitment(n.commitment) if n is not None else None
            if c is None or f"{s.hotkey}|{c.text()}" != s.key:
                continue
            cands.append((KEEPALIVE_STATUSES.index(s.status), s.first_seen_block, n.uid, s.key, n, float(score)))
        per_coldkey: dict[str, int] = {}
        res: list[tuple[int, float]] = []
        for *_, n, score in sorted(cands, key=lambda c: c[:4]):
            if per_coldkey.get(n.coldkey, 0) >= C.MAX_OPEN_PER_COLDKEY_TRACK or any(u == n.uid for u, _ in res):
                continue
            per_coldkey[n.coldkey] = per_coldkey.get(n.coldkey, 0) + 1
            res.append((n.uid, score))
        out[track_id] = res
    return out


def _no_base(sub: ST.Submission, detail: str = "") -> str:
    why = f"declared base model {sub.base_model or '(none)'} is not available on this validator; the near-duplicate check waits for it"
    return f"{why} ({detail})" if detail else why


class _Prepared:
    def __init__(self, sub: ST.Submission, path: str, weight_bytes: int):
        self.sub, self.path, self.weight_bytes = sub, path, weight_bytes
        self.infra_failure = False  # the last _serve failed for a host reason (docker, socket), not the model's
        self.probs: list[Any] = []  # the answers of the last _serve (behavioural near-duplicate check)


class SpecEvaluator:
    def __init__(self, cfg: C.SpecConfig, *, fetcher: F.Fetcher | None = None, backend_factory: BackendFactory | None = None,
                 base_provider: T.BaseProvider | None = None, clock: Callable[[], float] = time.time,
                 seed_source: Callable[[str, int], int] | None = None,
                 free_bytes: Callable[[str], int] | None = None, base_fetcher: B.BaseFetcher | None = None):
        """``base_fetcher``: downloads missing declared bases (spec/bases.py). By default only the production setup (no
        injected ``fetcher``, ``base_dir`` set, ``base_fetch`` on) gets one; tests pass their own with a fake hub."""
        self.cfg = cfg
        self.fetcher = fetcher or F.HfFetcher()
        self.backend_factory = backend_factory or self._docker_backend
        self.base_provider = base_provider or C.dir_base_provider(cfg.base_dir)
        if base_fetcher is None and fetcher is None and cfg.base_dir and cfg.base_fetch:
            base_fetcher = B.BaseFetcher.from_config(cfg)
        self.base_fetcher = base_fetcher
        self.clock = clock
        self._free_bytes = free_bytes or (lambda path: shutil.disk_usage(path).free)
        self.seed_source = seed_source or (lambda track, block: int.from_bytes(
            hashlib.sha256(f"{track}|{block}|{os.urandom(16).hex()}".encode()).digest()[:8], "big"))
        self.state = ST.load(cfg.state_path)
        self._lock = asyncio.Lock()
        self._pools: dict[str, list[dict]] = {}
        self._playoff_waits: set[str] = set()   # tracks whose cohort playoff waited for budget in the last cycle
        self._base_ids: dict[tuple, str] = {}    # (base dir, its files and sizes) -> tensor hash of the base model
        self._seed = T.load_sketch_seed(cfg.work_dir)   # secret sample positions of the delta sketch; never logged
        self._seed_id = T.seed_id(self._seed)
        os.makedirs(os.path.join(cfg.work_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(cfg.work_dir, "sketches"), exist_ok=True)

    # ------------------------------------------------------------------ public

    async def evaluate_cycle(self, snapshot: Snapshot) -> dict[str, list[tuple[int, float]]]:
        if self.base_fetcher is not None:
            self.base_fetcher.start_preseed(self.cfg.base_preseed)    # once, in the background
        async with self._lock:
            now = self.clock()
            if self.base_fetcher is not None:
                self.base_fetcher.new_cycle()
            active = self.cfg.active_tracks(now)
            self._pools.clear()
            for tdef in active:
                self._sync(tdef, snapshot, now)
            ST.save(self.state, self.cfg.state_path)
            # the hourly budget is shared: the track evaluated least recently goes first (1.2.0rc6; before, the first track
            # in the schedule always went first and could use the whole budget while the other waited)
            # a cohort playoff that waits for a full hour's budget goes first and holds the budget for itself (review G1):
            # the other tracks do not take the slots it waits for, so the wait ends when the rolling hour has emptied
            order = self._track_order(active)
            self._playoff_waits = set()
            for tdef in order:
                try:
                    await self._run_track(tdef, snapshot, now)
                except Exception:  # noqa: BLE001 - one failing track must not stop the others
                    log.exception("track %s: evaluation failed", tdef.track_id)
                ST.save(self.state, self.cfg.state_path)
                if tdef.track_id in self._playoff_waits:
                    log.info("track %s: the other tracks wait this cycle (the cohort playoff holds the budget)",
                             tdef.track_id)
                    break
            self._cleanup(active)
            return self.leaders(active, snapshot)

    def leaders(self, active: Sequence[C.TrackDef], snapshot: Snapshot) -> dict[str, list[tuple[int, float]]]:
        return compute_leaders(self.state, [t.track_id for t in active], snapshot)

    def keepalive(self, active: Sequence[C.TrackDef], snapshot: Snapshot) -> dict[str, list[tuple[int, float]]]:
        return compute_keepalive(self.state, [t.track_id for t in active], snapshot,
                                 {t.track_id: self._entry_floor(t) for t in active}, now=self.clock())

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
            if sub is not None and sub.status != ST.WITHDRAWN:
                sub.uid, sub.coldkey = n.uid, n.coldkey
                continue
            if sub is not None:
                # the same hotkey holds this commitment again (re-registered after a deregistration, or committed back):
                # a new entry in a later cohort; the withdrawn entry's first pass and score are not restored (1.2.0rc7).
                # The withdrawn one stays on record under its own key and keeps its hashes (the copy check decides).
                old_key = f"{key}#withdrawn@{sub.first_seen_block}"
                sub.key = old_key
                ts.submissions[old_key] = ts.submissions.pop(key)
                for h, owner_key in list(ts.hashes.items()):
                    if owner_key == key:
                        ts.hashes[h] = old_key
                ts.log(now, f"{key[:12]}: committed again after its withdrawal (uid {n.uid}), a new entry")
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
                 if s is not sub and s.coldkey == sub.coldkey and s.status in (ST.QUEUED, ST.PENDING_CONFIRM, ST.PENDING_VERIFY)]
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

    def _track_order(self, active: Sequence[C.TrackDef]) -> list[C.TrackDef]:
        """A track whose cohort playoff waited for budget first, then the track evaluated least recently."""
        return sorted(active, key=lambda t: (t.track_id not in self._playoff_waits, self._last_served(t)))

    def _last_served(self, tdef: C.TrackDef) -> float:
        """When a model of this track was last evaluated (0 = never); ties keep the schedule order (sorted is stable)."""
        ts = self.state.track(tdef.track_id)
        return max((s.last_eval_ts or 0.0 for s in ts.submissions.values() if s.status != ST.QUEUED), default=0.0)

    def _budget(self, now: float) -> int:
        self.state.eval_times = [t for t in self.state.eval_times if t > now - 3600]
        return max(0, self.cfg.max_evals_per_hour - len(self.state.eval_times))

    def _cleanup(self, active: Sequence[C.TrackDef]) -> None:
        keep = set()
        for ts in self.state.tracks.values():
            for s in ts.submissions.values():
                if s.status == ST.CHAMPION:     # served every round; a pending entry is fetched again (pinned revision)
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
            if sub.delta_checked:
                return kept
            # kept by a release whose base check could pass without the base: check it before it is served again
            if not await self._refresh_sketches(tdef, ts, sub):
                return self._renewal_wait(ts, kept, now)
            return await self._checked(ts, kept, now, self._near_check)
        sub.last_eval_ts = now
        try:
            ins = await self.fetcher.fetch(commitment, dest, category=tdef.category, hotkey=sub.hotkey, rules=self.cfg.rules)
        except F.FetchError as exc:
            if exc.kind == "rejected":
                sub.status, sub.reason = ST.REJECTED, str(exc)[:300]
                ts.log(now, f"rejected {sub.key[:12]}: {sub.reason[:120]}")
            elif sub.hash_named:
                # fetched before (1.2.0rc6: a pending entry's files are not kept): a hub, network or disk failure is not
                # the miner's; the publish deadline does not apply, MAX_ATTEMPTS failed fetches in a row do
                sub.attempts += 1
                if sub.attempts >= C.MAX_ATTEMPTS:
                    sub.status, sub.reason = ST.REJECTED, f"no longer available ({sub.attempts} fetches failed): {exc}"[:300]
                    ts.log(now, f"rejected {sub.key[:12]}: {sub.reason[:120]}")
                else:
                    sub.reason = f"waiting for the repo: {exc}"[:300]
            elif snap.block > sub.first_seen_block + C.PUBLISH_DEADLINE_BLOCKS:
                sub.status, sub.reason = ST.REJECTED, f"not available within {C.PUBLISH_DEADLINE_BLOCKS} blocks: {exc}"[:300]
                ts.log(now, f"expired {sub.key[:12]}")
            else:
                sub.reason = f"waiting for the repo: {exc}"[:300]
            return None
        assert ins.hashes is not None
        if sub.hash_named and ins.hashes.named != sub.hash_named:
            # fetched again (a pending entry's files are not kept, 1.2.0rc6): only the weights that took the first draw
            shutil.rmtree(dest, ignore_errors=True)
            sub.status, sub.reason = ST.REJECTED, "fetched again: other weights than at its first check"
            ts.log(now, f"rejected {sub.key[:12]}: {sub.reason}")
            return None
        with open(os.path.join(dest, MARKER), "w") as fh:
            json.dump({"hash": ins.hashes.named, "weight_bytes": ins.hashes.weight_bytes, "files": ins.facts.files}, fh)
        if sub.hash_named and sub.delta_checked and sub.status == ST.PENDING_CONFIRM:
            # a pending entry fetched again with the weights of its first check: as if its files had been kept (``_kept``)
            return _Prepared(sub, dest, ins.hashes.weight_bytes)
        sub.hash_named, sub.hash_unnamed = ins.hashes.named, ins.hashes.unnamed
        sub.base_model = str((ins.facts.manifest or {}).get("base_model", ""))
        await self._fetch_base(sub)
        if not await self._refresh_sketches(tdef, ts, sub):
            return self._renewal_wait(ts, _Prepared(sub, dest, ins.hashes.weight_bytes), now)
        return await self._checked(ts, _Prepared(sub, dest, ins.hashes.weight_bytes), now, self._copy_check)

    async def _checked(self, ts: ST.TrackState, p: _Prepared, now: float,
                       check: Callable[[ST.TrackState, ST.Submission, str], tuple[str, str]]) -> _Prepared | None:
        """Run a copy check; a refused or unverifiable submission is not served (its status and reason say why)."""
        sub = p.sub
        # reads about half of the candidate and of its base from disk: never on the event loop
        verdict, why = await asyncio.get_running_loop().run_in_executor(None, check, ts, sub, p.path)
        if not verdict:
            return p
        if verdict == ST.PENDING_VERIFY:
            self._hold(ts, sub, why, now, self._base_wait_kind(sub))
        else:
            sub.status, sub.reason = verdict, why
            ts.log(now, f"rejected {sub.key[:12]}: {why[:120]}")
        shutil.rmtree(p.path, ignore_errors=True)
        return None

    def _renewal_wait(self, ts: ST.TrackState, p: _Prepared, now: float) -> None:
        """Earlier sketches must be renewed before ``p`` can be checked, and the disk is too full for the re-fetch: held
        (``_run_track`` queues it again next cycle) instead of checked against an incomplete set of sketches."""
        self._hold(ts, p.sub, RENEWAL_WAIT, now, ST.WAIT_INFRA)
        shutil.rmtree(p.path, ignore_errors=True)
        return None

    async def _fetch_base(self, sub: ST.Submission) -> None:
        """Download the declared base of ``sub`` if it is missing here and the base fetcher may fetch it now."""
        if self.base_fetcher is not None and sub.base_model and self.base_provider(sub.base_model) is None:
            await self.base_fetcher.ensure(sub.base_model)

    def _base_detail(self, sub: ST.Submission) -> str:
        return self.base_fetcher.reason(sub.base_model) if self.base_fetcher is not None and sub.base_model else ""

    def _base_wait_kind(self, sub: ST.Submission) -> str:
        """WAIT_INFRA if the declared base is missing for this host's reason (``BaseFetcher.waits_here``), else WAIT_MINER
        (the base is here and the weights are not comparable, or the base itself cannot be had)."""
        missing = bool(sub.base_model) and self.base_provider(sub.base_model) is None
        here = missing and self.base_fetcher is not None and self.base_fetcher.waits_here(sub.base_model)
        return ST.WAIT_INFRA if here else ST.WAIT_MINER

    @staticmethod
    def _hold(ts: ST.TrackState, sub: ST.Submission, why: str, now: float, kind: str = ST.WAIT_MINER) -> None:
        """PENDING_VERIFY: kept, not served, not ranked, not crowned; checked again every cycle. ``kind``: whose wait it
        is (``ST.WAIT_INFRA`` holds back the cohort's first crowning)."""
        sub.status, sub.reason, sub.wait_kind = ST.PENDING_VERIFY, why, kind
        if ts.champion == sub.key:
            ts.champion = ""            # no award while the copy check is pending
        log.warning("track %s: %s (uid %d) is pending verification: %s", sub.track_id, sub.key[:12], sub.uid, why)
        ts.log(now, f"pending verification {sub.key[:12]}: {why[:120]}")

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

    def _copy_check(self, ts: ST.TrackState, sub: ST.Submission, model_dir: str) -> tuple[str, str]:
        """Exact tensor hashes (re-sharding and renaming proof), then the near-duplicate check.

        Returns ``(verdict, reason)``: verdict "" = passed, ``ST.REJECTED`` or ``ST.PENDING_VERIFY``."""
        why = self._exact_check(ts, sub)
        if why:
            return ST.REJECTED, why
        return self._near_check(ts, sub, model_dir)

    def _exact_check(self, ts: ST.TrackState, sub: ST.Submission) -> str:
        """Exact tensor hashes against every earlier commitment of the track. Returns a rejection reason. A withdrawn
        commitment of the same coldkey does not count (its model committed again after a deregistration)."""
        for h in (sub.hash_named, sub.hash_unnamed):
            owner_key = ts.hashes.get(h)
            owner = ts.submissions.get(owner_key) if owner_key else None
            if owner is None or owner is sub:
                continue
            if owner.status == ST.WITHDRAWN and owner.coldkey == sub.coldkey:
                # re-registration (1.2.0rc7): the same coldkey commits its withdrawn model again, under a new uid or a
                # new hotkey. Not a copy of itself; the hash passes to the new entry. Another coldkey stays refused.
                continue
            if self._earlier(owner, sub) or owner.status in (ST.CHAMPION, ST.FORMER):
                return f"identical tensors to an earlier commitment (uid {owner.uid})"
            owner.status, owner.reason = ST.REJECTED, f"identical tensors to an earlier commitment (uid {sub.uid})"
            if ts.champion == owner.key:
                ts.champion = ""
        ts.hashes[sub.hash_named] = sub.key
        ts.hashes[sub.hash_unnamed] = sub.key
        return ""

    def _sketch_current(self, sub: ST.Submission) -> bool:
        path = self._sketch_path(sub)
        try:
            return os.path.exists(path) and T.sketch_seed_id(path) == self._seed_id
        except Exception:   # unreadable or damaged file (zipfile.BadZipFile from np.load, ...): renew it
            return False

    async def _refresh_sketches(self, tdef: C.TrackDef, ts: ST.TrackState, sub: ST.Submission) -> bool:
        """Before ``sub`` is checked: recompute the sketch of every earlier checked submission on the same base whose
        stored sketch was taken with another seed or rule (rc5 sketches, a changed seed, a lost work directory). Sketches
        are only comparable at the same positions. The weights are re-fetched (the commitment pins the revision) unless
        they are still on disk, and removed again afterwards unless the submission is kept anyway.

        False = a re-fetch would leave less than ``min_free_bytes`` free: ``sub`` must wait (``RENEWAL_WAIT``). Any
        other failure on one earlier submission is logged and skipped (its sketch stays old and is not compared)."""
        base_dir = self.base_provider(sub.base_model) if sub.base_model else None
        if base_dir is None:
            return True
        base_id = self._base_identity(base_dir)
        loop = asyncio.get_running_loop()
        for other in sorted(ts.submissions.values(), key=lambda s: (s.first_seen_block, s.uid, s.key)):
            if (other is sub or not self._earlier(other, sub) or not other.delta_checked
                    or other.status in (ST.REJECTED, ST.WITHDRAWN) or self._resolved_base(other) != base_id
                    or self._sketch_current(other)):
                continue
            dest = self._model_dir(other)
            try:
                kept = self._kept(other, dest)
                if kept is None:
                    free = self._free_bytes(self.cfg.work_dir)
                    if free < self.cfg.min_free_bytes:
                        log.warning("track %s: sketch of %s not renewed, %.0f GB free, %.0f GB needed before a download",
                                    tdef.track_id, other.key[:12], free / 1e9, self.cfg.min_free_bytes / 1e9)
                        return False
                    try:
                        ins = await self.fetcher.fetch(SpecCommitment(other.track_id, other.repo, other.revision), dest,
                                                       category=tdef.category, hotkey=other.hotkey, rules=self.cfg.rules)
                    except F.FetchError as exc:
                        log.warning("track %s: sketch of %s cannot be renewed: %s", tdef.track_id, other.key[:12], exc)
                        continue
                    if ins.hashes is None or ins.hashes.named != other.hash_named:
                        log.warning("track %s: sketch of %s cannot be renewed: other weights than at its check",
                                    tdef.track_id, other.key[:12])
                        shutil.rmtree(dest, ignore_errors=True)
                        continue
                    with open(os.path.join(dest, MARKER), "w") as fh:
                        json.dump({"hash": ins.hashes.named, "weight_bytes": ins.hashes.weight_bytes, "files": ins.facts.files}, fh)
                sk = await loop.run_in_executor(None, lambda: T.delta_sketch_ex(dest, base_dir, seed=self._seed))
                T.save_sketch(sk.deltas, self._sketch_path(other), sk.seed_id)
                ts.log(self.clock(), f"{other.key[:12]}: sketch renewed for the current sampling rule")
            except Exception as exc:    # ENOSPC, a damaged download, ...: not this cycle's whole track
                log.warning("track %s: sketch of %s cannot be renewed: %r", tdef.track_id, other.key[:12], exc)
            finally:
                # a kept submission's directory is checked against its MARKER before it is served (``_kept``)
                if other.status not in (ST.CHAMPION, ST.PENDING_CONFIRM):
                    shutil.rmtree(dest, ignore_errors=True)
        return True

    def _base_identity(self, base_dir: str) -> str:
        """Tensor hash of a base model directory: the same base under another name (an alias, a mirror, another
        directory) has the same identity. Cached per directory content, computed once per process."""
        key = (os.path.realpath(base_dir), tuple(sorted(F.list_dir(base_dir).items())))
        if key not in self._base_ids:
            self._base_ids[key] = T.canonical_hashes(base_dir).named
        return self._base_ids[key]

    def _resolved_base(self, sub: ST.Submission) -> str:
        """The base identity of an earlier submission (resolved now if a state written before ``base_id`` lacks it)."""
        if sub.base_id or not sub.delta_checked:
            return sub.base_id
        base_dir = self.base_provider(sub.base_model) if sub.base_model else None
        return self._base_identity(base_dir) if base_dir else ""

    def _near_check(self, ts: ST.TrackState, sub: ST.Submission, model_dir: str) -> tuple[str, str]:
        """Weight change against the resolved declared base, compared with earlier submissions on the same base.

        Without the base on this host the submission is not served and not crowned: it waits in PENDING_VERIFY and is
        checked again once the base is there. A submission that was checked before stays checked."""
        base_dir = self.base_provider(sub.base_model) if sub.base_model else None
        if base_dir is None:
            if sub.delta_checked:
                return "", ""
            return ST.PENDING_VERIFY, _no_base(sub, self._base_detail(sub))
        sub.base_id = self._base_identity(base_dir)
        sub.sketch_rule = T.SKETCH_RULE
        sk = T.delta_sketch_ex(model_dir, base_dir, seed=self._seed)
        sketch = sk.deltas
        cov = T.sketch_coverage(sketch, model_dir, base_dir)
        fp8 = T.coverage_note(cov)
        if not sketch:
            return ST.PENDING_VERIFY, (f"no weight of the model matches the declared base model {sub.base_model} "
                                       f"(name, shape): the near-duplicate check cannot be made{fp8}")
        if not cov.enough():
            return ST.PENDING_VERIFY, (f"the weights comparable with the declared base model {sub.base_model} hold only "
                                       f"{cov.fraction:.1%} of its float parameters ({T.MIN_SKETCH_COVERAGE:.0%} needed): "
                                       f"the near-duplicate check cannot be made{fp8}")
        sub.delta_checked = True
        if not sk.changed:
            return ST.REJECTED, "weights are identical to the declared base model"
        for other in sorted(ts.submissions.values(), key=lambda s: (s.first_seen_block, s.uid, s.key)):
            if other is sub or other.status in (ST.REJECTED, ST.WITHDRAWN) or self._resolved_base(other) != sub.base_id:
                continue
            if not self._earlier(other, sub) or not self._sketch_current(other):
                continue    # another seed or rule samples other positions: not comparable (_refresh_sketches renews it)
            stats = T.compare_sketches(T.load_sketch(self._sketch_path(other)), sketch)
            if stats.derived():
                sub.derived_from, sub.derived_how = other.key, "weights"
                ts.log(self.clock(), f"{sub.key[:12]} is derived from {other.key[:12]} (cos {stats.cosine:.3f}, rel {stats.relative_l2:.3f})")
                break
        T.save_sketch(sketch, self._sketch_path(sub), sk.seed_id)
        return "", ""

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
        p.probs = list(probs)
        return items

    async def _run_track(self, tdef: C.TrackDef, snap: Snapshot, now: float) -> None:
        ts = self.state.track(tdef.track_id)
        for s in ts.ordered(ST.PENDING_VERIFY):
            if s.reason == RENEWAL_WAIT:
                s.status, s.reason = ST.QUEUED, ""     # the disk check runs again before every renewal fetch
            elif s.base_id and s.base_model and s.sketch_rule < T.SKETCH_RULE and self.base_provider(s.base_model) is not None:
                # held by an older copy-check rule although the base is here (e.g. rc5: another dtype than the base)
                s.status, s.reason = ST.QUEUED, ""
                ts.log(now, f"{s.key[:12]}: copy-check rule changed, queued for the copy check again")
            elif not s.base_id and s.base_model:
                await self._fetch_base(s)
                if self.base_provider(s.base_model) is not None:
                    s.status, s.reason = ST.QUEUED, ""
                    ts.log(now, f"{s.key[:12]}: declared base model now available, queued for the copy check")
                elif self._base_detail(s):
                    s.reason, s.wait_kind = _no_base(s, self._base_detail(s)), self._base_wait_kind(s)
        for s in list(ts.submissions.values()):
            # admitted by a release whose copy check passed without the base: held at once if the base is missing here
            # (with the base, the check runs before it is served again)
            if s.status in (ST.CHAMPION, ST.PENDING_CONFIRM) and not s.delta_checked:
                await self._fetch_base(s)
                if not s.base_model or self.base_provider(s.base_model) is None:
                    self._hold(ts, s, _no_base(s, self._base_detail(s)), now, self._base_wait_kind(s))
        playoff = self._cohort_step(ts, now)
        budget = self._budget(now)
        if budget < 1:
            return
        if playoff:
            await self._run_playoff(tdef, ts, playoff, snap, now, budget)
            return
        # an entry whose first draw was lost before 1.2.0rc7 (40-draw retention) can never be confirmed: it must not
        # head the queue, or its group is prepared and dropped every cycle and no later group is confirmed (review F3)
        due = [s for s in ts.ordered(ST.PENDING_CONFIRM)
               if not s.confirm_draw and s.confirm_after_ts is not None and now >= s.confirm_after_ts and s.draw1 in ts.draws]
        if due:
            # one first draw per confirmation group: the confirmation draw excludes the first draws of the whole group,
            # and two first draws together can leave too few non-proc items (SampleError on every cycle)
            due = [s for s in due if s.draw1 == due[0].draw1]
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
            lost = [p for p in prepared if p.sub.draw1 not in ts.draws]
            if lost:   # its first draw was pruned from the state: a disjoint confirmation is impossible, so it waits
                for p in lost:
                    log.error("track %s: %s cannot be confirmed: its first draw %s is no longer in the state",
                              tdef.track_id, p.sub.key[:12], p.sub.draw1)
                    ts.log(now, f"confirmation of {p.sub.key[:12]} waits: first draw {p.sub.draw1} not in the state")
                prepared = [p for p in prepared if p not in lost]
                if not prepared:
                    return
            for p in prepared:
                exclude |= set(ts.draws[p.sub.draw1].item_ids)
        seed = self.seed_source(tdef.track_id, snap.block)
        try:
            rows, fresh = S.draw(self._pool(tdef.category), n=self.cfg.sample_size, seed=seed, use_counts=ts.used, exclude=exclude)
        except S.SampleError as exc:
            log.warning("track %s: no draw this cycle: %s", tdef.track_id, exc)
            ts.log(now, f"no draw: {exc}; refill the sealed pool")
            return
        draw_id = self._record_draw(ts, rows, fresh, seed, snap, now)
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
        try:
            twins = self._behaviour_twins(ts, [p for p in group if p.sub.key in results], prepared, rows, draw_id, now)
        except Exception:   # a measurement bug must not discard the group's scores
            log.exception("track %s: behavioural near-duplicate check failed, no twins this group", tdef.track_id)
            twins = {}
        self._decide(ts, tdef, [p.sub for p in prepared if p.sub.key in results], results, champ, confirm_mode, draw_id, seed, now,
                     twins)
        if confirm_mode:
            self._cohort_step(ts, now)    # a lone confirmed entry of a complete cohort is crowned in the same cycle

    def _record_draw(self, ts: ST.TrackState, rows: list[dict], fresh: bool, seed: int, snap: Snapshot, now: float) -> str:
        draw_id = hashlib.sha256(f"{ts.track_id}|{seed}".encode()).hexdigest()[:16]
        ts.draws[draw_id] = ST.Draw(draw_id, now, snap.block, seed, [r["id"] for r in rows], fresh)
        for r in rows:
            ts.used[r["id"]] = ts.used.get(r["id"], 0) + 1
        self._prune_draws(ts)
        if not fresh:
            ts.log(now, "sealed pool exhausted: some items were reused; refill the pool")
        ST.save(self.state, self.cfg.state_path)
        return draw_id

    @staticmethod
    def _prune_draws(ts: ST.TrackState) -> None:
        """Keep every draw a pending entry still needs (its first draw, which its confirmation must exclude, and the draw
        it passed its confirmation on) and the newest ``MAX_UNREFERENCED_DRAWS`` others (1.2.0rc7; before, the 40 newest
        draws, so a pending entry could lose its first draw and never be confirmed)."""
        needed = {d for s in ts.submissions.values() if s.status == ST.PENDING_CONFIRM for d in (s.draw1, s.confirm_draw) if d}
        rest = [d for d in sorted(ts.draws.values(), key=lambda d: d.ts) if d.draw_id not in needed]
        for old in rest[:-MAX_UNREFERENCED_DRAWS]:
            ts.draws.pop(old.draw_id, None)

    def _behaviour_twins(self, ts: ST.TrackState, served: list[_Prepared], challengers: list[_Prepared], rows: list[dict],
                         draw_id: str, now: float) -> dict[str, list[str]]:
        """For each challenger: the earlier submissions of the group (incumbent, parents, other challengers) whose answers
        on this draw it shares (``BehaviourStats.twin``). Every measured pair is kept in the challenger's history, so the
        state holds the real noise for a later calibration. A twin must be beaten by the full margin; the first one
        becomes ``derived_from`` if the weights did not name a parent."""
        out: dict[str, list[str]] = {}
        for c in challengers:
            if c.sub.key not in {p.sub.key for p in served}:
                continue
            measured: dict[str, dict[str, float]] = {}
            for o in served:
                if o.sub is c.sub or not self._earlier(o.sub, c.sub):
                    continue
                st = behaviour_stats(rows, o.probs, c.probs)
                measured[o.sub.key] = st.as_dict()
                if st.twin():
                    out.setdefault(c.sub.key, []).append(o.sub.key)
                    ts.log(now, f"{c.sub.key[:12]} answers like {o.sub.key[:12]} (jsd {st.jsd_bits:.4f} bits, same order "
                                f"{st.same_order:.1%} of {st.items})")
            if measured:
                c.sub.history.append({"draw": draw_id, "kind": "behaviour", "vs": measured, "twins": out.get(c.sub.key, [])})
            if out.get(c.sub.key) and not c.sub.derived_from:
                c.sub.derived_from, c.sub.derived_how = out[c.sub.key][0], "behaviour"
        return out

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
                champ: ST.Submission | None, confirm_mode: bool, draw_id: str, seed: int, now: float,
                twins: Mapping[str, Sequence[str]] | None = None) -> None:
        """``twins``: per challenger, the earlier submissions it answers like on this draw (``_behaviour_twins``)."""
        twins = twins or {}
        total = {k: SC.score_items(v).total for k, v in results.items()}
        tiers = {k: nonproc_scores(v) for k, v in results.items()}

        def scored(s: ST.Submission) -> None:
            s.score = total[s.key]
            s.nonproc, s.by_tier = tiers[s.key][0], dict(tiers[s.key][1])

        line = f"draw {draw_id[:8]} non-proc: " + ", ".join(
            f"{k[:12]} {'-' if np_ is None else f'{np_:.2f}'}" for k, (np_, _) in tiers.items())
        ts.log(now, line)
        log.info("track %s: %s", tdef.track_id, line)
        if champ is not None:
            scored(champ)
            champ.last_eval_ts = now
        for pk in {c.derived_from for c in challengers if c.derived_from}:
            if pk in results and pk in ts.submissions and (champ is None or pk != champ.key):
                scored(ts.submissions[pk])
        current = champ
        promoted = False
        for ch in challengers:
            scored(ch)
            ch.last_eval_ts = now
            inc_items = results.get(current.key) if current else None
            se = SC.paired_se(results[ch.key], inc_items, seed) if inc_items is not None else 0.0
            inc_total = total[current.key] if current else None
            parent_ok, parent_gain = True, None
            # the weight parent (unless it is the incumbent, which the margin already covers) and every behavioural twin,
            # the incumbent included: the full margin over each, also on the confirmation draw
            must_beat = [k for k in twins.get(ch.key, ()) if k in results]
            par = ts.submissions.get(ch.derived_from) if ch.derived_from else None
            if par is not None and par.key in results and par.key != (current.key if current else "") and par.key not in must_beat:
                must_beat.append(par.key)
            for pk in must_beat:
                pse = SC.paired_se(results[ch.key], results[pk], seed)
                gain = total[ch.key] - total[pk]
                parent_gain = gain if parent_gain is None else min(parent_gain, gain)
                parent_ok = parent_ok and gain >= M.required_margin(pse)
            if not confirm_mode:
                fp = M.first_pass(ch.score, inc_total, se, entry_floor=self._entry_floor(tdef))
                ch.history.append({"draw": draw_id, "kind": "first", "score": ch.score, "incumbent": inc_total, "gain": fp.gain, "delta": fp.delta,
                                   "nonproc": ch.nonproc, "by_tier": dict(ch.by_tier)})
                if fp.passed and parent_ok:
                    ch.status, ch.reason = ST.PENDING_CONFIRM, ""
                    ch.delta, ch.first_gain, ch.draw1 = fp.delta, fp.gain, draw_id
                    ch.confirm_draw, ch.confirm_score = "", None
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
            ch.history.append({"draw": draw_id, "kind": "confirm", "score": ch.score, "incumbent": inc_total,
                               "nonproc": ch.nonproc, "by_tier": dict(ch.by_tier)})
            if ok and current is None:
                # vacant track (1.2.0rc7): not crowned yet; the cohort playoff decides (``_cohort_step``)
                ch.confirm_draw, ch.confirm_score = draw_id, ch.score
                _, open_ = self._cohort(ts, ch.first_seen_block)
                ts.log(now, f"{ch.key[:12]} passed the confirmation draw ({ch.score:.2f}); waits for its cohort "
                            f"(first seen by block {ch.first_seen_block}, {len(open_)} entries still open)")
                continue
            if ok:
                if current is not None:
                    current.status = ST.FORMER
                ch.status, ch.reason, ch.champion_since_ts = ST.CHAMPION, "", now
                ts.champion, current, promoted = ch.key, ch, True
                ts.log(now, f"{ch.key[:12]} is the new champion ({ch.score:.2f})")
            else:
                ch.status = ST.NOT_BETTER
                ch.reason = "confirmation draw: gain too small" if inc_total is not None else "confirmation draw: below the entry floor"
                if not parent_ok:
                    ch.reason += f"; does not beat its parent ({parent_gain:+.2f})"
                ts.log(now, f"{ch.key[:12]} {ch.reason}")

    # ------------------------------------------------------- first champion: cohort playoff

    def _cohort(self, ts: ST.TrackState, block: int) -> tuple[list[ST.Submission], list[ST.Submission]]:
        """The cohort of ``block`` on a vacant track: (its entries that passed their confirmation and wait for the playoff,
        best confirmation score first; the entries first seen in that block or earlier that are still open)."""
        ready = [s for s in ts.submissions.values()
                 if s.status == ST.PENDING_CONFIRM and s.confirm_draw and s.first_seen_block == block]
        ready.sort(key=lambda s: (-(s.confirm_score if s.confirm_score is not None else -math.inf),
                                  s.first_seen_block, s.uid, s.key))
        open_ = [s for s in ts.submissions.values() if s.first_seen_block <= block and self._cohort_open(ts, s)]
        return ready, sorted(open_, key=lambda s: (s.first_seen_block, s.uid, s.key))

    @staticmethod
    def _cohort_open(ts: ST.TrackState, s: ST.Submission) -> bool:
        """Still undecided in a way that holds back its cohort's crowning: queued (unless the miner's repo is not
        available), waiting for this host (``wait_kind`` WAIT_INFRA, set where the wait starts; never the reason text,
        which can embed the miner's base id), or pending a confirmation that can still run."""
        if s.status == ST.QUEUED:
            return not s.reason.startswith("waiting for the repo")
        if s.status == ST.PENDING_VERIFY:
            return s.wait_kind == ST.WAIT_INFRA
        if s.status == ST.PENDING_CONFIRM:
            return not s.confirm_draw and s.draw1 in ts.draws    # a lost first draw can never be confirmed: it waits alone
        return False

    def _cohort_step(self, ts: ST.TrackState, now: float) -> list[ST.Submission]:
        """The earliest cohort with confirmed entries, on a vacant track: nothing while one of its entries is still open; a
        lone confirmed entry is crowned at once (it cleared the floor on both its draws, as before 1.2.0rc7); several
        are returned for the playoff. With a champion, waiting entries go back to the normal challenger path. Open entries
        hold the cohort back for at most COHORT_MAX_WAIT_S after the earliest passed confirmation of its ready entries."""
        if ts.champion:
            self._release_waiting(ts, now, "")
            return []
        blocks = [s.first_seen_block for s in ts.submissions.values() if s.status == ST.PENDING_CONFIRM and s.confirm_draw]
        if not blocks:
            return []
        block = min(blocks)
        ready, open_ = self._cohort(ts, block)
        if open_:
            since = min((ts.draws[s.confirm_draw].ts for s in ready if s.confirm_draw in ts.draws), default=now)
            if now - since < COHORT_MAX_WAIT_S:
                return []
            log.info("track %s: cohort of block %d complete by its deadline (%.0f h); %d open entries no longer wait: %s",
                     ts.track_id, block, COHORT_MAX_WAIT_S / 3600, len(open_), ", ".join(s.key[:12] for s in open_))
        if len(ready) == 1:
            self._crown(ts, ready[0], now, f"the only confirmed entry of its cohort (block {block})")
            return []
        return ready

    def _crown(self, ts: ST.TrackState, sub: ST.Submission, now: float, why: str) -> None:
        sub.status, sub.reason, sub.champion_since_ts = ST.CHAMPION, "", now
        sub.confirm_draw, sub.confirm_score = "", None
        ts.champion, ts.playoff_aborts = sub.key, 0
        ts.log(now, f"{sub.key[:12]} is the new champion ({sub.score:.2f}): {why}")
        log.info("track %s: %s (uid %d) is the new champion: %s", ts.track_id, sub.key[:12], sub.uid, why)
        self._release_waiting(ts, now, sub.key)

    @staticmethod
    def _release_waiting(ts: ST.TrackState, now: float, champion: str) -> None:
        """Entries still waiting for a playoff once the track has a champion: they stay pending and are confirmed against
        the champion with the full margin (``M.confirm``, ``new_incumbent``) on the next confirmation draw."""
        for s in ts.ordered(ST.PENDING_CONFIRM):
            if s.confirm_draw and s.key != champion:
                s.confirm_draw, s.confirm_score = "", None
                s.reason = "a champion was crowned: confirms against it with the full margin"
                ts.log(now, f"{s.key[:12]} back on the challenger path ({s.reason})")

    async def _run_playoff(self, tdef: C.TrackDef, ts: ST.TrackState, ready: list[ST.Submission], snap: Snapshot,
                           now: float, budget: int) -> None:
        """The confirmed entries of a complete cohort, best confirmation score first and at most ``budget`` of them, on one
        shared fresh draw. Every one of them must be ready to serve; otherwise the playoff waits for the next cycle.

        It runs only with a full field: every entry, or a whole hour's evaluations (``max_evals_per_hour``; a configured
        limit of 1 an hour: the best confirmation alone). The confirmation scores come from different draws, so the field
        is not cut by whatever is left of the hour (review F5). Tracks are served least recently served first, so a
        declining playoff track goes first once the hour's budget is back. Starvation note: while another track keeps
        taking freed slots one at a time, the playoff can wait up to about an hour (until the rolling window empties)."""
        need = min(len(ready), self.cfg.max_evals_per_hour)
        if budget < need:
            log.info("track %s: cohort playoff of %d entries waits for budget (%d left, %d needed)", tdef.track_id,
                     len(ready), budget, need)
            self._playoff_waits.add(tdef.track_id)
            return
        field = ready[:budget]
        free = self._free_bytes(self.cfg.work_dir)
        if free < self.cfg.min_free_bytes:
            log.warning("track %s: cohort playoff skipped, %.0f GB free on the work directory's filesystem, %.0f GB needed",
                        tdef.track_id, free / 1e9, self.cfg.min_free_bytes / 1e9)
            return
        prepared: list[_Prepared] = []
        for s in field:
            p = await self._prepare(tdef, s, snap, now)
            if p is None:      # a fetch failure (counted toward MAX_ATTEMPTS) or a status change: all or nobody
                ts.log(now, f"cohort playoff waits: {s.key[:12]} could not be prepared ({s.reason[:80]})")
                return
            prepared.append(p)
        # the pools are small: the playoff draw may reuse items of the entries' earlier draws (least-used items first)
        seed = self.seed_source(tdef.track_id, snap.block)
        try:
            rows, fresh = S.draw(self._pool(tdef.category), n=self.cfg.sample_size, seed=seed, use_counts=ts.used)
        except S.SampleError as exc:
            log.warning("track %s: no playoff draw this cycle: %s", tdef.track_id, exc)
            ts.log(now, f"no draw: {exc}; refill the sealed pool")
            return
        draw_id = self._record_draw(ts, rows, fresh, seed, snap, now)
        ts.log(now, f"cohort playoff on draw {draw_id[:8]}: " + ", ".join(f"uid {p.sub.uid}" for p in prepared))
        results: dict[str, list[SC.ItemScore]] = {}
        failed: list[_Prepared] = []
        for p in prepared:
            items = await self._serve(p, rows, self.clock())
            if items is None:
                failed.append(p)
                continue
            p.sub.attempts = 0
            results[p.sub.key] = items
        if len(failed) == len(prepared):
            ts.log(now, "cohort playoff: no model could be served; nothing decided, next cycle again")
            return
        if any(p.infra_failure for p in failed):
            ts.playoff_aborts += 1
            if ts.playoff_aborts < C.MAX_ATTEMPTS:
                ts.log(now, "cohort playoff: a model could not be served for a host reason; nothing decided, next cycle again")
                return
            # the same participants keep failing to start on this host (review F4): decided among those served
            ts.log(now, f"cohort playoff: {ts.playoff_aborts} playoffs abandoned for a host reason; decided among the "
                        f"{len(prepared) - len(failed)} entries served")
        for p in failed:    # the model's own failure: counted, and it takes no part in this playoff
            p.sub.attempts += 1
            if p.sub.attempts >= C.MAX_ATTEMPTS:
                p.sub.status = ST.REJECTED
                p.sub.reason = "could not be served after repeated attempts: " + p.sub.reason
        served = [p for p in prepared if p.sub.key in results]
        try:
            twins = self._behaviour_twins(ts, served, served, rows, draw_id, now)
        except Exception:   # noqa: BLE001 - a measurement bug must not discard the playoff's scores
            log.exception("track %s: behavioural near-duplicate check failed, no twins this playoff", tdef.track_id)
            twins = {}
        ts.playoff_aborts = 0
        self._decide_playoff(ts, tdef, [p.sub for p in served], results, draw_id, seed, now, twins)

    def _decide_playoff(self, ts: ST.TrackState, tdef: C.TrackDef, field: list[ST.Submission],
                        results: dict[str, list[SC.ItemScore]], draw_id: str, seed: int, now: float,
                        twins: Mapping[str, Sequence[str]]) -> None:
        """The best score on the playoff draw that clears the entry floor is crowned, unless it answers like (or derives
        from) another entry of the playoff and does not beat it by the full margin there. Ties go to the earlier
        commitment. If nobody clears the floor, nobody is crowned and the entries served are not better."""
        total = {k: SC.score_items(v).total for k, v in results.items()}
        floor = self._entry_floor(tdef)
        for s in field:
            np_, by_tier = nonproc_scores(results[s.key])
            s.score, s.nonproc, s.by_tier, s.last_eval_ts = total[s.key], np_, dict(by_tier), now
            s.history.append({"draw": draw_id, "kind": "playoff", "score": s.score, "field": len(field),
                              "nonproc": s.nonproc, "by_tier": dict(s.by_tier)})
        line = f"playoff draw {draw_id[:8]}: " + ", ".join(f"uid {s.uid} {total[s.key]:.2f}" for s in field)
        ts.log(now, line)
        log.info("track %s: %s", tdef.track_id, line)
        for s in sorted(field, key=lambda s: (-total[s.key], s.first_seen_block, s.uid, s.key)):
            if total[s.key] < floor:
                break
            must_beat = [k for k in twins.get(s.key, ()) if k in results]
            if s.derived_from in results and s.derived_from not in must_beat:
                must_beat.append(s.derived_from)
            short = [k for k in must_beat
                     if total[s.key] - total[k] < M.required_margin(SC.paired_se(results[s.key], results[k], seed))]
            if short:
                ts.log(now, f"{s.key[:12]} not crowned: does not beat {short[0][:12]} by the full margin on the playoff draw")
                continue
            self._crown(ts, s, now, f"best of the cohort playoff of {len(field)} on draw {draw_id[:8]}")
            return
        for s in field:
            s.status, s.reason = ST.NOT_BETTER, "cohort playoff draw: below the entry floor"
            s.confirm_draw, s.confirm_score = "", None
            ts.log(now, f"{s.key[:12]} {s.reason}")
