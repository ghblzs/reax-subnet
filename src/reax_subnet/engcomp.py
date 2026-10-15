"""Engine competition ``reax-eng/1`` (unreleased; docs/ENGINE-COMPETITION.md).

Competitors do not serve traffic. A competitor hotkey commits a pinned engine image on chain::

    reax-eng/1:<track>:<image>@sha256:<64 hex>

The validator pulls that exact digest, runs it next to the track's reference engine on its own GPU in the same
session (no network, read-only, the same pinned weights mounted read-only) and drives both with the same fresh,
secretly perturbed decisions. This module holds the pure parts: commitment parsing, the correctness gate
(logit distance against the reference engine's own answers, s1scoring.logit_distance), the speed score, the
champion rule and the weight split. engrunner.py starts engines and measures them.

Rules (all from the validator's own measurements, nothing the engine reports is trusted):

* correctness: every timed answer is compared with the reference engine's answer for the same item:
  mean distance <= mean_tau, every item <= item_tau, top choice agrees on >= min_top_agreement of the items;
* anti-cache probes: items that repeat an earlier state with a different question (and the reverse) must also
  match the reference, so an engine that caches by a partial key fails;
* determinism: a re-run of some items at concurrency 1 must reproduce the timed answers within determinism_tau;
* speed: throughput ratio candidate / reference at the track's concurrency, median over interleaved runs;
  concurrency-1 latency may not get worse than max_latency_ratio x the reference;
* champion: the best valid submission that beats the reference by min_gain; a challenger must beat the
  champion's score by margin (copying the champion therefore never wins); an identical digest counts only for
  the hotkey that committed it first.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional, Sequence

from .s1scoring import logit_distance

PROTOCOL_ENG = "reax-eng/1"
# Image references are kept short enough that the whole commitment fits the chain's 128-byte raw field.
_COMMIT_RE = re.compile(
    r"^reax-eng/1:(?P<track>[a-z0-9][a-z0-9-]{0,15}):"
    r"(?P<image>[a-z0-9][a-z0-9._/-]{0,79}(?::[0-9]{1,5}/[a-z0-9._/-]{1,60})?)"
    r"@sha256:(?P<digest>[0-9a-f]{64})$"
)
MAX_COMMITMENT_BYTES = 128


@dataclass(frozen=True)
class TrackParams:
    """Release constants of one competition track (to calibrate per GPU class before activation)."""

    name: str
    pool: str  # the reax-s1 pool whose model and temperature the engine serves
    temperature: float
    timed_items: int = 256  # unique items per timed run
    concurrency: int = 16  # in-flight requests during a timed run
    runs: int = 2  # interleaved reference/candidate pairs; the score is the median ratio
    latency_items: int = 24  # items sent one at a time for the latency gate
    probe_pairs: int = 16  # anti-cache probe pairs (same state + other question, same question + other state)
    determinism_items: int = 32  # timed items re-run at concurrency 1
    mean_tau: float = 0.10  # stricter than serving (0.20): the tolerance must not become a speed budget
    item_tau: float = 1.0  # stricter than serving (2.0)
    min_top_agreement: float = 0.99
    determinism_tau: float = 0.05  # the reference is bit-identical at concurrency 1; batched engines move ~0.01
    max_latency_ratio: float = 1.25
    min_gain: float = 0.03  # must beat the reference by 3 % to count at all
    margin: float = 0.03  # must beat the champion's score by 3 % to take the title
    confirmations: int = 2  # separate evaluations (fresh items, later cycle) that must all clear the bar
    confirm_gap_blocks: int = 1800  # ... at least this many blocks apart (6 h at 12 s), also across restarts
    eval_budget_s: float = 2700.0  # one evaluation, start to finish; over budget = the candidate fails
    max_infra_attempts: int = 3  # inconclusive tries (pull/start trouble) before a submission is recorded as failed
    pause_idle: bool = True  # pause the idle engine during every measurement (False only for calibration)
    reference_drift: float = 0.12  # a reference run outside +-12 % of its own solo baseline is inconclusive (docs: ENGINE-COMPETITION.md)
    start_timeout_s: float = 900.0
    request_timeout_s: float = 60.0

    def __post_init__(self) -> None:
        ok = (
            self.temperature > 0 and self.timed_items >= 8 and 1 <= self.concurrency <= 256 and self.runs >= 1
            and 0 < self.mean_tau <= self.item_tau and 0 < self.min_top_agreement <= 1
            and self.determinism_tau > 0 and self.max_latency_ratio >= 1 and self.min_gain >= 0 and self.margin >= 0
            and self.determinism_items <= self.timed_items and self.confirmations >= 1
            and self.confirm_gap_blocks >= 0 and self.eval_budget_s > 0 and self.max_infra_attempts >= 1
            and 0 < self.reference_drift < 1
        )
        if not ok:
            raise ValueError(f"invalid track parameters for {self.name}")


@dataclass(frozen=True)
class RewardParams:
    """How the competition's share of the miner weights is split (docs/ENGINE-COMPETITION.md section 6)."""

    share: float = 0.20  # of the miner part of every weight vector, only while a champion exists (80/20 split)
    champion_start: float = 1.0  # champion's part of the share when crowned
    champion_floor: float = 0.5  # ... decaying linearly to this; the decayed part goes back to the serving pools
    decay_blocks: int = 100_800  # 14 days at 12 s
    # Runner-ups (halving by rank) would receive the decayed part instead. Off: a rebuilt copy of the champion or of
    # the reference with a lucky measurement could hold a runner-up slot (docs/ENGINE-COMPETITION.md section 6).
    runner_ups: int = 0

    def __post_init__(self) -> None:
        if not (0 <= self.share <= 0.5 and 0 <= self.champion_floor <= self.champion_start <= 1
                and self.decay_blocks >= 1 and self.runner_ups >= 0):
            raise ValueError("invalid engine competition reward parameters")


@dataclass(frozen=True)
class Submission:
    uid: int
    hotkey: str
    track: str
    image: str
    digest: str
    block: int = 0  # on-chain commitment block (0 = unknown)

    @property
    def ref(self) -> str:
        return f"{self.image}@sha256:{self.digest}"


def commitment_text(track: str, image: str, digest: str) -> str:
    text = f"{PROTOCOL_ENG}:{track}:{image}@sha256:{digest}"
    if parse_engine_commitment(text) is None:
        raise ValueError("not a valid reax-eng/1 commitment (image name too long or invalid characters?)")
    return text


def parse_engine_commitment(text: Any) -> Optional[tuple[str, str, str]]:
    """``reax-eng/1:<track>:<image>@sha256:<digest>`` -> (track, image, digest); anything else -> None."""
    if not isinstance(text, str) or len(text.encode("utf-8")) > MAX_COMMITMENT_BYTES:
        return None
    m = _COMMIT_RE.match(text.strip())
    if m is None or ".." in m["image"] or "//" in m["image"]:
        return None
    return m["track"], m["image"], m["digest"]


# ------------------------------------------------------------------------------------------------ measurements
@dataclass
class EvalResult:
    """One validator's measurement of one engine digest against the reference in the same session."""

    digest: str
    track: str
    ok: bool
    reason: str = ""
    score: float = 0.0  # median throughput ratio candidate / reference
    throughput: float = 0.0  # decisions/s, median over the runs
    ref_throughput: float = 0.0
    latency_p50_s: float = math.inf
    ref_latency_p50_s: float = math.inf
    mean_distance: float = math.inf
    max_distance: float = math.inf
    top_agreement: float = 0.0
    probe_max_distance: float = math.inf
    determinism_max_distance: float = math.inf
    items: int = 0
    block: int = 0

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        for k, v in out.items():
            if isinstance(v, float) and not math.isfinite(v):
                out[k] = None
        return out

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> "EvalResult":
        kw = {}
        for f in cls.__dataclass_fields__.values():
            if f.name not in raw:
                continue
            v = raw[f.name]
            if v is None and f.type == "float":
                v = math.inf
            kw[f.name] = v
        return cls(**kw)


Probs = Sequence[float]


def _argmax(p: Probs) -> int:
    return max(range(len(p)), key=lambda i: (p[i], -i))


@dataclass(frozen=True)
class Correctness:
    ok: bool
    reason: str
    mean_distance: float
    max_distance: float
    top_agreement: float


def check_correctness(answers: Mapping[str, Optional[Probs]], references: Mapping[str, Probs],
                      params: TrackParams, top_agreement: bool = True) -> Correctness:
    """Every reference item must be answered and match within the track tolerances.

    ``top_agreement=False`` skips the top-choice rate (small sets, where one honest near-tie would be > 1 %).
    """
    if not references:
        return Correctness(False, "no reference answers", math.inf, math.inf, 0.0)
    ds, agree = [], 0
    for ident in sorted(references):
        ref = references[ident]
        got = answers.get(ident)
        if got is None or len(got) != len(ref):
            return Correctness(False, f"missing or malformed answer for item {ident[:12]}", math.inf, math.inf, 0.0)
        ds.append(logit_distance(got, ref, params.temperature))
        agree += _argmax(got) == _argmax(ref)
    mean = math.fsum(ds) / len(ds)
    worst = max(ds)
    top = agree / len(ds)
    if worst > params.item_tau:
        return Correctness(False, f"item distance {worst:.3f} > {params.item_tau}", mean, worst, top)
    if mean > params.mean_tau:
        return Correctness(False, f"mean distance {mean:.3f} > {params.mean_tau}", mean, worst, top)
    if top_agreement and top < params.min_top_agreement:
        return Correctness(False, f"top choice agrees on {top:.1%} < {params.min_top_agreement:.0%}", mean, worst, top)
    return Correctness(True, "", mean, worst, top)


def max_distance(a: Mapping[str, Optional[Probs]], b: Mapping[str, Probs], temperature: float) -> float:
    worst = 0.0
    for ident, ref in b.items():
        got = a.get(ident)
        if got is None or len(got) != len(ref):
            return math.inf
        worst = max(worst, logit_distance(got, ref, temperature))
    return worst


def median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    if not s:
        return math.nan
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else 0.5 * (s[mid - 1] + s[mid])


def speed_score(cand_throughputs: Sequence[float], ref_throughputs: Sequence[float]) -> float:
    """Median of the per-pair ratios (pairs measured back to back, so slow drift of the machine cancels)."""
    ratios = [c / r for c, r in zip(cand_throughputs, ref_throughputs) if r > 0 and math.isfinite(c)]
    return median(ratios) if ratios else 0.0


# ------------------------------------------------------------------------------------------------ champion state
@dataclass
class Champion:
    digest: str
    hotkey: str
    uid: int
    score: float
    crowned_block: int


@dataclass
class TrackState:
    results: dict[str, EvalResult] = field(default_factory=dict)  # digest -> latest result
    first_seen: dict[str, tuple[int, str]] = field(default_factory=dict)  # digest -> (block, hotkey) first committed
    champion: Optional[Champion] = None
    # A challenger that cleared the bar once: {"digest", "hotkey", "measurements": [[block, score], ...]}.
    challenger: Optional[dict[str, Any]] = None
    attempts: dict[str, int] = field(default_factory=dict)  # digest -> inconclusive tries so far
    not_owner: set[str] = field(default_factory=set)  # "digest:hotkey" pairs whose image is bound to another hotkey
    history: list[dict[str, Any]] = field(default_factory=list)  # title changes, newest last (max 50)

    def to_json(self) -> dict[str, Any]:
        return {
            "results": {d: r.to_json() for d, r in self.results.items()},
            "first_seen": {d: [b, h] for d, (b, h) in self.first_seen.items()},
            "champion": asdict(self.champion) if self.champion else None,
            "challenger": self.challenger,
            "attempts": self.attempts,
            "not_owner": sorted(self.not_owner),
            "history": self.history[-50:],
        }

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> "TrackState":
        st = cls()
        st.results = {d: EvalResult.from_json(r) for d, r in raw.get("results", {}).items()}
        st.first_seen = {d: (int(v[0]), str(v[1])) for d, v in raw.get("first_seen", {}).items()}
        ch = raw.get("champion")
        st.champion = Champion(**ch) if ch else None
        st.challenger = raw.get("challenger")
        st.attempts = {d: int(n) for d, n in raw.get("attempts", {}).items()}
        st.not_owner = set(raw.get("not_owner", []))
        st.history = list(raw.get("history", []))[-50:]
        return st


def observe(state: TrackState, subs: Sequence[Submission], block: int,
            first_committer_only: bool = True) -> list[Submission]:
    """Record first sightings and return the *eligible* submissions.

    Order = the on-chain commitment block when known, else the block this validator first saw it.
    ``first_committer_only`` (runners without image-to-hotkey binding): a digest committed by a second hotkey is
    ignored for that hotkey. With binding (the image names its hotkey) every committer is eligible here and the
    runner rejects the ones the image is not bound to (``not_owner``), so nobody can squat a published digest.
    """
    for s in sorted(subs, key=lambda s: (s.block or block, s.uid)):
        seen = state.first_seen.get(s.digest)
        when = s.block or block
        if seen is None or when < seen[0]:
            state.first_seen[s.digest] = (when, s.hotkey)
    eligible = []
    for s in subs:
        if f"{s.digest}:{s.hotkey}" in state.not_owner:
            continue
        if first_committer_only and state.first_seen[s.digest][1] != s.hotkey:
            continue
        eligible.append(s)
    return eligible


def pending(state: TrackState, eligible: Sequence[Submission], limit: int) -> list[Submission]:
    """Not yet evaluated digests, oldest commitment first, at most ``limit``."""
    todo = [s for s in eligible if s.digest not in state.results]
    todo.sort(key=lambda s: (state.attempts.get(s.digest, 0), s.block or state.first_seen[s.digest][0], s.uid))
    return todo[:limit]


def _valid(r: Optional[EvalResult], params: TrackParams) -> bool:
    return r is not None and r.ok and r.score >= 1.0 + params.min_gain


def update_champion(state: TrackState, eligible: Sequence[Submission], params: TrackParams, block: int) -> Optional[Champion]:
    """Apply the champion rule over the currently committed, eligible submissions."""
    live = {s.digest: s for s in eligible}
    ch = state.champion
    if ch is not None:
        sub = live.get(ch.digest)
        r = state.results.get(ch.digest)
        if sub is None or sub.hotkey != ch.hotkey or not _valid(r, params):
            state.history.append({"block": block, "event": "vacated", "digest": ch.digest, "hotkey": ch.hotkey})
            ch = None
        else:
            ch.score = r.score  # re-measured in a later session
            ch.uid = sub.uid
    contenders = sorted(
        (s for s in eligible if _valid(state.results.get(s.digest), params)),
        key=lambda s: (-state.results[s.digest].score, state.first_seen[s.digest][0], s.uid),
    )
    best = contenders[0] if contenders else None
    if best is not None and (ch is None or (best.digest != ch.digest
                                            and state.results[best.digest].score >= ch.score * (1.0 + params.margin))):
        r = state.results[best.digest]
        c = state.challenger
        if c is None or c.get("digest") != best.digest or c.get("hotkey") != best.hotkey:
            c = {"digest": best.digest, "hotkey": best.hotkey, "measurements": []}
        last = max((m[0] for m in c["measurements"]), default=None)
        if last is None or r.block - last >= max(1, params.confirm_gap_blocks):
            c["measurements"].append([r.block, r.score])
        state.challenger = c
        if len(c["measurements"]) >= params.confirmations:
            # Winner's curse: the title is held at the lowest of the confirming measurements.
            score = min(m[1] for m in c["measurements"])
            state.history.append({"block": block, "event": "crowned", "digest": best.digest, "hotkey": best.hotkey,
                                  "score": round(score, 4)})
            ch = Champion(best.digest, best.hotkey, best.uid, score, block)
            state.challenger = None
    else:
        state.challenger = None
    state.champion = ch
    return ch


def needs_remeasurement(state: TrackState, eligible: Sequence[Submission]) -> list[Submission]:
    """The pending challenger (for its confirmation) and, when it is challenged, the champion."""
    out = []
    c = state.challenger
    if c is not None:
        out += [s for s in eligible if s.digest == c["digest"] and s.hotkey == c["hotkey"]]
    ch = state.champion
    if ch is not None and out:
        out += [s for s in eligible if s.digest == ch.digest and s.hotkey == ch.hotkey]
    return out


def competition_weights(state: TrackState, eligible: Sequence[Submission], params: TrackParams,
                        reward: RewardParams, block: int) -> dict[int, float]:
    """Split of the competition share inside the track (sums to 1, or empty when there is no champion)."""
    ch = state.champion
    if ch is None:
        return {}
    age = max(0, block - ch.crowned_block)
    frac = reward.champion_start - (reward.champion_start - reward.champion_floor) * min(1.0, age / reward.decay_blocks)
    others = sorted(
        (s for s in eligible if s.digest != ch.digest and _valid(state.results.get(s.digest), params)),
        key=lambda s: (-state.results[s.digest].score, state.first_seen[s.digest][0], s.uid),
    )[: reward.runner_ups]
    weights = {ch.uid: frac}
    if not others:
        return weights  # the decayed part (1 - frac) stays with the serving pools
    ranks = [0.5 ** i for i in range(len(others))]
    total = math.fsum(ranks)
    for s, r in zip(others, ranks):
        weights[s.uid] = weights.get(s.uid, 0.0) + (1.0 - frac) * r / total
    return weights


def merge_weights(base: Mapping[int, float], comp: Mapping[int, float], share: float,
                  owner_uid: Optional[int] = None, burn_fraction: float = 0.0) -> dict[int, float]:
    """Give the competition ``share`` x (its total) of the *miner* part of ``base``.

    The owner's burn fraction is kept as it is; everything else in ``base`` (serving miners, and the owner's
    fallback for pools without an eligible miner) is scaled down by the same factor. With no champion, or an empty
    base (nothing is set at all), ``base`` is returned unchanged.
    """
    total = math.fsum(comp.values())
    if not comp or share <= 0 or total <= 0 or not base:
        return dict(base)
    burn = burn_fraction if owner_uid is not None and owner_uid in base else 0.0
    miner_part = 1.0 - burn
    k = 1.0 - share * total
    out = {}
    for uid, w in base.items():
        out[uid] = burn + (w - burn) * k if uid == owner_uid else w * k
    for uid, w in comp.items():
        out[uid] = out.get(uid, 0.0) + share * miner_part * w
    norm = math.fsum(out.values())
    return {uid: w / norm for uid, w in sorted(out.items())}
