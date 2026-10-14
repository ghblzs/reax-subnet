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
    determinism_items: int = 16  # timed items re-run at concurrency 1
    mean_tau: float = 0.10  # stricter than serving (0.20): the tolerance must not become a speed budget
    item_tau: float = 1.0  # stricter than serving (2.0)
    min_top_agreement: float = 0.99
    determinism_tau: float = 0.10
    max_latency_ratio: float = 1.25
    min_gain: float = 0.03  # must beat the reference by 3 % to count at all
    margin: float = 0.03  # must beat the champion's score by 3 % to take the title
    start_timeout_s: float = 900.0
    request_timeout_s: float = 60.0

    def __post_init__(self) -> None:
        ok = (
            self.temperature > 0 and self.timed_items >= 8 and 1 <= self.concurrency <= 256 and self.runs >= 1
            and 0 < self.mean_tau <= self.item_tau and 0 < self.min_top_agreement <= 1
            and self.determinism_tau > 0 and self.max_latency_ratio >= 1 and self.min_gain >= 0 and self.margin >= 0
            and self.determinism_items <= self.timed_items
        )
        if not ok:
            raise ValueError(f"invalid track parameters for {self.name}")


@dataclass(frozen=True)
class RewardParams:
    """How the competition's share of the miner weights is split (docs/ENGINE-COMPETITION.md section 6)."""

    share: float = 0.10  # of the miner part of every weight vector, only while a champion exists
    champion_start: float = 1.0  # champion's part of the share when crowned
    champion_floor: float = 0.5  # ... decaying linearly to this
    decay_blocks: int = 100_800  # 14 days at 12 s
    runner_ups: int = 3  # the decayed part goes to the next best valid submissions (halving by rank)

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
    history: list[dict[str, Any]] = field(default_factory=list)  # title changes, newest last (max 50)

    def to_json(self) -> dict[str, Any]:
        return {
            "results": {d: r.to_json() for d, r in self.results.items()},
            "first_seen": {d: [b, h] for d, (b, h) in self.first_seen.items()},
            "champion": asdict(self.champion) if self.champion else None,
            "history": self.history[-50:],
        }

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> "TrackState":
        st = cls()
        st.results = {d: EvalResult.from_json(r) for d, r in raw.get("results", {}).items()}
        st.first_seen = {d: (int(v[0]), str(v[1])) for d, v in raw.get("first_seen", {}).items()}
        ch = raw.get("champion")
        st.champion = Champion(**ch) if ch else None
        st.history = list(raw.get("history", []))[-50:]
        return st


def observe(state: TrackState, subs: Sequence[Submission], block: int) -> list[Submission]:
    """Record first sightings and return the submissions that are *eligible*: the first committer of each digest.

    A digest committed by a second hotkey (a straight copy) is ignored for that hotkey.
    """
    for s in sorted(subs, key=lambda s: s.uid):
        state.first_seen.setdefault(s.digest, (block, s.hotkey))
    eligible = []
    for s in subs:
        first_block, first_hotkey = state.first_seen[s.digest]
        if first_hotkey == s.hotkey:
            eligible.append(s)
    return eligible


def pending(state: TrackState, eligible: Sequence[Submission], limit: int) -> list[Submission]:
    """Not yet evaluated digests, oldest commitment first, at most ``limit``."""
    todo = [s for s in eligible if s.digest not in state.results]
    todo.sort(key=lambda s: (state.first_seen[s.digest][0], s.uid))
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
    if contenders:
        best = contenders[0]
        best_score = state.results[best.digest].score
        if ch is None or (best.digest != ch.digest and best_score >= ch.score * (1.0 + params.margin)):
            state.history.append({"block": block, "event": "crowned", "digest": best.digest, "hotkey": best.hotkey,
                                  "score": round(best_score, 4)})
            ch = Champion(best.digest, best.hotkey, best.uid, best_score, block)
    state.champion = ch
    return ch


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
    if not others:
        return {ch.uid: 1.0}
    weights = {ch.uid: frac}
    ranks = [0.5 ** i for i in range(len(others))]
    total = math.fsum(ranks)
    for s, r in zip(others, ranks):
        weights[s.uid] = weights.get(s.uid, 0.0) + (1.0 - frac) * r / total
    return weights


def merge_weights(base: Mapping[int, float], comp: Mapping[int, float], share: float) -> dict[int, float]:
    """``(1 - share) * base + share * comp``; with no champion the base vector is returned unchanged."""
    if not comp or share <= 0:
        return dict(base)
    if not base:
        return dict(comp)
    out = {uid: (1.0 - share) * w for uid, w in base.items()}
    for uid, w in comp.items():
        out[uid] = out.get(uid, 0.0) + share * w
    norm = math.fsum(out.values())
    return {uid: w / norm for uid, w in sorted(out.items())}
