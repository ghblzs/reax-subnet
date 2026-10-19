"""Margin and confirmation rules (docs/SPECIALIST-TRACKS.md), as pure functions.

* A challenger must beat the incumbent by ``delta = max(1.5 points, 2 x paired SE)`` on a fresh shared draw.
* It is then confirmed on a second, disjoint fresh draw taken at least 12 hours later: the gain over the incumbent
  there must again be positive and at least ``delta / 2``. Only then does it become the incumbent.
* Challengers are handled in commitment order. One that passed the confirmation earlier in the same round becomes the
  incumbent for the challengers after it, and they must beat it by the full margin on the same (second) draw.
* The same holds for any incumbent the first pass was not measured against: one crowned in an earlier round, or any
  incumbent at all when the track was vacant at the first pass. Half of a margin taken against another model (or the
  entry floor itself, on a vacant track) says nothing about this pair.
* With no incumbent the first model must reach the entry floor on both draws.

Decision rule v3 (variance-aware, ``V3_DECISIONS``; docs/SPECIALIST-TRACKS.md "Decision rule v3"): the same ideas, but
the evidence is counted in standard errors and a close call gets one more draw instead of a verdict on one noisy draw.
Per challenger, only draws measured against the SAME incumbent are pooled: ``z = mean(gain) / (rms(se) / sqrt(k))``.

* First stage: ``z >= Z_PASS`` and ``gain >= MARGIN_FLOOR`` (exactly the margin rule above) -> pending confirmation.
  ``Z_CLOSE <= z < Z_PASS`` -> ONE more first-stage draw, disjoint from the first; the pooled z of the two must reach
  ``Z_PASS`` (and the pooled mean ``MARGIN_FLOOR``). ``z < Z_CLOSE`` -> not better.
* Confirmation: crown when the pooled z over all draws against that incumbent is ``>= Z_CROWN``, the confirmation gain
  is positive and the pooled mean ``>= MARGIN_FLOOR``. Between ``Z_EXTRA`` and ``Z_CROWN`` ONE extra confirmation draw;
  after it the pooled z must reach ``Z_CROWN`` and the two confirmation gains must add up above zero.
* Against an incumbent the first stage was not measured against (a new champion, or one crowned earlier in the same
  round) only the draws against that incumbent count: the same thresholds. A tie inside the noise therefore goes to
  the earlier commitment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

MARGIN_FLOOR = 1.5
SE_MULTIPLE = 2.0
CONFIRM_DELAY_S = 12 * 3600
CONFIRM_FRACTION = 0.5

# Decision rule v3. Z_PASS = 2 is the old "gain >= 2 x SE", Z_CROWN = 2.75 is the pooled bar for the crowning (about the
# strength of the old two-draw rule), Z_CLOSE / Z_EXTRA decide which borderline results get one more draw.
Z_PASS = 2.0
Z_CLOSE = 1.0
Z_EXTRA = 1.5
Z_CROWN = 2.75
V3_DECISIONS = False    # OFF until the dated #92 notice (critic H1); default of ``SpecConfig.v3_decisions`` (env REAX_SPEC_V3_DECISIONS=0 restores the old rule, no state migration)
V3_RULE = "v3"          # ``Submission.decision_rule`` of an entry whose first stage was decided under v3
MAX_CLOSE_DRAWS = 1     # extra first-stage draws per submission
MAX_EXTRA_CONFIRMS = 1  # extra confirmation draws per submission
Z_CONFIRM_MIN = 0.5     # the confirmation draw itself must show at least this many SE (it must confirm, not just not contradict)
Z_RECORD_MAX = 99.0     # recorded z values are clamped (no Infinity in the state file)
CLOSE_MIN_GAIN = 0.5 * MARGIN_FLOOR  # smallest first-stage gain that earns a close-call draw


def pooled(gains: Sequence[float], ses: Sequence[float]) -> tuple[float, float]:
    """(mean gain, its standard error) over draws against the same incumbent: ``rms(se) / sqrt(k)``. One draw is
    returned unchanged (no rounding), so a single-draw z is exactly gain / se."""
    k = len(gains)
    if k == 0:
        return 0.0, math.inf
    if k == 1:
        return float(gains[0]), float(ses[0])
    mean = sum(gains) / k
    rms = math.sqrt(sum(s * s for s in ses) / k)
    return mean, rms / math.sqrt(k)


def z_of(mean: float, se: float) -> float:
    """mean / se with the degenerate cases settled: no usable SE gives 0, a zero SE gives +-inf by the sign of the gain."""
    if math.isnan(mean) or math.isnan(se) or math.isinf(se):
        return 0.0
    if se <= 0:
        return Z_RECORD_MAX if mean > 0 else (-Z_RECORD_MAX if mean < 0 else 0.0)
    return max(-Z_RECORD_MAX, min(Z_RECORD_MAX, mean / se))   # finite: the value is written to the state JSON


def reaches(mean: float, se: float, z: float) -> bool:
    """``mean >= z x se``, written as a product so one draw equals the old ``gain >= 2 x SE`` bit for bit."""
    if math.isnan(mean) or math.isnan(se):
        return False
    return mean >= z * se


def required_margin(se: float) -> float:
    return max(MARGIN_FLOOR, SE_MULTIPLE * se)


@dataclass(frozen=True)
class FirstPass:
    passed: bool
    gain: float
    delta: float


def first_pass(challenger: float, incumbent: float | None, se: float, *, entry_floor: float = 0.0) -> FirstPass:
    """Draw 1. ``incumbent`` None = vacant track: the entry floor is the bar."""
    if incumbent is None:
        return FirstPass(challenger >= entry_floor, challenger - entry_floor, entry_floor)
    delta = required_margin(se)
    gain = challenger - incumbent
    return FirstPass(gain >= delta, gain, delta)


def confirmation_due(first_pass_ts: float, now: float, delay_s: float = CONFIRM_DELAY_S) -> bool:
    return now >= first_pass_ts + delay_s


def confirm(challenger: float, incumbent: float | None, first_delta: float, *, entry_floor: float = 0.0,
            promoted_this_round: bool = False, new_incumbent: bool = False, se: float = 0.0) -> bool:
    """Draw 2. Against the incumbent of the first pass: gain >= first_delta / 2 (and > 0). Against a challenger
    promoted earlier in the same round, or any other incumbent the first pass was not measured against
    (``new_incumbent``): gain >= the full margin for this pair. Vacant: the entry floor."""
    if incumbent is None:
        return challenger >= entry_floor
    gain = challenger - incumbent
    if promoted_this_round or new_incumbent:
        return gain >= required_margin(se)
    return gain > 0 and gain >= CONFIRM_FRACTION * first_delta


def first_stage_v3(gain: float, se: float, pooled_gain: float, pooled_se: float, *, close_used: bool) -> str:
    """Verdict of a first-stage draw against an incumbent: "pass", "close" (one more draw) or "fail".

    ``gain``/``se``: this draw. ``pooled_gain``/``pooled_se``: this draw pooled with the earlier first-stage draws
    against the same incumbent (equal to the draw's own when there are none). ``close_used``: the one extra draw was
    already taken (whatever incumbent it was measured against)."""
    if reaches(pooled_gain, pooled_se, Z_PASS) and pooled_gain >= MARGIN_FLOOR:
        return "pass"
    if close_used:
        return "fail"
    # a close call must at least be half the margin floor: a tiny gain with a tiny SE can never pass, its draw is waste
    if (not reaches(pooled_gain, pooled_se, Z_PASS) and reaches(pooled_gain, pooled_se, Z_CLOSE)
            and pooled_gain >= CLOSE_MIN_GAIN):
        return "close"
    return "fail"


def confirm_v3(draws: Sequence[tuple[float, float]], confirm: Sequence[tuple[float, float]], *, extra_used: bool) -> str:
    """Verdict of a confirmation draw: "crown", "extra" (one more confirmation draw) or "fail".

    ``draws``: (gain, se) of every draw of the current stage against THIS incumbent, the one just taken last (first stage
    included when it was measured against the same incumbent). ``confirm``: (gain, se) of the confirmation draws among
    them, the new one last. Pairs, so a gain can never be paired with another draw's SE.
    ``extra_used``: this is already the extra confirmation draw."""
    mean, se = pooled([g for g, _ in draws], [e for _, e in draws])
    strong = reaches(mean, se, Z_CROWN) and mean >= MARGIN_FLOOR
    cmean, cse = pooled([g for g, _ in confirm], [e for _, e in confirm])
    confirmed = cmean > 0 and reaches(cmean, cse, Z_CONFIRM_MIN)
    if not extra_used:
        if strong and confirmed:
            return "crown"
        if reaches(mean, se, Z_EXTRA) and (not reaches(mean, se, Z_CROWN) or not confirmed) and cmean > 0:
            return "extra"
        return "fail"
    return "crown" if strong and confirmed else "fail"
