"""Margin and confirmation rules (docs/SPECIALIST-TRACKS.md), as pure functions.

* A challenger must beat the incumbent by ``delta = max(1.5 points, 2 x paired SE)`` on a fresh shared draw.
* It is then confirmed on a second, disjoint fresh draw taken at least 12 hours later: the gain over the incumbent
  there must again be positive and at least ``delta / 2``. Only then does it become the incumbent.
* Challengers are handled in commitment order. One that passed the confirmation earlier in the same round becomes the
  incumbent for the challengers after it, and they must beat it by the full margin on the same (second) draw.
* With no incumbent the first model must reach the entry floor on both draws.
"""

from __future__ import annotations

from dataclasses import dataclass

MARGIN_FLOOR = 1.5
SE_MULTIPLE = 2.0
CONFIRM_DELAY_S = 12 * 3600
CONFIRM_FRACTION = 0.5


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
            promoted_this_round: bool = False, se: float = 0.0) -> bool:
    """Draw 2. Against the old incumbent: gain >= first_delta / 2 (and > 0). Against a challenger promoted earlier in
    the same round: gain >= the full margin for this pair. Vacant: the entry floor."""
    if incumbent is None:
        return challenger >= entry_floor
    gain = challenger - incumbent
    if promoted_this_round:
        return gain >= required_margin(se)
    return gain > 0 and gain >= CONFIRM_FRACTION * first_delta
