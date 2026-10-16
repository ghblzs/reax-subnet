"""Frozen thresholds of the engine competition (unreleased; docs/ENGINE-COMPETITION.md).

Every number that decides whether an engine is correct enough, fast enough or crowned lives in this one block. It is
not tuned to any submission: a change is a new release (new CHANGELOG entry, new digest below, champion re-measured).
The validator logs ``thresholds_sha256`` at start and writes it into its state file, so a published result can always
be tied to the exact rules that produced it, and a state written under other rules is noticed.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

# One block. Names are TrackParams fields (engcomp.py); its defaults are read from here.
THRESHOLDS: dict[str, Any] = {
    # correctness against the reference engine's own answers (logit distance, s1scoring.logit_distance)
    "mean_tau": 0.10,  # stricter than serving (0.20): the tolerance must not become a speed budget
    "item_tau": 1.0,  # stricter than serving (2.0)
    "min_top_agreement": 0.99,  # share of items whose top choice equals the reference's
    "determinism_tau": 0.05,  # re-run at concurrency 1 must reproduce the timed answers
    "probe_pairs": 16,  # anti-cache probe pairs
    # speed
    "min_gain": 0.03,  # must beat the reference by 3 % to count at all
    "margin": 0.03,  # must beat the champion (and a pending challenger) by 3 % to displace it
    "max_latency_ratio": 1.25,  # concurrency-1 latency vs the reference
    # evidence per evaluation
    "timed_items": 256,  # unique items per timed run
    "runs": 2,  # interleaved reference/candidate pairs
    "latency_items": 24,
    "determinism_items": 32,
    # confirmation: separate evaluations (new containers, fresh secret items); they may follow in the same cycle
    "confirmations": 2,
    "confirm_gap_blocks": 0,
}


def canonical(thresholds: Mapping[str, Any]) -> bytes:
    return json.dumps(dict(thresholds), sort_keys=True, separators=(",", ":")).encode("utf-8")


def thresholds_sha256(thresholds: Mapping[str, Any] = THRESHOLDS) -> str:
    """sha256 of the canonical JSON of the block (or of any mapping of the same fields, e.g. a track in use)."""
    return hashlib.sha256(canonical(thresholds)).hexdigest()


# Pinned in tests/test_engcomp.py (and checked against the tracks in use at start): editing THRESHOLDS without
# updating this digest and the CHANGELOG fails the suite.
FROZEN_SHA256 = "1d4b170f11c8d028a4b0a988996aa6f3332d4be3cfcf3de8d44299d09340c07a"
