#!/usr/bin/env python3
"""reax-validator for the 1.2 release end-to-end run (e2e_release.py). LOCAL CHAINS ONLY.

Runs the real ``reax_subnet.cli.validator_main`` with exactly two test substitutions:

* the specialist evaluator is a fake whose per-track results come from the JSON file ``REAX_E2E_SPEC_RESULTS``
  (``{"track": [[uid, score], ...]}``; a missing file = no results yet). Everything after the evaluator is real:
  SpecialistAdapter, CompetitionDriver, EngineCompetition.weights -> combine_weights;
* the competition clock runs at 2026-10-16T13:00Z + the process run time, so that the two use-case tracks of
  competition_split.TRACK_SCHEDULE (16 Oct 12:00Z - 30 Oct) are active on a development network today. The engine
  competition itself is active at once on a development network (no start time).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone

if not os.environ.get("REAX_NETWORK", "").startswith(("ws://127.0.0.1", "ws://localhost")):
    sys.exit("e2e_release_validator.py only runs against a local chain")

from reax_subnet import cli, engvalidator  # noqa: E402
from reax_subnet.competitions import SpecialistAdapter  # noqa: E402

SHIFT = datetime(2026, 10, 16, 13, 0, tzinfo=timezone.utc) - datetime.now(timezone.utc)


def shifted_now() -> datetime:
    return datetime.now(timezone.utc) + SHIFT


class FakeSpecEvaluator:
    def __init__(self, path: str):
        self.path = path

    async def evaluate_cycle(self, snapshot):
        try:
            with open(self.path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return {}
        return {t: [(int(u), float(s)) for u, s in rows] for t, rows in data.items()}


async def _no_gpu(_gpu):  # no nvidia-smi on this host: skip the memory check
    return None


def fake_specialist(cfg, e):
    return SpecialistAdapter(FakeSpecEvaluator(os.environ["REAX_E2E_SPEC_RESULTS"]), interval_s=10.0, free_mib=_no_gpu)


_Engine = engvalidator.EngineCompetition


class ShiftedEngine(_Engine):
    def __init__(self, *a, **kw):
        kw["utcnow"] = shifted_now
        super().__init__(*a, **kw)


cli._specialist = fake_specialist
engvalidator.EngineCompetition = ShiftedEngine  # cli._engine_competition imports the class from the module at call time

if __name__ == "__main__":
    cli.validator_main()
