"""``reax-spec-eval``: run the validator's specialist evaluation locally on the PUBLIC dev split.

Same rule checks, same prompt, same container, same scorer as the validator; only the data differs (public dev rows
instead of a fresh sealed draw), so a score here is a preview, not a ranking. Needs Docker, the NVIDIA runtime and
a GPU with room for the model; ``--dry-run`` only runs the rule checks and the tensor hash.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from typing import Any

from . import fetch as F
from . import prompt as P
from . import rules as R
from . import scoring as SC
from .evaluator import score_model
from .vllm import GPU_BUDGET_GB, GPU_TOTAL_GB, VLLM_IMAGE, VllmDockerBackend


def load_dev(path: str, category: str | None) -> list[dict[str, Any]]:
    """``path`` is a dev .jsonl file, a directory holding dev.jsonl, or the dataset root (needs --category)."""
    candidates = [path]
    if os.path.isdir(path):
        candidates = [os.path.join(path, "dev.jsonl"), os.path.join(path, "dev.clean.jsonl")]
        if category:
            candidates += [os.path.join(path, category, "dev", "dev.clean.jsonl"), os.path.join(path, category, "dev", "dev.jsonl")]
    for c in candidates:
        if os.path.isfile(c):
            with open(c, encoding="utf-8") as fh:
                return [json.loads(line) for line in fh if line.strip()]
    raise SystemExit(f"no dev rows found at {path}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="reax-spec-eval", description=__doc__.split("\n\n")[0])
    ap.add_argument("model_dir", help="local model directory (safetensors, config.json, tokenizer, LICENSE, reax-specialist.json)")
    ap.add_argument("--dev", required=True, help="public dev split: a .jsonl file, a directory with dev.jsonl, or the dataset root")
    ap.add_argument("--category", required=True, help="track category, e.g. demand_forecasting")
    ap.add_argument("--hotkey", default=None, help="your hotkey (ss58); checked against reax-specialist.json")
    ap.add_argument("--limit", type=int, default=0, help="score a random subset of this many dev rows (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-device", default="0")
    ap.add_argument("--gpu-total-gb", type=float, default=GPU_TOTAL_GB)
    ap.add_argument("--gpu-budget-gb", type=float, default=GPU_BUDGET_GB)
    ap.add_argument("--image", default=VLLM_IMAGE)
    ap.add_argument("--no-manifest-check", action="store_true", help="skip the reax-specialist.json checks (before you wrote it)")
    ap.add_argument("--dry-run", action="store_true", help="rule checks and tensor hash only; no container")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    return ap


def format_report(score: SC.Score, tiers: dict[str, SC.Score]) -> str:
    lines = [f"score {score.total:.2f} = 0.75 x competence {score.competence:.2f} + 0.25 x calibration {score.calibration:.2f}"
             f"  (n={score.n}, failed items={score.failures})"]
    for kind, d in sorted(score.primitives.items()):
        lines.append(f"  {kind:7s} n={int(d['n']):4d} accuracy {d['accuracy']:.3f} competence {d['competence']:.1f} calibration {d['calibration']:.1f}")
    for tier, t in tiers.items():
        if tier:
            lines.append(f"  tier {tier:12s} n={t.n:4d} score {t.total:.2f}")
    return "\n".join(lines)


async def run(args: argparse.Namespace) -> int:
    ins = F.inspect_dir(args.model_dir, category=args.category, hotkey=args.hotkey, rules=R.TrackRules(),
                        require_manifest=not args.no_manifest_check)
    report: dict[str, Any] = {"violations": ins.violations}
    if ins.violations:
        print("REFUSED by the submission rules:", file=sys.stderr)
        for v in ins.violations:
            print(f"  - {v}", file=sys.stderr)
        if args.json:
            print(json.dumps(report))
        return 2
    assert ins.hashes is not None
    report.update(hash=ins.hashes.named, params=ins.hashes.params, weight_bytes=ins.hashes.weight_bytes)
    if not args.json:
        print(f"rules ok; {ins.hashes.params / 1e9:.2f} B parameters; tensor hash {ins.hashes.named[:16]}")
    if args.dry_run:
        if args.json:
            print(json.dumps(report))
        return 0
    rows = load_dev(args.dev, args.category)
    if args.limit and args.limit < len(rows):
        rows = random.Random(args.seed).sample(rows, args.limit)
    rows.sort(key=lambda r: r["id"])
    async with VllmDockerBackend(args.model_dir, ins.hashes.weight_bytes, gpu_device=args.gpu_device, gpu_total_gb=args.gpu_total_gb,
                                 budget_gb=args.gpu_budget_gb, image=args.image) as backend:
        items, score = await score_model(backend, rows)
    tiers = SC.by_tier(items)
    report.update(prompt=P.PROMPT_ID, **score.as_dict())
    print(json.dumps(report) if args.json else format_report(score, tiers))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(run(args))
    except Exception as exc:  # noqa: BLE001 - a CLI: print the reason, not a traceback
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
