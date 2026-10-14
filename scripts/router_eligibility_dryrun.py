#!/usr/bin/env python3
"""Read-only dry run of the router's track-record rule against a validator state file.

Usage: router_eligibility_dryrun.py STATE_JSON [--commitments FILE] [--min-ema X] [--min-rounds N] [--cooldown-s S]

Reads the state file only. Without --commitments every scored miner with the s1-fast role is treated as present on chain
(the state file already requires hotkey and role to match what the validator saw). Prints one line per scored UID.
"""
import argparse
import json
import sys
import time
from types import SimpleNamespace

from reax_subnet.router import Router, RouterConfig
from reax_subnet.s1validator import POOL_PARAMS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("--min-ema", type=float, default=RouterConfig.min_ema)
    ap.add_argument("--min-rounds", type=int, default=RouterConfig.min_rounds)
    ap.add_argument("--cooldown-s", type=float, default=RouterConfig.strike_cooldown_s)
    a = ap.parse_args()
    raw = json.load(open(a.state))
    cfg = SimpleNamespace(min_ema=a.min_ema, min_rounds=a.min_rounds, strike_cooldown_s=a.cooldown_s)
    probe = SimpleNamespace(cfg=cfg)
    now = time.time()
    kept, rows = [], []
    for uid, m in sorted(raw["miners"].items(), key=lambda kv: int(kv[0])):
        model, _, manifest = m.get("role", "").partition(":")
        if model not in POOL_PARAMS:
            continue
        reason = Router.exclusion(probe, m, m["hotkey"], (model, manifest), now)
        rows.append((int(uid), m["ema"], m["rounds"], m.get("strikes", 0), m["last_score"], reason or "ELIGIBLE"))
        if not reason:
            kept.append(int(uid))
    print(f"state rounds={raw.get('rounds')} saved_at={raw.get('saved_at')} rule: ema>={a.min_ema} rounds>={a.min_rounds}")
    print("uid    ema  rounds strikes last_score  verdict")
    for uid, ema, rounds, strikes, last, verdict in rows:
        print(f"{uid:>3} {ema:6.3f} {rounds:6d} {strikes:7d} {last:10.3f}  {verdict}")
    print(f"scored s1-fast miners: {len(rows)}; eligible: {len(kept)} {kept}")
    old = [u for u, e, *_ in rows if e >= 0.5]
    print(f"old rule (ema>=0.5 only) would have kept {len(old)}")


if __name__ == "__main__":
    sys.exit(main())
