#!/usr/bin/env bash
DATA=${DATA:-/tmp/reax-localnet}
[ -f "$DATA/pids" ] && kill $(cat "$DATA/pids") 2>/dev/null || true
rm -f "$DATA/pids"
