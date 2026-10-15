#!/usr/bin/env bash
# Prints the versions the specialist-track security notes depend on, from the pinned vLLM image (no network, nothing mounted).
# Needs docker and the image pulled. Exit 1 if jinja2 is older than 3.1.6 (sandbox-escape fixes: CVE-2024-56201/56326, CVE-2025-27516).
set -euo pipefail
IMAGE="${1:-$(python3 -c 'from reax_subnet.spec.vllm import VLLM_IMAGE; print(VLLM_IMAGE)')}"
out=$(docker run --rm --network none --entrypoint python3 "$IMAGE" -c \
  'import jinja2, vllm; print(jinja2.__version__, vllm.__version__)')
echo "image $IMAGE: jinja2 / vllm = $out"
jv=${out%% *}
python3 - "$jv" <<'P'
import sys
v = tuple(int(x) for x in sys.argv[1].split(".")[:3])
sys.exit(0 if v >= (3, 1, 6) else 1)
P
