# Competition worker on a separate keyless GPU box

`reax-competition-worker` runs the competitions (engine measurements and specialist-track evaluations) and nothing else:
it reads the chain **read-only**, holds **no wallet, no hotkey and no token**, and sets no weights. It writes
`engine-competition.json` and `spec-state.json` atomically into its state directory. A validator started with
`REAX_COMPETITIONS_MODE=remote` reads those two files every round and pays exactly like a validator that runs the
competitions itself (see [docs/COMPETITIONS.md](../../docs/COMPETITIONS.md), "Remote mode").

Why: the competitions run code and models that miners chose, in containers on a GPU, under a root docker daemon. An escape
from such a container is root on the machine it runs on. On this box that is worth nothing to an attacker; on the
validator host it would be the hotkey and the paying customers' traffic. (Security review of 15 Oct 2026, options D and E.)

## Needs

* A Linux box with docker, the NVIDIA container toolkit (patched: toolkit >= 1.17.8 with `disable-cuda-compat-lib-hook = true`,
  runc >= 1.2.8 / 1.3.3, current driver) and **one GPU with at least 30 GB free** (engine: 2 x 12.5 GB; specialist: up to 25 GB,
  one at a time under one lock). The measurements are timed: use a card of the validator's class or recalibrate.
* About 500 GB of disk for `REAX_COMPETITION_DIR`, on a filesystem that does not hold `/`; the worker skips a specialist
  cycle with less than 60 GB free (nothing is recorded against any submission). Put `/var/lib/docker` on a separate filesystem too.
* The Plumb-4B weights (`REAX_S1_FAST_WEIGHTS`) and the sealed pools (`REAX_SPEC_SEALED_DIR`, read-only, from the subnet owner).
* Outbound access to the chain RPC, the image registries and Hugging Face. No inbound port.

## Setup

```bash
git clone <this repository> && cd reax-subnet   # the release you run on the validator
export REAX_UID=$(id -u) REAX_GID=$(id -g) REAX_DOCKER_GID=$(getent group docker | cut -d: -f3)
export REAX_S1_FAST_WEIGHTS=/srv/reax/plumb-4b REAX_SPEC_SEALED_DIR=/srv/reax/sealed
export REAX_COMPETITION_DIR=/srv/reax/competitions REAX_HF_CACHE=/srv/reax/hf
hf download crh225/plumb-4b --revision 55de037801a8a9b9de3db5c0e16cef86210c2186 --local-dir "$REAX_S1_FAST_WEIGHTS"
mkdir -p "$REAX_COMPETITION_DIR/state" "$REAX_HF_CACHE"
docker compose -f deploy/competition-worker/compose.yml up -d --build
docker compose -f deploy/competition-worker/compose.yml logs -f competition-worker
```

Both competitions run by default (`command: ["--engine", "--spec"]`). The reference engine image is a **release constant**
(`reax_subnet.config.ENGINE_REFERENCE_IMAGE`); do not set it. `REAX_ENGINE_DOCKER_RUNTIME=runsc` / `REAX_SPEC_DOCKER_RUNTIME=runsc`
run the candidates under gVisor (check the driver and the card are supported first, then recalibrate). With
`REAX_ENGINE_NVML_CHECK=1` the worker samples `nvidia-smi pmon` for the PIDs of a paused candidate during every reference
window and fails a candidate that keeps using the GPU.

## Getting the state to the validator

The validator reads `engine-competition.json` and `spec-state.json` from `REAX_COMPETITIONS_STATE_DIR` (mounted read-only).
A file older than **26 hours** (by the `saved_at` inside it) counts as missing: that competition's share goes to the inference
part, with one warning in the validator log. The worker refreshes `saved_at` every 5 minutes while it is alive.

Copy the directory to the validator host every few minutes. `rsync` writes a temporary file and renames it, so the validator
never sees a half-written file:

```bash
# on the worker box, e.g. from cron or a systemd timer, every 5 minutes; the key can only write that one directory
rsync -a --chmod=F644 "$REAX_COMPETITION_DIR/state/" validator-host:/srv/reax/competitions/state/
```

On the validator host that directory is the `${REAX_COMPETITION_DIR}/state` of `deploy/validator-1.2/compose.yml` with the
`competition-worker` service removed. Use a dedicated ssh key for the copy, restricted to `rsync` into that directory
(`command="rrsync /srv/reax/competitions/state"` in `authorized_keys`).

A compromised worker box can write false results into these files, so it counts as part of the subnet owner's trusted
infrastructure. What it cannot do is reach a hotkey: the validator still checks, on every read, that the champion's hotkey
holds its uid and still commits to the measured image or model.

## Check

```bash
ls -l "$REAX_COMPETITION_DIR/state"                       # engine-competition.json, spec-state.json, fresh mtime
python3 -c "import json,time;print(time.time()-json.load(open('$REAX_COMPETITION_DIR/state/spec-state.json'))['saved_at'])"   # < 600 s
docker compose -f deploy/competition-worker/compose.yml run --rm --no-deps competition-worker --engine --once   # one pass
```
