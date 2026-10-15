# REAX subnet

REAX is a [Bittensor](https://bittensor.com) subnet (Finney, **netuid 92**) for typed decisions. **Miners serve pinned
REAX reference models** — `s1-fast` (Plumb-4B) or `s1-pro` (Surogate Rune 26B-A4B v3) — in a runtime sidecar on
loopback, and **validators check them** by recomputing fresh decision answers on their own runtime of the same pinned
model and comparing in logit space. Matching miners share the weight of their pool; a small part depends on speed.

Release 1.1 ships the **reax-s1 mechanism** (default, `REAX_MECHANISM=s1`, protocol `reax-s1/1`). The earlier
mechanism (`REAX_MECHANISM=mpm1`: Qwen3-4B, protocol `reax-mpm/1`, `POST /v1/decide`) is kept unchanged as a fallback.

## Quick start

* **Miner:** [docs/MINER.md](docs/MINER.md) — one NVIDIA GPU (24 GB or more for `s1-fast`), the runtime sidecar and
  the `reax-miner` process. The Compose examples build images locally; the `ghcr.io/reaxlabs/reax-miner` and
  `ghcr.io/reaxlabs/reax-s1-fast-runtime` images are also published for direct use when a release is available; the
  tiered-pool runtimes are `reax-s1-cpu-runtime` (amd64 + arm64), `reax-s1-lite-runtime` (CUDA) and `reax-s1-lite-runtime-jetson`.
* **Validator:** [docs/VALIDATOR.md](docs/VALIDATOR.md) — the reference runtime sidecar plus `reax-validator`.
  The Compose examples build locally; `ghcr.io/reaxlabs/reax-validator` is also published for direct use when a release is available.
* Settings for both: [docs/CONFIG.md](docs/CONFIG.md). Compose examples: `deploy/`.

Servers need only the hotkey; keep coldkeys offline. `REAX_NETWORK` and `REAX_NETUID` default to `finney` and `92`;
set them explicitly for testnet or local endpoints.

## Documentation

* Design of the s1 mechanism: [docs/DESIGN-v1.1.md](docs/DESIGN-v1.1.md)
* GPU calibration behind the scoring thresholds: [docs/CALIBRATION.md](docs/CALIBRATION.md)
* Threat model: [docs/THREAT-MODEL.md](docs/THREAT-MODEL.md)
* Emissions, owner share and burn: [docs/EMISSIONS.md](docs/EMISSIONS.md)
* Disabled features and switches: [docs/FEATURE-FLAGS.md](docs/FEATURE-FLAGS.md)
* Fallback mechanism and wire protocol: [docs/MPM.md](docs/MPM.md)
* Optional client-facing router that forwards signed requests to miners: [docs/ROUTER.md](docs/ROUTER.md)

## Development

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e ".[model,test]"
pytest                      # unit + in-process adversarial tests; real-model tests run if the test model is cached
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for pull-request rules.

### Local chain end-to-end

Linux x86_64, no Docker needed; CPU is enough with the small test models. The harness exercises the mpm1 mechanism;
the s1 paths are covered by the pytest suite with a hash test runtime (and by `scripts/localnet/e2e_s1.py` with real
runtimes on GPUs):

```bash
python scripts/localnet/fetch_node.py /tmp/subtensor-node   # pinned Rao Foundation localnet image, layers hash-checked
NODE_DIR=/tmp/subtensor-node scripts/localnet/start.sh && sleep 30
hf download Qwen/Qwen2.5-0.5B-Instruct --revision 7ae557604adf67be50417f59c2c2f167def9a775
hf download Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca   # adversary model
python scripts/localnet/e2e.py /tmp/reax-e2e --host 127.0.0.1 --offline
scripts/localnet/stop.sh
```

Wallet and registration commands in the docs use bittensor-cli 9.x (`pip install bittensor-cli==9.23.2`, in its
own virtual environment).

## Risks

* **No promises.** Rewards are variable subnet emissions determined by the Bittensor network and by validator
  scoring. Nothing here promises income, token value or returns, and rewards can be zero.
* **Costs.** Running a miner costs hardware and power. Registering a neuron on the subnet costs TAO, and a
  registration can be lost to deregistration.
* **Software is provided as-is** under the MIT licence, without warranty; read the code and the
  [threat model](docs/THREAT-MODEL.md) before you run it.
* **Keys.** Keep wallets and coldkeys off mining and validating machines; only the hotkey belongs on a server.
  Never commit wallets, mnemonics, keys, tokens or validator state.
* Scoring or manifest changes require a new `WEIGHTS_VERSION_KEY`, a CHANGELOG entry and a calibration note.

## Security

Report vulnerabilities privately; see [SECURITY.md](SECURITY.md).

## Licence

MIT, see [LICENSE](LICENSE). Third-party attributions and included licence texts:
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) (also copied into the runtime images at `/licenses/`).
