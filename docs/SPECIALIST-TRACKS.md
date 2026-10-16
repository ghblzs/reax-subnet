# Specialist tracks (`reax-spec/1`)

A specialist track is a competition for one use case, for example demand forecasting. You fine-tune a small
open-weight model for that use case, commit it on chain, and validators score it on items nobody has seen. The best
model of a track is its champion. How much a champion earns is the track's share of the competition part of the
rewards; see COMPETITIONS.md. Nothing on this page is a promise of any payout.

Specialist models are described as **Jev-compatible specialist models**: they answer the same typed decision
contract (choice, noul, score) as the reference models. They are not Jev models.

## Schedule

* The first two tracks run together: **demand forecasting** and **risk assessment**, open for commitments and scored from
  **16 Oct 2026, 12:00 UTC** to 30 Oct 2026, 12:00 UTC (the schedule is a release constant in `competition_split.py`).
* A track runs for **14 days**, then the next track starts (the rotation). Tracks and dates are announced before they
  open; a commitment for a track that is not active is ignored.

## What you submit

A Hugging Face model repository, pinned to one revision, containing **weights only**:

| Requirement | Rule |
|---|---|
| Weights | `*.safetensors` only. No pickle, `.bin`, `.pt`, GGUF or ONNX file may be in the repo. |
| Code | None is ever run. `trust_remote_code` is never enabled; a `config.json` with `auto_map` is refused. |
| Size | At most **9 billion parameters** (counted from the safetensors headers) and at most 20 GiB of weights. |
| Architecture | One of the architectures the pinned vLLM image serves **and** that has a fixed chat template in this release: Qwen3 (dense and MoE), Qwen3.5, Gemma 3, Llama. (Gemma 4 is not accepted in this release: it has no fixed template.) |
| Licence | Apache-2.0, MIT, BSD-2-Clause or BSD-3-Clause (other licences only if the subnet owner has approved them in writing), plus a `LICENSE` file, so that the model can be hosted and served. |
| Tokenizer | `tokenizer_config.json` and `tokenizer.json` (or `tokenizer.model`). Your **chat template is never used**: the validator renders the decision prompt with a fixed template per architecture family (below). A `chat_template.jinja`, a `chat_template` in `tokenizer_config.json` and any `*.jinja` are ignored (not even downloaded); `tokenizer.model` / `*.tiktoken` are ignored when `tokenizer.json` exists. |
| Manifest | `reax-specialist.json` in the repo root (below). |

```json
{"schema": "reax-specialist/1", "category": "demand_forecasting", "hotkey": "<your hotkey ss58>",
 "base_model": "<hf id>@<revision of the base you fine-tuned>", "licence": "apache-2.0"}
```

The manifest ties the model to your hotkey and to the track. `base_model` must be the model you fine-tuned: it is
used to compare your weight changes with other submissions.

Commit on chain (the Commitments pallet), **while the repo is still private if you like**:

```
reax-spec/1:<track_id>:<hf_repo>@<40-hex revision>
```

for example `reax-spec/1:demand_forecasting:acme/df-4b@0123456789abcdef0123456789abcdef01234567`. The revision must
be the full 40-character commit hash, in lower case, never a branch name. The whole text must fit 128 bytes.

* **Make the repo public within 7,200 blocks (about 24 hours) of the commitment** and keep it public. A repo that is
  still not downloadable after that is dropped.
* Limits per coldkey: at most 2 open submissions per track and 4 new submissions per 7 days.
* A new commitment from the same hotkey replaces the old one; the old model no longer counts.

### Chat templates

A chat template is code (Jinja) that the validator would otherwise parse inside the vLLM container that holds the GPU, so
a submission's template is never rendered. Each allowed architecture family maps to one template shipped in the package
(`src/reax_subnet/spec/templates/`: `qwen.jinja` for Qwen3, Qwen3-MoE and Qwen3.5, `gemma3.jinja`, `llama3.jinja`),
passed to vLLM with `--chat-template` and `--chat-template-content-format string`. The decision prompt is one system and
one user message, so what a model sees is the family's standard turn layout around the fixed prompt (Qwen: ChatML, with the
empty think block). `reax-spec-eval` uses the same templates, so your local number matches. Fine-tune with that layout.

## How a model is scored

1. **Checks, no GPU.** The validator reads the file list, `config.json`, the manifest and the safetensors headers as
   data and checks every rule above. A refused submission is never loaded.
2. **Serving.** The validator downloads the pinned revision (weights, json, tokenizer, licence only) and serves it in
   its own pinned vLLM container: no network, read-only file system, no capabilities, the model mounted read-only, and
   a GPU memory cap of at most 25 GB. No code from your repo runs.
3. **Prompt and readout are fixed** (`reax-decide-prompt/1`, `src/reax_subnet/spec/prompt.py`): the state as JSON, the
   question, and the options labelled A, B, C ...; one forward pass; your answer is the softmax (temperature 2) over
   the logits of the option letters at the first generated position. The same renderer and scorer ship with
   `reax-spec-eval`, so you can train against exactly this format.
4. **Fresh sealed items.** Every round draws a new random sample (600 items by default, about a third per primitive,
   at least half programmatic-gold items, with items from new domains and new task families) from the sealed pool.
   Every model of a round answers the same items. Items that were used are retired and published later as training data.
5. **Score** `S = 0.75 x competence + 0.25 x calibration`, 0 to 100.
   * *Competence* is the JevBench O1S chance-corrected score per primitive (choice, noul, score), averaged. A noul
     answer between 0.20 and 0.80 counts as no decision; a score answer gets partial credit by distance.
   * *Calibration* is `100 x max(0, 1 - Brier / Brier_uniform)`: honest probabilities are paid, a model that puts all its
     mass on one option is punished when it is wrong.
   * An invalid or missing answer counts as a wrong answer with the uniform probability.

## Becoming champion

* A challenger must beat the incumbent by `max(1.5 points, 2 x standard error)` on the same fresh draw. The standard
  error is the paired bootstrap error of the score difference over the shared items.
* It is then **confirmed on a second, disjoint fresh draw taken at least 12 hours later**, where it must beat the
  incumbent again by at least half of that margin. Only then does it become the incumbent.
* Challengers are handled in order of their commitment block (then UID). A challenger that is confirmed first becomes
  the incumbent for the ones after it, and they must beat it by the full margin.
* The incumbent is measured again every time a group is evaluated. The validator runs at most a fixed number of model
  evaluations per hour (6 by default, including the incumbent), so a long queue waits; the queue is ordered by
  commitment block.
* The first model of an empty track has to reach the entry floor on both draws: the s1-fast model's score on the sealed
  pool plus 1.5 points (demand_forecasting 24.8, risk_assessment 33.7; see COMPETITIONS.md).
* Ranks 2 and 3 of a track are the most recent former champions that are still committed.

### Ranks

The payout ranks of a track (75 / 17 / 8 % of the track's share) follow one fixed order, not the scores:

1. the current champion;
2. the former champion crowned most recently before it;
3. the one crowned before that.

Only submissions whose hotkey still holds the commitment count; a missing one leaves its rank to the next in this
order. The scores shown next to the ranks are each model's latest score, but the former champions are not all measured
on every draw, so scores from different draws are never compared to reorder the ranks: a former champion with a higher
score on an older draw does not outrank the champion that beat it. The entry floor still applies: a leader whose
latest score is at or below the floor earns nothing, and the ones after it move up one rank. A rank nobody holds goes
to the inference part.

## Copies

* **Tensor hash.** The hash is over every tensor (name, dtype, shape, bytes), not over files, so re-sharding the same
  weights gives the same hash. A second hash ignores tensor names. The earliest commitment owns a hash; later
  submissions with the same hash are refused before they use any GPU time.
* **Near-duplicates.** If the base model is available on the validator, the sampled weight change of your model (your
  weights minus the declared base) is compared with those of earlier submissions. Cosine above 0.98 and relative
  distance below 0.05 mark yours as *derived*: it must then also beat its parent by the full margin in the same
  round. A model identical to its declared base is refused.
* **Commit before publish.** Earlier commitments are handled first, and a copier can only copy after the original's
  commitment. The owner reviews a new champion before it is paid.

## Try it locally

```
huggingface-cli download decisionmodels/sn92-specialists --repo-type dataset --local-dir sn92-specialists
pip install -e '.[spec]'
reax-spec-eval ./my-model --dev ./sn92-specialists --category demand_forecasting --hotkey <your hotkey>
```

The public train and dev data (CC-BY-4.0, 20 categories, `<category>/train.jsonl` and `<category>/dev.jsonl`) is at
[huggingface.co/datasets/decisionmodels/sn92-specialists](https://huggingface.co/datasets/decisionmodels/sn92-specialists);
the sealed split stays private and is what validators score on.

`--dev` takes the public dev split: a `.jsonl` file, a directory holding `dev.jsonl`, or the dataset root. The command
runs the same rule checks, the same prompt, the same container and the same scorer as the validator, on the dev rows
instead of a fresh sealed draw, so the number is a preview, not a ranking. It needs Docker with the NVIDIA runtime and
a GPU with about 25 GB free. `--dry-run` only checks the rules and prints the tensor hash; `--limit N` scores a random
subset; `--no-manifest-check` skips the manifest before you have written it.

## Operator notes (validators)

| Variable | Default | Meaning |
|---|---|---|
| `REAX_SPEC_SEALED_DIR` | required | host path of the sealed pool (`<category>/sealed/sealed.clean.jsonl`), mounted read-only; never inside a container the miner influences |
| `REAX_SPEC_STATE_DIR` | `~/.reax` | holds `spec-state.json` (champions, pending confirmations, queue, hash registry, used item ids) and `spec-work/` |
| `REAX_SPEC_TRACKS_FILE` | the release schedule (development networks only; ignored on finney) | JSON list of `{"track_id", "category", "start": "YYYY-MM-DDTHH:MM:SSZ", "days": 14}` |
| `REAX_SPEC_BASE_DIR` | none | base models for the near-duplicate check, as `<hf id with / replaced by __>@<revision>/` directories. Without it only exact hashes are compared |
| `REAX_SPEC_GPU_DEVICE` / `REAX_SPEC_GPU_TOTAL_GB` / `REAX_SPEC_GPU_BUDGET_GB` | `0` / card size from nvidia-smi / `25` | GPU for the candidate container and its memory budget |
| `REAX_VLLM_DISABLE_UVA` | unset | `1` on confidential-compute GPUs: the candidate container gets a read-only mount of `reax_subnet/spec/ccfix/sitecustomize.py` (via `PYTHONPATH`) that turns vLLM's UVA buffers off, otherwise inputs go stale |
| `REAX_SPEC_UID` | `10003:10003` | uid:gid of the vLLM container: never root, and not the uid of the validator/worker or of the engine containers (10002). The model directory is made world-readable after the download; if the socket (mode 0666 from uvicorn) cannot be connected to, set this to the worker's own `uid:gid` |
| `REAX_SPEC_DOCKER_RUNTIME` | none | `docker run --runtime`, e.g. `runsc` (gVisor), like `REAX_ENGINE_DOCKER_RUNTIME` |
| `REAX_SPEC_CPUS` | `8` | `--cpus` of the vLLM container (memory and swap are 48g, nofile 4096, logs 2 x 10 MB) |
| `REAX_SPEC_MIN_FREE_GB` | `60` | free space required on the filesystem of `REAX_SPEC_WORK_DIR` before a submission is downloaded; below it the cycle is skipped and nothing is recorded against any submission |
| `REAX_SPEC_SAMPLE_SIZE` | `600` | items per draw |
| `REAX_SPEC_MAX_EVALS_PER_HOUR` | `6` | model evaluations per hour, all tracks together |
| `REAX_SPEC_VLLM_IMAGE` | the pinned image of `runtimes/s1-pro` | vLLM image by digest |

**Where it runs.** In `REAX_COMPETITIONS_MODE=local` the validator process runs the evaluator; in `remote` mode it is
`reax-competition-worker --spec` (on the validator host without wallets or tokens, or on a separate keyless box), which
writes `spec-state.json`, and the validator only reads it (docs/COMPETITIONS.md, "Where the competitions run"). The worker
needs `REAX_SPEC_SEALED_DIR`, a writable `REAX_SPEC_WORK_DIR` on a host path that is mounted at the same path into the
worker container (the vLLM sockets are created under `<work dir>/sock/`, the models under `<work dir>/models/`) and the
docker socket. The pinned vLLM image must ship `jinja2 >= 3.1.6`; `scripts/check-vllm-image.sh` prints the versions.

The evaluator class is `reax_subnet.spec.evaluator.SpecEvaluator`; `await evaluate_cycle(snapshot)` returns
`{track_id: [(uid, score), ...]}` in rank order (see "Ranks") for the active tracks. Turning the ranking into weights is done by a
separate layer (`competitions.SpecialistAdapter` and `competition_split.combine_weights`, see COMPETITIONS.md).
