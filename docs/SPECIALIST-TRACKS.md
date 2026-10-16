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
4. **Fresh sealed items.** Every round draws a new random sample (300 items by default, about a third per primitive)
   from the sealed pool: 15 % programmatic-gold items from the published procedural families and 85 % LLM-written
   items, split 50 / 25 / 25 between same-family items, items from new domains and items from new task families
   (since 1.2.0rc6; rc5 drew 600 items, half of them programmatic). Every model of a round answers the same items. Items
   that were used are retired and published later as training data. Each tier keeps its share of the draw: a tier that
   runs out of unused items reuses its own least-used items first, and only a tier with too few items at all hands the
   rest to the other LLM-written tiers. The programmatic tier never takes another tier's share and never holds more
   than 15 % of the draw: without enough LLM-written items the draw is smaller (below 200 items there is no draw).
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
  the incumbent for the ones after it, and they must beat it by the full margin. This holds whether it was confirmed
  in the same round or an earlier one, and also when the track was vacant at a challenger's first draw: an incumbent
  the first draw was not measured against has to be beaten by the full margin on the confirmation draw.
* The incumbent is measured again every time a group is evaluated. The validator runs at most a fixed number of model
  evaluations per hour (6 by default, including the incumbent), so a long queue waits; the queue is ordered by
  commitment block.
* The first model of an empty track has to reach the entry floor on both draws: the s1-fast model's score on release
  draws plus 1.5 points (demand_forecasting 39.7, risk_assessment 33.2; see COMPETITIONS.md).
* Only the champion's model files stay on the validator between rounds. An entry pending confirmation is downloaded
  again for its confirmation draw (the same pinned revision) and must have exactly the weights of its first check;
  other weights under that revision reject it.
* Ranks 2 and 3 of a track are the most recent former champions that are still committed.

### Ranks

The payout ranks of a track (75 / 17 / 8 % of the track's share, or of the 95 % left after the keep-alive when the
track has keep-alive entries, see "Keep-alive while you wait") follow one fixed order, not the scores:

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
* **Near-duplicates.** The sampled weight change of your model (your weights minus the declared base) is compared with
  those of earlier submissions on the same base. "Same base" means the same base weights (their tensor hash), not the
  same `base_model` text: declaring a mirror or another name of the base changes nothing. Cosine above 0.98 and
  relative distance below 0.05 mark yours as *derived*: it must then also beat its parent by the full margin in the
  same round. A model identical to its declared base is refused. Independent fine-tunes of the same base are not
  affected: only a change that nearly equals an earlier one counts.
* **Any float dtype.** The weight change is taken after decoding both sides and rounding your values to the base
  tensor's dtype, so a fine-tune saved in F32 or F16 of a BF16 base is compared like the same fine-tune saved in BF16
  (in both directions: rounding an F32 model to BF16 does not hide a copy). The compared tensors (same name and shape
  as in the base; a `base_model.model.` name prefix from a PEFT export is ignored) must hold at least 90 % of the
  base's float parameters. FP8 and int-quantized tensors (GPTQ, AWQ) are not compared: save the weights in the base
  model's dtype (bf16 for Qwen3-4B) or in F16/F32.
* **Sample positions are secret.** The weight change is sampled at positions drawn from a secret seed of each
  validator (one value in 4093 per tensor, and at least 64 values of small tensors such as norms). Changing a few
  weights to look independent therefore needs luck, not knowledge of the rule.
* **Same answers.** On the shared draw, the validator also compares your answer probabilities item by item with
  those of every earlier submission served on the same draw (the champion included). If they are practically the same
  (mean Jensen-Shannon divergence below 0.005 bits, or below 0.05 bits with the options in the same order on at least
  99 % of the items that have three or more options; yes/no items do not count for the order),
  yours is *derived* from it, whatever the weights look like (permuted neurons, rescaled layers): it must beat that
  submission by the full margin, so a copy that answers like the champion cannot be crowned. Independent fine-tunes
  of the same base answer differently enough not to be affected.
* **Pending verification.** If the validator does not have the declared base (an unknown name or revision), or the
  tensors your model shares with the declared base (name and shape) hold less than 90 % of its float parameters, the
  near-duplicate check cannot be made. The submission is then *pending verification*: it is kept and keeps its place in
  the commitment order, but it is not evaluated, ranked, crowned or paid. A missing base is checked again every cycle
  and the submission goes ahead as soon as the base is available. Too little overlap with a base that is there does not
  change by waiting: it is checked again only after a release changes this rule; otherwise commit a fixed model.
  A champion whose check could not be made is held the same way and loses the title until it passes. The status and
  the reason are in the state file and the validator log. Exact copies (same tensor hash) are refused at once, with or
  without the base.
* **Commit before publish.** Earlier commitments are handled first, and a copier can only copy after the original's
  commitment. The owner reviews a new champion before it is paid.

## Payment needs the owner's approval

A winning entry is paid only after the subnet owner has checked and approved it. Each paid rank (1, 2 and 3) needs an
approval signed by the owner for exactly your hotkey and your model (`<repo>@<revision>`). Until then your share goes to
the inference part, your rank is kept, and nobody below you moves up. A new revision needs a new approval. Details:
COMPETITIONS.md, "Owner approval".

## Keep-alive while you wait

A competition hotkey has one commitment, so it cannot also serve inference: without help it would earn nothing while
its entry waits for evaluation, waits for its confirmation draw or waits for the owner's approval, and it would be
deregistered as the lowest-emission UID once its immunity period ends (which also drops the submission). From
1.2.0rc6 the validator therefore gives every **qualifying entry** a small keep-alive weight. It is a participation
keep-alive, not an award: it does **not** need the owner's approval and it does not change the ranks.

* **Who qualifies.** The hotkey's current commitment is a submission that passed the copy checks (exact hashes and the
  near-duplicate check against its declared base), whose latest score on a shared draw is **above the track's entry
  floor**, and which is waiting for a decision: an entry **pending confirmation** (for at most 24 hours after its first
  pass) or a **new champion** waiting for the owner's approval of its award (for at most 24 hours after it was crowned).
  The same draw's score on its **non-proc items** (same-family, new-domain and new-family; not the programmatic
  families) must also be above `KEEPALIVE_NONPROC_FLOOR` (release constant: 49.8 on demand forecasting, 37.8 on risk
  assessment, the s1-fast model's score on the non-proc items of release draws + 1.5 points), so a model that only
  learned the public procedural families does not qualify. An entry last scored before this rule waits for its next draw.
  Once the award is approved, the champion is paid by its rank only and no longer holds a keep-alive slot.
* **Who does not.** Queued or not yet evaluated entries, entries at or below the floor or the non-proc floor, decided entries (former
  champions, entries that did not beat the incumbent, rejected or withdrawn entries), exact copies, near copies that
  did not beat their parent and entries pending verification. Registering many hotkeys with weak or copied models
  earns nothing, and the keep-alive is never a standing payment.
* **How much.** `KEEPALIVE_SHARE` (release constant, 5 %) of the track's share (15 % in the 50/50 profile), split evenly
  among the track's qualifying entries: `0.15 x 0.05 / n` of the miner weights each. The ranks (75 / 17 / 8 %) are paid
  from the other 95 %. With no qualifying entry nothing is reserved and the weights are exactly the 1.2.0rc5 weights.
* **Limit.** At most 2 keep-alive slots per coldkey and track (the open-submission limit); the champion comes first,
  then entries pending confirmation, each in commitment order.
* The keep-alive keeps a waiting entry above the lowest-emission UIDs; it does not reserve a UID or guarantee that a
  hotkey is never deregistered.

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
| `REAX_SPEC_BASE_DIR` | none (`${REAX_COMPETITION_DIR}/spec-bases` in the compose files) | base models for the near-duplicate check, as `<hf id with / replaced by __>@<revision>/` directories. **Required for awards:** a submission whose declared base is not here waits in *pending verification* (not evaluated, not paid). The evaluator downloads a missing base itself (see "Base models" below); `grep "pending verification"` in the log lists the bases it could not get. A second directory name with the same weights serves an alias |
| `REAX_SPEC_BASE_FETCH` | `1` | `0` turns the automatic base download off (then the operator places every base) |
| `REAX_SPEC_BASE_PRESEED` | none | `<hf id>@<revision>,...` downloaded in the background when the evaluator starts (not counted against the cycle budget) |
| `REAX_SPEC_BASE_BUDGET_GB` | `40` | base downloads per cycle; a base that does not fit is deferred to the next cycle |
| `REAX_SPEC_BASE_MIN_FREE_GB` | `50` | free space that must remain on the filesystem of `REAX_SPEC_BASE_DIR` after a base download; otherwise the base is skipped this cycle (log line `must stay free`) |
| `REAX_SPEC_BASE_RETRY_CYCLES` | `6` | a base that could not be fetched (gated, private, not found, refused) is tried again after this many cycles |
| `REAX_SPEC_GPU_DEVICE` / `REAX_SPEC_GPU_TOTAL_GB` / `REAX_SPEC_GPU_BUDGET_GB` | `0` / card size from nvidia-smi / `25` | GPU for the candidate container and its memory budget |
| `REAX_VLLM_DISABLE_UVA` | unset | `1` on confidential-compute GPUs: the candidate container gets a read-only mount of `reax_subnet/spec/ccfix/sitecustomize.py` (via `PYTHONPATH`) that turns vLLM's UVA buffers off, otherwise inputs go stale |
| `REAX_SPEC_UID` | `10003:10003` | uid:gid of the vLLM container: never root, and not the uid of the validator/worker or of the engine containers (10002). The model directory is made world-readable after the download; if the socket (mode 0666 from uvicorn) cannot be connected to, set this to the worker's own `uid:gid` |
| `REAX_SPEC_DOCKER_RUNTIME` | none | `docker run --runtime`, e.g. `runsc` (gVisor), like `REAX_ENGINE_DOCKER_RUNTIME` |
| `REAX_SPEC_CPUS` | `8` | `--cpus` of the vLLM container (memory and swap are 48g, nofile 4096, logs 2 x 10 MB) |
| `REAX_SPEC_SKETCH_SEED` | none (a random seed created once in `REAX_SPEC_WORK_DIR/sketch-seed`, mode 0600) | secret seed of the near-duplicate sample positions. Keep it for the life of the validator and never publish it; when it changes, the stored sketches of earlier submissions are recomputed (their weights are downloaded again) |
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

**Base models.** When the base a submission declares (`base_model` in `reax-specialist.json`) is not under
`REAX_SPEC_BASE_DIR`, the evaluator downloads it there itself, the same way it downloads a submission: the pinned
40-character revision only, public repos only (no token), only `*.safetensors`, json, tokenizer and licence files, the
same size caps, an allowlisted architecture and no `auto_map`. The base is only read for the copy check and never served,
so its licence is not restricted and it needs no manifest. The download goes to a `.tmp-...` directory and is renamed
into place after the checks, so no cycle and no other process sees a half-written base; leftover `.tmp-...` directories
of a dead process are removed at the next start. A base that cannot be fetched (gated like Gemma or Llama, private, not
found, too large, refused by a rule) leaves the submission in *pending verification* with the reason, and the download is
tried again after `REAX_SPEC_BASE_RETRY_CYCLES` cycles. For gated bases, download them yourself (with your own Hugging
Face account) into `<base dir>/<hf id with / replaced by __>@<revision>/`; the evaluator picks them up in the next cycle.
Downloaded bases are kept.

The evaluator class is `reax_subnet.spec.evaluator.SpecEvaluator`; `await evaluate_cycle(snapshot)` returns
`{track_id: [(uid, score), ...]}` in rank order (see "Ranks") for the active tracks. Turning the ranking into weights is done by a
separate layer (`competitions.SpecialistAdapter` and `competition_split.combine_weights`, see COMPETITIONS.md).
