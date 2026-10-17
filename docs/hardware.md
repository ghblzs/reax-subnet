# Hardware by pool (release 1.2)

One hotkey serves one pool. The validator checks you against the reference model's answers, so what counts is not
the card but that your runtime reproduces the reference (see [CALIBRATION.md](CALIBRATION.md)).

| Pool | Share of the miner part | Model | GPU | Measured | Runtime |
|---|---|---|---|---|---|
| s1-fast | 0.35 | Plumb-4B | one NVIDIA GPU with **≥ 24 GB** | RTX PRO 6000 Blackwell, H100 80 GB, RTX 4090, RTX 5090, RTX 3090, L40S, RTX A6000 | `runtimes/s1-fast` |
| s1-pro | 0.65 | Surogate Rune 26B-A4B v3 | one NVIDIA GPU of the **80–96 GB class** (52 GB bf16 weights + KV cache) | RTX PRO 6000 96 GB and H100 80 GB (bit-identical results). H200 and B200: expected to work, **not yet measured**. A100 80 GB: **not measured, at your own risk**. GB10 / DGX Spark (128 GB unified, arm64): bit-identical **with `REAX_VLLM_INVARIANT_ROUTER=1`** (see below) | `runtimes/s1-pro`, deterministic build **required** |

Cards below the class (for example 48 GB for s1-pro) cannot hold the weights plus KV cache; a quantised build
is not the reference and is struck.

## Validator GPU budget (one 96 GB card)

Measured on an RTX PRO 6000 Blackwell Server Edition (97,887 MiB):

| Part | VRAM |
|---|---|
| s1-fast reference runtime | 12.3 GB |
| s1-pro reference runtime (`RUNE_GPU_MEM_UTIL=0.60`, `RUNE_MAX_MODEL_LEN=8192`, bit-identical to the calibration runs) | 58.7 GB |
| one competition measurement at a time (engine 2 x 12.5 GB, or specialist up to 25 GB) | up to 25 GB |
| **Total** | **about 96 GB** |

## Confidential-compute GPUs

On confidential-compute GPUs (e.g. some cloud RTX PRO 6000 / H100 CC instances) set `REAX_VLLM_DISABLE_UVA=1` for the
s1-pro runtime. Without it vLLM returns garbage there, because a zero-copy (UVA) GPU view of pinned host memory does not
see later host writes. The flag loads `runtimes/vllm-ccfix/sitecustomize.py`, which switches vLLM to explicit-copy
buffers; results are unchanged. For a miner, add three flags to the s1-pro runtime container:

```bash
docker run ... \
  -v "$(pwd)/runtimes/vllm-ccfix:/opt/reax-ccfix:ro" -e PYTHONPATH=/opt/reax-ccfix -e REAX_VLLM_DISABLE_UVA=1 \
  reax-s1-pro-runtime:1.2.0rc3
```

## GB10 (DGX Spark): s1-pro router fix

On a GB10 (DGX Spark: sm_121, arm64, 128 GB unified memory) the s1-pro runtime matches the reference only while vLLM
batches several requests. `reax-selfcheck --model s1-pro` at its default concurrency 1 fails there: the MoE router GEMM
is batch-variant. On family-120 Blackwell, vLLM 0.30.0's `GateLinear` runs it as a plain cuBLAS call
(`torch.mm(x, W.T, out_dtype=torch.float32)`) that `VLLM_BATCH_INVARIANT` does not cover, and cuBLAS on GB10 chooses
that GEMM's algorithm by the number of rows. A prompt that runs alone therefore gets slightly different router logits
than inside a batch, and near-ties select other experts. Set `REAX_VLLM_INVARIANT_ROUTER=1`: the flag loads
`runtimes/vllm-routerfix/sitecustomize.py`, which copies the rows into zero-padded contiguous buffers of exactly 4,096
rows, so every GEMM call has one shape and one memory layout. On GB10 each row's output then no longer depends on the
rest of the batch. That is measured on GB10 only, not a cuBLAS guarantee: on any other GPU, run `reax-selfcheck` with
and without the flag before mining. The fix is written for the pinned vLLM 0.30.0 and the deterministic build: with the
flag set and another vLLM, or if the patch cannot be applied, every Python process in the container exits at start-up
with the reason, so the runtime never serves unpatched by accident. For a miner, add three flags to the s1-pro runtime
container:

```bash
docker run ... \
  -v "$(pwd)/runtimes/vllm-routerfix:/opt/reax-routerfix:ro" -e PYTHONPATH=/opt/reax-routerfix -e REAX_VLLM_INVARIANT_ROUTER=1 \
  reax-s1-pro-runtime:1.2.0rc3
```

Measured on 17 Oct 2026 on one GB10 (driver 580.178.04, CUDA 13.0), 640 reference decisions per run, all runs of a row in
one container. The image was built from `runtimes/s1-pro/Dockerfile` unchanged on the linux/arm64 variant of the pinned
vLLM 0.30.0 digest, and Docker got the GPU through CDI (`--device nvidia.com/gpu=all`):

| `reax-selfcheck --model s1-pro` | concurrency 1 | concurrency 16 | concurrency 64 |
|---|---|---|---|
| without the flag | **FAIL** (two runs, identical): median 0.125, max 4.29, 5 items above the strike line, 3.2 % of rounds struck; 4.7 decisions/s | **FAIL** in one of two runs (max 2.63, 1 item above the strike line; 17.0/s), PASS in the other (median 0.0, max 0.96; 21.1/s) | PASS (two runs): median 0.0, max 0.96; 29.0–29.1/s |
| with `REAX_VLLM_INVARIANT_ROUTER=1` | **PASS** (two runs): median, p99 and max 0.0, 100 % of rounds score; 4.6 decisions/s | PASS (two runs): median, p99 and max 0.0; 16.6–20.6/s | PASS (two runs): median, p99 and max 0.0; 28.2–30.3/s |

Each fix is its own `sitecustomize.py`; mount only the one your GPU needs.

## s1-pro needs the deterministic runtime

The pool is scored against the deterministic reference build (`RUNE_DETERMINISTIC=1`, the default in
`runtimes/s1-pro`). Other builds are not accepted:

* **Stock vLLM** (compiled, batch-variant kernels) gives different numbers under different load: logit distance
  to the reference 0.146 median with items up to 3.9; it scores only about 65 % of its rounds (CALIBRATION.md).
* **FP8 / online-quantised** builds score about 12 % of rounds and are struck in about 22 % of them
  (CALIBRATION.md), which keeps the moving average below the 0.3 needed for any weight.

Run `reax-selfcheck --model s1-pro` against your running sidecar before you commit on chain; it must print PASS.

## s1-pro miner quickstart

1. **Get the weights with your own Hugging Face account.** The weights are gated: accept the gate for
   `surogate/rune-26b-a4b-GGUF` on huggingface.co, run `hf auth login`, then
   `hf download surogate/rune-26b-a4b-GGUF --revision c6b360d47895bb77bdf3805a13ee5a5557ab1921 --local-dir /srv/reax/rune-v3`
   (about 52 GB). Never mirror the weights for anyone else.
2. **Build and start the sidecar and the miner:** `docker build -t reax:1.2.0rc3 .` and
   `docker build -f runtimes/s1-pro/Dockerfile -t reax-s1-pro-runtime:1.2.0rc3 .`. Run the sidecar on loopback
   `127.0.0.1:8102` with the weights read-only at `/model` and the shared token file, exactly like the s1-fast
   example in `deploy/docker-compose.s1.yml`; the miner uses `REAX_MODEL=s1-pro` and
   `REAX_S1_PRO_RUNTIME_URL=http://127.0.0.1:8102`. Start-up verifies all 18 weight files (a few minutes).
3. **Self-check:** `REAX_S1_PRO_RUNTIME_URL=http://127.0.0.1:8102 REAX_RUNTIME_TOKEN_FILE=<token file> reax-selfcheck --model s1-pro`.
4. **Commitment:** the miner publishes `reax-s1/1:s1-pro:<manifest_id>` on start (`REAX_PUBLISH_COMMITMENT=1`);
   the manifest id of this release is `5b11440bf57b5434`, so the commitment is `reax-s1/1:s1-pro:5b11440bf57b5434`.
   A different manifest id means a different release and the validator does not query you.

More than one GPU or machine for the same hotkey: run one sidecar per GPU and list them in
`REAX_S1_PRO_RUNTIME_URLS` (MINER.md, "Several GPUs or machines behind one hotkey").

Details: [MINER.md](MINER.md). Validators: [VALIDATOR.md](VALIDATOR.md) and `deploy/validator-1.2/README.md`.
