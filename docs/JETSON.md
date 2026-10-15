# Jetson and ARM for SN92: what works, what is hard, effort

Status: 15 Oct 2026. We have no Jetson hardware. The cloud providers we have keys for (Lium, RunPod, Shadeform) had no
ARM + NVIDIA machine available today: Shadeform lists GH200 at Lambda (USD 2.29/h) and Latitude (USD 4.23/h), both
sold out. So the ARM parts below were tested in **real arm64 containers under qemu emulation** on an x86 build host (correct
results, not representative speed), and the Jetson GPU recipe is written but **not run**.

## Short answer

| | Works today? | Evidence | Effort to production |
|---|---|---|---|
| Miner process on aarch64 (Python, wallet signing) | yes | every dependency has an aarch64 wheel (bittensor-wallet 4.1.1 `manylinux_2_28_aarch64`, aiohttp, cryptography, numpy; the rest pure Python) | small: build and publish an arm64 miner image |
| **s1-cpu pool on ARM** (Gutsy 0.8B, llama.cpp NEON) | **yes** | arm64 sidecar image built from the same Dockerfile; answers vs x86: mean 0.035, max 0.087 on 30 items (tolerance: mean 0.20) | **1–2 days**: CI build, pin, docs |
| s1-lite on an x86 NVIDIA card (incl. 8–16 GB consumer cards) | yes | CUDA image built and calibrated on an RTX 5070 Ti 16 GB (s1-lite about 27–30 ms per decision) and an RTX A6000; cards agree within mean 0.08 | small |
| **s1-lite on Jetson Orin** (JetPack 6, sm_87 iGPU) | recipe only | `runtimes/s1-gguf/Dockerfile.jetson` (L4T r36.4 base, llama.cpp CUDA arch 87) | **3–5 days with a device** (or a beta tester), about 1–2 weeks elapsed |
| s1-lite on Jetson Thor (JetPack 7, SBSA, sm_110) | not started | `Dockerfile.cuda` with `CUDA_ARCHS=110` on an arm64 CUDA 13 base should work (Thor is SBSA) | 2–3 days with a device; low priority (USD 3.5k devkit, few miners) |

## Is a Jetson too small?
No. A decision is one forward pass of the prompt (typically 150–400 tokens), and nothing is generated.

| Device | Unified memory | s1-lite (Plumb-4B Q8_0, 4.5 GB) | s1-cpu (Gutsy 0.8B, 0.8 GB) |
|---|---|---|---|
| Orin Nano 4 GB | 4 GB | no (does not fit next to the OS) | yes (6 ARM cores) |
| Orin Nano 8 GB / Super | 8 GB | yes, tight: 4.5 GB model + ~1.5 GB OS + miner | yes |
| Orin NX 8 / 16 GB | 8 / 16 GB | yes | yes |
| AGX Orin 32 / 64 GB | 32 / 64 GB | yes; several sidecars possible = several UIDs, each measured honestly | yes |
| AGX Thor | 128 GB | yes (needs the SBSA image) | yes |

- Expected s1-lite speed on an Orin Nano 8 GB Super: prefill of a 4B Q8 model at roughly 300–600 tokens/s. That is
  about 0.5–1.5 s per decision, or 0.7–2 decisions/s.
- This is an **estimate** from the RTX/A6000 measurements scaled by memory bandwidth (102 GB/s vs 448–768 GB/s) and
  from public Orin llama.cpp numbers. It is **not measured**.
- On these numbers the node passes the latency curve (1 s → 12 s for 4 decisions) and reaches about 20–50 % of the
  s1-lite capacity target (4/s). An AGX Orin should reach the target.
- So a Jetson earns a real, smaller share in the small-GPU pool. That is the intended result: bigger hardware earns
  more, and a Jetson is still on board.

## What is hard (honest list)
1. **Container base for Orin.** Orin's GPU is an integrated Tegra GPU. Generic `nvidia/cuda` arm64 images (SBSA) do
   not see it on JetPack 6. The image must start from NVIDIA's L4T base (`nvcr.io/nvidia/l4t-jetpack:r36.4.0`) and run
   with `--runtime nvidia`.
   - The community `dustynv/llama_cpp` images are stale (May 2025), so we compile llama.cpp ourselves in our
     Dockerfile.
   - JetPack 7.2.1 now also supports Orin and is SBSA-aligned. Once miners move to it, one arm64 CUDA image could
     serve Orin and Thor, which needs checking on hardware.
2. **No cloud Jetson.** Nobody rents Orin by the hour. Real testing needs a device: a beta tester
   or a USD ~250 Orin Nano Super devkit.
   - GH200 (ARM + H100, SBSA) can test the arm64 CUDA build when Lambda or Latitude has stock, but not the Tegra
     path.
3. **Building on the device.** llama-cpp-python has no prebuilt aarch64 CUDA wheels. The Jetson image compiles it,
   about 20–40 min on an Orin. We should publish the built image so miners pull it instead of building.
4. **Numerical agreement.** The validator recomputes answers on its own backend. The pool tolerance must cover CUDA on
   Orin vs CUDA/CPU on the validator. Measured so far: x86 CPUs on three machines are identical; ARM vs x86 0.035; CUDA vs CPU 0.03 (s1-cpu) and 0.07 (s1-lite); 5070 Ti vs A6000 0.08. All of these are well inside the 0.20 round tolerance. Orin
   must be measured on hardware before release. If it is outside tolerance, the fallback is a per-pool tolerance
   bump, decided with data.
5. **Memory pressure on 8 GB.** The OS, Docker, the miner and a 4.5 GB model leave little headroom. Q4_K_M (2.7 GB)
   would be comfortable but is a different pool, because its answers differ more than the tolerance (mean 0.60 from Q8). A
   first beta should tell us whether 8 GB devices are stable with Q8.
6. **Power modes.** Orin "Super"/MAXN modes change speed by about 1.7×. The capacity probes measure whatever the
   device delivers, so this is fair by construction, but docs should tell people to set MAXN.

## What was built and tested here
- `runtimes/s1-gguf/Dockerfile`: CPU sidecar, one file for linux/amd64 and linux/arm64 (portable flags: x86-64
  AVX2/FMA, ARMv8.2-a+dotprod = Graviton2+, Orin's Cortex-A78AE, Raspberry Pi 5; not Pi 4). arm64 built with BuildKit
  with BuildKit under qemu on an x86 host: about 25 min, mostly compiling llama.cpp under emulation. The image runs, and its answers match x86 (see above). A slim arm64 miner image (`Dockerfile` target `miner-s1`, hash-locked wheels for all platforms) also builds and imports `bittensor` 11.1.0 on aarch64.
- `runtimes/s1-gguf/Dockerfile.cuda`: x86 CUDA 12.8 sidecar for Turing to Blackwell (sm 75–120), built and run on
  RTX A6000 and RTX 5070 Ti pods.
- `runtimes/s1-gguf/Dockerfile.jetson`: L4T r36.4 base, CUDA arch 87, `--runtime nvidia`. Not run.
- Miner kit (`reaxctl`): detects a Jetson (`/etc/nv_tegra_release`, L4T R36+) and treats unified memory as VRAM.
  It recommends s1-lite on Orin ≥ 8 GB and s1-cpu on any other aarch64 Linux host, and renders the compose file with
  `runtime: nvidia` and the Jetson image. Unit-tested with fakes; not run on a device.

## Next steps (after GO for a testnet beta)
1. Publish the arm64 CPU image and the x86 CUDA image (CI with native arm64 runners), pinned by digest.
2. Send a beta tester the Jetson recipe:
   - build `Dockerfile.jetson` on the device;
   - run `scripts/calibration_gguf/run.py` (200 items, about 5 min) and send back the JSONL;
   - we compare it with the validator backend and set the s1-lite tolerance and capacity target from real Orin data.
3. When GH200 stock appears (Lambda via Shadeform, about USD 2.29/h, under 1 h needed): build `Dockerfile.cuda` for
   arm64 SBSA and calibrate. This covers Thor-style SBSA and ARM server GPUs.
