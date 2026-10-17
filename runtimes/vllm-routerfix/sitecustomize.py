# REAX vLLM fix for GPUs where the MoE router GEMM is batch-variant (measured: GB10 / DGX Spark, sm_121, arm64).
# Mount this directory read-only at /opt/reax-routerfix, set PYTHONPATH=/opt/reax-routerfix and REAX_VLLM_INVARIANT_ROUTER=1.
# Measured on GB10: with the patch the s1-pro (Rune, deterministic) self-check is bit-identical to the reference at
# concurrency 1, 16 and 64; without it, concurrency 1 fails (docs/hardware.md).
# vLLM 0.30.0's GateLinear sends family-120 Blackwell (and every GPU without its specialised kernels) to
# torch.mm(x, W.T, out_dtype=float32), a cuBLAS call that VLLM_BATCH_INVARIANT does not cover. On GB10 cuBLAS picks that
# GEMM's algorithm by row count, so a prompt alone gets slightly different router logits than inside a batch and a
# near-tie selects other experts. The patch copies the rows into zero-padded contiguous buffers of exactly _ROWS rows,
# so every GEMM call has one shape and one memory layout; on GB10 each row's output then no longer depends on the rest
# of the batch (measured, not guaranteed by cuBLAS for every GPU: run reax-selfcheck on yours).
# Written for the deterministic build (RUNE_DETERMINISTIC=1, the default: eager, no CUDA graphs). Like vllm-ccfix it runs
# in every Python process of the container (vLLM, the decision adapter and its workers, warm-up) and imports torch and
# vLLM's router module there. If the flag is set and the patch cannot be applied, every one of those processes exits at
# start-up with the reason, so the runtime never serves unpatched by accident.
import os, sys
if os.environ.get("REAX_VLLM_INVARIANT_ROUTER") == "1":
    try:
        import inspect
        import torch
        import vllm
        from vllm.model_executor.layers.fused_moe.router import gate_linear as _gl

        _VLLM = "0.30.0"  # the s1-pro runtime's pinned vLLM; the tier logic below mirrors its GateLinear.forward
        _ROWS = 4096  # one fixed GEMM shape; the measured runs used 4096 (a different size can change the numbers)

        if vllm.__version__ != _VLLM:
            raise RuntimeError(f"vLLM {vllm.__version__} is not {_VLLM}; re-check GateLinear.forward before using the fix")
        _orig_forward = _gl.GateLinear.forward
        _src = inspect.getsource(_orig_forward)
        if "allow_cublas_router_gemm" not in _src or "out_dtype=torch.float32" not in _src:
            raise RuntimeError("GateLinear.forward has no cuBLAS bf16->fp32 tier")

        def _mm(rows, weight):
            return torch.mm(rows, weight.T, out_dtype=torch.float32)

        def _fixed_shape_mm(x, weight):
            out = torch.empty(x.shape[0], weight.shape[0], dtype=torch.float32, device=x.device)
            for start in range(0, x.shape[0], _ROWS):
                n = min(_ROWS, x.shape[0] - start)
                rows = x.new_zeros(_ROWS, x.shape[1])  # contiguous, same layout for every chunk
                rows[:n] = x[start:start + n]
                out[start:start + n] = _mm(rows, weight)[:n]
            return out

        def _forward(self, x):
            # Replace only the tier that GateLinear.forward would send to cuBLAS; every other tier keeps vLLM's code.
            takes_cublas_tier = (
                x.dtype == torch.bfloat16
                and self.allow_cublas_router_gemm
                and not (self.allow_ll_bf16_gemm and x.shape[0] <= 16)
                and not self.allow_fp32_router_gemm
                and not self.allow_bf16x3_router_gemm
            )
            if takes_cublas_tier:
                return _fixed_shape_mm(x, self.weight), None
            return _orig_forward(self, x)

        _gl.GateLinear.forward = _forward
        print(f"reax: fixed-shape MoE router GEMM enabled ({_ROWS} rows, vLLM {_VLLM})", file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"reax: REAX_VLLM_INVARIANT_ROUTER=1 but the router fix was not applied: {e}")
