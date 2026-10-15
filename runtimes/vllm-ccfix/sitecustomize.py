# REAX vLLM fix for confidential-compute (CC) GPUs (e.g. some cloud RTX PRO 6000 / H100 CC instances).
# Mount this directory read-only at /opt/reax-ccfix, set PYTHONPATH=/opt/reax-ccfix and REAX_VLLM_DISABLE_UVA=1.
# Measured: with the patch the s1-pro (Rune, deterministic) results are bit-identical to the stored calibration runs.
# REAX validator on a confidential-compute (CC) GPU: a UVA (zero-copy) GPU view of pinned host memory does not see later
# host writes there, so vLLM's UvaBuffer inputs go stale. Force vLLM's NonUvaBuffer path (explicit copies; numerics unchanged).
import os, sys
if os.environ.get("REAX_VLLM_DISABLE_UVA") == "1":
    try:
        import vllm.utils.platform_utils as _pu
        _pu.is_uva_available = lambda: False
        print("reax: vLLM UVA disabled (CC GPU)", file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print("reax: UVA patch not applied:", e, file=sys.stderr)
