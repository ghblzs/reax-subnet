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
