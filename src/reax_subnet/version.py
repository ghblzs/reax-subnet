"""Release and protocol identifiers.

``WEIGHTS_VERSION_KEY`` is sent with every ``set_weights`` call. The subnet
owner sets the chain hyperparameter ``weights_version`` to the oldest
accepted key, which makes the chain reject weights from outdated validators.
Bump it on every release that changes scoring.
"""

__version__ = "1.2.0rc8.post2"

# Wire protocol of the rc3 mechanism (REAX_MECHANISM=mpm1). The reax-s1 protocol id lives in
# s1protocol.py. A miner rejects any other value.
PROTOCOL = "reax-mpm/1"

# major * 1000 + minor * 10 + patch  (1.0.0 -> 1000)
WEIGHTS_VERSION_KEY = 1230
