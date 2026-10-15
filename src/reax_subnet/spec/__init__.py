"""Specialist (use-case fine-tune) track evaluator, ``reax-spec/1``.

Miners commit a weights-only model repo for one track (a JevBench use case); validators serve it in a pinned,
network-less vLLM container and score it on fresh sealed items. New code only: nothing here is imported by the
1.1 miner or validator paths. See docs/SPECIALIST-TRACKS.md.
"""

PROTOCOL_SPEC = "reax-spec/1"
