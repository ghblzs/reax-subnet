"""On-chain commitment ``reax-spec/1:<track_id>:<hf_repo>@<40-hex revision>``.

Parsed strictly, like the ``reax-s1/1`` commitments in manifest.py: anything that does not match exactly is ``None``.
The Commitments pallet stores at most 128 bytes, so longer text cannot be a real commitment either.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from . import PROTOCOL_SPEC

MAX_COMMITMENT_BYTES = 128
_TRACK = re.compile(r"[a-z][a-z0-9_]{1,47}")
_REPO_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}")
_REVISION = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class SpecCommitment:
    track_id: str
    repo: str
    revision: str

    def text(self) -> str:
        return f"{PROTOCOL_SPEC}:{self.track_id}:{self.repo}@{self.revision}"


def _valid_repo(repo: str) -> bool:
    owner, sep, name = repo.partition("/")
    if not sep or "/" in name:
        return False
    for part in (owner, name):
        if not _REPO_PART.fullmatch(part) or ".." in part or "--" in part or part.endswith((".", "-")):
            return False
    return True


def parse_commitment(text: Any) -> SpecCommitment | None:
    """``reax-spec/1:<track_id>:<repo>@<revision>`` -> SpecCommitment; anything else -> None."""
    if not isinstance(text, str) or not text.isascii() or len(text.encode()) > MAX_COMMITMENT_BYTES:
        return None
    parts = text.strip().split(":")
    if len(parts) != 3 or parts[0] != PROTOCOL_SPEC or not _TRACK.fullmatch(parts[1]):
        return None
    repo, sep, revision = parts[2].partition("@")
    if not sep or not _valid_repo(repo) or not _REVISION.fullmatch(revision):
        return None
    return SpecCommitment(parts[1], repo, revision)
