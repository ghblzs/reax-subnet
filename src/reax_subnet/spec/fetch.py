"""Get a submission onto local disk and read its facts, without running any of its code.

``inspect_dir`` works on a local directory (also used by the miner CLI). ``HfFetcher`` resolves the pinned revision on the
Hugging Face Hub, checks the file list before downloading, downloads only weights/json/tokenizer/licence files in a
child process with a hard timeout (no token, public repos only), and then runs the same directory inspection.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from . import rules as R
from . import tensors as T
from .commitment import SpecCommitment

DOWNLOAD_TIMEOUT_S = 3600
LISTING_TIMEOUT_S = 60


class FetchError(RuntimeError):
    """kind: 'unavailable' (not found/private/network: retry until the publish deadline) or 'rejected' (rule violation)."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


@dataclass
class Inspected:
    path: str
    facts: R.RepoFacts
    hashes: T.WeightHashes | None
    violations: list[str]


def list_dir(model_dir: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for root, _d, files in os.walk(model_dir):
        for f in files:
            p = os.path.join(root, f)
            if os.path.islink(p):
                out[os.path.relpath(p, model_dir)] = -1
            else:
                out[os.path.relpath(p, model_dir)] = os.path.getsize(p)
    return out


def inspect_dir(model_dir: str, *, category: str, hotkey: str | None, rules: R.TrackRules = R.TrackRules(),
                require_manifest: bool = True) -> Inspected:
    """Read config, manifest and safetensors headers as data; hash the tensors; apply every rule."""
    files = list_dir(model_dir)
    problems: list[str] = [f"{n}: symlink" for n, s in files.items() if s < 0]
    files = {n: s for n, s in files.items() if s >= 0}

    def load(name: str) -> Mapping[str, Any] | None:
        if name not in files:
            return None
        try:
            v = R.read_json_file(os.path.join(model_dir, name))
            return v if isinstance(v, dict) else None
        except (ValueError, OSError) as exc:
            problems.append(f"{name}: {exc}")
            return None

    config, manifest = load("config.json"), load(R.MANIFEST_NAME)
    tok_cfg = load("tokenizer_config.json")
    tmpl = tok_cfg.get("chat_template") if tok_cfg else None
    tmpl_chars = len(tmpl) if isinstance(tmpl, str) else len(json.dumps(tmpl)) if tmpl else 0
    if "tokenizer_config.json" not in files or not any(n in files for n in ("tokenizer.json", "tokenizer.model")):
        problems.append("tokenizer files missing (tokenizer_config.json and tokenizer.json or tokenizer.model)")
    params = wbytes = None
    hashes = None
    try:
        tensors = T.read_model(model_dir)
        params, wbytes = T.param_count(tensors), T.weight_bytes(tensors)
    except T.SafetensorsError as exc:
        problems.append(f"safetensors: {exc}")
    facts = R.RepoFacts(files=files, config=config, manifest=manifest, params=params, weight_bytes=wbytes,
                        chat_template_chars=tmpl_chars, problems=problems)
    violations = R.check_facts(facts, category=category, hotkey=hotkey, rules=rules)
    if not require_manifest:
        violations = [v for v in violations if R.MANIFEST_NAME not in v]
    if not violations:   # hashing reads every byte: only for acceptable submissions
        hashes = T.canonical_hashes(model_dir)
    return Inspected(model_dir, facts, hashes, violations)


@dataclass(frozen=True)
class RepoListing:
    sha: str
    files: dict[str, int]
    private: bool
    gated: bool


class Fetcher(Protocol):
    async def fetch(self, commitment: SpecCommitment, dest: str, *, category: str, hotkey: str,
                    rules: R.TrackRules) -> Inspected: ...


_DOWNLOAD_SNIPPET = (
    "import sys; from huggingface_hub import snapshot_download; "
    "snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3], "
    "allow_patterns=sys.argv[4].split(','), token=False, etag_timeout=30)"
)


class HfFetcher:
    """Hugging Face Hub fetcher. Needs ``huggingface_hub`` (pulled in by the ``model`` extra)."""

    def __init__(self, timeout_s: float = DOWNLOAD_TIMEOUT_S):
        self.timeout_s = timeout_s

    def _listing(self, c: SpecCommitment) -> RepoListing:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError, RevisionNotFoundError
        try:
            info = HfApi().model_info(c.repo, revision=c.revision, files_metadata=True, token=False, timeout=LISTING_TIMEOUT_S)
        except (RepositoryNotFoundError, RevisionNotFoundError) as exc:
            raise FetchError("unavailable", f"repo or revision not found (still private?): {exc}") from exc
        except (HfHubHTTPError, OSError) as exc:
            raise FetchError("unavailable", f"hub error: {exc}") from exc
        files = {s.rfilename: int(s.size or 0) for s in info.siblings or []}
        return RepoListing(sha=str(info.sha), files=files, private=bool(info.private), gated=bool(info.gated))

    async def fetch(self, commitment: SpecCommitment, dest: str, *, category: str, hotkey: str,
                    rules: R.TrackRules = R.TrackRules()) -> Inspected:
        loop = asyncio.get_running_loop()
        listing = await loop.run_in_executor(None, self._listing, commitment)
        if listing.sha != commitment.revision:
            raise FetchError("rejected", "hub revision does not match the committed revision")
        if listing.gated or listing.private:
            raise FetchError("unavailable", "repo is gated or private")
        bad = R.check_listing(listing.files, rules)
        if bad:
            raise FetchError("rejected", "; ".join(bad))
        shutil.rmtree(dest, ignore_errors=True)
        os.makedirs(dest, exist_ok=True)
        env = {"PATH": os.environ.get("PATH", ""), "HOME": dest, "HF_HUB_DISABLE_TELEMETRY": "1"}
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-I", "-c", _DOWNLOAD_SNIPPET, commitment.repo, commitment.revision, dest,
            ",".join(R.DOWNLOAD_PATTERNS), env=env, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
        try:
            _o, err = await asyncio.wait_for(proc.communicate(), self.timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            shutil.rmtree(dest, ignore_errors=True)
            raise FetchError("unavailable", "download timed out") from None
        if proc.returncode != 0:
            shutil.rmtree(dest, ignore_errors=True)
            raise FetchError("unavailable", "download failed: " + err.decode("utf-8", "replace")[-300:])
        shutil.rmtree(os.path.join(dest, ".cache"), ignore_errors=True)
        ins = await loop.run_in_executor(None, lambda: inspect_dir(dest, category=category, hotkey=hotkey, rules=rules))
        if ins.violations:
            shutil.rmtree(dest, ignore_errors=True)
            raise FetchError("rejected", "; ".join(ins.violations))
        return ins
