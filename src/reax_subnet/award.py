"""Owner approval of competition awards (docs/COMPETITIONS.md, "Owner approval").

A winning entry is paid only after the subnet owner has checked and approved it. An approval names exactly one
award: ``(track_id, hotkey, submission_id)``, where ``submission_id`` is ``sha256:<64 hex>`` (the image digest) for the
engine track and ``<hf repo>@<40 hex revision>`` for a specialist track. Another image, another model revision or
another hotkey needs a new approval.

The approvals file lives on the validator host only (``REAX_AWARD_APPROVALS``; never mounted into the keyless
competition worker). Each entry carries an Ed25519 signature over a canonical JSON payload; the verifying PUBLIC key is
``AWARD_APPROVAL_PUBKEY`` (release constant) or ``REAX_AWARD_APPROVAL_PUBKEY`` (hex). The private key is kept off the
validator host. Fail closed: no key, a missing or unreadable file, a bad signature, another key, an expired or
malformed entry -> no approval; the share goes to the inference part (never burned).

``reax-award-approve`` signs approvals (``keygen``, ``sign``, ``verify``).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Optional

from Crypto.PublicKey import ECC  # pycryptodome: already installed with bittensor (eth-account -> eth-keyfile)
from Crypto.Signature import eddsa

log = logging.getLogger("reax.award")


# Release constant: the REAX subnet owner's award approval key (Ed25519, 32 bytes hex), set 16 Oct 2026. The private key
# never sits on a validator host. REAX_AWARD_APPROVAL_PUBKEY overrides it; with neither, no award is ever paid.
AWARD_APPROVAL_PUBKEY: Optional[str] = "89beb19ceade780adce8f819ba12c3f48e12520e6e033c7204772c72a7d52e61"
APPROVALS_VERSION = 1
DOMAIN = b"reax-award-approval/1\n"  # signatures cannot be replayed as anything else
MAX_FILE_BYTES = 4 * 1024 * 1024
CLOCK_SKEW_S = 600
ENGINE_TRACK_ID = "s1-fast"

_HEX32 = re.compile(r"[0-9a-f]{64}")
_KEYS = {"track_id", "hotkey", "submission_id", "approved_at", "expires_at"}


def engine_submission_id(digest: str) -> str:
    return f"sha256:{digest}"


def spec_submission_id(repo: str, revision: str) -> str:
    return f"{repo}@{revision}"


def canonical(payload: Mapping[str, Any]) -> bytes:
    return DOMAIN + json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def check_payload(p: Any) -> Optional[str]:
    """None when ``p`` is a well-formed payload, else the reason."""
    if not isinstance(p, dict) or set(p) != _KEYS:
        return f"payload must have exactly {', '.join(sorted(_KEYS))}"
    for k in ("track_id", "hotkey", "submission_id"):
        if not isinstance(p[k], str) or not p[k] or len(p[k]) > 256:
            return f"{k} must be a non-empty string"
    if not _int(p["approved_at"]) or not (p["expires_at"] is None or _int(p["expires_at"])):
        return "approved_at must be an integer, expires_at an integer or null"
    return None


def load_public_key(hex_key: str):
    hex_key = (hex_key or "").strip().lower()
    if not _HEX32.fullmatch(hex_key):
        raise ValueError("an approval public key is 32 bytes as 64 hex characters")
    return eddsa.import_public_key(bytes.fromhex(hex_key))


def load_private_key(path: str):
    """PEM (PKCS#8, as written by ``keygen``) or a 32-byte seed as 64 hex characters."""
    with open(path, "rb") as fh:
        data = fh.read()
    text = data.decode("ascii", "replace").strip()
    if _HEX32.fullmatch(text.lower()):
        return eddsa.import_private_key(bytes.fromhex(text))
    return ECC.import_key(data)


def public_hex(key) -> str:
    return key.public_key().export_key(format="raw").hex() if key.has_private() else key.export_key(format="raw").hex()


def sign(private_key, payload: Mapping[str, Any]) -> dict[str, Any]:
    why = check_payload(dict(payload))
    if why:
        raise ValueError(why)
    sig = eddsa.new(private_key, "rfc8032").sign(canonical(payload))
    return {"payload": dict(payload), "signature": sig.hex()}


def verify_entry(public_key, entry: Any) -> Optional[str]:
    """None when the entry's signature is valid for ``public_key``, else the reason."""
    if not isinstance(entry, dict) or set(entry) != {"payload", "signature"}:
        return "entry must be an object with payload and signature"
    why = check_payload(entry["payload"])
    if why:
        return why
    sig = entry["signature"]
    if not isinstance(sig, str) or not re.fullmatch(r"[0-9a-f]{128}", sig):
        return "signature must be 64 bytes hex"
    try:
        eddsa.new(public_key, "rfc8032").verify(canonical(entry["payload"]), bytes.fromhex(sig))
    except ValueError:
        return "bad signature (tampered, or signed with another key)"
    return None


class AwardGate:
    """Answers ``approved(track_id, hotkey, submission_id)`` from the signed approvals file. Never raises."""

    def __init__(self, path: str, pubkey_hex: Optional[str], *, wall: Callable[[], float] = time.time):
        self.path, self._wall = path, wall
        self._key, self._key_error = None, ""
        try:
            self._key = load_public_key(pubkey_hex or "")
        except ValueError as exc:
            self._key_error = f"no valid approval public key ({exc})"
        self._stamp: Any = None
        self._entries: list[dict[str, Any]] = []
        self._problem = ""
        self._logged: dict[tuple[str, str, str], str] = {}
        self._logged_problem = ""

    @classmethod
    def from_env(cls, env: Mapping[str, str], **kw) -> "AwardGate":
        return cls((env.get("REAX_AWARD_APPROVALS") or "").strip(),
                   (env.get("REAX_AWARD_APPROVAL_PUBKEY") or "").strip() or AWARD_APPROVAL_PUBKEY, **kw)

    def _refresh(self) -> None:
        if self._key is None:
            self._entries, self._problem = [], self._key_error
            return
        if not self.path:
            self._entries, self._problem = [], "REAX_AWARD_APPROVALS is not set"
            return
        try:
            st = os.stat(self.path)
            stamp = (st.st_mtime_ns, st.st_size, st.st_ino)
            if stamp == self._stamp:
                return
            if st.st_size > MAX_FILE_BYTES:
                raise ValueError("file too large")
            with open(self.path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except FileNotFoundError:
            self._stamp, self._entries, self._problem = None, [], f"approvals file {self.path} missing"
            return
        except (OSError, ValueError) as exc:
            self._stamp, self._entries, self._problem = None, [], f"approvals file {self.path} unreadable ({exc})"
            return
        self._stamp = stamp
        if not isinstance(raw, dict) or raw.get("version") != APPROVALS_VERSION or not isinstance(raw.get("approvals"), list):
            self._entries, self._problem = [], f"approvals file {self.path} is not version {APPROVALS_VERSION} with a list"
            return
        good: list[dict[str, Any]] = []
        for i, entry in enumerate(raw["approvals"]):
            why = verify_entry(self._key, entry)
            if why:
                log.warning("award approvals: entry %d of %s ignored: %s", i, self.path, why)
                continue
            good.append(entry["payload"])
        self._entries, self._problem = good, ""

    def approved(self, track_id: str, hotkey: str, submission_id: str) -> bool:
        try:
            self._refresh()
        except Exception as exc:  # noqa: BLE001 - fail closed, never stop the validator
            self._stamp, self._entries, self._problem = None, [], f"internal error ({type(exc).__name__})"
        if self._problem and self._problem != self._logged_problem:
            log.warning("award approvals: %s; every competition award goes to the inference part", self._problem)
        self._logged_problem = self._problem
        now = self._wall()
        ok, why = False, "awaiting owner approval"
        for p in self._entries:
            if (p["track_id"], p["hotkey"], p["submission_id"]) != (track_id, hotkey, submission_id):
                continue
            if p["approved_at"] > now + CLOCK_SKEW_S:
                why = "approval dated in the future"
            elif p["expires_at"] is not None and now >= p["expires_at"]:
                why = "approval expired; awaiting owner approval"
            else:
                ok, why = True, "approved by the owner"
                break
        ident = (track_id, hotkey, submission_id)
        status = "approved" if ok else why
        if self._logged.get(ident) != status:
            (log.info if ok else log.warning)("award [%s] hotkey %s submission %s: %s%s", track_id, hotkey[:16],
                                              submission_id, status, "" if ok else " (its share goes to the inference part)")
            self._logged[ident] = status
        return ok


# ------------------------------------------------------------------------------------------------ CLI
def _write_atomic(path: str, data: dict[str, Any], mode: int = 0o644) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=".approvals-", dir=directory)
    with os.fdopen(fd, "w") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.chmod(tmp, mode)
    os.replace(tmp, path)


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog="reax-award-approve",
                                 description="Sign owner approvals of competition awards (docs/COMPETITIONS.md).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    kg = sub.add_parser("keygen", help="new Ed25519 key (keep it OFF the validator host); prints the public key")
    kg.add_argument("--out", required=True, help="private key file to create (PEM, mode 600)")
    sg = sub.add_parser("sign", help="sign one approval and add it to an approvals file (or print it)")
    sg.add_argument("--key", required=True, help="private key file (PEM or 64 hex seed)")
    sg.add_argument("--track", required=True, help=f"track id: {ENGINE_TRACK_ID} (engine) or a specialist track id")
    sg.add_argument("--hotkey", required=True)
    sg.add_argument("--submission", required=True,
                    help="engine: sha256:<64 hex image digest>; specialist: <hf repo>@<40 hex revision>")
    sg.add_argument("--expires-days", type=float, default=None, help="approval lapses after this many days")
    sg.add_argument("--approvals", default="", help="approvals file to add the entry to (created if missing)")
    vf = sub.add_parser("verify", help="check every entry of an approvals file")
    vf.add_argument("--approvals", required=True)
    vf.add_argument("--pubkey", default="", help="hex public key (default: REAX_AWARD_APPROVAL_PUBKEY / release key)")
    args = ap.parse_args(argv)

    if args.cmd == "keygen":
        if os.path.exists(args.out):
            sys.exit(f"{args.out} exists; refusing to overwrite a key")
        key = ECC.generate(curve="ed25519")
        fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(key.export_key(format="PEM"))
        print(public_hex(key))
        return

    if args.cmd == "sign":
        if args.track == ENGINE_TRACK_ID and not re.fullmatch(r"sha256:[0-9a-f]{64}", args.submission):
            sys.exit("engine track: --submission must be sha256:<64 hex image digest>")
        if args.track != ENGINE_TRACK_ID and not re.fullmatch(r"[^@\s]+/[^@\s]+@[0-9a-f]{40}", args.submission):
            sys.exit("specialist track: --submission must be <hf repo>@<40 hex revision>")
        key = load_private_key(args.key)
        now = int(time.time())
        payload = {"track_id": args.track, "hotkey": args.hotkey, "submission_id": args.submission, "approved_at": now,
                   "expires_at": None if args.expires_days is None else now + int(args.expires_days * 86400)}
        entry = sign(key, payload)
        if not args.approvals:
            print(json.dumps(entry, sort_keys=True))
            return
        try:
            with open(args.approvals, encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            data = {"version": APPROVALS_VERSION, "approvals": []}
        data["approvals"].append(entry)
        _write_atomic(args.approvals, data)
        print(f"approved {args.track} {args.hotkey} {args.submission} (key {public_hex(key)[:16]}...) -> {args.approvals}")
        return

    pub = args.pubkey or os.environ.get("REAX_AWARD_APPROVAL_PUBKEY") or AWARD_APPROVAL_PUBKEY or ""
    try:
        key = load_public_key(pub)
        with open(args.approvals, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        sys.exit(str(exc))
    bad = 0
    for i, entry in enumerate(data.get("approvals", []) if isinstance(data, dict) else []):
        why = verify_entry(key, entry)
        bad += bool(why)
        p = entry.get("payload", {}) if isinstance(entry, dict) else {}
        print(f"{i}: {'OK ' if not why else 'BAD'} {p.get('track_id')} {p.get('hotkey')} {p.get('submission_id')}"
              f" expires {p.get('expires_at')}" + (f" -- {why}" if why else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
