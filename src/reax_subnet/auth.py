"""Hotkey signatures on both directions of a validator <-> miner exchange.

Requests use the standard ``btauth/1`` scheme from the Bittensor SDK, bound to
the receiving miner's hotkey. Responses are signed the same way by the miner,
bound to the requesting validator and to the request id, under a distinct
method name so a signed response can never be replayed as a request.

Binding both directions to hotkeys means a miner cannot answer with another
miner's endpoint: the other miner would refuse a request addressed to a
different hotkey, and its signed answer would not carry the right hotkey.
"""

from __future__ import annotations

from typing import Any, Mapping

from bittensor import http_auth

DECIDE_PATH = "/v1/decide"
RESPONSE_METHOD = "REAX-RESPONSE"

AuthError = http_auth.AuthError
NonceStore = http_auth.InMemoryNonceStore


class DuplicateAuthHeader(http_auth.AuthError):
    """A btauth header appears more than once (header smuggling)."""


def _auth_header_values(headers: Mapping[str, str]) -> dict[str, list[str]]:
    seen: dict[str, list[str]] = {}
    for key, value in headers.items():  # multidicts (aiohttp) yield every duplicate here
        k = key.lower()
        if k.startswith("x-bittensor-"):
            seen.setdefault(k, []).append(value)
    return seen


def check_single_auth_headers(headers: Mapping[str, str]) -> None:
    """Refuse any request/response that carries a btauth header twice.

    The SDK verifier folds headers into a dict (last value wins) while a cheap pre-check may read the
    first one; with one value per header both always see the same sender.
    """
    for k, values in _auth_header_values(headers).items():
        if len(values) > 1:
            raise DuplicateAuthHeader(f"header {k} appears {len(values)} times")


def claimed_hotkey(headers: Mapping[str, str]) -> str | None:
    """The sender hotkey a request *claims*, before any verification.

    Used only to reject unknown senders cheaply before spending CPU on
    signature checks; the claim is verified afterwards and must equal the
    verified signer. None if the header is missing or appears more than once.
    """
    values = _auth_header_values(headers).get(http_auth.HEADER_HOTKEY.lower(), [])
    return values[0] if len(values) == 1 else None


def sign_request(wallet: Any, body: bytes, miner_hotkey: str, path: str = DECIDE_PATH) -> dict[str, str]:
    return http_auth.sign(wallet, method="POST", path=path, body=body, receiver_ss58=miner_hotkey)


def verify_request(
    headers: Mapping[str, str], body: bytes, self_hotkey: str, nonce_store: NonceStore, path: str = DECIDE_PATH
) -> str:
    check_single_auth_headers(headers)
    caller = http_auth.verify(
        headers, body, method="POST", path=path, self_hotkey_ss58=self_hotkey, nonce_store=nonce_store
    )
    return caller.hotkey_ss58


def _response_path(request_id: str, path: str = DECIDE_PATH) -> str:
    return f"{path}/{request_id}"


def sign_response(
    wallet: Any, body: bytes, request_id: str, validator_hotkey: str, path: str = DECIDE_PATH
) -> dict[str, str]:
    return http_auth.sign(
        wallet, method=RESPONSE_METHOD, path=_response_path(request_id, path), body=body,
        receiver_ss58=validator_hotkey,
    )


def verify_response(
    headers: Mapping[str, str],
    body: bytes,
    request_id: str,
    self_hotkey: str,
    miner_hotkey: str,
    nonce_store: NonceStore,
    max_age: float,
    path: str = DECIDE_PATH,
) -> None:
    check_single_auth_headers(headers)
    caller = http_auth.verify(
        headers,
        body,
        method=RESPONSE_METHOD,
        path=_response_path(request_id, path),
        self_hotkey_ss58=self_hotkey,
        nonce_store=nonce_store,
        max_age=max_age,
    )
    if caller.hotkey_ss58 != miner_hotkey:
        raise http_auth.BadSignature("response is signed by a different hotkey than the miner queried")
