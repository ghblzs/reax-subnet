import time

import pytest

from reax_subnet import auth


@pytest.fixture
def keys(make_wallet):
    return make_wallet("//Validator"), make_wallet("//Miner"), make_wallet("//Other")


def test_request_roundtrip(keys):
    v, m, _ = keys
    h = auth.sign_request(v, b"body", m.hotkey.ss58_address)
    assert auth.claimed_hotkey(h) == v.hotkey.ss58_address
    assert auth.verify_request(h, b"body", m.hotkey.ss58_address, auth.NonceStore()) == v.hotkey.ss58_address


def test_request_wrong_receiver_tamper_replay(keys):
    v, m, o = keys
    h = auth.sign_request(v, b"body", m.hotkey.ss58_address)
    with pytest.raises(auth.AuthError):
        auth.verify_request(h, b"body", o.hotkey.ss58_address, auth.NonceStore())
    with pytest.raises(auth.AuthError):
        auth.verify_request(h, b"bodY", m.hotkey.ss58_address, auth.NonceStore())
    store = auth.NonceStore()
    auth.verify_request(h, b"body", m.hotkey.ss58_address, store)
    with pytest.raises(auth.AuthError):
        auth.verify_request(h, b"body", m.hotkey.ss58_address, store)


def test_stale_request(keys):
    v, m, _ = keys
    from bittensor import http_auth

    h = http_auth.sign(v, method="POST", path=auth.DECIDE_PATH, body=b"x", receiver_ss58=m.hotkey.ss58_address,
                       nonce_ns=time.time_ns() - 60 * 10**9)
    with pytest.raises(auth.AuthError):
        auth.verify_request(h, b"x", m.hotkey.ss58_address, auth.NonceStore())


def test_response_binding(keys):
    v, m, o = keys
    vk, mk = v.hotkey.ss58_address, m.hotkey.ss58_address
    h = auth.sign_response(m, b"resp", "req1", vk)
    auth.verify_response(h, b"resp", "req1", vk, mk, auth.NonceStore(), max_age=30)
    # different request id
    with pytest.raises(auth.AuthError):
        auth.verify_response(h, b"resp", "req2", vk, mk, auth.NonceStore(), max_age=30)
    # signed by a different miner than the one queried
    h2 = auth.sign_response(o, b"resp", "req1", vk)
    with pytest.raises(auth.AuthError):
        auth.verify_response(h2, b"resp", "req1", vk, mk, auth.NonceStore(), max_age=30)


def test_response_cannot_be_reflected_as_request(keys):
    v, m, _ = keys
    # A miner-signed response addressed to the validator must not verify as a request to anyone.
    h = auth.sign_response(m, b"resp", "req1", v.hotkey.ss58_address)
    with pytest.raises(auth.AuthError):
        auth.verify_request(h, b"resp", v.hotkey.ss58_address, auth.NonceStore())
