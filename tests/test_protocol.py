import json

import pytest

from reax_subnet.protocol import (
    MAX_ITEMS,
    MAX_REQUEST_BYTES,
    DecideRequest,
    DecideResponse,
    Item,
    ProtocolError,
    normalise,
    parse_request,
    parse_response,
)
from reax_subnet.version import PROTOCOL


def _req(n=2):
    items = tuple(Item(f"i{k}", f"question {k}?", ("yes", "no", "maybe")) for k in range(n))
    return DecideRequest("r1", "hash-test", items)


def _payload(**over):
    obj = json.loads(_req().to_bytes())
    obj.update(over)
    return json.dumps(obj).encode()


def test_request_roundtrip():
    req = _req(3)
    assert parse_request(req.to_bytes()) == req


@pytest.mark.parametrize(
    "mutate",
    [
        lambda o: o.update(extra=1),
        lambda o: o.pop("model"),
        lambda o: o.update(protocol="reax-mpm/0"),
        lambda o: o.update(request_id="bad id!"),
        lambda o: o.update(model="UPPER"),
        lambda o: o.update(items=[]),
        lambda o: o.update(items=[o["items"][0]] * 2),  # duplicate ids
        lambda o: o["items"][0].update(options=["a"]),
        lambda o: o["items"][0].update(options=["a", "a"]),
        lambda o: o["items"][0].update(options=[str(i) for i in range(9)]),
        lambda o: o["items"][0].update(question=""),
        lambda o: o["items"][0].update(question="x" * 2001),
        lambda o: o["items"][0].update(question="bell\x07"),
        lambda o: o["items"][0].update(options=["a", 3]),
        lambda o: o["items"][0].update(extra=True),
        lambda o: o.update(items=[dict(o["items"][0], id=f"x{i}") for i in range(MAX_ITEMS + 1)]),
    ],
)
def test_request_rejects(mutate):
    obj = json.loads(_req().to_bytes())
    mutate(obj)
    with pytest.raises(ProtocolError):
        parse_request(json.dumps(obj).encode())


def test_request_rejects_non_json_and_oversize():
    with pytest.raises(ProtocolError):
        parse_request(b"\xff\xfe")
    with pytest.raises(ProtocolError):
        parse_request(b"[" * 100_000)
    with pytest.raises(ProtocolError):
        parse_request(b" " * (MAX_REQUEST_BYTES + 1))


def _resp(req, probs_by_item=None, **over):
    obj = {
        "protocol": PROTOCOL,
        "request_id": req.request_id,
        "miner_version": "1.0.0",
        "results": [{"id": it.item_id, "probs": (probs_by_item or {}).get(it.item_id, [0.5, 0.3, 0.2])} for it in req.items],
    }
    obj.update(over)
    return json.dumps(obj).encode()


def test_response_ok_and_renormalised():
    req = _req()
    out = parse_response(_resp(req, {"i0": [0.5, 0.3, 0.2005]}), req)
    assert abs(sum(out["i0"]) - 1.0) < 1e-12
    assert set(out) == {"i0", "i1"}


@pytest.mark.parametrize(
    "raw_mutation",
    [
        {"request_id": "other"},
        {"protocol": "x"},
        {"results": []},
        {"miner_version": 5},
        {"extra": 1},
    ],
)
def test_response_rejects_envelope(raw_mutation):
    req = _req()
    with pytest.raises(ProtocolError):
        parse_response(_resp(req, **raw_mutation), req)


@pytest.mark.parametrize(
    "probs",
    [[0.5, 0.5], [0.5, 0.3, 0.3], [1.2, -0.1, -0.1], [True, False, False], ["0.5", 0.3, 0.2], [0.4, 0.3, None]],
)
def test_response_rejects_bad_probs(probs):
    req = _req()
    with pytest.raises(ProtocolError):
        parse_response(_resp(req, {"i0": probs}), req)


def test_response_rejects_nan_and_infinity_tokens():
    req = _req()
    raw = _resp(req).replace(b"0.5", b"NaN", 1)
    with pytest.raises(ProtocolError):
        parse_response(raw, req)
    raw = _resp(req).replace(b"0.5", b"1e999", 1)
    with pytest.raises(ProtocolError):
        parse_response(raw, req)


def test_response_rejects_duplicate_and_unknown_ids():
    req = _req()
    obj = json.loads(_resp(req))
    obj["results"][1]["id"] = "i0"
    with pytest.raises(ProtocolError):
        parse_response(json.dumps(obj).encode(), req)
    obj["results"][1]["id"] = "zzz"
    with pytest.raises(ProtocolError):
        parse_response(json.dumps(obj).encode(), req)


def test_normalise_bounds():
    assert normalise([1, 0], 2) == (1.0, 0.0)
    with pytest.raises(ProtocolError):
        normalise([0.0, 0.0], 2)


def test_response_bytes_roundtrip():
    req = _req()
    resp = DecideResponse(req.request_id, (("i0", (0.2, 0.3, 0.5)), ("i1", (1.0, 0.0, 0.0))), "1.0.0rc1")
    assert parse_response(resp.to_bytes(), req)["i0"] == (0.2, 0.3, 0.5)


def test_huge_integer_probability_is_protocol_error():
    req = _req()
    raw = _resp(req).replace(b'"probs": [', b'"probs": [' + b"9" * 400 + b",", 1)
    with pytest.raises(ProtocolError):
        parse_response(raw, req)
    raw = _resp(req).replace(b'"probs": [', b'"probs": [' + b"9" * 5000 + b",", 1)  # beyond int digit limit
    with pytest.raises(ProtocolError):
        parse_response(raw, req)
