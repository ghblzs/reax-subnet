"""SidecarBackend retries once when a pooled runtime connection was closed under it."""
import asyncio
import json

import pytest

from reax_subnet.s1backends import PROBE_DECISION, SidecarBackend

ANSWER = {"type": "noul", "probabilities": {"true": 0.75, "false": 0.25}, "confidence": 0.75, "noul": 0.75}


async def _serve(drops):
    calls = {"n": 0}

    async def handle(reader, writer):
        await reader.readuntil(b"\r\n\r\n")
        calls["n"] += 1
        if calls["n"] <= drops:
            writer.close()  # connection dropped before any response byte
            return
        body = json.dumps({"model": "s1-fast", "answers": {"q": ANSWER},
                           "usage": {"input_tokens": 9, "output_tokens": 0}}).encode()
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n"
                     b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1], calls


@pytest.mark.asyncio
async def test_one_dropped_connection_is_retried():
    server, port, calls = await _serve(drops=1)
    backend = SidecarBackend("s1-fast", f"http://127.0.0.1:{port}", "tok", timeout_s=5)
    try:
        (answer,) = await backend.decide([PROBE_DECISION])
        assert answer["probabilities"]["true"] == pytest.approx(0.75)
        assert calls["n"] == 2
    finally:
        await backend.close()
        server.close()


@pytest.mark.asyncio
async def test_repeated_drops_still_fail():
    import aiohttp

    server, port, calls = await _serve(drops=10)
    backend = SidecarBackend("s1-fast", f"http://127.0.0.1:{port}", "tok", timeout_s=5)
    try:
        with pytest.raises((aiohttp.ServerDisconnectedError, aiohttp.ClientOSError)):
            await backend.decide([PROBE_DECISION])
        assert calls["n"] == 2
    finally:
        await backend.close()
        server.close()


async def _serve_split(body: bytes, pause_s: float = 0.2):
    """Sends the headers and the first half of the body, pauses, then sends the rest."""

    async def handle(reader, writer):
        await reader.readuntil(b"\r\n\r\n")
        half = len(body) // 2
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n"
                     b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body[:half])
        await writer.drain()
        await asyncio.sleep(pause_s)
        writer.write(body[half:])
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1]


@pytest.mark.asyncio
async def test_body_split_across_segments_is_read_whole():
    body = json.dumps({"model": "s1-fast", "answers": {"q": ANSWER}, "pad": "x" * 4000}).encode()
    server, port = await _serve_split(body)
    backend = SidecarBackend("s1-fast", f"http://127.0.0.1:{port}", "tok", timeout_s=5)
    try:
        (answer,) = await backend.decide([PROBE_DECISION])
        assert answer["probabilities"]["true"] == pytest.approx(0.75)
    finally:
        await backend.close()
        server.close()


@pytest.mark.asyncio
async def test_oversized_runtime_body_is_refused():
    from reax_subnet.s1backends import MAX_RUNTIME_RESPONSE_BYTES, SidecarError

    body = b'{"pad": "' + b"x" * (MAX_RUNTIME_RESPONSE_BYTES + 10) + b'"}'
    server, port = await _serve_split(body, pause_s=0.0)
    backend = SidecarBackend("s1-fast", f"http://127.0.0.1:{port}", "tok", timeout_s=5)
    try:
        with pytest.raises(SidecarError, match="too large"):
            await backend.decide([PROBE_DECISION])
    finally:
        await backend.close()
        server.close()
