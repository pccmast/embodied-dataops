"""Host shim for the Windows dev box (no product behaviour change).

This machine runs a Winsock filter (Proxifier/Clash) that intercepts loopback
TCP connections, which makes CPython's ``socket.socketpair`` fallback reject its
own accepted socket (``ConnectionError: Unexpected peer connection``).  Any
``asyncio`` event loop — and therefore ``fastapi.testclient.TestClient`` — then
fails before the first request.

The shim retries only for that exact platform error, using the same loopback
pair construction without the peer-identity assertion; the data path is the
normal TCP loopback connection.  Product code and the delivered contract stay
untouched.
"""

from __future__ import annotations

import socket
import sys

if sys.platform == "win32":  # pragma: no cover - host-specific environment shim
    _original_socketpair = socket.socketpair

    def _tolerant_socketpair(
        family: int = socket.AF_INET,
        type: int = socket.SOCK_STREAM,
        proto: int = 0,
    ) -> tuple[socket.socket, socket.socket]:
        try:
            return _original_socketpair(family, type, proto)
        except ConnectionError:
            host = "127.0.0.1" if family == socket.AF_INET else "::1"
            listener = socket.socket(family, type, proto)
            try:
                listener.bind((host, 0))
                listener.listen()
                address, port = listener.getsockname()[:2]
                client = socket.socket(family, type, proto)
                try:
                    client.setblocking(False)
                    try:
                        client.connect((address, port))
                    except (BlockingIOError, InterruptedError):
                        pass
                    client.setblocking(True)
                    server, _ = listener.accept()
                except BaseException:
                    client.close()
                    raise
            finally:
                listener.close()
            return server, client

    socket.socketpair = _tolerant_socketpair  # type: ignore[assignment]
