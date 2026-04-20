"""Dashboard HTTP server — stdlib-only, no framework dependencies.

The server reads from an :class:`InstanceSource`, so the request handlers don't
care whether snapshots come from local files or (someday) a network ingest
endpoint.
"""

from __future__ import annotations

import errno
import json
import logging
import re
import socket
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from ralph.dashboard.sources import InstanceSource, LocalFilesSource
from ralph.dashboard.static import render_index

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
PORT_FALLBACK_RANGE = 10

_PID_RE = re.compile(r"^/api/instances/(?P<pid>\d+)/tail$")

# Strip terminal control sequences that Claude + friends emit so the <pre> in
# the dashboard shows readable text instead of "\x1b[32m..." garble.
# Matches CSI (most SGR/cursor codes), OSC (title/hyperlink), and single-char
# ESC escapes. Intentionally conservative — we leave printable characters alone.
_ANSI_RE = re.compile(
    rb"\x1b\[[0-?]*[ -/]*[@-~]"          # CSI  ESC [ ... final-byte
    rb"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"  # OSC  ESC ] ... BEL | ESC \
    rb"|\x1b[@-Z\\-_]"                   # 2-byte ESC sequences (ESC =, ESC >, ESC N, ...)
)


def _strip_terminal_noise(data: bytes) -> bytes:
    """Remove ANSI escape sequences and normalise carriage returns for the tail view."""
    cleaned = _ANSI_RE.sub(b"", data)
    # Collapse CRLF → LF and bare CR → LF; bare CRs otherwise render as
    # overprints in <pre> and make the tail look scrambled.
    cleaned = cleaned.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return cleaned


def _make_handler(source: InstanceSource, refresh_seconds: float) -> type[BaseHTTPRequestHandler]:
    """Return a handler class bound to this source — avoids module globals."""

    class DashboardRequestHandler(BaseHTTPRequestHandler):
        # Silence per-request logs; we emit our own startup banner.
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            logger.debug("%s - %s", self.address_string(), format % args)

        def do_GET(self) -> None:  # noqa: N802 (required by BaseHTTPRequestHandler)
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/" or path == "/index.html":
                self._serve_index(parsed.query)
                return
            if path == "/api/health":
                self._send_json(HTTPStatus.OK, {"ok": True})
                return
            if path == "/api/instances":
                self._serve_instances()
                return
            m = _PID_RE.match(path)
            if m:
                pid = int(m.group("pid"))
                qs = parse_qs(parsed.query)
                try:
                    max_bytes = int(qs.get("bytes", ["8192"])[0])
                except (ValueError, TypeError):
                    max_bytes = 8192
                self._serve_tail(pid, max_bytes)
                return

            self._send_text(HTTPStatus.NOT_FOUND, "not found")

        # ---- Responders ----------------------------------------------

        def _serve_index(self, query: str) -> None:
            qs = parse_qs(query)
            refresh = refresh_seconds
            if "refresh" in qs:
                try:
                    refresh = float(qs["refresh"][0])
                except (ValueError, TypeError):
                    pass
            html = render_index(refresh)
            body = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _serve_instances(self) -> None:
            try:
                snapshots = source.list_instances()
            except Exception as exc:  # pragma: no cover — defensive
                logger.exception("list_instances failed: %s", exc)
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
                return
            payload = {"instances": [s.to_dict() for s in snapshots]}
            self._send_json(HTTPStatus.OK, payload)

        def _serve_tail(self, pid: int, max_bytes: int) -> None:
            if max_bytes <= 0 or max_bytes > 1024 * 1024:
                max_bytes = 8192
            try:
                # Read a bit more raw data than requested so that after stripping
                # ANSI we still serve ~max_bytes of visible content.
                raw = source.get_tail(pid, max_bytes * 2)
            except Exception as exc:  # pragma: no cover — defensive
                logger.exception("get_tail failed: %s", exc)
                self._send_text(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
                return
            data = _strip_terminal_noise(raw)
            if len(data) > max_bytes:
                data = data[-max_bytes:]
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        # ---- Helpers -------------------------------------------------

        def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_text(self, status: HTTPStatus, text: str) -> None:
            body = text.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return DashboardRequestHandler


def _bind_with_fallback(
    host: str, port: int, handler_cls: type[BaseHTTPRequestHandler]
) -> tuple[ThreadingHTTPServer, int]:
    """Try the requested port, then increment up to PORT_FALLBACK_RANGE times."""
    last_exc: OSError | None = None
    for offset in range(PORT_FALLBACK_RANGE + 1):
        attempt = port + offset
        try:
            server = ThreadingHTTPServer((host, attempt), handler_cls)
            return server, server.server_address[1]
        except OSError as exc:
            if exc.errno in (errno.EADDRINUSE, errno.EACCES):
                last_exc = exc
                continue
            raise
    assert last_exc is not None
    raise last_exc


def run_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    refresh_seconds: float = 1.5,
    source: InstanceSource | None = None,
) -> None:
    """Start the dashboard server and block until interrupted."""
    if source is None:
        source = LocalFilesSource()

    handler_cls = _make_handler(source, refresh_seconds)
    server, bound_port = _bind_with_fallback(host, port, handler_cls)

    url = f"http://{host}:{bound_port}/"
    if host not in ("127.0.0.1", "localhost"):
        logger.warning(
            "Dashboard is binding %s — non-loopback without auth. "
            "Anyone on your network can see ralph progress.",
            host,
        )
    print(f"Ralph dashboard listening on {url}")
    if bound_port != port:
        print(f"  (port {port} was busy, fell back to {bound_port})")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down dashboard.")
    finally:
        server.shutdown()
        server.server_close()


def start_server_in_thread(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    refresh_seconds: float = 1.5,
    source: InstanceSource | None = None,
) -> tuple[ThreadingHTTPServer, int, threading.Thread]:
    """Start the server in a background thread. Returns (server, bound_port, thread).

    Primarily intended for tests — the caller is responsible for shutting down.
    """
    if source is None:
        source = LocalFilesSource()
    handler_cls = _make_handler(source, refresh_seconds)
    server, bound_port = _bind_with_fallback(host, port, handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, bound_port, thread


def ephemeral_socket_port() -> int:
    """Pick a random available port — for tests that want to avoid collisions."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
