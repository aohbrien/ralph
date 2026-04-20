"""Integration-style tests for the dashboard HTTP server.

We bind to an ephemeral port on loopback and hit the endpoints with
stdlib urllib — no new test dependencies required.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import pytest

from ralph.dashboard.server import (
    ephemeral_socket_port,
    start_server_in_thread,
)
from ralph.dashboard.sources import LocalFilesSource


def _get(url: str) -> tuple[int, bytes, str]:
    with urllib.request.urlopen(url, timeout=3) as resp:
        return resp.status, resp.read(), resp.headers.get("Content-Type", "")


@pytest.fixture
def running_server(tmp_path: Path):
    source = LocalFilesSource(instances_dir=tmp_path)
    port = ephemeral_socket_port()
    server, bound_port, thread = start_server_in_thread(
        host="127.0.0.1", port=port, source=source
    )
    base = f"http://127.0.0.1:{bound_port}"
    try:
        yield base, tmp_path
    finally:
        server.shutdown()
        server.server_close()


def test_health_endpoint(running_server) -> None:
    base, _ = running_server
    status, body, ctype = _get(f"{base}/api/health")
    assert status == 200
    assert ctype.startswith("application/json")
    assert json.loads(body) == {"ok": True}


def test_instances_empty(running_server) -> None:
    base, _ = running_server
    status, body, _ = _get(f"{base}/api/instances")
    assert status == 200
    assert json.loads(body) == {"instances": []}


def test_instances_includes_live_writer(running_server, write_instance) -> None:
    base, tmp_path = running_server
    write_instance(tmp_path, pid=os.getpid())
    status, body, _ = _get(f"{base}/api/instances")
    payload = json.loads(body)
    assert status == 200
    assert len(payload["instances"]) == 1
    assert payload["instances"][0]["pid"] == os.getpid()
    assert payload["instances"][0]["status"] == "running"


def test_tail_endpoint(running_server, write_instance) -> None:
    base, tmp_path = running_server
    pid = os.getpid()
    write_instance(tmp_path, pid=pid)
    (tmp_path / f"{pid}.tail").write_bytes(b"hello world\n")

    status, body, ctype = _get(f"{base}/api/instances/{pid}/tail?bytes=1024")
    assert status == 200
    assert ctype.startswith("text/plain")
    assert body == b"hello world\n"


def test_tail_missing_returns_empty(running_server) -> None:
    base, _ = running_server
    status, body, _ = _get(f"{base}/api/instances/99999999/tail")
    assert status == 200
    assert body == b""


def test_index_served(running_server) -> None:
    base, _ = running_server
    status, body, ctype = _get(f"{base}/")
    assert status == 200
    assert ctype.startswith("text/html")
    assert b"Ralph Dashboard" in body


def test_tail_strips_ansi_escapes(running_server, write_instance) -> None:
    """Tail should strip CSI colour codes and normalise carriage returns."""
    base, tmp_path = running_server
    pid = os.getpid()
    write_instance(tmp_path, pid=pid)
    # Typical Claude-style output: green "ok" + CRLF + a progress-bar line using CR.
    raw = (
        b"\x1b[32mok\x1b[0m done\r\n"
        b"progress 40%\r"
        b"progress 50%\r"
        b"progress 60%\n"
        b"\x1b]0;title\x07tail"
    )
    (tmp_path / f"{pid}.tail").write_bytes(raw)

    status, body, _ = _get(f"{base}/api/instances/{pid}/tail?bytes=1024")
    assert status == 200
    text = body.decode()
    # No ESC bytes survived.
    assert "\x1b" not in text
    # Visible content is preserved.
    assert "ok done" in text
    assert "progress 60%" in text
    assert "tail" in text
    # CRs normalised — no bare \r.
    assert "\r" not in text


def test_not_found(running_server) -> None:
    base, _ = running_server
    try:
        urllib.request.urlopen(f"{base}/api/does-not-exist", timeout=3)
    except urllib.error.HTTPError as exc:
        assert exc.code == 404
    else:  # pragma: no cover
        pytest.fail("expected 404")
