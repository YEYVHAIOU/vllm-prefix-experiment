#!/usr/bin/env python3
"""EnvBench replay with per-lane persistent stdlib HTTP connections.

The original replay implementation remains unchanged. This wrapper:
1. keeps metrics polling on the original urllib implementation;
2. replaces only inference request transport with thread-local
   http.client.HTTPConnection objects;
3. records connect/send/wait-for-headers timings in requests.csv;
4. preserves the original stream parser and request timing fields.
"""

from __future__ import annotations

import argparse
import http.client
import io
import json
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


MODES = ("legacy_urllib", "httpclient_per_lane")
_thread_local = threading.local()
_original_urlopen = urllib.request.urlopen


def parse_wrapper_args() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--http-connection-mode",
        choices=MODES,
        default="legacy_urllib",
    )
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return str(known.http_connection_mode)


def original_fetch_text(url: str, timeout: float = 10.0) -> str:
    request = urllib.request.Request(
        url,
        headers={"Accept": "text/plain"},
    )
    with _original_urlopen(request, timeout=timeout) as response:
        return response.read().decode(
            "utf-8",
            errors="replace",
        )


def close_thread_connection() -> None:
    connection = getattr(_thread_local, "connection", None)
    if connection is not None:
        try:
            connection.close()
        except Exception:
            pass
    _thread_local.connection = None
    _thread_local.connection_key = None


def get_connection(
    parsed: urllib.parse.SplitResult,
    timeout: float,
) -> tuple[http.client.HTTPConnection, bool, float]:
    scheme = parsed.scheme.lower()
    host = parsed.hostname
    if not host:
        raise urllib.error.URLError(
            f"Missing hostname in URL: {parsed.geturl()}"
        )

    if scheme == "https":
        port = parsed.port or 443
        connection_type = http.client.HTTPSConnection
    elif scheme == "http":
        port = parsed.port or 80
        connection_type = http.client.HTTPConnection
    else:
        raise urllib.error.URLError(
            f"Unsupported URL scheme: {scheme}"
        )

    key = (scheme, host, port)
    connection = getattr(_thread_local, "connection", None)
    current_key = getattr(
        _thread_local,
        "connection_key",
        None,
    )

    if connection is not None and current_key == key:
        connection.timeout = timeout
        return connection, True, 0.0

    close_thread_connection()

    connection = connection_type(
        host,
        port,
        timeout=timeout,
    )

    connect_start = time.perf_counter()
    connection.connect()
    connect_seconds = time.perf_counter() - connect_start

    _thread_local.connection = connection
    _thread_local.connection_key = key
    return connection, False, connect_seconds


class PersistentHTTPResponse:
    def __init__(
        self,
        response: http.client.HTTPResponse,
    ) -> None:
        self._response = response
        self.status = int(response.status)
        self.reason = response.reason
        self.headers = response.headers
        self._fully_consumed = False

    def __iter__(self):
        while True:
            line = self._response.readline()
            if not line:
                self._fully_consumed = True
                break

            # Consume the remaining HTTP body before exposing [DONE].
            # The original replay breaks as soon as it sees [DONE]; draining
            # first keeps the connection reusable without adding hidden delay
            # after the original request-end timestamp.
            if line.strip() == b"data: [DONE]":
                self._response.read()
                self._fully_consumed = True
                yield line
                break

            yield line

    def read(self, amount: int = -1) -> bytes:
        data = self._response.read(amount)
        if amount is None or amount < 0:
            self._fully_consumed = True
        return data

    def getcode(self) -> int:
        return self.status

    def close(self) -> None:
        if not self._fully_consumed:
            try:
                self._response.read()
                self._fully_consumed = True
            except Exception:
                close_thread_connection()
        try:
            self._response.close()
        except Exception:
            close_thread_connection()

    def __enter__(self) -> "PersistentHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.close()
        return False


def httpclient_urlopen(
    url: str | urllib.request.Request,
    data: bytes | None = None,
    timeout: float | object = 10.0,
    *args: Any,
    **kwargs: Any,
) -> PersistentHTTPResponse:
    del args, kwargs

    if isinstance(url, urllib.request.Request):
        target_url = url.full_url
        method = url.get_method()
        headers = dict(url.header_items())
        body = url.data if data is None else data
    else:
        target_url = str(url)
        method = "POST" if data is not None else "GET"
        headers = {}
        body = data

    request_timeout = (
        float(timeout)
        if isinstance(timeout, (int, float))
        else 10.0
    )
    parsed = urllib.parse.urlsplit(target_url)
    path = urllib.parse.urlunsplit(
        ("", "", parsed.path or "/", parsed.query, "")
    )

    timing: dict[str, Any] = {
        "http_connection_reused": None,
        "http_connect_seconds": None,
        "http_send_seconds": None,
        "http_wait_headers_seconds": None,
    }
    _thread_local.last_transport_timing = timing

    try:
        connection, reused, connect_seconds = get_connection(
            parsed,
            request_timeout,
        )
        timing["http_connection_reused"] = int(reused)
        timing["http_connect_seconds"] = connect_seconds

        send_start = time.perf_counter()
        connection.request(
            method=method,
            url=path,
            body=body,
            headers=headers,
        )
        timing["http_send_seconds"] = (
            time.perf_counter() - send_start
        )

        headers_start = time.perf_counter()
        response = connection.getresponse()
        timing["http_wait_headers_seconds"] = (
            time.perf_counter() - headers_start
        )

    except (
        OSError,
        http.client.HTTPException,
    ) as exc:
        close_thread_connection()
        raise urllib.error.URLError(exc) from exc

    if response.status >= 400:
        response_body = response.read()
        status = int(response.status)
        reason = str(response.reason or "")
        response_headers = response.headers
        response.close()
        raise urllib.error.HTTPError(
            target_url,
            status,
            reason,
            response_headers,
            io.BytesIO(response_body),
        )

    return PersistentHTTPResponse(response)


def find_new_summary(start_time: float) -> Path | None:
    root: Path | None = None
    for index, arg in enumerate(sys.argv):
        if arg == "--run-dir" and index + 1 < len(sys.argv):
            root = Path(sys.argv[index + 1])
            break
        if arg.startswith("--run-dir="):
            root = Path(arg.split("=", 1)[1])
            break

    if root is None:
        root = Path("results/continuum/runtime_full")
    if not root.is_dir():
        return None

    candidates = [
        path
        for path in root.rglob("summary.json")
        if path.stat().st_mtime >= start_time - 2.0
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda path: path.stat().st_mtime,
    )


def record_mode(summary_path: Path, mode: str) -> None:
    data = json.loads(
        summary_path.read_text(encoding="utf-8")
    )
    experiment = data.setdefault("experiment", {})
    experiment["http_connection_mode"] = mode
    summary_path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (summary_path.parent / "http_connection_mode.txt").write_text(
        mode + "\n",
        encoding="utf-8",
    )


def main() -> None:
    mode = parse_wrapper_args()
    print(f"HTTP connection mode: {mode}")

    import run_envbench_runtime_concurrent as replay

    # Do not alter /metrics polling. It remains on original urllib.
    replay.fetch_text = original_fetch_text

    if mode == "httpclient_per_lane":
        urllib.request.urlopen = httpclient_urlopen

        original_stream_request = replay.stream_request

        def instrumented_stream_request(
            *args: Any,
            **kwargs: Any,
        ) -> dict[str, Any]:
            _thread_local.last_transport_timing = {
                "http_connection_reused": None,
                "http_connect_seconds": None,
                "http_send_seconds": None,
                "http_wait_headers_seconds": None,
            }
            result = original_stream_request(
                *args,
                **kwargs,
            )
            result.update(
                getattr(
                    _thread_local,
                    "last_transport_timing",
                    {},
                )
            )
            return result

        replay.stream_request = instrumented_stream_request

    start_time = time.time()
    try:
        replay.main()
    finally:
        close_thread_connection()
        urllib.request.urlopen = _original_urlopen

    summary_path = find_new_summary(start_time)
    if summary_path is not None:
        record_mode(summary_path, mode)
        print(f"HTTP mode recorded:   {summary_path}")
    else:
        print(
            "Warning: no new summary.json found",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
