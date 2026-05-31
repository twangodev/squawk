from __future__ import annotations

import re
import threading
from http import HTTPStatus
from pathlib import Path

import httpx
import pytest
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Request, Response

from squawk.download import (
    FetchResult,
    WriteMode,
    fetch_object,
    run_mirror,
    write_mode,
)
from squawk.models import ObjectStat, RemoteObject

CHUNK = 1 << 20


def _response(status: int) -> httpx.Response:
    return httpx.Response(status_code=status, request=httpx.Request("GET", "http://x"))


@pytest.mark.parametrize(
    ("status", "resuming", "expected"),
    [
        (HTTPStatus.PARTIAL_CONTENT, True, WriteMode.APPEND),
        (HTTPStatus.PARTIAL_CONTENT, False, WriteMode.TRUNCATE),
        (HTTPStatus.OK, True, WriteMode.TRUNCATE),
        (HTTPStatus.OK, False, WriteMode.TRUNCATE),
        (HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, True, WriteMode.TRUNCATE),
    ],
)
def test_write_mode(status: int, resuming: bool, expected: WriteMode) -> None:
    assert write_mode(_response(status), resuming=resuming) is expected


class ServerClient:
    """A ``SwiftClient``-shaped double that streams from a pytest-httpserver instance."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=30)

    def _url(self, container: str, key: str) -> str:
        return f"{self._base_url}/{container}/{key}"

    def stat(self, container: str, key: str) -> ObjectStat:
        response = self._client.head(self._url(container, key))
        response.raise_for_status()
        return ObjectStat(
            size=int(response.headers["Content-Length"]),
            etag=response.headers.get("ETag", "").strip('"'),
        )

    def get(self, container: str, key: str, *, start: int = 0) -> httpx.Response:
        headers = {"Range": f"bytes={start}-"} if start else {}
        request = self._client.build_request(
            "GET", self._url(container, key), headers=headers
        )
        return self._client.send(request, stream=True)


def _head_response(size: int, etag: str = "") -> Response:
    """A Swift-style HEAD: Content-Length declared, no body.

    Werkzeug recomputes Content-Length from the (empty) body unless we opt out,
    which would make every stat report size 0.
    """
    response = Response(status=HTTPStatus.OK)
    response.headers["Content-Length"] = str(size)
    if etag:
        response.headers["ETag"] = f'"{etag}"'
    response.automatically_set_content_length = False
    return response


def _ranged_handler(body: bytes, etag: str = "etag-1") -> object:
    def handle(request: Request) -> Response:
        if request.method == "HEAD":
            return _head_response(len(body), etag)
        range_header = request.headers.get("Range")
        if range_header:
            start = int(re.match(r"bytes=(\d+)-", range_header).group(1))
            tail = body[start:]
            return Response(
                tail,
                status=HTTPStatus.PARTIAL_CONTENT,
                headers={
                    "Content-Length": str(len(tail)),
                    "Content-Range": f"bytes {start}-{len(body) - 1}/{len(body)}",
                },
            )
        return Response(
            body, status=HTTPStatus.OK, headers={"Content-Length": str(len(body))}
        )

    return handle


def _whole_file_handler(body: bytes) -> object:
    """Server that ignores Range and always replies 200 with the entire body."""

    def handle(request: Request) -> Response:
        if request.method == "HEAD":
            return _head_response(len(body))
        return Response(
            body, status=HTTPStatus.OK, headers={"Content-Length": str(len(body))}
        )

    return handle


def _aborting_handler(declared_size: int) -> object:
    """Server that promises ``declared_size`` bytes but sends two chunks then closes."""

    def handle(request: Request) -> Response:
        if request.method == "HEAD":
            return _head_response(declared_size)

        def stream() -> object:
            yield b"A" * CHUNK
            yield b"B" * CHUNK

        return Response(
            stream(),
            status=HTTPStatus.OK,
            headers={"Content-Length": str(declared_size)},
        )

    return handle


def _remote_object(container: str = "c", key: str = "obj.zip") -> RemoteObject:
    return RemoteObject(container=container, key=key, rel_path=Path(key))


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("squawk.download._sleep_backoff", lambda attempt: None)


def test_full_download(httpserver: HTTPServer, tmp_path: Path) -> None:
    body = bytes(range(256)) * 4096
    obj = _remote_object()
    httpserver.expect_request(f"/{obj.container}/{obj.key}").respond_with_handler(
        _ranged_handler(body, etag="full-etag")
    )

    client = ServerClient(httpserver.url_for(""))
    result = fetch_object(client, obj, tmp_path)

    dest = tmp_path / obj.rel_path
    assert result.status == "ok"
    assert result.size == len(body)
    assert result.etag == "full-etag"
    assert dest.read_bytes() == body
    assert not dest.with_name(dest.name + ".part").exists()


def test_resume_from_partial(httpserver: HTTPServer, tmp_path: Path) -> None:
    body = bytes(range(256)) * 4096
    obj = _remote_object()
    dest = tmp_path / obj.rel_path
    part = dest.with_name(dest.name + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    prefix_len = 5000
    part.write_bytes(body[:prefix_len])

    seen_ranges: list[str | None] = []

    def recording_handler(request: Request) -> Response:
        if request.method == "GET":
            seen_ranges.append(request.headers.get("Range"))
        return _ranged_handler(body)(request)

    httpserver.expect_request(f"/{obj.container}/{obj.key}").respond_with_handler(
        recording_handler
    )

    client = ServerClient(httpserver.url_for(""))
    result = fetch_object(client, obj, tmp_path)

    assert result.status == "ok"
    assert dest.read_bytes() == body
    assert not part.exists()
    assert seen_ranges == [f"bytes={prefix_len}-"]


def test_whole_file_200_does_not_corrupt(
    httpserver: HTTPServer, tmp_path: Path
) -> None:
    body = bytes(range(256)) * 4096
    obj = _remote_object()
    dest = tmp_path / obj.rel_path
    part = dest.with_name(dest.name + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    part.write_bytes(body[:5000])

    httpserver.expect_request(f"/{obj.container}/{obj.key}").respond_with_handler(
        _whole_file_handler(body)
    )

    client = ServerClient(httpserver.url_for(""))
    result = fetch_object(client, obj, tmp_path)

    assert result.status == "ok"
    assert dest.read_bytes() == body
    assert not part.exists()


def test_idempotent_second_run_makes_no_get(
    httpserver: HTTPServer, tmp_path: Path
) -> None:
    body = bytes(range(256)) * 256
    obj = _remote_object()
    get_count = {"n": 0}

    def counting_handler(request: Request) -> Response:
        if request.method == "GET":
            get_count["n"] += 1
        return _ranged_handler(body)(request)

    httpserver.expect_request(f"/{obj.container}/{obj.key}").respond_with_handler(
        counting_handler
    )

    client = ServerClient(httpserver.url_for(""))
    first = fetch_object(client, obj, tmp_path)
    assert first.status == "ok"
    assert get_count["n"] == 1

    second = fetch_object(client, obj, tmp_path)
    assert second.status == "skip"
    assert get_count["n"] == 1


def test_aborted_stream_keeps_part_and_no_final(
    httpserver: HTTPServer, tmp_path: Path
) -> None:
    declared = CHUNK * 8
    obj = _remote_object()
    httpserver.expect_request(f"/{obj.container}/{obj.key}").respond_with_handler(
        _aborting_handler(declared)
    )

    client = ServerClient(httpserver.url_for(""))
    result = fetch_object(client, obj, tmp_path)

    dest = tmp_path / obj.rel_path
    part = dest.with_name(dest.name + ".part")
    assert result.status == "fail"
    assert not dest.exists()
    assert part.exists()
    assert part.stat().st_size < declared


def test_run_mirror_one_failure_does_not_abort_batch(
    httpserver: HTTPServer, tmp_path: Path
) -> None:
    good_body = bytes(range(256)) * 256
    good = RemoteObject(container="c", key="good.zip", rel_path=Path("good.zip"))
    bad = RemoteObject(container="c", key="bad.zip", rel_path=Path("bad.zip"))

    httpserver.expect_request("/c/good.zip").respond_with_handler(
        _ranged_handler(good_body)
    )
    httpserver.expect_request("/c/bad.zip").respond_with_handler(
        _aborting_handler(CHUNK * 8)
    )

    client = ServerClient(httpserver.url_for(""))
    seen: list[FetchResult] = []
    lock = threading.Lock()

    def on_result(result: FetchResult) -> None:
        with lock:
            seen.append(result)

    results = run_mirror(
        client, [good, bad], tmp_path, max_workers=2, on_result=on_result
    )

    by_key = {r.key: r.status for r in results}
    assert by_key == {"good.zip": "ok", "bad.zip": "fail"}
    assert {r.key for r in seen} == {"good.zip", "bad.zip"}
    assert (tmp_path / "good.zip").read_bytes() == good_body
