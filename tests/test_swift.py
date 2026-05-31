from __future__ import annotations

import httpx
import pytest
import respx

from squawk.models import ObjectStat
from squawk.swift import SwiftClient

ENDPOINT = "https://swift.example/v1/AUTH_test"
CONTAINER = "tartanaviation-audio"


@pytest.fixture
def client() -> SwiftClient:
    return SwiftClient(ENDPOINT)


@respx.mock
def test_list_container_paginates_until_empty_page(client: SwiftClient) -> None:
    page1 = [f"a/{i:05d}.zip" for i in range(3)]
    page2 = [f"b/{i:05d}.zip" for i in range(2)]
    route = respx.get(f"{ENDPOINT}/{CONTAINER}/")
    route.side_effect = [
        httpx.Response(200, text="\n".join(page1) + "\n"),
        httpx.Response(200, text="\n".join(page2) + "\n"),
        httpx.Response(200, text=""),
    ]

    names = client.list_container(CONTAINER)

    assert names == page1 + page2
    assert route.call_count == 3


@respx.mock
def test_list_container_does_not_stop_on_short_nonempty_page(
    client: SwiftClient,
) -> None:
    route = respx.get(f"{ENDPOINT}/{CONTAINER}/")
    route.side_effect = [
        httpx.Response(200, text="only/one.zip\n"),
        httpx.Response(200, text=""),
    ]

    names = client.list_container(CONTAINER)

    assert names == ["only/one.zip"]
    assert route.call_count == 2


@respx.mock
def test_list_container_passes_marker_from_last_name(client: SwiftClient) -> None:
    route = respx.get(f"{ENDPOINT}/{CONTAINER}/")
    route.side_effect = [
        httpx.Response(200, text="x/first.zip\nx/last.zip\n"),
        httpx.Response(200, text=""),
    ]

    client.list_container(CONTAINER)

    assert "marker" not in route.calls[0].request.url.params
    assert route.calls[1].request.url.params["marker"] == "x/last.zip"


@respx.mock
def test_list_container_ignores_blank_lines(client: SwiftClient) -> None:
    route = respx.get(f"{ENDPOINT}/{CONTAINER}/")
    route.side_effect = [
        httpx.Response(200, text="a.zip\n\n  \nb.zip\n"),
        httpx.Response(200, text=""),
    ]

    assert client.list_container(CONTAINER) == ["a.zip", "b.zip"]


@respx.mock
def test_list_container_passes_prefix(client: SwiftClient) -> None:
    route = respx.get(f"{ENDPOINT}/{CONTAINER}/")
    route.side_effect = [
        httpx.Response(200, text="kagc/a.zip\n"),
        httpx.Response(200, text=""),
    ]

    client.list_container(CONTAINER, prefix="kagc/")

    assert route.calls[0].request.url.params["prefix"] == "kagc/"


@respx.mock
def test_stat_parses_size_and_unquoted_etag(client: SwiftClient) -> None:
    respx.head(f"{ENDPOINT}/{CONTAINER}/kagc/a.zip").mock(
        return_value=httpx.Response(
            200,
            headers={
                "Content-Length": "12345",
                "ETag": '"d41d8cd98f00b204e9800998ecf8427e"',
            },
        )
    )

    stat = client.stat(CONTAINER, "kagc/a.zip")

    assert stat == ObjectStat(size=12345, etag="d41d8cd98f00b204e9800998ecf8427e")


@respx.mock
def test_container_totals_parses_bytes_and_count(client: SwiftClient) -> None:
    respx.head(f"{ENDPOINT}/{CONTAINER}/").mock(
        return_value=httpx.Response(
            204,
            headers={"X-Container-Bytes-Used": "999", "X-Container-Object-Count": "7"},
        )
    )

    assert client.container_totals(CONTAINER) == (999, 7)


@respx.mock
def test_get_without_start_sends_no_range_header(client: SwiftClient) -> None:
    route = respx.get(f"{ENDPOINT}/{CONTAINER}/kagc/a.zip").mock(
        return_value=httpx.Response(200, content=b"data")
    )

    response = client.get(CONTAINER, "kagc/a.zip")

    assert response.status_code == 200
    assert "Range" not in route.calls[0].request.headers
    response.close()


@respx.mock
def test_get_with_start_sends_range_header(client: SwiftClient) -> None:
    route = respx.get(f"{ENDPOINT}/{CONTAINER}/kagc/a.zip").mock(
        return_value=httpx.Response(206, content=b"tail")
    )

    response = client.get(CONTAINER, "kagc/a.zip", start=1024)

    assert route.calls[0].request.headers["Range"] == "bytes=1024-"
    response.close()


@respx.mock
def test_get_returns_streaming_response(client: SwiftClient) -> None:
    respx.get(f"{ENDPOINT}/{CONTAINER}/kagc/a.zip").mock(
        return_value=httpx.Response(200, content=b"streamed")
    )

    response = client.get(CONTAINER, "kagc/a.zip")

    assert b"".join(response.iter_raw()) == b"streamed"
    response.close()


def test_injected_client_is_used() -> None:
    transport = httpx.MockTransport(
        lambda request: (
            httpx.Response(200, text="a.zip\n")
            if request.method == "GET"
            else httpx.Response(204, headers={"Content-Length": "1", "ETag": "x"})
        )
    )
    injected = httpx.Client(transport=transport)
    client = SwiftClient(ENDPOINT, client=injected)

    stat = client.stat(CONTAINER, "a.zip")

    assert stat.size == 1
    injected.close()


@respx.mock
def test_list_container_requests_trailing_slash_form(client: SwiftClient) -> None:
    route = respx.get(f"{ENDPOINT}/{CONTAINER}/")
    route.side_effect = [
        httpx.Response(200, text="a.zip\n"),
        httpx.Response(200, text=""),
    ]

    client.list_container(CONTAINER)

    assert route.calls[0].request.url.path.endswith(f"/{CONTAINER}/")


def test_default_client_follows_redirects() -> None:
    assert SwiftClient(ENDPOINT)._client.follow_redirects is True
