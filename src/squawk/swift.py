from __future__ import annotations

import httpx

from squawk.models import ObjectStat

LIST_TIMEOUT = httpx.Timeout(120.0)
STAT_TIMEOUT = httpx.Timeout(60.0)
GET_TIMEOUT = httpx.Timeout(connect=30.0, read=600.0, write=600.0, pool=600.0)


class SwiftClient:
    """Swift v1 REST access for one storage endpoint.

    Listing uses the plaintext, marker-paginated object index. boto3's S3
    ListObjects misbehaves against this endpoint, so the native Swift REST
    listing is the authoritative enumeration.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.Client | None = None,
        verify: bool | str = True,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._client = (
            client
            if client is not None
            else httpx.Client(verify=verify, follow_redirects=True)
        )

    def list_container(self, container: str, prefix: str = "") -> list[str]:
        names: list[str] = []
        marker = ""
        while True:
            params: dict[str, str] = {}
            if prefix:
                params["prefix"] = prefix
            if marker:
                params["marker"] = marker
            response = self._client.get(
                self._url(container), params=params, timeout=LIST_TIMEOUT
            )
            response.raise_for_status()
            page = [line for line in response.text.splitlines() if line.strip()]
            if not page:
                return names
            names += page
            marker = page[-1]

    def stat(self, container: str, key: str) -> ObjectStat:
        response = self._client.head(self._url(container, key), timeout=STAT_TIMEOUT)
        response.raise_for_status()
        return ObjectStat(
            size=int(response.headers.get("Content-Length", 0)),
            etag=response.headers.get("ETag", "").strip('"'),
        )

    def container_totals(self, container: str) -> tuple[int, int]:
        response = self._client.head(self._url(container), timeout=STAT_TIMEOUT)
        response.raise_for_status()
        return (
            int(response.headers["X-Container-Bytes-Used"]),
            int(response.headers["X-Container-Object-Count"]),
        )

    def get(self, container: str, key: str, *, start: int = 0) -> httpx.Response:
        headers = {"Range": f"bytes={start}-"} if start else {}
        request = self._client.build_request(
            "GET", self._url(container, key), headers=headers, timeout=GET_TIMEOUT
        )
        return self._client.send(request, stream=True)

    def _url(self, container: str, key: str = "") -> str:
        # The bare container path 301-redirects to the trailing-slash form; request it
        # directly so pagination never loses the marker query to a redirect.
        base = f"{self._endpoint}/{container}"
        return f"{base}/{key}" if key else f"{base}/"
