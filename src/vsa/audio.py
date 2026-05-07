"""Download signed-URL audio to /tmp with size and type guards."""

from pathlib import Path

import httpx


class AudioFetchError(Exception):
    """Base error for audio fetch failures."""


class ContentTooLargeError(AudioFetchError):
    """Response declares Content-Length larger than the configured max."""


class InvalidContentTypeError(AudioFetchError):
    """Response Content-Type is not in the allowed audio set."""


class AudioFetcher:
    def __init__(self, max_bytes: int, allowed_types: set[str]) -> None:
        self._max_bytes = max_bytes
        self._allowed_types = allowed_types

    async def fetch(self, url: str) -> Path:
        async with httpx.AsyncClient() as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                length_header = response.headers.get("content-length")
                if length_header is not None:
                    length = int(length_header)
                    if length > self._max_bytes:
                        raise ContentTooLargeError(
                            f"content-length {length} exceeds max_bytes {self._max_bytes}"
                        )
                # Body fetch will be implemented as later tests demand it.
                raise NotImplementedError("body fetch not yet implemented")
