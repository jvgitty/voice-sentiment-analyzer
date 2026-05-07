"""Download signed-URL audio to /tmp with size and type guards."""

import os
import tempfile
from pathlib import Path

import httpx


class AudioFetchError(Exception):
    """Base error for audio fetch failures."""


class ContentTooLargeError(AudioFetchError):
    """Response declares (or streams) more bytes than the configured max."""


class InvalidContentTypeError(AudioFetchError):
    """Response Content-Type is not in the allowed audio set."""


class AudioFetcher:
    def __init__(self, max_bytes: int, allowed_types: set[str]) -> None:
        self._max_bytes = max_bytes
        self._allowed_types = allowed_types

    async def fetch(self, url: str) -> Path:
        tmp_path: Path | None = None
        try:
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

                    content_type = (
                        response.headers.get("content-type", "")
                        .split(";")[0]
                        .strip()
                        .lower()
                    )
                    if content_type not in self._allowed_types:
                        raise InvalidContentTypeError(
                            f"content-type {content_type!r} not in allowed set"
                        )

                    fd, str_path = tempfile.mkstemp(prefix="vsa-fetch-", suffix=".audio")
                    tmp_path = Path(str_path)
                    bytes_written = 0
                    with os.fdopen(fd, "wb") as f:
                        async for chunk in response.aiter_bytes():
                            bytes_written += len(chunk)
                            if bytes_written > self._max_bytes:
                                raise ContentTooLargeError(
                                    f"streamed bytes exceed max_bytes {self._max_bytes}"
                                )
                            f.write(chunk)

                    return tmp_path
        except BaseException:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()
            raise
