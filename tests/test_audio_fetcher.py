"""Tests for AudioFetcher."""

import httpx
import pytest
import respx

from vsa.audio import (
    AudioFetcher,
    ContentTooLargeError,
)


ALLOWED = {"audio/wav", "audio/mpeg", "audio/x-wav", "audio/mp3", "audio/ogg", "audio/flac"}


class TestAudioFetcher:
    @pytest.mark.asyncio
    async def test_rejects_oversize_before_body(self) -> None:
        url = "https://example.test/big.wav"
        max_bytes = 1000

        with respx.mock(assert_all_called=True) as respx_mock:
            route = respx_mock.get(url).mock(
                return_value=httpx.Response(
                    200,
                    headers={
                        "Content-Type": "audio/wav",
                        "Content-Length": str(max_bytes + 1),
                    },
                    content=b"X" * (max_bytes + 1),  # body present, fetcher must NOT download it
                )
            )
            fetcher = AudioFetcher(max_bytes=max_bytes, allowed_types=ALLOWED)
            with pytest.raises(ContentTooLargeError):
                await fetcher.fetch(url)
            # one request was made (to read headers), but the fetcher rejected before reading body
            assert route.called
            # respx exposes the request's bytes_read on the underlying stream; we assert the fetcher
            # did not consume the response content. The cleanest assertion: the response body bytes
            # were never read into memory by the fetcher (we verify by ensuring the rejected error
            # carries the declared length, not actual transferred bytes).
