"""FastAPI HTTP service entrypoint."""

import os

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException

from vsa.audio import AudioFetcher
from vsa.auth import AuthError, AuthVerifier, CallbackSigner
from vsa.pipeline import Pipeline
from vsa.schema import AnalyzeRequest, AnalyzeResult, CallbackBody

app = FastAPI(title="Voice Sentiment Analyzer", version="0.1.1")


_ALLOWED_AUDIO_TYPES = {
    "audio/wav",
    "audio/mpeg",
    "audio/x-wav",
    "audio/mp3",
    "audio/ogg",
    "audio/flac",
    # Pixel voice notes and other modern recorders. These containers
    # require ffmpeg-based normalization downstream (see vsa.preprocess);
    # the AudioFetcher accepts them, the Pipeline's preprocessing step
    # transcodes them to 16 kHz mono PCM WAV before the wave reader.
    "audio/mp4",
    "audio/m4a",
    "audio/aac",
    "audio/webm",
    # "x-" prefixed variants. Older servers (and Supabase Storage in
    # particular for m4a) emit these instead of the RFC-registered
    # forms above. Same audio data, different label.
    "audio/x-m4a",
    "audio/x-aac",
    "audio/x-mpeg",
}
_DEFAULT_MAX_AUDIO_BYTES = 50 * 1024 * 1024  # 50MB


def _verifier() -> AuthVerifier:
    return AuthVerifier(api_key=os.environ.get("API_KEY", ""))


def _check_auth(
    authorization: str | None = Header(default=None),
    verifier: AuthVerifier = Depends(_verifier),
) -> None:
    try:
        verifier.verify(authorization)
    except AuthError as e:
        raise HTTPException(status_code=401, detail=str(e)) from e


def _audio_fetcher() -> AudioFetcher:
    max_bytes = int(os.environ.get("MAX_AUDIO_BYTES", _DEFAULT_MAX_AUDIO_BYTES))
    return AudioFetcher(max_bytes=max_bytes, allowed_types=_ALLOWED_AUDIO_TYPES)


def _pipeline() -> Pipeline:
    return Pipeline()


@app.post(
    "/analyze",
    response_model=AnalyzeResult,
    dependencies=[Depends(_check_auth)],
)
async def analyze(
    request: AnalyzeRequest,
    fetcher: AudioFetcher = Depends(_audio_fetcher),
    pipeline: Pipeline = Depends(_pipeline),
) -> AnalyzeResult:
    audio_path = await fetcher.fetch(str(request.audio_url))
    try:
        result = await pipeline.analyze(audio_path)
    finally:
        if audio_path.exists():
            audio_path.unlink(missing_ok=True)

    # Hard transcription failure: the rest of the pipeline produced
    # nothing the supabase callback can render, so we explicitly mark
    # the run as failed rather than emit a "completed" callback with
    # an empty transcript. Downstream consumers (analyze-callback in
    # the supabase project) take a single failure code path on this
    # status — see PRD #1, Q7/Q8 in the design conversation.
    if result.transcription is None:
        error_messages = result.processing.errors or [
            "transcription returned no text"
        ]
        callback = CallbackBody(
            request_id=request.request_id,
            status="failed",
            metadata=request.metadata,
            result=result,
            error={"messages": error_messages},
        )
    else:
        callback = CallbackBody(
            request_id=request.request_id,
            status="completed",
            metadata=request.metadata,
            result=result,
        )
    body_bytes = callback.model_dump_json().encode("utf-8")
    signature = CallbackSigner.sign(body_bytes, request.callback_secret)

    async with httpx.AsyncClient() as client:
        await client.post(
            str(request.callback_url),
            content=body_bytes,
            headers={
                "Content-Type": "application/json",
                "X-Signature-256": signature,
            },
        )

    return result
