"""FastAPI HTTP service entrypoint."""

import os

from fastapi import Depends, FastAPI, Header, HTTPException

from vsa.auth import AuthError, AuthVerifier
from vsa.schema import AnalyzeRequest, AnalyzeResult

app = FastAPI(title="Voice Sentiment Analyzer", version="0.1.0")


def _verifier() -> AuthVerifier:
    api_key = os.environ.get("API_KEY", "")
    return AuthVerifier(api_key=api_key)


def _check_auth(
    authorization: str | None = Header(default=None),
    verifier: AuthVerifier = Depends(_verifier),
) -> None:
    try:
        verifier.verify(authorization)
    except AuthError as e:
        raise HTTPException(status_code=401, detail=str(e)) from e


@app.post(
    "/analyze",
    response_model=AnalyzeResult,
    dependencies=[Depends(_check_auth)],
)
async def analyze(request: AnalyzeRequest) -> AnalyzeResult:
    raise NotImplementedError("not yet wired")
