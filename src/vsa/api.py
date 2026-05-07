"""FastAPI HTTP service entrypoint."""

from fastapi import FastAPI

from vsa.schema import AnalyzeRequest, AnalyzeResult

app = FastAPI(title="Voice Sentiment Analyzer", version="0.1.0")


@app.post("/analyze", response_model=AnalyzeResult)
async def analyze(request: AnalyzeRequest) -> AnalyzeResult:
    raise NotImplementedError("not yet wired")
