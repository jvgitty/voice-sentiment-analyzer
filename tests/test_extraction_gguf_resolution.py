"""Unit tests for the GGUF-path resolution in :mod:`vsa.extraction.llm`.

These cover the two-branch logic that decides whether to use a pre-
baked local checkpoint or fall through to a HuggingFace download:

  * If ``LLM_MODEL_PATH`` points at an existing file, return it verbatim.
  * Otherwise call ``huggingface_hub.hf_hub_download`` with the
    configured repo + filename and return whatever it produces.

We mock ``hf_hub_download`` so these tests never touch the network and
never need a multi-GB GGUF on disk.
"""

from pathlib import Path

import pytest

from vsa.extraction import llm as llm_module


class TestResolveGgufPath:
    def test_returns_local_path_when_file_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When ``LLM_MODEL_PATH`` points at an existing file, use it
        directly — no network call. This is the "Parakeet-style baked
        into the image" path operators take when they pre-download the
        GGUF at build time.

        The mock on ``hf_hub_download`` is a tripwire: if the resolver
        ever falls through to the download path when a local file
        exists, this test will fail loudly."""
        baked_path = tmp_path / "qwen3.5-9b-q4_k_m.gguf"
        baked_path.write_bytes(b"fake gguf bytes")

        def _explode(**_kwargs: object) -> str:
            raise AssertionError(
                "hf_hub_download must not run when the local file exists"
            )

        monkeypatch.setattr(
            "huggingface_hub.hf_hub_download", _explode
        )

        resolved = llm_module._resolve_gguf_path(
            model_path=str(baked_path),
            gguf_repo="bartowski/whatever",
            gguf_file="whatever.gguf",
        )
        assert resolved == str(baked_path)

    def test_downloads_from_hf_when_local_file_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the local path doesn't exist, fall through to
        ``huggingface_hub.hf_hub_download`` and return its result.

        Default deployment shape on Fly: image bakes Parakeet but not
        Qwen, so the first /analyze call hits this branch and pulls
        ~5.5 GB from HuggingFace on demand. Subsequent loads on the
        same Machine hit HF Hub's local cache and skip the network."""
        downloaded_path = tmp_path / "downloaded.gguf"
        downloaded_path.write_bytes(b"fake downloaded bytes")

        called_with: dict[str, str] = {}

        def _fake_download(
            repo_id: str, filename: str, **_kwargs: object
        ) -> str:
            called_with["repo_id"] = repo_id
            called_with["filename"] = filename
            return str(downloaded_path)

        monkeypatch.setattr(
            "huggingface_hub.hf_hub_download", _fake_download
        )

        resolved = llm_module._resolve_gguf_path(
            model_path=str(tmp_path / "does-not-exist.gguf"),
            gguf_repo="bartowski/Qwen_Qwen3.5-9B-Instruct-GGUF",
            gguf_file="Qwen_Qwen3.5-9B-Instruct-Q4_K_M.gguf",
        )

        assert resolved == str(downloaded_path)
        assert called_with["repo_id"] == "bartowski/Qwen_Qwen3.5-9B-Instruct-GGUF"
        assert called_with["filename"] == "Qwen_Qwen3.5-9B-Instruct-Q4_K_M.gguf"

    def test_empty_model_path_falls_through_to_download(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty / unset ``LLM_MODEL_PATH`` is treated the same as
        a missing file — the resolver falls through to HuggingFace
        rather than crashing on the truthy check."""
        downloaded_path = tmp_path / "downloaded.gguf"
        downloaded_path.write_bytes(b"fake")

        monkeypatch.setattr(
            "huggingface_hub.hf_hub_download",
            lambda **_: str(downloaded_path),
        )

        resolved = llm_module._resolve_gguf_path(
            model_path="",
            gguf_repo="any/repo",
            gguf_file="any.gguf",
        )
        assert resolved == str(downloaded_path)


class TestLlmExtractorConfig:
    def test_constructor_reads_env_vars_with_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All tunable env vars flow into the LlmExtractor instance.
        Defaults apply when env vars are unset; explicit values
        override. Constructor is still cheap — no model load, no file
        IO."""
        for var in (
            "LLM_MODEL_PATH",
            "LLM_GGUF_REPO",
            "LLM_GGUF_FILE",
            "LLM_CONTEXT_SIZE",
            "LLM_THREADS",
            "LLM_N_GPU_LAYERS",
        ):
            monkeypatch.delenv(var, raising=False)

        from vsa.extraction.llm import (
            DEFAULT_CONTEXT_SIZE,
            DEFAULT_GGUF_FILE,
            DEFAULT_GGUF_REPO,
            DEFAULT_MODEL_PATH,
            DEFAULT_N_GPU_LAYERS,
            DEFAULT_THREADS,
            LlmExtractor,
        )

        e = LlmExtractor()
        assert e._model_path == DEFAULT_MODEL_PATH
        assert e._gguf_repo == DEFAULT_GGUF_REPO
        assert e._gguf_file == DEFAULT_GGUF_FILE
        assert e._context_size == DEFAULT_CONTEXT_SIZE
        assert e._threads == DEFAULT_THREADS
        assert e._n_gpu_layers == DEFAULT_N_GPU_LAYERS
        # No model loaded yet — construction is cheap.
        assert e._model is None

    def test_env_var_overrides_take_effect(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_MODEL_PATH", "/custom/path.gguf")
        monkeypatch.setenv("LLM_GGUF_REPO", "my-org/my-model")
        monkeypatch.setenv("LLM_GGUF_FILE", "my-quant.gguf")
        monkeypatch.setenv("LLM_CONTEXT_SIZE", "16384")
        monkeypatch.setenv("LLM_THREADS", "4")
        monkeypatch.setenv("LLM_N_GPU_LAYERS", "20")

        from vsa.extraction.llm import LlmExtractor

        e = LlmExtractor()
        assert e._model_path == "/custom/path.gguf"
        assert e._gguf_repo == "my-org/my-model"
        assert e._gguf_file == "my-quant.gguf"
        assert e._context_size == 16384
        assert e._threads == 4
        assert e._n_gpu_layers == 20

    def test_n_gpu_layers_zero_is_cpu_only_signal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An explicit ``LLM_N_GPU_LAYERS=0`` means "force CPU." Useful
        on dev machines without a CUDA build of llama-cpp-python, or
        when debugging GPU vs CPU output divergence."""
        monkeypatch.setenv("LLM_N_GPU_LAYERS", "0")
        from vsa.extraction.llm import LlmExtractor

        e = LlmExtractor()
        assert e._n_gpu_layers == 0

    def test_invalid_int_env_var_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A bogus int env var must not crash extractor construction —
        fall back to the default rather than failing at app startup."""
        monkeypatch.setenv("LLM_CONTEXT_SIZE", "not-a-number")
        monkeypatch.setenv("LLM_THREADS", "")

        from vsa.extraction.llm import (
            DEFAULT_CONTEXT_SIZE,
            DEFAULT_THREADS,
            LlmExtractor,
        )

        e = LlmExtractor()
        assert e._context_size == DEFAULT_CONTEXT_SIZE
        assert e._threads == DEFAULT_THREADS
