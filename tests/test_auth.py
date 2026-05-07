"""Tests for AuthVerifier and CallbackSigner."""

import pytest

from vsa.auth import AuthError, AuthVerifier


class TestAuthVerifier:
    def test_rejects_missing_header(self) -> None:
        verifier = AuthVerifier(api_key="right")
        with pytest.raises(AuthError):
            verifier.verify(None)
