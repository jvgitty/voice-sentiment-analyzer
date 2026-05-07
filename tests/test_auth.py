"""Tests for AuthVerifier and CallbackSigner."""

import pytest

from vsa.auth import AuthError, AuthVerifier, CallbackSigner


class TestAuthVerifier:
    def test_rejects_missing_header(self) -> None:
        verifier = AuthVerifier(api_key="right")
        with pytest.raises(AuthError):
            verifier.verify(None)

    def test_rejects_wrong_token(self) -> None:
        verifier = AuthVerifier(api_key="right")
        with pytest.raises(AuthError):
            verifier.verify("Bearer wrong")

    def test_accepts_correct_token(self) -> None:
        verifier = AuthVerifier(api_key="right")
        assert verifier.verify("Bearer right") is None


class TestCallbackSigner:
    def test_sign_produces_verifiable_hmac(self) -> None:
        body = b'{"hello": "world"}'
        secret = "shared-secret-1234567890"
        signature = CallbackSigner.sign(body, secret)
        assert signature.startswith("sha256=")
        assert CallbackSigner.verify(body, secret, signature) is True
