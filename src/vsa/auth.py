"""Inbound auth verification and outbound callback signing."""

import hashlib
import hmac


class AuthError(Exception):
    """Raised when an inbound request fails authentication."""


class CallbackSigner:
    """HMAC-SHA256 signing/verification for outbound webhook callbacks."""

    @staticmethod
    def sign(body: bytes, secret: str) -> str:
        digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
        return f"sha256={digest}"

    @staticmethod
    def verify(body: bytes, secret: str, signature: str) -> bool:
        expected = CallbackSigner.sign(body, secret)
        return hmac.compare_digest(expected, signature)


class AuthVerifier:
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def verify(self, auth_header: str | None) -> None:
        if auth_header is None:
            raise AuthError("missing authorization header")
        prefix = "Bearer "
        if not auth_header.startswith(prefix):
            raise AuthError("malformed authorization header")
        token = auth_header[len(prefix):]
        if not hmac.compare_digest(token, self._api_key):
            raise AuthError("invalid api key")
