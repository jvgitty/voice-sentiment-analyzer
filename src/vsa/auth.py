"""Inbound auth verification and outbound callback signing."""

import hmac


class AuthError(Exception):
    """Raised when an inbound request fails authentication."""


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
