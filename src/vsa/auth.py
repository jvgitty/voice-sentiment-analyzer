"""Inbound auth verification and outbound callback signing."""


class AuthError(Exception):
    """Raised when an inbound request fails authentication."""


class AuthVerifier:
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def verify(self, auth_header: str | None) -> None:
        if auth_header is None:
            raise AuthError("missing authorization header")
