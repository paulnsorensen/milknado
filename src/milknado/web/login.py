# pyright: reportAny=false, reportExplicitAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnannotatedClassAttribute=false, reportUnnecessaryCast=false, reportUnnecessaryIsInstance=false
"""In-memory local browser login token."""

from __future__ import annotations

import hmac
import secrets


class LaunchToken:
    """One process-local token exchanged for the login cookie."""

    cookie_name: str = "milknado_login"

    def __init__(self, value: str | None = None) -> None:
        self.value: str = value or secrets.token_urlsafe(32)

    def verify(self, candidate: str | None) -> bool:
        return bool(candidate) and hmac.compare_digest(self.value, candidate)
