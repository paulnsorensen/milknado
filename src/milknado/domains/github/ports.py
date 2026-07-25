"""Outbound GitHub Projects contract owned by the GitHub domain."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from milknado.domains.github.models import GithubField, GithubItem, GithubProject


class GithubProjectPort(Protocol):
    def preflight(self) -> None: ...

    def project_view(self, owner: str, number: int) -> GithubProject: ...

    def item_list(self, owner: str, number: int) -> list[GithubItem]: ...

    def item_add(self, owner: str, number: int, issue_url: str) -> str: ...

    def item_edit(
        self,
        project_id: str,
        item_id: str,
        field_id: str,
        *,
        single_select_option_id: str | None = None,
        text: str | None = None,
    ) -> None: ...

    def field_list(self, owner: str, number: int) -> list[GithubField]: ...

    def field_create(self, number: int, owner: str, name: str, options: Sequence[str]) -> str: ...

    def field_create_text(self, number: int, owner: str, name: str) -> str: ...

    def issue_create(self, owner: str, repo: str, title: str, body: str) -> str: ...

    def issue_edit_body(self, issue_url: str, body: str) -> None: ...
