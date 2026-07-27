"""Validated contracts for data crossing the GitHub slice boundary."""

from __future__ import annotations

from typing import Annotated

import msgspec

NonEmptyString = Annotated[str, msgspec.Meta(min_length=1)]


class _GithubModel(msgspec.Struct, frozen=True, kw_only=True):
    pass


class GithubProject(_GithubModel, frozen=True, kw_only=True):
    id: NonEmptyString
    title: str


class GithubItem(_GithubModel, frozen=True, kw_only=True):
    id: NonEmptyString | None = None
    title: str = "(untitled item)"
    body: str = ""
    url: str | None = None


class GithubFieldOption(_GithubModel, frozen=True, kw_only=True):
    id: NonEmptyString
    name: NonEmptyString


class GithubField(_GithubModel, frozen=True, kw_only=True):
    id: NonEmptyString
    name: NonEmptyString
    options: tuple[GithubFieldOption, ...] = ()


class GithubIssue(_GithubModel, frozen=True, kw_only=True):
    title: str
    body: str
    url: NonEmptyString
    number: int | None = None


class GithubBindingConfig(_GithubModel, frozen=True, kw_only=True):
    github_repo: str
    github_project: str | None = None

    def __post_init__(self) -> None:
        if self.github_project is not None:
            owner, separator, number = self.github_project.partition("/")
            if not separator or not owner.strip() or not number.strip().isdigit():
                raise ValueError(
                    f"malformed `github_project` frontmatter: {self.github_project!r}"
                )
            object.__setattr__(self, "github_project", f"{owner.strip()}/{number.strip()}")
        owner, separator, repo = self.github_repo.partition("/")
        if not separator or not owner.strip() or not repo.strip():
            raise ValueError(f"malformed `github_repo` frontmatter: {self.github_repo!r}")
        object.__setattr__(self, "github_repo", f"{owner.strip()}/{repo.strip()}")

    @property
    def project(self) -> tuple[str, int]:
        if self.github_project is None:
            raise ValueError("github_project is required")
        owner, number = self.github_project.split("/", maxsplit=1)
        return owner, int(number)

    @property
    def repo(self) -> tuple[str, str]:
        owner, repo = self.github_repo.split("/", maxsplit=1)
        return owner, repo


def decode_github_binding(raw: object) -> GithubBindingConfig:
    return msgspec.convert(raw, type=GithubBindingConfig, strict=True)
