"""Validated contracts for data crossing the GitHub slice boundary."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator

NonEmptyString = Annotated[StrictStr, Field(min_length=1)]


class _GithubModel(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class GithubProject(_GithubModel):
    id: NonEmptyString
    title: StrictStr


class GithubItem(_GithubModel):
    id: NonEmptyString | None = None
    title: StrictStr = "(untitled item)"
    body: StrictStr = ""
    url: StrictStr | None = None


class GithubFieldOption(_GithubModel):
    id: NonEmptyString
    name: NonEmptyString


class GithubField(_GithubModel):
    id: NonEmptyString
    name: NonEmptyString
    options: tuple[GithubFieldOption, ...] = ()


class GithubIssue(_GithubModel):
    title: StrictStr
    body: StrictStr
    number: StrictInt | None = None
    url: NonEmptyString


class GithubBindingConfig(_GithubModel):
    github_project: StrictStr | None = None
    github_repo: StrictStr

    @field_validator("github_project")
    @classmethod
    def validate_project(cls, value: str | None) -> str | None:
        if value is None:
            return None
        owner, separator, number = value.partition("/")
        if not separator or not owner.strip() or not number.strip().isdigit():
            raise ValueError(f"malformed `github_project` frontmatter: {value!r}")
        return f"{owner.strip()}/{number.strip()}"

    @field_validator("github_repo")
    @classmethod
    def validate_repo(cls, value: str) -> str:
        owner, separator, repo = value.partition("/")
        if not separator or not owner.strip() or not repo.strip():
            raise ValueError(f"malformed `github_repo` frontmatter: {value!r}")
        return f"{owner.strip()}/{repo.strip()}"

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
