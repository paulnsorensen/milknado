# Contributing to milknado

Thanks for your interest. Contributions of all sizes are welcome.

## Filing issues

- Search [open issues](https://github.com/paulnsorensen/milknado/issues)
  before opening a new one.
- Use the bug-report or feature-request template when it fits.
- For security vulnerabilities, do **not** open a public issue — see
  [`SECURITY.md`](./SECURITY.md).

## Setting up locally

```sh
git clone https://github.com/paulnsorensen/milknado.git
cd milknado
just install
```

## Running checks

```sh
just build
```

`just build` runs lint autofixes, the full test suite, and the coverage
gate used before opening a PR.

## Submitting a pull request

1. Create a topic branch from `main`.
2. Keep commits focused and descriptive.
3. Use a [Conventional Commits](https://www.conventionalcommits.org)
   style PR title.
4. Fill out the pull request template.
5. Wait for CI to go green and address review feedback.

## Code of Conduct

Participation in this project is governed by the
[Contributor Covenant](./CODE_OF_CONDUCT.md). By contributing you
agree to abide by it.

## Licensing

By submitting a contribution you agree that it will be licensed under
the same terms as the project itself (see [`LICENSE`](./LICENSE)).
