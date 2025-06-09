# Contributing

Thank you for considering a contribution! This project welcomes pull requests
from everyone. Please read the guidelines below before opening a PR.

## Best practices

- Keep pull requests focused and limited in scope.
- Write clear commit messages.
- Ensure all existing and new tests pass.
- Run `pre-commit` on the files you touched.

## Running checks

Run the linting and formatting hooks with pre-commit:

```bash
pre-commit run --files AGENTS.md CONTRIBUTORS.md
```

Adjust the file paths to match your changes. A full run without the `--files`
argument checks every file.

Run the test suite with doctests enabled:

```bash
pytest
```

The configuration runs doctests across the project and ignores the documentation
files `docs/`, `AGENTS.md` and `CONTRIBUTORS.md`.

## Documentation style

Docstrings should follow the Google style. Where practical, include doctest
examples directly in the docstring so the documentation doubles as tests. Use
`yaml_disk` and `print_directory` to create and display file trees in examples.
These helpers are automatically available in the doctest namespace when the
packages are installed, so explicit imports are unnecessary. If additional
modules are commonly required in doctests, add them in `conftest.py` via the
`doctest_namespace` fixture.
