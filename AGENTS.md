# Contributor Guidance

This project follows a few conventions to keep contributions consistent and easy
for others to pick up.

## Development workflow

1. **Formatting and linting**: run `pre-commit` before pushing any changes.
    The repository uses Ruff, docformatter, prettier for YAML, codespell and
    other hooks from `.pre-commit-config.yaml`. Run only on the files you have
    changed using:

    ```bash
    pre-commit run --files path/to/file1 path/to/file2
    ```

    Running with no arguments will check all files.

2. **Tests**: use `pytest` which includes doctest runs. Example:

    ```bash
    pytest
    ```

    Pytest is configured to ignore `docs/`, `AGENTS.md` and `CONTRIBUTORS.md`
    during doctest collection. Doctests are preferred over separate unit tests
    whenever possible so the examples serve as documentation and tests.

3. **Documentation**: use Google style docstrings and keep examples runnable.
    When file system layouts are required use `yaml_disk` and `print_directory`
    to create data structures and display them. These helpers are available in
    doctests automatically when the packages are installed.

4. **Shared doctest imports**: common utilities can be added to the doctest
    namespace via the `doctest_namespace` fixture in `conftest.py`. This avoids
    repetitious imports in doctest blocks and keeps examples concise.

## Examples

Create some files with `yaml_disk` and show them with `print_directory`:

```python
>>> contents = """
... foo:
...   bar.txt: Hello
... """
>>> with yaml_disk(contents) as root:
...     print_directory(root)
├── foo
│   └── bar.txt
```

`yaml_disk` writes the directory structure and `print_directory` prints a tree
for readability.
