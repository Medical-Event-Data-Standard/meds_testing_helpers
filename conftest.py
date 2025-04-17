"""Manipulate doctest namespace."""

import tempfile
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _setup_doctest_namespace(
    doctest_namespace: dict[str, Any],
) -> None:
    doctest_namespace.update(
        {
            "tempfile": tempfile,
        }
    )
