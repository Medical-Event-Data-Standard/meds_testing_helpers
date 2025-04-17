"""Manipulate doctest namespace."""

import importlib
import tempfile
from typing import Any

import pytest

import meds_testing_helpers.pytest_plugin
import meds_testing_helpers.static_sample_data

importlib.reload(meds_testing_helpers.pytest_plugin)
importlib.reload(meds_testing_helpers.static_sample_data)


@pytest.fixture(autouse=True)
def _setup_doctest_namespace(doctest_namespace: dict[str, Any]) -> None:
    doctest_namespace.update({"tempfile": tempfile})
