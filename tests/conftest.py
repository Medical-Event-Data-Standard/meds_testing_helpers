"""This file ensures that the necessary modules are reloaded for accurate coverage tracking."""

import importlib

import meds_testing_helpers.pytest_plugin
import meds_testing_helpers.static_sample_data

importlib.reload(meds_testing_helpers.pytest_plugin)
importlib.reload(meds_testing_helpers.static_sample_data)
