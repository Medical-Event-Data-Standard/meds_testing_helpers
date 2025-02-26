# MEDS Testing Helpers

[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![tests](https://github.com/Medical-Event-Data-Standard/meds_testing_helpers/actions/workflows/tests.yaml/badge.svg)](https://github.com/Medical-Event-Data-Standard/meds_testing_helpers/actions/workflows/test.yaml)
[![codecov](https://codecov.io/gh/Medical-Event-Data-Standard/meds_testing_helpers/branch/main/graph/badge.svg?token=F9NYFEN5FX)](https://codecov.io/gh/Medical-Event-Data-Standard/meds_testing_helpers)
[![code-quality](https://github.com/Medical-Event-Data-Standard/meds_testing_helpers/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/Medical-Event-Data-Standard/meds_testing_helpers/actions/workflows/code-quality-main.yaml)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/Medical-Event-Data-Standard/meds_testing_helpers#license)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Medical-Event-Data-Standard/meds_testing_helpers/pulls)
[![contributors](https://img.shields.io/github/contributors/Medical-Event-Data-Standard/meds_testing_helpers.svg)](https://github.com/Medical-Event-Data-Standard/meds_testing_helpers/graphs/contributors)

Provides various utilities for testing and benchmarking MEDS packages and tools, including pytest helpers,
fixtures, sample datasets, and capabilities to build larger sample datasets for benchmarking purposes.

## Testing Helpers

TODO

## Building Sample Datasets

Builds a sample MEDS dataset in the form of a dictionary of time deltas between events (as floats), code
vocabulary indices per time delta, numeric values (or NaNs) per time delta, and static codes per subject.

To install, run `pip install TODO`.

After installation, a dataset can be generated using

```bash
build_sample_MEDS_dataset dataset_spec=mimic N_subjects=500
```

You can use `dataset_spec=sample` for a dataset with fewer events per subject or `dataset_spec=mimic` for a
MIMIC-IV MEDS cohort like dataset. Add `do_overwrite=True` to overwrite an existing dataset. You can see the
full configuration options by running `sample_MEDS --help`.
