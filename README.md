# MEDS Sample Dataset Builder

[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![tests](https://github.com/Medical-Event-Data-Standard/meds_sample_dataset_builder/actions/workflows/tests.yaml/badge.svg)](https://github.com/Medical-Event-Data-Standard/meds_sample_dataset_builder/actions/workflows/test.yaml)
[![codecov](https://codecov.io/gh/Medical-Event-Data-Standard/meds_sample_dataset_builder/branch/main/graph/badge.svg?token=F9NYFEN5FX)](https://codecov.io/gh/Medical-Event-Data-Standard/meds_sample_dataset_builder)
[![code-quality](https://github.com/Medical-Event-Data-Standard/meds_sample_dataset_builder/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/Medical-Event-Data-Standard/meds_sample_dataset_builder/actions/workflows/code-quality-main.yaml)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/Medical-Event-Data-Standard/meds_sample_dataset_builder#license)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Medical-Event-Data-Standard/meds_sample_dataset_builder/pulls)
[![contributors](https://img.shields.io/github/contributors/Medical-Event-Data-Standard/meds_sample_dataset_builder.svg)](https://github.com/Medical-Event-Data-Standard/meds_sample_dataset_builder/graphs/contributors)

Builds a sample MEDS dataset in the form of a dictionary of time deltas between events (as floats), code
vocabulary indices per time delta, numeric values (or NaNs) per time delta, and static codes per subject.

To install, clone this directory and run `pip install sample_dataset_builder/` from the root directory of the
repository (one directory up from where this file is located).

After installation, a dataset can be generated using

```bash
sample_MEDS dataset_spec=mimic N_subjects=500
```

You can use `dataset_spec=sample` for a dataset with fewer events per subject or `dataset_spec=mimic` for a
MIMIC-IV MEDS cohort like dataset. Add `do_overwrite=True` to overwrite an existing dataset. You can see the
full configuration options by running `sample_MEDS --help`.
