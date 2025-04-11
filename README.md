# MEDS Testing Helpers

[![PyPI - Version](https://img.shields.io/pypi/v/meds_testing_helpers)](https://pypi.org/project/meds_testing_helpers/)
[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Documentation Status](https://readthedocs.org/projects/meds-testing-helpers/badge/?version=latest)](https://meds-testing-helpers.readthedocs.io/en/latest/?badge=latest)
[![tests](https://github.com/Medical-Event-Data-Standard/meds_testing_helpers/actions/workflows/tests.yaml/badge.svg)](https://github.com/Medical-Event-Data-Standard/meds_testing_helpers/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/Medical-Event-Data-Standard/meds_testing_helpers/branch/main/graph/badge.svg?token=F9NYFEN5FX)](https://codecov.io/gh/Medical-Event-Data-Standard/meds_testing_helpers)
[![code-quality](https://github.com/Medical-Event-Data-Standard/meds_testing_helpers/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/Medical-Event-Data-Standard/meds_testing_helpers/actions/workflows/code-quality-main.yaml)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/Medical-Event-Data-Standard/meds_testing_helpers#license)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Medical-Event-Data-Standard/meds_testing_helpers/pulls)
[![contributors](https://img.shields.io/github/contributors/Medical-Event-Data-Standard/meds_testing_helpers.svg)](https://github.com/Medical-Event-Data-Standard/meds_testing_helpers/graphs/contributors)

Provides various utilities for testing and benchmarking MEDS packages and tools, including pytest helpers,
fixtures, sample datasets, and capabilities to build larger sample datasets for benchmarking purposes.

## Installation

```bash
pip install meds_testing_helpers
```

## Testing Helpers

After installing this package via pip, you can use the provided pytest fixtures and helpers to test your MEDS
pipelines and tools. These include fixtures for static datasets shipped with this package and fixtures that
generate larger datasets on the fly for benchmarking purposes.

### Simple Static Dataset

You can use the fixture
[`simple_static_MEDS`](src/meds_testing_helpers/static_sample_data/simple_static_sharded_by_split.yaml) to
access a simple, static dataset that is sharded by split (e.g., shard names are of the form `train/0.parquet`.
To use this fixture, simply add it as an argument to your test function in pytest:

```python
# test_my_pipeline.py


def test_my_pipeline(simple_static_MEDS):
    # The simple static dataset will be stored on disk in a temporary directory in a path given by
    # the `simple_static_MEDS` input variable.
    pass
```

Note that you can also import this static dataset directly in yaml form then convert it to a MEDS dataset in a
simple object-oriented format that can be written to disk via:

```python
from meds_testing_helpers.static_sample_data import SIMPLE_STATIC_SHARDED_BY_SPLIT
from meds_testing_helpers.dataset import MEDSDataset

data = MEDSDataset.from_yaml(SIMPLE_STATIC_SHARDED_BY_SPLIT)
data.write(...)
```

### Simple Static Dataset with Tasks

You can use the fixture
[`simple_static_MEDS_with_task`](src/meds_testing_helpers/static_sample_data/simple_static_sharded_by_split_with_tasks.yaml)
to access a dataset that is identical to the `simple_static_MEDS` dataset, but augmented with a prediction
task named `boolean_value_task` that has a boolean label. _Note that this formulation of including tasks
relies on file storage conventions that are not mandated within MEDS; namely that tasks are stored in a
`task_labels` subdirectory of the raw dataset directory._

### Generated Datasets (useful for benchmarking)

You can use the fixture `generated_sample_MEDS` to generate a sample dataset that is similar to the static
dataset discussed above dynamically with a controllable number of patients (controlled via the pytest argument
`--generated-dataset-N`). This dataset is generated on the fly and is not stored on disk, so will take some
time to generate depending on the number of patients. The dataset is generated according to the
[relevant configuration file](src/meds_testing_helpers/configs/dataset_spec/data_generator). Over time, more
configs and data generation specifications will be added. Like the static datasets, as a pytest fixture this
can be accessed via a temporary path:

```python
# test_my_pipeline.py


def test_my_pipeline(generated_sample_MEDS):
    # The generated dataset will be stored on disk in a temporary directory in a path given by
    # the `generated_sample_MEDS` input variable.
    pass
```

You can also control the seed of the generation process via the pytest argument `--generated-dataset-seed`.

## Building Sample Datasets

This package also contains an executable to generate sample MEDS datasets and store them to disk. It is this
command that backs the `generated_sample_MEDS` pytest fixtures. This CLI tool uses hydra to manage
configuration options and generate datasets according to the configuration. You can run the command as
follows:

```bash
build_sample_MEDS_dataset dataset_spec=sample N_subjects=500
```

Add `do_overwrite=True` to overwrite an existing dataset. You can see the full configuration options by
running `sample_MEDS --help`.

## Inferring Dataset Generation Configs

To generate a dataset similar to a local dataset on disk, you can also use the `infer_MEDS_sample_gen_config`
CLI command to infer the dataset generation configuration from a local dataset. This command will output a
yaml file to a specified path on disk that can be used as input to the `build_sample_MEDS_dataset` command to
generate a dataset similar to the local dataset along some limited axes. There is not a clean way currently to
use the yaml file on disk other than making a new `dataset_spec` configuration file and referencing that
directly via Hydra.

```bash
infer_MEDS_sample_gen_config dataset_dir=/path/to/local/dataset output_fp=/path/to/output.yaml
```
