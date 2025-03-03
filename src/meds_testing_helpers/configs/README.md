# Hydra Configs for CLI Applications

This directory contains the [Hydra](https://hydra.cc/) configuration files for the CLI applications in this
package:

- [`generate_dataset.yaml`](generate_dataset.yaml) for the `build_sample_MEDS_dataset` CLI application.
- [`infer_dataset_config.yaml`](infer_dataset_config.yaml) for the `infer_MEDS_sample_gen_config` CLI
    application.

The [`dataset_spec/data_generator`](dataset_spec/data_generator) directory contains the dataset specification
files for different generation configurations and can be sub-selected through the normal Hydra means.
