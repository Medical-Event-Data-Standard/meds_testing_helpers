# Static Datasets

For testing purposes, one may also want a set of simple, fully static datasets that can be used reliably
without the cost or variability of generating data over time. In this folder, there are a nested collection of
YAML files that contain such datasets in YAML form.

Currently, the supported datasets are as follows:

## `simple_static_sharded_by_split.yaml`

This is a legacy dataset used in previous testing code. It is currently in use, but has a few issues that make
it likely to be deprecated soon. These include:

- It is sharded by split, rather than relying on the more general `metadata/subject_splits.parquet` file.
- It does not use the proper birth or death codes.

This dataset can also be generated under the generation config `dataset_spec/data_generator: sample`

This dataset does not have any task associated with it, and has an incomplete code metadata file (this is not
an error as code metadata files are not guaranteed to be complete under the MEDS schema).
