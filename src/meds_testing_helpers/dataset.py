import json
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from meds import (
    DatasetMetadata,
    code_metadata_filepath,
    code_metadata_schema,
    data_schema,
    data_subdirectory,
    dataset_metadata_filepath,
    subject_split_schema,
    subject_splits_filepath,
)


class MEDSDataset:
    """A minimal helper class for working with MEDS Datasets intended for use in testing, not production.

    This class is intended to be used for testing and development purposes only. It is not intended to be used
    in production code, and as such has only been optimized for small datasets, and may not support all
    aspects of the MEDS schema.

    Attributes:
        root_dir: The root directory of the dataset, if provided. If specified, data will be read from this
            root directory upon access, not stored in memory. If not provided, the below parameters must be
            provided upon initialization.
        data_shards: A dictionary of data shards, where the keys are the shard names and the values are the
            data tables. Upon access of this attribute, the data will be returned as pyarrow tables. Upon
            specification in the constructor, polars dataframes are expected instead.
        dataset_metadata: The metadata for the dataset, stored as a DatasetMetadata object (which is just a
            type-annotated dictionary).
        code_metadata: The metadata for the codes. Upon access of this attribute, the data will be returned as
            a pyarrow table. Upon specification in the constructor, a polars dataframe is expected instead.
        subject_splits: The subject splits for the dataset. Optional. Upon access of this attribute, the data
            will be returned as a pyarrow table. Upon specification in the constructor, a polars dataframe is
            expected instead. If not specified for an otherwise valid dataset, `None` will be returned.

    Examples:
        >>> data_shards = {
        ...     "0": pl.DataFrame({"subject_id": [0], "time": [0], "numeric_value": [None], "code": ["A"]}),
        ...     "1": pl.DataFrame({"subject_id": [1], "time": [0], "numeric_value": [1.0], "code": ["B"]}),
        ... }
        >>> dataset_metadata = DatasetMetadata(
        ...     dataset_name="test",
        ...     dataset_version="0.0.1",
        ...     etl_name="foo",
        ...     etl_version="0.0.1",
        ...     meds_version="0.3.3",
        ...     created_at="1/1/2025",
        ...     extension_columns=[],
        ... )
        >>> code_metadata = pl.DataFrame({
        ...     "code": ["A", "B"], "description": ["foo", "bar"],
        ...     "parent_codes": pl.Series([None, None], dtype=pl.List(pl.Utf8)),
        ... })
        >>> subject_splits = None
        >>> D = MEDSDataset(
        ...     data_shards=data_shards,
        ...     dataset_metadata=dataset_metadata,
        ...     code_metadata=code_metadata,
        ...     subject_splits=subject_splits
        ... )
        >>> D # doctest: +NORMALIZE_WHITESPACE
        MEDSDataset(data_shards={'0': {'subject_id': [0],
                                       'time': [0],
                                       'numeric_value': [None],
                                       'code': ['A']},
                                 '1': {'subject_id': [1],
                                       'time': [0],
                                       'numeric_value': [1.0],
                                       'code': ['B']}},
                    dataset_metadata={'dataset_name': 'test',
                                      'dataset_version': '0.0.1',
                                      'etl_name': 'foo',
                                      'etl_version': '0.0.1',
                                      'meds_version': '0.3.3',
                                      'created_at': '1/1/2025',
                                      'extension_columns': []},
                    code_metadata={'code': ['A', 'B'],
                                   'description': ['foo', 'bar'],
                                   'parent_codes': [None, None]})
        >>> print(D)
        MEDSDataset:
        dataset_metadata:
          - dataset_name: test
          - dataset_version: 0.0.1
          - etl_name: foo
          - etl_version: 0.0.1
          - meds_version: 0.3.3
          - created_at: 1/1/2025
          - extension_columns: []
        data_shards:
          - 0:
            pyarrow.Table
            subject_id: int64
            time: timestamp[us]
            code: string
            numeric_value: float
            ----
            subject_id: [[0]]
            time: [[1970-01-01 00:00:00.000000]]
            code: [["A"]]
            numeric_value: [[null]]
          - 1:
            pyarrow.Table
            subject_id: int64
            time: timestamp[us]
            code: string
            numeric_value: float
            ----
            subject_id: [[1]]
            time: [[1970-01-01 00:00:00.000000]]
            code: [["B"]]
            numeric_value: [[1]]
        code_metadata:
          pyarrow.Table
          code: string
          description: string
          parent_codes: list<item: string>
            child 0, item: string
          ----
          code: [["A","B"]]
          description: [["foo","bar"]]
          parent_codes: [[null,null]]
        subject_splits: None
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     D2 = D.write(Path(tmpdir))
        ...     assert D == D2
        ...     print(f"repr: {repr(D2).replace(tmpdir, '...')}")
        ...     print(f"str: {str(D2).replace(tmpdir, '...')}")
        repr: MEDSDataset(root_dir=PosixPath('...'))
        str: MEDSDataset:
        stored in root_dir: ...
        dataset_metadata:
          - dataset_name: test
          - dataset_version: 0.0.1
          - etl_name: foo
          - etl_version: 0.0.1
          - meds_version: 0.3.3
          - created_at: 1/1/2025
          - extension_columns: []
        data_shards:
          - 0:
            pyarrow.Table
            subject_id: int64
            time: timestamp[us]
            code: string
            numeric_value: float
            ----
            subject_id: [[0]]
            time: [[1970-01-01 00:00:00.000000]]
            code: [["A"]]
            numeric_value: [[null]]
          - 1:
            pyarrow.Table
            subject_id: int64
            time: timestamp[us]
            code: string
            numeric_value: float
            ----
            subject_id: [[1]]
            time: [[1970-01-01 00:00:00.000000]]
            code: [["B"]]
            numeric_value: [[1]]
        code_metadata:
          pyarrow.Table
          code: string
          description: string
          parent_codes: list<item: string>
            child 0, item: string
          ----
          code: [["A","B"]]
          description: [["foo","bar"]]
          parent_codes: [[null,null]]
        subject_splits: None
    """

    def __init__(
        self,
        root_dir: Path | None = None,
        data_shards: dict[str, pl.DataFrame] | None = None,
        dataset_metadata: DatasetMetadata | None = None,
        code_metadata: pl.DataFrame | None = None,
        subject_splits: pl.DataFrame | None = None,
    ):
        if root_dir is None:
            if data_shards is None:
                raise ValueError("data_shards must be provided if root_dir is None")
            if dataset_metadata is None:
                raise ValueError("dataset_metadata must be provided if root_dir is None")
            if code_metadata is None:
                raise ValueError("code_metadata must be provided if root_dir is None")

        self.root_dir = root_dir
        self.data_shards = data_shards
        self.dataset_metadata = dataset_metadata
        self.code_metadata = code_metadata
        self.subject_splits = subject_splits

        # These will throw errors if the data is malformed.
        self.data_shards
        self.code_metadata
        self.subject_splits
        self.dataset_metadata

    @staticmethod
    def _align_df(df: pl.DataFrame, schema: pa.Schema) -> pa.Table:
        return df.select(schema.names).to_arrow().cast(schema)

    @property
    def dataset_metadata(self) -> DatasetMetadata:
        if self.root_dir is None:
            return self._dataset_metadata
        else:
            dataset_metadata_fp = self.root_dir / dataset_metadata_filepath
            return DatasetMetadata(**json.loads(dataset_metadata_fp.read_text()))

    @dataset_metadata.setter
    def dataset_metadata(self, value: DatasetMetadata | None):
        self._dataset_metadata = value

    def _shard_name(self, data_fp: Path) -> str:
        return data_fp.relative_to(self.root_dir / data_subdirectory).with_suffix("").as_posix()

    @property
    def shard_fps(self) -> list[Path] | None:
        if self.root_dir is None:
            return None
        else:
            return sorted(list((self.root_dir / data_subdirectory).rglob("*.parquet")))

    @property
    def _pl_shards(self) -> dict[str, pl.DataFrame]:
        if self._data_shards is None:
            return {self._shard_name(fp): pl.read_parquet(fp, use_pyarrow=True) for fp in self.shard_fps}
        else:
            return self._data_shards

    @property
    def data_shards(self) -> dict[str, pa.Table]:
        return {shard: self._align_df(df, data_schema()) for shard, df in self._pl_shards.items()}

    @data_shards.setter
    def data_shards(self, value: dict[str, pl.DataFrame] | None):
        self._data_shards = value

    @property
    def _pl_code_metadata(self) -> pl.DataFrame:
        if self._code_metadata is None:
            return pl.read_parquet(self.root_dir / code_metadata_filepath, use_pyarrow=True)
        else:
            return self._code_metadata

    @property
    def code_metadata(self) -> pa.Table:
        return self._align_df(self._pl_code_metadata, code_metadata_schema())

    @code_metadata.setter
    def code_metadata(self, value: pl.DataFrame | None):
        self._code_metadata = value

    @property
    def _pl_subject_splits(self) -> pl.DataFrame:
        if self.root_dir is None:
            return self._subject_splits

        subject_splits_fp = self.root_dir / subject_splits_filepath
        if subject_splits_fp.exists():
            return pl.read_parquet(subject_splits_fp, use_pyarrow=True)
        else:
            return None

    @property
    def subject_splits(self) -> pa.Table | None:
        pl_subject_splits = self._pl_subject_splits
        if pl_subject_splits is None:
            return None
        return self._align_df(pl_subject_splits, subject_split_schema)

    @subject_splits.setter
    def subject_splits(self, value: pl.DataFrame | None):
        self._subject_splits = value

    def write(self, output_dir: Path) -> "MEDSDataset":
        data_dir = output_dir / data_subdirectory
        data_dir.mkdir(parents=True, exist_ok=True)

        for shard, table in self.data_shards.items():
            pq.write_table(table, data_dir / f"{shard}.parquet")

        code_metadata_fp = output_dir / code_metadata_filepath
        code_metadata_fp.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(self.code_metadata, code_metadata_fp)

        dataset_metadata_fp = output_dir / dataset_metadata_filepath
        dataset_metadata_fp.parent.mkdir(parents=True, exist_ok=True)
        dataset_metadata_fp.write_text(json.dumps(self.dataset_metadata))

        if self.subject_splits is not None:
            subject_splits_fp = output_dir / subject_splits_filepath
            subject_splits_fp.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(self.subject_splits, subject_splits_fp)

        return MEDSDataset(root_dir=output_dir)

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        if self.root_dir is None:
            kwargs = {
                "data_shards": {k: v.to_dict(as_series=False) for k, v in self._pl_shards.items()},
                "dataset_metadata": self.dataset_metadata,
                "code_metadata": self._pl_code_metadata.to_dict(as_series=False),
            }
            if self.subject_splits is not None:
                kwargs["subject_splits"] = self._pl_subject_splits.to_dict(as_series=False)
            kwargs_str = ", ".join(f"{k}={repr(v)}" for k, v in kwargs.items())
            return f"{cls_name}({kwargs_str})"
        else:
            return f"{cls_name}(root_dir={repr(self.root_dir)})"

    def __str__(self) -> str:
        lines = []
        lines.append(f"{self.__class__.__name__}:")
        if self.root_dir is not None:
            lines.append(f"stored in root_dir: {str(self.root_dir.resolve())}")
        lines.append("dataset_metadata:")
        for k, v in self.dataset_metadata.items():
            lines.append(f"  - {k}: {v}")
        lines.append("data_shards:")
        for shard, table in self.data_shards.items():
            lines.append(f"  - {shard}:")
            lines.append("    " + str(table).replace("\n", "\n    "))
        lines.append("code_metadata:")
        lines.append("  " + str(self.code_metadata).replace("\n", "\n  "))
        if self.subject_splits is None:
            lines.append("subject_splits: None")
        else:
            lines.append("subject_splits:")
            lines.append("  " + str(self.subject_splits).replace("\n", "\n  "))
        return "\n".join(lines)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, MEDSDataset):
            return False

        if self.data_shards != other.data_shards:
            return False
        if self.dataset_metadata != other.dataset_metadata:
            return False
        if self.code_metadata != other.code_metadata:
            return False
        if self.subject_splits != other.subject_splits:
            return False

        return True
