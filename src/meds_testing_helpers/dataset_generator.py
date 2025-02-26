#!/usr/bin/env python

import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import hydra
import numpy as np
import polars as pl
from annotated_types import Ge, Gt
from meds import DatasetMetadata
from meds import __version__ as meds_version
from meds import (
    code_field,
    description_field,
    held_out_split,
    numeric_value_field,
    parent_codes_field,
    subject_id_field,
    time_field,
    train_split,
    tuning_split,
)
from omegaconf import DictConfig

from . import CONFIG_YAML, __package_name__, __version__
from .dataset import MEDSDataset

logger = logging.getLogger(__name__)


NUM = int | float
POSITIVE_INT = Annotated[int, Ge(0)]
NON_NEGATIVE_NUM = Annotated[NUM, Gt(-1)]


@dataclass
class DiscreteGenerator:
    """A class to generate random numbers from a list of options with given frequencies.

    This is largely just for type safety and to ease specification of the various things that need to be
    sampled to generate a dataset.

    Attributes:
        X: The list of options to sample from.
        freq: The frequency of each option. If None, all options are equally weighted.

    Raises:
        ValueError: If the frequencies are not all positive, the lengths of X and freq are not equal, or no
            options are provided.

    Examples:
        >>> x = DiscreteGenerator([1, 2, 3], [1, 2, 3])
        >>> rng = np.random.default_rng(1)
        >>> x.rvs(10, rng)
        array([3, 3, 1, 3, 2, 2, 3, 2, 3, 1])
        >>> rng = np.random.default_rng(1)
        >>> x.rvs(10, rng)
        array([3, 3, 1, 3, 2, 2, 3, 2, 3, 1])
        >>> x.rvs(10, rng)
        array([3, 3, 2, 3, 2, 2, 1, 2, 2, 2])
        >>> rng = np.random.default_rng(1)
        >>> DiscreteGenerator([1, 2, 3]).rvs(10, rng)
        array([2, 3, 1, 3, 1, 2, 3, 2, 2, 1])
        >>> rng = np.random.default_rng(1)
        >>> DiscreteGenerator(['a', 'b', 'c']).rvs(10, rng)
        array(['b', 'c', 'a', 'c', 'a', 'b', 'c', 'b', 'b', 'a'], dtype='<U1')
        >>> DiscreteGenerator([1, 2], [-1, 1])
        Traceback (most recent call last):
            ...
        ValueError: All frequencies should be positive.
        >>> DiscreteGenerator([1, 2], [1, 2, 3])
        Traceback (most recent call last):
            ...
        ValueError: Equal numbers of frequencies and options must be provided. Got 3 and 2.
        >>> DiscreteGenerator([])
        Traceback (most recent call last):
            ...
        ValueError: At least one option should be provided. Got length 0.
    """

    X: list[Any]
    freq: list[NON_NEGATIVE_NUM] | None = None

    def __post_init__(self):
        if self.freq is None:
            self.freq = [1] * len(self.X)
        if not all(f > 0 for f in self.freq):
            raise ValueError("All frequencies should be positive.")
        if len(self.freq) != len(self.X):
            raise ValueError(
                "Equal numbers of frequencies and options must be provided. "
                f"Got {len(self.freq)} and {len(self.X)}."
            )
        if len(self.freq) == 0:
            raise ValueError("At least one option should be provided. Got length 0.")

    @property
    def p(self) -> np.ndarray:
        return np.array(self.freq) / sum(self.freq)

    def rvs(self, size: int, rng: np.random.Generator) -> np.ndarray:
        return rng.choice(self.X, size=size, p=self.p, replace=True)


class DatetimeGenerator(DiscreteGenerator):
    """A class to generate random datetimes.

    This merely applies type-checking to the DiscreteGenerator class.

    Attributes:
        X: The list of datetimes to sample from.
        freq: The frequency of each option. If None, all options are equally weighted.

    Raises:
        ValueError: In addition to the base class errors, if any of the options are not datetimes.

    Examples:
        >>> rng = np.random.default_rng(1)
        >>> DatetimeGenerator([np.datetime64("2021-01-01"), np.datetime64("2022-02-02")]).rvs(10, rng)
        array(['2022-02-02', '2022-02-02', '2021-01-01', '2022-02-02',
               '2021-01-01', '2021-01-01', '2022-02-02', '2021-01-01',
               '2022-02-02', '2021-01-01'], dtype='datetime64[D]')
    """

    def __post_init__(self):
        if not all(isinstance(x, np.datetime64) for x in self.X):
            raise ValueError("All elements should be datetimes.")
        super().__post_init__()


class ProportionGenerator(DiscreteGenerator):
    """A class to generate random proportions.

    This merely applies type-checking to the DiscreteGenerator class.

    Attributes:
        X: The list of proportions to sample from.
        freq: The frequency of each option. If None, all options are equally weighted.

    Raises:
        ValueError: In addition to the base class errors, if any of the proportions are not numbers between 0
            and 1.

    Examples:
        >>> rng = np.random.default_rng(1)
        >>> ProportionGenerator([0, 1, 0.3]).rvs(10, rng)
        array([1. , 0.3, 0. , 0.3, 0. , 1. , 0.3, 1. , 1. , 0. ])
        >>> ProportionGenerator([1, 2])
        Traceback (most recent call last):
            ...
        ValueError: All elements should be numbers between 0 and 1.
        >>> ProportionGenerator(["a"])
        Traceback (most recent call last):
            ...
        ValueError: All elements should be numbers between 0 and 1.
    """

    def __post_init__(self):
        if not all(isinstance(x, (int, float)) and 0 <= x <= 1 for x in self.X):
            raise ValueError("All elements should be numbers between 0 and 1.")
        super().__post_init__()


class PositiveNumGenerator(DiscreteGenerator):
    """A class to generate random positive numbers.

    This merely applies type-checking to the DiscreteGenerator class.

    Attributes:
        X: The list of positive numbers to sample from.
        freq: The frequency of each option. If None, all options are equally weighted.

    Raises:
        ValueError: In addition to the base class errors, if any of the options are not positive numbers.

    Examples:
        >>> rng = np.random.default_rng(1)
        >>> PositiveNumGenerator([1, 2, 3.0]).rvs(10, rng)
        array([2., 3., 1., 3., 1., 2., 3., 2., 2., 1.])
        >>> PositiveNumGenerator([1, -1, 2])
        Traceback (most recent call last):
            ...
        ValueError: All elements should be positive numbers.
        >>> PositiveNumGenerator([0])
        Traceback (most recent call last):
            ...
        ValueError: All elements should be positive numbers.
        >>> PositiveNumGenerator(["a"])
        Traceback (most recent call last):
            ...
        ValueError: All elements should be positive numbers.
    """

    def __post_init__(self):
        if not all(isinstance(x, (int, float)) and x > 0 for x in self.X):
            raise ValueError("All elements should be positive numbers.")
        super().__post_init__()


class PositiveIntGenerator(PositiveNumGenerator):
    """A class to generate random positive integers.

    This merely applies type-checking to the DiscreteGenerator class.

    Attributes:
        X: The list of positive integers to sample from.
        freq: The frequency of each option. If None, all options are equally weighted.

    Raises:
        ValueError: In addition to the base class errors, if any of the options are not positive integers.

    Examples:
        >>> rng = np.random.default_rng(1)
        >>> PositiveIntGenerator([1, 2, 3]).rvs(10, rng)
        array([2, 3, 1, 3, 1, 2, 3, 2, 2, 1])
        >>> PositiveIntGenerator([0.1])
        Traceback (most recent call last):
            ...
        ValueError: All elements should be integers.
    """

    def __post_init__(self):
        if not all(isinstance(x, int) for x in self.X):
            raise ValueError("All elements should be integers.")
        super().__post_init__()


@dataclass
class MEDSDataDFGenerator:
    """A class to generate whole dataset objects in the form of static and dynamic measurements.

    Attributes:
        num_events_per_subject: A random variable for the number of events (unique timestamps) per subject.
        num_measurements_per_event: A random variable for the number of measurements (codes and values) per
            event.
        num_static_measurements_per_subject: A random variable for the number of static measurements per
            subject.
        frac_code_occurrences_with_value: A random variable for the proportion of occurrences of codes that
            have a value.
        time_between_events_per_subj: A random variable for the time between events for each subject.
        vocab_size: The number of unique codes.
        static_vocab_size: The number of unique static codes.
        start_datetime_per_subject: A random variable for the start datetime per subject.


    Raises:
        ValueError: Various validation errors for the input parameters will raise value errors.

    Examples:
        >>> rng = np.random.default_rng(1)
        >>> kwargs = dict(
        ...     start_datetime_per_subject=DatetimeGenerator([
        ...         np.datetime64("2021-01-01"), np.datetime64("2022-02-02"), np.datetime64("2023-03-03")
        ...     ]),
        ...     num_events_per_subject=PositiveIntGenerator([1, 2, 3]),
        ...     num_measurements_per_event=PositiveIntGenerator([1, 2, 3]),
        ...     num_static_measurements_per_subject=PositiveIntGenerator([2]),
        ...     frac_code_occurrences_with_value=ProportionGenerator([0, 0, 0.9]),
        ...     time_between_events_per_subj=PositiveNumGenerator([1, 2.5, 3]),
        ... )
        >>> DG = MEDSDataDFGenerator(vocab_size=16, static_vocab_size=4, **kwargs)
        >>> X = DG.sample(3, rng)
        >>> X # doctest: +NORMALIZE_WHITESPACE
        shape: (19, 4)
        ┌────────────┬─────────┬─────────────────────┬───────────────┐
        │ subject_id ┆ code    ┆ time                ┆ numeric_value │
        │ ---        ┆ ---     ┆ ---                 ┆ ---           │
        │ i64        ┆ str     ┆ datetime[μs]        ┆ f64           │
        ╞════════════╪═════════╪═════════════════════╪═══════════════╡
        │ 0          ┆ code_1  ┆ null                ┆ null          │
        │ 0          ┆ code_2  ┆ null                ┆ null          │
        │ 0          ┆ code_15 ┆ 2023-03-03 00:00:00 ┆ NaN           │
        │ 0          ┆ code_12 ┆ 2023-03-03 00:00:04 ┆ -0.944752     │
        │ 0          ┆ code_11 ┆ 2023-03-03 00:00:04 ┆ -0.09827      │
        │ …          ┆ …       ┆ …                   ┆ …             │
        │ 2          ┆ code_1  ┆ null                ┆ null          │
        │ 2          ┆ code_0  ┆ null                ┆ null          │
        │ 2          ┆ code_4  ┆ 2022-02-02 00:00:02 ┆ NaN           │
        │ 2          ┆ code_11 ┆ 2022-02-02 00:00:05 ┆ -0.345216     │
        │ 2          ┆ code_13 ┆ 2022-02-02 00:00:05 ┆ -1.481818     │
        └────────────┴─────────┴─────────────────────┴───────────────┘
        >>> MEDSDataDFGenerator(vocab_size="foo", static_vocab_size=4, **kwargs)
        Traceback (most recent call last):
            ...
        ValueError: vocab_size must be an integer.
        >>> MEDSDataDFGenerator(vocab_size=0, static_vocab_size=4, **kwargs)
        Traceback (most recent call last):
            ...
        ValueError: vocab_size must be positive.
        >>> MEDSDataDFGenerator(vocab_size=1, static_vocab_size=3.0, **kwargs)
        Traceback (most recent call last):
            ...
        ValueError: static_vocab_size must be an integer.
        >>> MEDSDataDFGenerator(vocab_size=1, static_vocab_size=-1, **kwargs)
        Traceback (most recent call last):
            ...
        ValueError: static_vocab_size must be positive.
    """

    num_events_per_subject: PositiveIntGenerator
    num_measurements_per_event: PositiveIntGenerator
    num_static_measurements_per_subject: PositiveIntGenerator
    frac_code_occurrences_with_value: ProportionGenerator
    time_between_events_per_subj: PositiveNumGenerator
    vocab_size: POSITIVE_INT
    static_vocab_size: POSITIVE_INT
    start_datetime_per_subject: DatetimeGenerator

    def __post_init__(self):
        if not isinstance(self.vocab_size, int):
            raise ValueError("vocab_size must be an integer.")
        if not isinstance(self.static_vocab_size, int):
            raise ValueError("static_vocab_size must be an integer.")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if self.static_vocab_size <= 0:
            raise ValueError("static_vocab_size must be positive.")

    def sample(self, N_subjects: int, rng: np.random.Generator) -> pl.DataFrame:
        avg_time_between_events_per_subj = self.time_between_events_per_subj.rvs(N_subjects, rng)
        n_static_measurements_per_subject = self.num_static_measurements_per_subject.rvs(N_subjects, rng)
        n_events_per_subject = self.num_events_per_subject.rvs(N_subjects, rng)
        n_measurements_per_event_per_subject = np.split(
            self.num_measurements_per_event.rvs(sum(n_events_per_subject), rng),
            np.cumsum(n_events_per_subject),
        )[:-1]

        dataset = {}
        dataset[subject_id_field] = []
        dataset[code_field] = []
        dataset[time_field] = []
        dataset[numeric_value_field] = []

        codes_value_props = self.frac_code_occurrences_with_value.rvs(self.vocab_size, rng)

        for subject, n_static_measurements, n_measurements_per_event, avg_time_between_events in zip(
            range(N_subjects),
            n_static_measurements_per_subject,
            n_measurements_per_event_per_subject,
            avg_time_between_events_per_subj,
        ):
            static_codes = rng.choice(self.static_vocab_size, size=n_static_measurements)
            dataset[code_field].extend(f"code_{i}" for i in static_codes)
            dataset[subject_id_field].extend([subject] * n_static_measurements)
            dataset[time_field].extend([None] * n_static_measurements)
            dataset[numeric_value_field].extend([None] * n_static_measurements)

            start_datetime = self.start_datetime_per_subject.rvs(1, rng)[0]

            n_events = len(n_measurements_per_event)
            timedeltas = rng.exponential(avg_time_between_events, size=n_events)

            for n, timedelta in zip(n_measurements_per_event, timedeltas):
                codes_obs = rng.choice(self.vocab_size, size=n)
                value_obs_p = codes_value_props[codes_obs]
                value_obs = rng.random(size=n) < value_obs_p
                value_num = rng.normal(size=n)

                values = np.where(value_obs, value_num, np.nan)

                codes = codes_obs + self.static_vocab_size
                dataset[code_field].extend([f"code_{i}" for i in codes])
                dataset[subject_id_field].extend([subject] * n)
                dataset[time_field].extend([start_datetime + np.timedelta64(int(timedelta), "s")] * n)
                dataset[numeric_value_field].extend(values)

        dataset[time_field] = np.array(dataset[time_field], dtype="datetime64[us]")
        return pl.DataFrame(dataset)


@dataclass
class MEDSDatasetGenerator:
    """A class to generate whole MEDS datasets, including core data and metadata.

    Note that these datasets are _not_ meaningful datasets, but rather are just random data for use in testing
    or benchmarking purposes.

    Args:
        data_generator: The data generator to use.
        shard_size: The number of subjects per shard. If None, the dataset will be split into two shards.
        train_frac: The fraction of subjects to use for training.
        tuning_frac: The fraction of subjects to use for tuning. If None, the remaining fraction will be used.
            If both tuning_frac and held_out_frac are None, the remaining fraction will be split evenly
            between the two.
        held_out_frac: The fraction of subjects to use for the held-out set. If None, the remaining fraction
            will be used. If both tuning_frac and held_out_frac are None, the remaining fraction will be split
            evenly between the two.
        dataset_name: The name of the dataset. If None, a default name will be generated.

    Examples:
        >>> rng = np.random.default_rng(1)
        >>> data_df_gen = MEDSDataDFGenerator(
        ...     vocab_size=16,
        ...     static_vocab_size=4,
        ...     start_datetime_per_subject=DatetimeGenerator([
        ...         np.datetime64("2021-01-01"), np.datetime64("2022-02-02"), np.datetime64("2023-03-03")
        ...     ]),
        ...     num_events_per_subject=PositiveIntGenerator([1, 2, 3]),
        ...     num_measurements_per_event=PositiveIntGenerator([1, 2, 3]),
        ...     num_static_measurements_per_subject=PositiveIntGenerator([2]),
        ...     frac_code_occurrences_with_value=ProportionGenerator([0, 0, 0.9]),
        ...     time_between_events_per_subj=PositiveNumGenerator([1, 2.5, 3]),
        ... )
        >>> G = MEDSDatasetGenerator(data_generator=data_df_gen, shard_size=3, dataset_name="MEDS_Sample")
        >>> dataset = G.sample(10, rng)
        >>> for k, v in dataset.dataset_metadata.items(): print(f"{k}: {v}")
        dataset_name: MEDS_Sample
        dataset_version: 0.0.1
        etl_name: meds_testing_helpers
        etl_version: 0.0.1
        meds_version: 0.3.3
        ...
        >>> dataset._pl_code_metadata # This is always empty for now as these codes are meaningless.
        shape: (0, 3)
        ┌──────┬─────────────┬──────────────┐
        │ code ┆ description ┆ parent_codes │
        │ ---  ┆ ---         ┆ ---          │
        │ str  ┆ str         ┆ list[str]    │
        ╞══════╪═════════════╪══════════════╡
        └──────┴─────────────┴──────────────┘
        >>> dataset._pl_subject_splits
        shape: (10, 2)
        ┌────────────┬──────────┐
        │ subject_id ┆ split    │
        │ ---        ┆ ---      │
        │ i64        ┆ str      │
        ╞════════════╪══════════╡
        │ 7          ┆ train    │
        │ 6          ┆ train    │
        │ 1          ┆ train    │
        │ 2          ┆ train    │
        │ 5          ┆ train    │
        │ 3          ┆ train    │
        │ 8          ┆ train    │
        │ 0          ┆ train    │
        │ 4          ┆ tuning   │
        │ 9          ┆ held_out │
        └────────────┴──────────┘
        >>> len(dataset.data_shards)
        3
        >>> dataset._pl_shards["0"]
        shape: (19, 4)
        ┌────────────┬─────────┬─────────────────────┬───────────────┐
        │ subject_id ┆ code    ┆ time                ┆ numeric_value │
        │ ---        ┆ ---     ┆ ---                 ┆ ---           │
        │ i64        ┆ str     ┆ datetime[μs]        ┆ f64           │
        ╞════════════╪═════════╪═════════════════════╪═══════════════╡
        │ 0          ┆ code_1  ┆ null                ┆ null          │
        │ 0          ┆ code_2  ┆ null                ┆ null          │
        │ 0          ┆ code_15 ┆ 2023-03-03 00:00:00 ┆ NaN           │
        │ 0          ┆ code_12 ┆ 2023-03-03 00:00:04 ┆ -0.944752     │
        │ 0          ┆ code_11 ┆ 2023-03-03 00:00:04 ┆ -0.09827      │
        │ …          ┆ …       ┆ …                   ┆ …             │
        │ 2          ┆ code_1  ┆ null                ┆ null          │
        │ 2          ┆ code_0  ┆ null                ┆ null          │
        │ 2          ┆ code_4  ┆ 2022-02-02 00:00:02 ┆ NaN           │
        │ 2          ┆ code_11 ┆ 2022-02-02 00:00:05 ┆ -0.345216     │
        │ 2          ┆ code_13 ┆ 2022-02-02 00:00:05 ┆ -1.481818     │
        └────────────┴─────────┴─────────────────────┴───────────────┘
        >>> dataset._pl_shards["1"]
        shape: (15, 4)
        ┌────────────┬─────────┬─────────────────────┬───────────────┐
        │ subject_id ┆ code    ┆ time                ┆ numeric_value │
        │ ---        ┆ ---     ┆ ---                 ┆ ---           │
        │ i64        ┆ str     ┆ datetime[μs]        ┆ f64           │
        ╞════════════╪═════════╪═════════════════════╪═══════════════╡
        │ 3          ┆ code_2  ┆ null                ┆ null          │
        │ 3          ┆ code_0  ┆ null                ┆ null          │
        │ 3          ┆ code_7  ┆ 2021-01-01 00:00:00 ┆ NaN           │
        │ 3          ┆ code_18 ┆ 2021-01-01 00:00:00 ┆ NaN           │
        │ 4          ┆ code_1  ┆ null                ┆ null          │
        │ …          ┆ …       ┆ …                   ┆ …             │
        │ 4          ┆ code_9  ┆ 2022-02-02 00:00:07 ┆ NaN           │
        │ 5          ┆ code_0  ┆ null                ┆ null          │
        │ 5          ┆ code_3  ┆ null                ┆ null          │
        │ 5          ┆ code_9  ┆ 2023-03-03 00:00:01 ┆ NaN           │
        │ 5          ┆ code_18 ┆ 2023-03-03 00:00:01 ┆ NaN           │
        └────────────┴─────────┴─────────────────────┴───────────────┘
        >>> dataset._pl_shards["2"]
        shape: (26, 4)
        ┌────────────┬─────────┬─────────────────────┬───────────────┐
        │ subject_id ┆ code    ┆ time                ┆ numeric_value │
        │ ---        ┆ ---     ┆ ---                 ┆ ---           │
        │ i64        ┆ str     ┆ datetime[μs]        ┆ f64           │
        ╞════════════╪═════════╪═════════════════════╪═══════════════╡
        │ 6          ┆ code_3  ┆ null                ┆ null          │
        │ 6          ┆ code_1  ┆ null                ┆ null          │
        │ 6          ┆ code_10 ┆ 2023-03-03 00:00:00 ┆ 2.871567      │
        │ 6          ┆ code_4  ┆ 2023-03-03 00:00:00 ┆ -1.554731     │
        │ 6          ┆ code_7  ┆ 2023-03-03 00:00:00 ┆ NaN           │
        │ …          ┆ …       ┆ …                   ┆ …             │
        │ 9          ┆ code_15 ┆ 2023-03-03 00:00:00 ┆ NaN           │
        │ 9          ┆ code_15 ┆ 2023-03-03 00:00:00 ┆ NaN           │
        │ 9          ┆ code_7  ┆ 2023-03-03 00:00:00 ┆ NaN           │
        │ 9          ┆ code_14 ┆ 2023-03-03 00:00:00 ┆ NaN           │
        │ 9          ┆ code_17 ┆ 2023-03-03 00:00:00 ┆ NaN           │
        └────────────┴─────────┴─────────────────────┴───────────────┘
    """

    data_generator: MEDSDataDFGenerator
    shard_size: POSITIVE_INT | None = None
    train_frac: float = 0.8
    tuning_frac: float | None = None
    held_out_frac: float | None = None
    dataset_name: str | None = None

    def __post_init__(self):
        if self.shard_size is not None and self.shard_size <= 0:
            raise ValueError(f"shard_size must be positive; got {self.shard_size}")
        if self.train_frac < 0 or self.train_frac > 1:
            raise ValueError(f"train_frac must be between 0 and 1; got {self.train_frac}")

        if self.tuning_frac is None and self.held_out_frac is None:
            leftover = 1 - self.train_frac
            self.tuning_frac = round(leftover / 2, 4)
            self.held_out_frac = round(leftover / 2, 4)
        elif self.tuning_frac is None:
            self.tuning_frac = 1 - self.train_frac - self.held_out_frac
        elif self.held_out_frac is None:
            self.held_out_frac = 1 - self.train_frac - self.tuning_frac

        if self.tuning_frac < 0 or self.tuning_frac > 1:
            raise ValueError(f"tuning_frac must be between 0 and 1; got {self.tuning_frac}")
        if self.held_out_frac < 0 or self.held_out_frac > 1:
            raise ValueError(f"held_out_frac must be between 0 and 1; got {self.held_out_frac}")

        if self.train_frac + self.tuning_frac + self.held_out_frac != 1:
            raise ValueError(
                "The sum of train_frac, tuning_frac, and held_out_frac must be 1. Got "
                f"{self.train_frac} + {self.tuning_frac} + {self.held_out_frac} = "
                f"{self.train_frac + self.tuning_frac + self.held_out_frac}."
            )

        if self.dataset_name is None:
            self.dataset_name = f"MEDS_Sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def sample(self, N_subjects: int, rng: np.random.Generator) -> MEDSDataset:
        n_shards = N_subjects // self.shard_size if self.shard_size is not None else 2
        subjects_per_shard = N_subjects // n_shards
        shard_sizes = [subjects_per_shard] * (n_shards - 1) + [
            N_subjects - subjects_per_shard * (n_shards - 1)
        ]

        data_shards = {}
        total_subjects = 0
        for i, size in enumerate(shard_sizes):
            data_shards[str(i)] = self.data_generator.sample(size, rng).with_columns(
                (pl.col(subject_id_field) + total_subjects).alias(subject_id_field)
            )
            total_subjects += size

        dataset_metadata = DatasetMetadata(
            dataset_name=self.dataset_name,
            dataset_version="0.0.1",
            etl_name=__package_name__,
            etl_version=__version__,
            meds_version=meds_version,
            created_at=datetime.now(),
            extension_columns=[],
        )

        code_metadata = pl.DataFrame(
            {
                code_field: pl.Series([], dtype=pl.Utf8),
                description_field: pl.Series([], dtype=pl.Utf8),
                parent_codes_field: pl.Series([], dtype=pl.List(pl.Utf8)),
            }
        )

        subjects = list(range(N_subjects))
        rng.shuffle(subjects)
        N_train = int(N_subjects * self.train_frac)
        N_tuning = int(N_subjects * self.tuning_frac)
        N_held_out = N_subjects - N_train - N_tuning

        split = [train_split] * N_train + [tuning_split] * N_tuning + [held_out_split] * N_held_out
        subject_splits = pl.DataFrame(
            {
                subject_id_field: pl.Series(subjects, dtype=pl.Int64),
                "split": pl.Series(split, dtype=pl.Utf8),
            }
        )

        return MEDSDataset(
            data_shards=data_shards,
            dataset_metadata=dataset_metadata,
            code_metadata=code_metadata,
            subject_splits=subject_splits,
        )


@hydra.main(version_base=None, config_path=str(CONFIG_YAML.parent), config_name=CONFIG_YAML.stem)
def main(cfg: DictConfig):
    """Generate a dataset of the specified size."""

    cfg = hydra.utils.instantiate(cfg)

    output_dir = Path(cfg.output_dir)

    if output_dir.exists():
        if output_dir.is_file():
            raise ValueError("Output directory is a file; expected a directory.")
        if cfg.do_overwrite:
            logger.warning("Output directory already exists. Overwriting.")
            shutil.rmtree(output_dir)
        elif (output_dir / "data").exists() or (output_dir / "metadata").exists():
            contents = [f"  - {p.relative_to(output_dir)}" for p in output_dir.rglob("*")]
            contents_str = "\n".join(contents)
            raise ValueError(
                f"Output directory is not empty! use --do-overwrite to overwrite. Contents:\n{contents_str}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    G = hydra.utils.instantiate(cfg.dataset_spec)
    rng = np.random.default_rng(cfg.seed)

    logger.info(f"Generating dataset with {cfg.N_subjects} subjects.")
    dataset = G.sample(cfg.N_subjects, rng)

    logger.info(f"Saving dataset to root directory {str(output_dir.resolve())}.")
    dataset.write(output_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
