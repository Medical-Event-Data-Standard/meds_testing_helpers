#!/usr/bin/env python

import dataclasses
import json
import logging
from pathlib import Path

import hydra
import numpy as np
import polars as pl
import polars.selectors as cs
from meds import (
    birth_code,
    code_field,
    dataset_metadata_filepath,
    death_code,
    held_out_split,
    subject_id_field,
    subject_splits_filepath,
    time_field,
    train_split,
    tuning_split,
)
from omegaconf import DictConfig
from yaml import dump as dump_yaml

try:
    from yaml import CDumper as Dumper
except ImportError:
    from yaml import Dumper

from . import INF_YAML
from .dataset_generator import (
    DatetimeGenerator,
    MEDSDataDFGenerator,
    MEDSDatasetGenerator,
    PositiveIntGenerator,
    PositiveTimeDeltaGenerator,
    ProportionGenerator,
)

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=str(INF_YAML.parent), config_name=INF_YAML.stem)
def main(cfg: DictConfig):
    """Infers the configuration parameters that would generate a dataset similar to the input."""

    output_fp = Path(cfg.output_fp)
    if output_fp.exists():
        if cfg.do_overwrite:
            logger.info(f"Overwriting existing file {output_fp}.")
        else:
            raise FileExistsError(f"Output file {output_fp} already exists.")

    data_dir = Path(cfg.dataset_dir) / "data"
    metadata_dir = Path(cfg.dataset_dir) / "metadata"

    shards = list(data_dir.rglob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No shards found in {data_dir}")

    dataset_metadata_fp = metadata_dir / dataset_metadata_filepath
    if not dataset_metadata_fp.exists():
        logger.warning(f"Dataset metadata file not found at {dataset_metadata_fp}!")
        dataset_name = "UNKNOWN"
    else:
        dataset_metadata = json.loads(dataset_metadata_fp.read_text())
        dataset_name = f"{dataset_metadata['dataset_name']}/SYNTHETIC"

    subject_splits_fp = metadata_dir / subject_splits_filepath
    if not subject_splits_fp.exists():
        train_frac = None
        tuning_frac = None
        held_out_frac = None
    else:
        subject_splits = pl.read_parquet(subject_splits_fp)
        split_cnts = subject_splits.group_by("split").agg(pl.col(subject_id_field).count().alias("count"))
        train_cnt = split_cnts.filter(pl.col("split") == train_split).select("count").first().item()
        tuning_cnt = split_cnts.filter(pl.col("split") == tuning_split).select("count").first().item()
        held_out_cnt = split_cnts.filter(pl.col("split") == held_out_split).select("count").first().item()

        total_cnt = train_cnt + tuning_cnt + held_out_cnt
        train_frac = train_cnt / total_cnt
        tuning_frac = tuning_cnt / total_cnt
        held_out_frac = 1 - train_frac - tuning_frac

    if cfg.seed is not None:
        rng = np.random.default_rng(cfg.seed)
    else:
        rng = np.random.default_rng()

    shards_to_examine = rng.shuffle(shards)[:3]

    code_col = pl.col(code_field)
    time_col = pl.col(time_field)
    is_static = time_col.is_null()
    is_dynamic = time_col.is_not_null()
    is_birth = code_col.str.starts_with(birth_code)
    is_death = code_col.str.starts_with(death_code)
    is_dynamic_data = is_dynamic & ~is_birth & ~is_death
    numerics_present = (
        pl.col("numeric_value").is_not_null()
        & pl.col("numeric_value").is_finite()
        & pl.col("numeric_value").is_not_nan()
    )

    static_codes = set()
    dynamic_codes = set()
    birth_codes = set()
    death_codes = set()
    birth_times = []
    death_times = []
    start_of_data_times = np.array([])
    num_events_per_subject = np.array([])
    num_measurements_per_event = np.array([])
    num_static_measurements_per_subject = np.array([])
    frac_subjects_with_birth = np.array([])
    frac_subjects_with_death = np.array([])
    n_subjects = np.array([])
    time_between_data_events = np.array([])
    time_between_birth_and_data = np.array([])
    time_between_data_and_death = np.array([])
    frac_dynamic_code_occurrences_with_values = []
    frac_static_code_occurrences_with_values = []

    for i, shard in enumerate(shards_to_examine):
        logger.info(f"Examining shard {shard}")
        df = pl.scan_parquet(shard)

        static_codes |= set(df.filter(is_static).select(code_col.unique()).collect())
        dynamic_codes |= set(df.filter(is_dynamic_data).select(code_col.unique()).collect())
        birth_codes |= set(df.filter(is_birth).select(code_col.unique()).collect())
        death_codes |= set(df.filter(is_death).select(code_col.unique()).collect())

        subject_stats = (
            df.group_by(subject_id_field)
            .agg(
                pl.when(is_birth).then(time_col).min().alias("birth_time"),
                pl.when(is_death).then(time_col).max().alias("death_time"),
                is_birth.any().alias("has_birth"),
                is_death.any().alias("has_death"),
                is_static.sum().alias("n_static_measurements"),
            )
            .collect()
        )

        birth_times.extend(subject_stats.select("birth_time").drop_nulls())
        death_times.extend(subject_stats.select("death_time").drop_nulls())
        n_subjects.append(len(subject_stats))
        frac_subjects_with_birth.append(subject_stats.select("has_birth").mean().item())
        frac_subjects_with_death.append(subject_stats.select("has_death").mean().item())
        num_static_measurements_per_subject.extend(subject_stats.select("n_static_measurements").drop_nulls())

        dynamic_df = df.filter(is_dynamic_data)

        subject_dynamic_stats = (
            dynamic_df.group_by(subject_id_field)
            .agg(
                time_col.min().alias("first_data_time"),
                time_col.max().alias("last_data_time"),
                time_col.n_unique().alias("n_events"),
                time_col.unique(maintain_order=True).diff().mean().alias("time_between_events"),
            )
            .collect()
        )

        start_of_data_times.extend(subject_dynamic_stats.select("first_data_time").drop_nulls())
        num_events_per_subject.extend(subject_dynamic_stats.select("n_events").drop_nulls())
        time_between_data_events.extend(subject_dynamic_stats.select("time_between_events").drop_nulls())

        boundary_deltas = subject_dynamic_stats.join(subject_stats, on=subject_id_field).select(
            (pl.col("first_data_time") - pl.col("birth_time"))
            .alias("time_between_birth_and_data")(pl.col("death_time") - pl.col("last_data_time"))
            .alias("time_between_data_and_death"),
        )

        time_between_birth_and_data.extend(boundary_deltas.select("time_between_birth_and_data").drop_nulls())
        time_between_data_and_death.extend(boundary_deltas.select("time_between_data_and_death").drop_nulls())

        num_measurements_per_event.extend(
            df.filter(is_dynamic_data).group_by(subject_id_field, time_col).count().collect()
        )
        frac_dynamic_code_occurrences_with_values.append(
            dynamic_df.group_by(code_col)
            .agg(
                numerics_present.sum().alias(f"n_values//{i}"),
                pl.col("*").count().alias(f"n_occurrences//{i}"),
            )
            .collect()
        )
        frac_static_code_occurrences_with_values.append(
            df.filter(is_static)
            .group_by(code_col)
            .agg(
                numerics_present.sum().alias(f"n_values//{i}"),
                pl.col("*").count().alias(f"n_occurrences//{i}"),
            )
            .collect()
        )

    static_vocab_size = len(static_codes)
    dynamic_vocab_size = len(dynamic_codes)
    birth_codes_vocab_size = len(birth_codes)
    death_codes_vocab_size = len(death_codes)

    total_subjects = n_subjects.sum()
    frac_subjects_with_birth = (frac_subjects_with_birth * n_subjects).sum() / total_subjects
    frac_subjects_with_death = (frac_subjects_with_death * n_subjects).sum() / total_subjects

    frac_dynamic_code_occurrences_with_values = (
        pl.join(frac_dynamic_code_occurrences_with_values, on=code_col, how="outer")
        .select(
            cs.starts_with("n_occurrences").sum().alias("n_occurrences"),
            cs.starts_with("n_values").sum().alias("n_values"),
        )
        .select(
            (pl.col("n_values") / pl.col("n_occurrences")).alias("frac_values"),
        )
        .collect()
        .to_numpy()
    )

    frac_static_code_occurrences_with_values = (
        pl.join(frac_static_code_occurrences_with_values, on=code_col, how="outer")
        .select(
            cs.starts_with("n_occurrences").sum().alias("n_occurrences"),
            cs.starts_with("n_values").sum().alias("n_values"),
        )
        .select(
            (pl.col("n_values") / pl.col("n_occurrences")).alias("frac_values"),
        )
        .collect()
        .to_numpy()
    )

    data_generator = MEDSDataDFGenerator(
        birth_datetime_per_subject=DatetimeGenerator(birth_times),
        start_data_datetime_per_subject=DatetimeGenerator(start_of_data_times),
        time_between_birth_and_data_per_subject=PositiveTimeDeltaGenerator(time_between_birth_and_data),
        time_between_data_and_death_per_subject=PositiveTimeDeltaGenerator(time_between_data_and_death),
        time_between_data_events_per_subject=PositiveTimeDeltaGenerator(time_between_data_events),
        num_events_per_subject=PositiveIntGenerator(num_events_per_subject),
        num_measurements_per_event=PositiveIntGenerator(num_measurements_per_event),
        num_static_measurements_per_subject=PositiveIntGenerator(num_static_measurements_per_subject),
        frac_dynamic_code_occurrences_with_value=ProportionGenerator(
            frac_dynamic_code_occurrences_with_values
        ),
        frac_static_code_occurrences_with_value=ProportionGenerator(frac_static_code_occurrences_with_values),
        static_vocab_size=static_vocab_size,
        dynamic_vocab_size=dynamic_vocab_size,
        frac_subjects_with_death=frac_subjects_with_death,
        frac_subjects_with_birth=frac_subjects_with_birth,
        birth_codes_vocab_size=birth_codes_vocab_size,
        death_codes_vocab_size=death_codes_vocab_size,
    )

    dataset_generator = MEDSDatasetGenerator(
        data_generator=data_generator,
        shard_size=np.median(n_subjects),
        train_frac=train_frac,
        tuning_frac=tuning_frac,
        held_out_frac=held_out_frac,
        dataset_name=dataset_name,
    )

    dataset_config = dataclasses.asdict(dataset_generator)
    output_fp.write_text(dump_yaml(dataset_config, Dumper=Dumper))


if __name__ == "__main__":
    main()
