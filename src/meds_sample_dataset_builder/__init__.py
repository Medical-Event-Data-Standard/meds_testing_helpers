from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files

from .dataset_generator import (
    SAMPLE_DATASET_T,
    MEDSDataDFGenerator,
    PositiveIntGenerator,
    PositiveNumGenerator,
    ProportionGenerator,
)

__package_name__ = "meds_sample_dataset_builder"
try:
    __version__ = version(__package_name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

CONFIG_YAML = files(__package_name__).joinpath("configs/generate_dataset.yaml")

__all__ = [
    MEDSDataDFGenerator.__name__,
    PositiveIntGenerator.__name__,
    PositiveNumGenerator.__name__,
    ProportionGenerator.__name__,
    "CONFIG_YAML",
    "SAMPLE_DATASET_T",
]
