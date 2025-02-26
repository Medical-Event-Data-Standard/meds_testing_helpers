from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files

__package_name__ = "meds_testing_helpers"
try:
    __version__ = version(__package_name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

CONFIG_YAML = files(__package_name__).joinpath("configs/generate_dataset.yaml")

__all__ = [
    "CONFIG_YAML",
    "__version__",
    "__package_name__",
]
