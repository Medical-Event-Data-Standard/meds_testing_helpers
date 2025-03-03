import importlib

import meds_testing_helpers.static_sample_data
from meds_testing_helpers.dataset import MEDSDataset


def recursive_check(d: dict, full_key: str | None = None):
    for k, v in d.items():
        if full_key is None:
            local_key = k
        else:
            local_key = f"{full_key}/{k}"
        if isinstance(v, dict):
            recursive_check(v, full_key=local_key)
        else:
            try:
                MEDSDataset.from_yaml(v)
            except Exception as e:
                raise AssertionError(f"Failed to parse {local_key}") from e


def test_static_datasets():
    # This reload ensures that pytest catches coverage for the `__init__.py` file in the static data.
    importlib.reload(meds_testing_helpers.static_sample_data)
    from meds_testing_helpers.static_sample_data import exported_yamls

    assert len(exported_yamls) > 0
    recursive_check(exported_yamls)
