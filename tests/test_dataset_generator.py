from meds_testing_helpers.dataset import MEDSDataset

# def test_dataset_generation(generated_mimic_like_MEDS):
#    """Test the generation of a dataset."""
#
#    # This will throw an error if the generated data is invalid
#    MEDSDataset(root_dir=generated_mimic_like_MEDS)


# def test_force_plugin_reload():
# """Ensures pytest_plugin.py is loaded after coverage tracking starts."""
# importlib.reload(meds_testing_helpers.pytest_plugin)


def test_dataset_generation(generated_sample_MEDS):
    """Test the generation of a dataset."""

    # This will throw an error if the generated data is invalid
    MEDSDataset(root_dir=generated_sample_MEDS)
