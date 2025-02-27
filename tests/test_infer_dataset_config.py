import subprocess
import tempfile
from pathlib import Path


def test_dataset_config_inference(simple_static_MEDS):
    """Test the inference of a config to generate a synthetic version of a dataset."""

    with tempfile.TemporaryDirectory() as temp_dir:
        output_fp = Path(temp_dir) / "generated_config.yaml"
        cmd = [
            "infer_MEDS_sample_gen_config",
            f"dataset_dir={str(simple_static_MEDS.resolve())}",
            f"output_fp={str(output_fp.resolve())}",
        ]

        out = subprocess.run(cmd, shell=False, check=False, capture_output=True)

        error_str = (
            f"Command failed with return code {out.returncode}.\n"
            f"Command stdout:\n{out.stdout.decode()}\n"
            f"Command stderr:\n{out.stderr.decode()}"
        )

        assert out.returncode == 0, error_str

        assert output_fp.exists(), f"Output file {output_fp} does not exist."
