import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class EstimateDitComplexityTest(unittest.TestCase):
    def test_runs_from_non_repo_cwd(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    str(repo / "scripts" / "estimate_dit_complexity.py"),
                    "--experiment",
                    "imagenet256_latent_dit_b2_moe_layerwise_d8_c1_ffnw05",
                    "--format",
                    "json",
                ],
                cwd=tmp,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("active_params_m", result.stdout)


if __name__ == "__main__":
    unittest.main()
