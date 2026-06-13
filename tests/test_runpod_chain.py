import base64
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from scripts.preflight_imagenet_latent_mmap import preflight_cache
from scripts.sky_runpod_chain import (
    RuntimeValues,
    build_jobs_launch_command,
    build_volume_apply_command,
    render_template_text,
    resolve_runtime_values,
    select_runpod_infra_from_datacenters,
    write_secret_file,
)


class RunPodChainTest(unittest.TestCase):
    def test_render_template_replaces_runtime_placeholders_only(self):
        text = (
            "name: ${CHAIN_ID}\n"
            "run: |\n"
            "  echo ${ARTIFACT_CLOUD_URI}/${CHAIN_ID}\n"
            "  echo ${GCP_SERVICE_ACCOUNT_JSON_B64:-missing}\n"
        )

        rendered = render_template_text(
            text,
            {
                "CHAIN_ID": "chain-a",
                "ARTIFACT_CLOUD_URI": "gs://nanoflow/runs",
            },
        )

        self.assertIn("name: chain-a", rendered)
        self.assertIn("gs://nanoflow/runs/chain-a", rendered)
        self.assertIn("${GCP_SERVICE_ACCOUNT_JSON_B64:-missing}", rendered)

    def test_render_template_rejects_missing_runtime_placeholder(self):
        with self.assertRaisesRegex(ValueError, "MISSING"):
            render_template_text("name: ${MISSING}\n", {})

    def test_runtime_values_use_supplied_chain_id_and_env_defaults(self):
        template = Path("cloud/runpod/imagenet256-dit-smoke-chain.yaml")
        args = _args(template=template, chain_id="manual-chain")
        values = resolve_runtime_values(
            args,
            {
                "RUNPOD_INFRA": "runpod/NL/EU-NL-1",
                "GPU_REQUEST": "H100-SXM:1",
                "VOLUME_NAME": "nf-test",
                "IMAGE_ID": "docker:image:tag",
            },
        )

        self.assertEqual(values.chain_id, "manual-chain")
        self.assertEqual(values.runpod_infra, "runpod/NL/EU-NL-1")
        self.assertEqual(values.gpu_request, "H100-SXM:1")
        self.assertEqual(values.volume_name, "nf-test")
        self.assertEqual(values.sync_image_id, "docker:image:tag")
        self.assertEqual(
            values.chain_template_path,
            "cloud/runpod/imagenet256-dit-smoke-chain.yaml",
        )
        self.assertTrue(str(values.rendered_path).endswith("manual-chain.yaml"))

    def test_dry_run_auto_infra_does_not_require_runpod_api_key(self):
        args = _args(
            template=Path("cloud/runpod/imagenet256-dit-smoke-chain.yaml"),
            chain_id="manual-chain",
            infra="auto",
        )

        values = resolve_runtime_values(args, {}, dry_run=True)

        self.assertEqual(values.runpod_infra, "runpod/NL/EU-NL-1")

    def test_auto_infra_skips_datacenters_unsupported_by_skypilot(self):
        datacenters = [
            {
                "id": "AP-JP-1",
                "location": "Japan",
                "storageSupport": True,
                "gpuAvailability": [{"displayName": "H100 SXM", "stockStatus": "High"}],
            },
            {
                "id": "US-CA-2",
                "location": "United States",
                "storageSupport": True,
                "gpuAvailability": [{"displayName": "H100 SXM", "stockStatus": "Low"}],
            },
        ]

        infra = select_runpod_infra_from_datacenters(
            datacenters,
            gpu_request="H100-SXM:1",
            preferred_datacenter="EU-NL-1",
        )

        self.assertEqual(infra, "runpod/US/US-CA-2")

    def test_auto_infra_rejects_unknown_stock_status(self):
        datacenters = [
            {
                "id": "US-GA-1",
                "storageSupport": True,
                "gpuAvailability": [
                    {"displayName": "H100 SXM", "stockStatus": "Unknown"}
                ],
            }
        ]

        with self.assertRaisesRegex(ValueError, "No SkyPilot-supported RunPod"):
            select_runpod_infra_from_datacenters(
                datacenters,
                gpu_request="H100-SXM:1",
                preferred_datacenter="EU-NL-1",
            )

    def test_auto_infra_prefers_storage_supported_available_gpu(self):
        datacenters = [
            {
                "id": "US-GA-1",
                "location": "Georgia",
                "storageSupport": True,
                "gpuAvailability": [{"displayName": "H100 SXM", "stockStatus": "High"}],
            },
            {
                "id": "EU-NL-1",
                "location": "Netherlands",
                "storageSupport": True,
                "gpuAvailability": [
                    {"displayName": "H100 SXM", "stockStatus": "Medium"}
                ],
            },
        ]

        infra = select_runpod_infra_from_datacenters(
            datacenters,
            gpu_request="H100-SXM:1",
            preferred_datacenter="EU-NL-1",
        )

        self.assertEqual(infra, "runpod/NL/EU-NL-1")

    def test_auto_infra_skips_datacenters_without_storage(self):
        datacenters = [
            {
                "id": "US-GA-1",
                "storageSupport": False,
                "gpuAvailability": [{"displayName": "H100 SXM", "stockStatus": "High"}],
            },
            {
                "id": "US-CA-2",
                "storageSupport": True,
                "gpuAvailability": [{"displayName": "H100 SXM", "stockStatus": "Low"}],
            },
        ]

        infra = select_runpod_infra_from_datacenters(
            datacenters,
            gpu_request="H100-SXM:1",
            preferred_datacenter="EU-NL-1",
        )

        self.assertEqual(infra, "runpod/US/US-CA-2")

    def test_build_launch_commands_include_volume_and_secret(self):
        with tempfile.TemporaryDirectory() as tmp:
            rendered = Path(tmp) / "chain.yaml"
            rendered.write_text("name: test\n")
            key = Path(tmp) / "gcp.json"
            key.write_text('{"type":"service_account"}')
            values = RuntimeValues(
                chain_id="chain-a",
                runpod_infra="runpod/NL/EU-NL-1",
                gpu_request="H100-SXM:1",
                volume_name="nf-test",
                volume_size="100Gi",
                image_id="docker:image:tag",
                sync_image_id="docker:image:tag",
                dataset_cloud_uri="gs://nanoflow/data",
                artifact_cloud_uri="gs://nanoflow/runs",
                chain_template_path="cloud/runpod/chain.yaml",
                rendered_path=rendered,
                gcp_credentials=key,
            )

            self.assertEqual(
                build_volume_apply_command(values),
                [
                    "sky",
                    "volumes",
                    "apply",
                    "--infra",
                    "runpod/NL/EU-NL-1",
                    "--name",
                    "nf-test",
                    "--type",
                    "runpod-network-volume",
                    "--size",
                    "100Gi",
                    "-y",
                ],
            )
            launch = build_jobs_launch_command(values)
            self.assertEqual(launch[:4], ["sky", "jobs", "launch", "-n"])
            self.assertNotIn("--config", launch)
            self.assertIn("--secret-file", launch)
            secret_path = Path(launch[launch.index("--secret-file") + 1])
            self.assertFalse(secret_path.exists())
            encoded = base64.b64encode(key.read_bytes()).decode("ascii")
            self.assertEqual(write_secret_file(values), secret_path)
            self.assertEqual(
                secret_path.read_text(),
                f"GCP_SERVICE_ACCOUNT_JSON_B64={encoded}\n",
            )
            self.assertNotIn(encoded, launch)
            self.assertEqual(launch[-2:], ["-y", str(rendered)])

    def test_smoke_chain_template_has_expected_serial_tasks(self):
        template = Path("cloud/runpod/imagenet256-dit-smoke-chain.yaml")
        text = template.read_text()
        values = {
            "CHAIN_ID": "chain-a",
            "RUNPOD_INFRA": "runpod/NL/EU-NL-1",
            "GPU_REQUEST": "H100-SXM:1",
            "VOLUME_NAME": "nf-test",
            "IMAGE_ID": "docker:image:tag",
            "SYNC_IMAGE_ID": "docker:image:tag",
            "DATASET_CLOUD_URI": "gs://nanoflow/data",
            "ARTIFACT_CLOUD_URI": "gs://nanoflow/runs",
            "CHAIN_TEMPLATE_PATH": "cloud/runpod/imagenet256-dit-smoke-chain.yaml",
            "RENDERED_CHAIN_PATH": ".nanoflow_runpod_chains/chain-a.yaml",
            "VOLUME_SIZE": "100Gi",
        }
        rendered = render_template_text(text, values)
        docs = list(yaml.safe_load_all(rendered))

        self.assertEqual(
            docs[0], {"name": "nf-imagenet256-dit-chain-a", "execution": "serial"}
        )
        self.assertEqual(
            [doc["name"] for doc in docs[1:]],
            [
                "sync_inputs",
                "masked_pretrain",
                "unmasked_finetune",
                "eval_generate",
                "sync_artifacts",
            ],
        )
        self.assertEqual(
            docs[1]["file_mounts"],
            {
                "/tmp/nanoflow-rendered-chain.yaml": ".nanoflow_runpod_chains/chain-a.yaml"
            },
        )
        joined = "\n".join(doc["run"] for doc in docs[1:])
        self.assertIn(
            "training.run_dir=/workspace/runs/$CHAIN_ID/masked_pretrain", joined
        )
        self.assertIn("training.resume=auto", joined)
        self.assertIn(
            "training.init_from=/workspace/runs/$CHAIN_ID/masked_pretrain/checkpoints/latest.pt",
            joined,
        )
        self.assertIn("model.masker=null", joined)
        self.assertIn("training.loss_mode=mse", joined)
        self.assertIn("eval.output_dir=/workspace/runs/$CHAIN_ID/eval_generate", joined)
        self.assertIn("runpod_gcs_rsync.sh", joined)
        self.assertIn(
            'cp /tmp/nanoflow-rendered-chain.yaml "/workspace/runs/$CHAIN_ID/chain.yaml"',
            joined,
        )
        self.assertNotIn("python scripts/sky_runpod_chain.py", joined)

    def test_latent_mmap_preflight_validates_arrays_and_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_mmap_cache(root)

            summary = preflight_cache(root, batch_size=2, num_workers=0)

            self.assertEqual(
                summary["splits"]["train"]["latents_shape"], [3, 4, 32, 32]
            )
            self.assertEqual(summary["splits"]["val"]["labels_shape"], [2])
            self.assertEqual(
                summary["batches"]["train"]["latents_shape"], [2, 4, 32, 32]
            )
            self.assertEqual(summary["batches"]["val"]["labels_shape"], [2])

    def test_latent_mmap_preflight_cli_runs_from_repo_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cache"
            _write_mmap_cache(root)
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/preflight_imagenet_latent_mmap.py",
                    "--cache-root",
                    str(root),
                    "--batch-size",
                    "2",
                    "--num-workers",
                    "0",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn('"cache_root"', result.stdout)
            self.assertIn('"train"', result.stdout)

    def test_chain_launcher_direct_dry_run_needs_no_runpod_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = os.environ.copy()
            env.pop("RUNPOD_API_KEY", None)
            env.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
            env["HOME"] = tmp
            output = Path(tmp) / "rendered.yaml"
            result = subprocess.run(
                [
                    "scripts/sky_runpod_chain.py",
                    "cloud/runpod/imagenet256-dit-smoke-chain.yaml",
                    "--dry-run",
                    "--chain-id",
                    "direct-dry-run",
                    "--output",
                    str(output),
                ],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertTrue(output.exists())
            self.assertIn("Rendered chain", result.stdout)
            self.assertIn("runpod/NL/EU-NL-1", output.read_text())

    def test_direct_dry_run_does_not_write_secret_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = os.environ.copy()
            env.pop("RUNPOD_API_KEY", None)
            env.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
            env["HOME"] = tmp
            output = Path(tmp) / "rendered.yaml"
            key = Path(tmp) / "gcp.json"
            key.write_text('{"type":"service_account"}')

            subprocess.run(
                [
                    "scripts/sky_runpod_chain.py",
                    "cloud/runpod/imagenet256-dit-smoke-chain.yaml",
                    "--dry-run",
                    "--chain-id",
                    "direct-dry-run",
                    "--output",
                    str(output),
                    "--gcp-credentials",
                    str(key),
                ],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertTrue(output.exists())
            self.assertFalse(output.with_suffix(".secrets.env").exists())

    def test_latent_mmap_preflight_rejects_bad_label_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_mmap_cache(root, bad_label=True)

            with self.assertRaisesRegex(ValueError, "label range"):
                preflight_cache(root, batch_size=2, num_workers=0)


def _args(**overrides):
    values = {
        "template": Path("cloud/runpod/imagenet256-dit-smoke-chain.yaml"),
        "chain_id": None,
        "infra": None,
        "gpu_request": None,
        "volume_name": None,
        "volume_size": None,
        "image_id": None,
        "sync_image_id": None,
        "dataset_cloud_uri": None,
        "artifact_cloud_uri": None,
        "gcp_credentials": None,
        "output": None,
    }
    values.update(overrides)
    return type("Args", (), values)()


def _write_mmap_cache(root: Path, bad_label: bool = False) -> None:
    metadata = {
        "storage_format": "mmap_npy_v1",
        "cache_version": 1,
        "vae": "stabilityai/sd-vae-ft-ema",
        "latent": {"shape": [4, 32, 32], "dtype": "float16"},
        "label": {"dtype": "int64"},
        "transform": {"image_size": 256, "crop": "resize"},
        "splits": {},
    }
    for split, count in [("train", 3), ("val", 2)]:
        split_dir = root / split
        split_dir.mkdir(parents=True)
        latents = np.zeros((count, 4, 32, 32), dtype=np.float16)
        labels = np.arange(count, dtype=np.int64)
        if bad_label and split == "train":
            labels[-1] = 1000
        np.save(split_dir / "latents.npy", latents)
        np.save(split_dir / "labels.npy", labels)
        with open(split_dir / "source_paths.txt", "w") as handle:
            for idx in range(count):
                handle.write(f"{split}/n00000000_{idx:08d}.JPEG\n")
        metadata["splits"][split] = {
            "count": count,
            "files": {
                "latents": f"{split}/latents.npy",
                "labels": f"{split}/labels.npy",
                "source_paths": f"{split}/source_paths.txt",
            },
        }
    (root / "metadata.json").write_text(json.dumps(metadata))


if __name__ == "__main__":
    unittest.main()
