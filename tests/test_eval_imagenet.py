import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import torch
import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from PIL import Image

from eval_imagenet import (
    CheckpointInfo,
    GenerationConfig,
    build_uniform_labels,
    cleanfid_stats_path,
    compute_imagenet_fid,
    endpoint_excluded_euler_grid,
    generate_imagenet_samples,
    generate_latents,
    load_checkpoint_weights,
    make_custom_fid_stats,
    sha256_file,
    _generation_config_from_hydra,
)


class DummyCFGModel(torch.nn.Module):
    null_token = 4

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x, t, labels=None):
        return torch.zeros_like(x) + self.anchor * 0


class DummyVAE:
    model_id = "dummy-vae"

    def decode(self, z):
        values = z.float().mean(dim=(1, 2, 3)).tanh().view(z.size(0), 1, 1, 1)
        return values.expand(z.size(0), 3, 256, 256)


class LinearVelocityModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x, t, labels=None):
        return x + self.anchor * 0


class EvalImageNetTest(unittest.TestCase):
    def test_uniform_labels_balanced_when_divisible(self):
        labels_1k = build_uniform_labels(1000, 1000)
        self.assertTrue(
            torch.equal(torch.bincount(labels_1k), torch.ones(1000, dtype=torch.long))
        )

        labels_50k = build_uniform_labels(50000, 1000)
        expected = torch.full((1000,), 50, dtype=torch.long)
        self.assertTrue(torch.equal(torch.bincount(labels_50k), expected))

    def test_endpoint_excluded_grid_does_not_include_one(self):
        grid = endpoint_excluded_euler_grid(4, device=torch.device("cpu"))
        torch.testing.assert_close(grid, torch.tensor([0.0, 0.25, 0.5, 0.75]))
        self.assertLess(float(grid[-1]), 1.0)
        with self.assertRaises(TypeError):
            endpoint_excluded_euler_grid(cast(Any, 1.5), device=torch.device("cpu"))

    def test_sample_generation_writes_pngs_labels_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "eval"
            cfg = GenerationConfig(
                output_dir=output_dir,
                checkpoint="dummy.pt",
                num_samples=8,
                batch_size=3,
                num_steps=2,
                guidance_scale=2.0,
                latent_shape=(4, 4, 4),
                seed=7,
                num_classes=4,
                image_size=256,
            )
            metadata = generate_imagenet_samples(
                DummyCFGModel(),
                DummyVAE(),
                cfg,
                checkpoint_info=CheckpointInfo("dummy.pt", "ema", 5),
                device=torch.device("cpu"),
            )

            self.assertTrue(metadata["complete"])
            pngs = sorted(output_dir.glob("*.png"))
            self.assertEqual(len(pngs), 8)
            with Image.open(pngs[0]) as img:
                self.assertEqual(img.mode, "RGB")
                self.assertEqual(img.size, (256, 256))

            with open(output_dir / "labels.json") as handle:
                labels_payload = json.load(handle)
            labels = [record["label"] for record in labels_payload["records"]]
            self.assertEqual(labels, [0, 1, 2, 3, 0, 1, 2, 3])

            with open(output_dir / "metadata.yaml") as handle:
                saved_metadata = yaml.safe_load(handle)
            self.assertEqual(
                saved_metadata["sampler"]["time_grid"], "endpoint_excluded"
            )
            self.assertEqual(saved_metadata["checkpoint"]["weights"], "ema")

            generate_imagenet_samples(
                DummyCFGModel(),
                DummyVAE(),
                cfg,
                checkpoint_info=CheckpointInfo("dummy.pt", "ema", 5),
                device=torch.device("cpu"),
            )

            changed_cfg = GenerationConfig(
                output_dir=output_dir,
                checkpoint="dummy.pt",
                num_samples=8,
                batch_size=3,
                num_steps=2,
                guidance_scale=2.0,
                latent_shape=(4, 4, 4),
                seed=8,
                num_classes=4,
                image_size=256,
            )
            with self.assertRaisesRegex(ValueError, "different eval config"):
                generate_imagenet_samples(
                    DummyCFGModel(),
                    DummyVAE(),
                    changed_cfg,
                    checkpoint_info=CheckpointInfo("dummy.pt", "ema", 5),
                    device=torch.device("cpu"),
                )

    def test_heun_solver_uses_predictor_corrector(self):
        labels = torch.tensor([0])
        latent_shape = (1,)
        seed = 13
        euler = generate_latents(
            LinearVelocityModel(),
            [0],
            labels,
            latent_shape=latent_shape,
            num_steps=1,
            guidance_scale=1.0,
            seed=seed,
            device=torch.device("cpu"),
            solver="euler",
        )
        heun = generate_latents(
            LinearVelocityModel(),
            [0],
            labels,
            latent_shape=latent_shape,
            num_steps=1,
            guidance_scale=1.0,
            seed=seed,
            device=torch.device("cpu"),
            solver="heun",
        )
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        noise = torch.randn(*latent_shape, generator=gen).view(1, *latent_shape)

        torch.testing.assert_close(euler, noise * 2.0)
        torch.testing.assert_close(heun, noise * 2.5)

    def test_sample_generation_writes_grid_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "eval"
            grid_path = output_dir / "grid.png"
            cfg = GenerationConfig(
                output_dir=output_dir,
                checkpoint="dummy.pt",
                num_samples=8,
                batch_size=4,
                num_steps=2,
                guidance_scale=2.0,
                latent_shape=(4, 4, 4),
                seed=9,
                num_classes=4,
                image_size=256,
                solver="heun",
                grid_path=grid_path,
                grid_nrow=4,
            )

            metadata = generate_imagenet_samples(
                DummyCFGModel(),
                DummyVAE(),
                cfg,
                checkpoint_info=CheckpointInfo("dummy.pt", "raw", 7),
                device=torch.device("cpu"),
            )

            self.assertEqual(metadata["sampler"]["type"], "ode_heun")
            self.assertEqual(metadata["sampler"]["num_steps"], 2)
            self.assertEqual(metadata["grid"]["nrow"], 4)
            sample_pngs = sorted(output_dir.glob("[0-9][0-9][0-9][0-9][0-9][0-9].png"))
            self.assertEqual(len(sample_pngs), 8)
            with Image.open(grid_path) as img:
                self.assertEqual(img.mode, "RGB")
                self.assertEqual(img.size, (4 * 256, 2 * 256))

            generate_imagenet_samples(
                DummyCFGModel(),
                DummyVAE(),
                cfg,
                checkpoint_info=CheckpointInfo("dummy.pt", "raw", 7),
                device=torch.device("cpu"),
            )

    def test_generation_config_validation_rejects_bad_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_batch_cfg = GenerationConfig(
                output_dir=Path(tmp) / "eval",
                checkpoint="dummy.pt",
                batch_size=cast(Any, 1.5),
            )
            with self.assertRaises(TypeError):
                generate_imagenet_samples(
                    DummyCFGModel(),
                    DummyVAE(),
                    bad_batch_cfg,
                    device=torch.device("cpu"),
                )

        unsafe_clean_cfg = GenerationConfig(
            output_dir=Path.cwd(),
            checkpoint="dummy.pt",
            num_samples=1,
            batch_size=1,
            num_steps=1,
            clean_output_dir=True,
        )
        with self.assertRaisesRegex(ValueError, "unsafe output_dir"):
            generate_imagenet_samples(
                DummyCFGModel(),
                DummyVAE(),
                unsafe_clean_cfg,
                device=torch.device("cpu"),
            )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "not_eval"
            output_dir.mkdir()
            (output_dir / "metadata.yaml").write_text("kind: other\n")
            wrong_metadata_cfg = GenerationConfig(
                output_dir=output_dir,
                checkpoint="dummy.pt",
                num_samples=1,
                batch_size=1,
                num_steps=1,
                clean_output_dir=True,
            )
            with self.assertRaisesRegex(ValueError, "non-eval metadata"):
                generate_imagenet_samples(
                    DummyCFGModel(),
                    DummyVAE(),
                    wrong_metadata_cfg,
                    device=torch.device("cpu"),
                )

        with tempfile.TemporaryDirectory() as tmp:
            bad_solver_cfg = GenerationConfig(
                output_dir=Path(tmp) / "eval",
                checkpoint="dummy.pt",
                num_samples=1,
                batch_size=1,
                num_steps=1,
                solver=cast(Any, "bogus"),
            )
            with self.assertRaisesRegex(ValueError, "solver"):
                generate_imagenet_samples(
                    DummyCFGModel(),
                    DummyVAE(),
                    bad_solver_cfg,
                    device=torch.device("cpu"),
                )

    def test_load_checkpoint_weights_prefers_ema_in_auto_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            model = torch.nn.Linear(1, 1)
            raw_state = {
                name: torch.zeros_like(value)
                for name, value in model.state_dict().items()
            }
            ema_state = {
                name: torch.ones_like(value)
                for name, value in model.state_dict().items()
            }
            torch.save(
                {
                    "model_state": raw_state,
                    "ema_state": ema_state,
                    "train_progress": {"num_epochs_completed": 3},
                },
                path,
            )

            info = load_checkpoint_weights(model, path, "auto")
            self.assertEqual(info.weights, "ema")
            self.assertEqual(info.epoch, 3)
            self.assertEqual(info.sha256, sha256_file(path))
            for value in model.state_dict().values():
                self.assertTrue(torch.equal(value, torch.ones_like(value)))

    def test_compute_fid_uses_custom_stats_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_dir = root / "samples"
            sample_dir.mkdir()
            Image.new("RGB", (8, 8)).save(sample_dir / "000000.png")
            calls = []

            class FakeFid:
                @staticmethod
                def compute_fid(*args, **kwargs):
                    calls.append((args, kwargs))
                    return 12.5

            fake_cleanfid = types.SimpleNamespace(
                fid=FakeFid,
                __file__=str(root / "cleanfid" / "__init__.py"),
            )
            with patch.dict(sys.modules, {"cleanfid": fake_cleanfid}):
                result = compute_imagenet_fid(
                    sample_dir,
                    custom_stats_name="nanoflow_imagenet256_val_real_tf_legacy",
                    mode="legacy_tensorflow",
                    device="cpu",
                    output_path=root / "metrics.yaml",
                )

            self.assertEqual(result["score"], 12.5)
            self.assertEqual(
                calls[0][1]["dataset_name"], "nanoflow_imagenet256_val_real_tf_legacy"
            )
            self.assertEqual(calls[0][1]["dataset_split"], "custom")
            self.assertEqual(calls[0][1]["dataset_res"], "na")
            self.assertEqual(calls[0][1]["model_name"], "inception_v3")
            with open(root / "metrics.yaml") as handle:
                saved = yaml.safe_load(handle)
            self.assertEqual(saved["num_samples"], 1)

    def test_force_stats_rebuild_tolerates_missing_kid_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "real"
            image_dir.mkdir()
            package_dir = root / "cleanfid"
            package_dir.mkdir()
            (package_dir / "__init__.py").write_text("")
            fake_cleanfid = types.SimpleNamespace(
                __file__=str(package_dir / "__init__.py")
            )
            stats_dir = package_dir / "stats"
            stats_dir.mkdir()
            stats_path = stats_dir / (
                "nanoflow_imagenet256_val_real_tf_legacy_"
                "legacy_tensorflow_custom_na.npz"
            )
            stats_path.write_bytes(b"old")
            calls = []

            class FakeFid:
                @staticmethod
                def make_custom_stats(*args, **kwargs):
                    calls.append((args, kwargs))
                    cleanfid_stats_path(
                        "nanoflow_imagenet256_val_real_tf_legacy",
                        "legacy_tensorflow",
                    ).write_bytes(b"new")

            fake_cleanfid.fid = FakeFid
            with patch.dict(sys.modules, {"cleanfid": fake_cleanfid}):
                metadata = make_custom_fid_stats(
                    image_dir,
                    force=True,
                    device="cpu",
                )

            self.assertEqual(len(calls), 1)
            self.assertEqual(stats_path.read_bytes(), b"new")
            self.assertEqual(metadata["stats_sha256"], sha256_file(stats_path))

    def test_eval_hydra_config_is_composable_and_structured(self):
        config_dir = str(Path(__file__).resolve().parents[1] / "configs")
        GlobalHydra.instance().clear()
        try:
            with initialize_config_dir(config_dir=config_dir, version_base=None):
                cfg = compose(config_name="eval_imagenet")
                self.assertEqual(cfg.eval.generation.num_samples, 10000)
                cfg_with_grid = compose(
                    config_name="eval_imagenet",
                    overrides=[
                        "eval.generation.solver=heun",
                        "eval.generation.grid_path=grid.png",
                    ],
                )
                self.assertEqual(cfg_with_grid.eval.generation.solver, "heun")
                self.assertEqual(cfg_with_grid.eval.generation.grid_path, "grid.png")
                runtime_cfg = _generation_config_from_hydra(
                    OmegaConf.merge(cfg_with_grid.eval, {"output_dir": "/tmp/eval"})
                )
                self.assertEqual(runtime_cfg.solver, "heun")
                with self.assertRaises(Exception):
                    compose(
                        config_name="eval_imagenet",
                        overrides=["eval.generation.solver=bogus"],
                    )
                with self.assertRaises(Exception):
                    compose(
                        config_name="eval_imagenet",
                        overrides=["eval.generation.not_a_field=1"],
                    )
        finally:
            GlobalHydra.instance().clear()


if __name__ == "__main__":
    unittest.main()
