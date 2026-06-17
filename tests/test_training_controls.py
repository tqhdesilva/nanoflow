import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from callbacks import CheckpointCallback, RunDirCallback
from config import InitFromWeights, LossMode, OptimizerType, TrainingConfig
from flow import CondOT
from models import MLP
from train import Trainer, _build_callbacks


class LRRecorder:
    def __init__(self):
        self.values = []

    def on_train_step_end(self, trainer):
        self.values.append(trainer.optimizer.param_groups[0]["lr"])


class FakeCheckpointTrainer:
    def state_dict(self):
        return {
            "train_progress": {
                "num_epochs_completed": 1,
                "num_steps_completed": 2,
            }
        }


class InterruptAfterStep:
    def on_train_step_end(self, trainer):
        raise SystemExit(0)


class LifecycleRecorder:
    def __init__(self):
        self.train_end_calls = 0
        self.cleanup_calls = 0

    def on_train_end(self, trainer):
        self.train_end_calls += 1

    def on_train_cleanup(self, trainer):
        self.cleanup_calls += 1


class ZeroTargetFlow:
    def interpolate(self, x_0, eps, t):
        return x_0

    def target(self, x_0, eps, t):
        return torch.zeros_like(x_0)


class AuxMaskModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x, t, labels=None, return_aux=False):
        pred = torch.full_like(x, 100.0) * self.weight
        pred[:, :, :, : x.size(-1) // 2] = 0
        loss_mask = torch.zeros(x.size(0), 1, x.size(2), x.size(3), device=x.device)
        loss_mask[:, :, :, : x.size(-1) // 2] = 1
        if return_aux:
            return {"pred": pred, "loss_mask": loss_mask}
        return {"pred": pred, "loss_mask": loss_mask}


class ActiveMaskerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.masker = object()

    def forward(self, x, t):
        return torch.zeros_like(x) * self.weight


class TrainingControlsTest(unittest.TestCase):
    def _make_toy_trainer(self, training):
        return Trainer(
            model=MLP(hidden_dim=8, num_layers=1, time_dim=8),
            flow=CondOT(),
            training=training,
            device=torch.device("cpu"),
        )

    def test_loss_mode_is_enum_validated(self):
        cfg = OmegaConf.structured(TrainingConfig)
        self.assertEqual(cfg.loss_mode, LossMode.mse)

        cfg.loss_mode = "masked_mse"
        self.assertEqual(cfg.loss_mode, LossMode.masked_mse)

        with self.assertRaises(Exception):
            cfg.loss_mode = "bad"

    def test_init_from_weights_is_enum_validated(self):
        cfg = OmegaConf.structured(TrainingConfig)
        self.assertEqual(cfg.init_from_weights, InitFromWeights.raw)

        cfg.init_from_weights = "ema"
        self.assertEqual(cfg.init_from_weights, InitFromWeights.ema)

        with self.assertRaises(Exception):
            cfg.init_from_weights = "bad"

    def test_optimizer_is_enum_validated(self):
        cfg = OmegaConf.structured(TrainingConfig)
        self.assertEqual(cfg.optimizer, OptimizerType.adam)

        cfg.optimizer = "adamw"
        self.assertEqual(cfg.optimizer, OptimizerType.adamw)

        with self.assertRaises(Exception):
            cfg.optimizer = "bad"

    def test_adamw_and_weight_decay_are_configurable(self):
        training = TrainingConfig(
            optimizer=OptimizerType.adamw,
            weight_decay=0.01,
        )
        trainer = self._make_toy_trainer(training)

        self.assertIsInstance(trainer.optimizer, torch.optim.AdamW)
        self.assertEqual(trainer.optimizer.param_groups[0]["weight_decay"], 0.01)

    def test_masked_mse_uses_aux_loss_mask(self):
        training = TrainingConfig(
            epochs=1,
            batch_size=2,
            eval_every=0,
            checkpoint_every=0,
            loss_mode=LossMode.masked_mse,
        )
        trainer = Trainer(
            model=AuxMaskModel(),
            flow=ZeroTargetFlow(),
            training=training,
            device=torch.device("cpu"),
        )
        x = torch.randn(2, 1, 4, 4)

        loss, pred = trainer._compute_loss((x,))

        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(pred.shape, x.shape)

    def test_masked_mse_requires_aux_output(self):
        training = TrainingConfig(
            epochs=1,
            batch_size=2,
            eval_every=0,
            checkpoint_every=0,
            loss_mode=LossMode.masked_mse,
        )
        trainer = Trainer(
            model=MLP(hidden_dim=8, num_layers=1, time_dim=8),
            flow=ZeroTargetFlow(),
            training=training,
            device=torch.device("cpu"),
        )
        x = torch.randn(2, 2)

        with self.assertRaisesRegex(ValueError, "masked_mse"):
            trainer._compute_loss((x,))

    def test_mse_rejects_dict_output(self):
        training = TrainingConfig(
            epochs=1,
            batch_size=2,
            eval_every=0,
            checkpoint_every=0,
            loss_mode=LossMode.mse,
        )
        trainer = Trainer(
            model=AuxMaskModel(),
            flow=ZeroTargetFlow(),
            training=training,
            device=torch.device("cpu"),
        )
        x = torch.randn(2, 1, 4, 4)

        with self.assertRaisesRegex(ValueError, "loss_mode=mse"):
            trainer._compute_loss((x,))

    def test_mse_rejects_active_masker_in_training(self):
        training = TrainingConfig(
            epochs=1,
            batch_size=2,
            eval_every=0,
            checkpoint_every=0,
            loss_mode=LossMode.mse,
        )
        trainer = Trainer(
            model=ActiveMaskerModel(),
            flow=ZeroTargetFlow(),
            training=training,
            device=torch.device("cpu"),
        )
        x = torch.randn(2, 1, 4, 4)

        with self.assertRaisesRegex(ValueError, "masker"):
            trainer._compute_loss((x,))

    def test_explicit_run_dir_overrides_generated_run_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            explicit = Path(tmp) / "stable-run"
            cb = RunDirCallback(
                runs_dir=str(Path(tmp) / "ignored"),
                run_prefix="ignored",
                cfg=OmegaConf.create({"training": {"run_dir": str(explicit)}}),
                run_dir=str(explicit),
            )
            try:
                self.assertEqual(cb.run_dir, explicit)
                self.assertEqual(cb.ckpt_dir, explicit / "checkpoints")
                self.assertEqual(cb.tb_dir, explicit / "tensorboard")
                self.assertTrue(cb.ckpt_dir.exists())
                self.assertTrue(cb.tb_dir.exists())
                self.assertTrue((explicit / "metadata.yaml").exists())
            finally:
                cb.on_train_cleanup(None)

    def test_generated_run_dir_uses_runs_dir_and_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = RunDirCallback(
                runs_dir=tmp,
                run_prefix="toy",
                cfg=OmegaConf.create({}),
            )
            try:
                self.assertEqual(cb.run_dir.parent, Path(tmp))
                self.assertTrue(cb.run_dir.name.startswith("toy_"))
                self.assertTrue(cb.ckpt_dir.exists())
            finally:
                cb.on_train_cleanup(None)

    def test_auto_resume_requires_explicit_run_dir_for_training_callbacks(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir_cb = RunDirCallback(
                runs_dir=tmp,
                run_prefix="toy",
                cfg=OmegaConf.create({}),
            )
            cfg = OmegaConf.create(
                {
                    "training": {
                        "checkpoint_every": 1,
                        "resume": "auto",
                        "run_dir": None,
                        "epochs": 1,
                        "batch_size": 2,
                        "log_every": 10,
                        "eval_every": 0,
                        "p_uncond": None,
                    },
                    "sample_logger": None,
                }
            )
            try:
                with self.assertRaisesRegex(ValueError, "training.run_dir"):
                    _build_callbacks(cfg, run_dir_cb)
            finally:
                run_dir_cb.on_train_cleanup(None)

    def test_auto_resume_missing_checkpoint_starts_from_scratch(self):
        training = TrainingConfig(
            epochs=1,
            batch_size=2,
            eval_every=0,
            checkpoint_every=1,
        )
        trainer = self._make_toy_trainer(training)
        with tempfile.TemporaryDirectory() as tmp:
            cb = CheckpointCallback(Path(tmp) / "checkpoints", 1, resume="auto")
            cb.on_train_start(trainer)

        self.assertEqual(trainer.epoch, 0)
        self.assertEqual(trainer.step, 0)

    def test_auto_resume_loads_latest_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "checkpoints"
            data = TensorDataset(torch.randn(6, 2))
            loader = DataLoader(data, batch_size=2, shuffle=False)
            training = TrainingConfig(
                epochs=1,
                batch_size=2,
                warmup_epochs=1,
                ema_decay=0.9,
                eval_every=0,
                checkpoint_every=1,
            )
            trainer = self._make_toy_trainer(training)
            trainer.fit(loader, loader, callbacks=[CheckpointCallback(ckpt_dir, 1)])
            ckpt = torch.load(
                ckpt_dir / "latest.pt", weights_only=True, map_location="cpu"
            )

            restored = self._make_toy_trainer(training)
            restored.lr_scheduler = restored._build_scheduler(
                restored.optimizer, restored.training, len(loader)
            )
            CheckpointCallback(ckpt_dir, 1, resume="auto").on_train_start(restored)

            self.assertEqual(
                restored.epoch,
                ckpt["train_progress"]["num_epochs_completed"],
            )
            self.assertEqual(
                restored.step,
                ckpt["train_progress"]["num_steps_completed"],
            )
            self.assertEqual(restored.losses, ckpt["losses"])
            for name, value in restored.raw_module.state_dict().items():
                torch.testing.assert_close(value, ckpt["model_state"][name])
            self.assertTrue(restored.optimizer.state_dict()["state"])
            self.assertEqual(
                restored.lr_scheduler.state_dict(), ckpt["lr_scheduler_state"]
            )
            for name, value in restored.ema_model.module.state_dict().items():
                torch.testing.assert_close(value, ckpt["ema_state"][name])
            self.assertEqual(
                int(restored.ema_model.n_averaged.item()), ckpt["ema_n_averaged"]
            )

    def test_legacy_ema_resume_keeps_average_initialized(self):
        data = TensorDataset(torch.randn(6, 2))
        loader = DataLoader(data, batch_size=2, shuffle=False)
        training = TrainingConfig(
            epochs=1,
            batch_size=2,
            ema_decay=0.9,
            eval_every=0,
            checkpoint_every=1,
        )
        trainer = self._make_toy_trainer(training)
        trainer.fit(loader, loader)
        ckpt = trainer.state_dict()
        del ckpt["ema_n_averaged"]

        restored = self._make_toy_trainer(training)
        restored.load_state_dict(ckpt)

        self.assertEqual(int(restored.ema_model.n_averaged.item()), 1)

    def test_init_from_loads_model_weights_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = TensorDataset(torch.randn(4, 2))
            loader = DataLoader(data, batch_size=2, shuffle=False)
            training = TrainingConfig(
                epochs=1,
                batch_size=2,
                eval_every=0,
                checkpoint_every=1,
            )
            source = self._make_toy_trainer(training)
            source.fit(loader, loader)
            ckpt_path = Path(tmp) / "seed.pt"
            torch.save(source.state_dict(), ckpt_path)

            restored = self._make_toy_trainer(training)
            CheckpointCallback(
                Path(tmp) / "dest" / "checkpoints",
                1,
                init_from=str(ckpt_path),
            ).on_train_start(restored)

            self.assertEqual(restored.epoch, 0)
            self.assertEqual(restored.step, 0)
            self.assertEqual(restored.losses, [])
            self.assertEqual(restored.optimizer.state_dict()["state"], {})
            for name, value in restored.raw_module.state_dict().items():
                torch.testing.assert_close(value, source.raw_module.state_dict()[name])

    def test_init_from_ema_requires_ema_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            training = TrainingConfig(
                epochs=1,
                batch_size=2,
                eval_every=0,
                checkpoint_every=1,
            )
            source = self._make_toy_trainer(training)
            ckpt_path = Path(tmp) / "seed.pt"
            torch.save(source.state_dict(), ckpt_path)
            restored = self._make_toy_trainer(training)

            with self.assertRaisesRegex(ValueError, "EMA"):
                CheckpointCallback(
                    Path(tmp) / "dest" / "checkpoints",
                    1,
                    init_from=str(ckpt_path),
                    init_from_weights="ema",
                ).on_train_start(restored)

    def test_init_from_ema_loads_ema_weights_and_resets_live_ema(self):
        with tempfile.TemporaryDirectory() as tmp:
            training = TrainingConfig(
                epochs=1,
                batch_size=2,
                ema_decay=0.9,
                eval_every=0,
                checkpoint_every=1,
            )
            source = self._make_toy_trainer(training)
            ckpt = source.state_dict()
            ckpt["ema_state"] = {
                name: torch.full_like(value, 3.0)
                for name, value in ckpt["model_state"].items()
            }
            ckpt_path = Path(tmp) / "seed.pt"
            torch.save(ckpt, ckpt_path)

            restored = self._make_toy_trainer(training)
            CheckpointCallback(
                Path(tmp) / "dest" / "checkpoints",
                1,
                init_from=str(ckpt_path),
                init_from_weights="ema",
            ).on_train_start(restored)

            for value in restored.raw_module.state_dict().values():
                torch.testing.assert_close(value, torch.full_like(value, 3.0))
            for value in restored.ema_model.module.state_dict().values():
                torch.testing.assert_close(value, torch.full_like(value, 3.0))
            self.assertEqual(int(restored.ema_model.n_averaged.item()), 0)

    def test_auto_resume_wins_over_init_from(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_ckpt_dir = Path(tmp) / "run" / "checkpoints"
            seed_ckpt = Path(tmp) / "seed.pt"
            training = TrainingConfig(
                epochs=1,
                batch_size=2,
                eval_every=0,
                checkpoint_every=1,
            )
            resume_source = self._make_toy_trainer(training)
            resume_source.epoch = 5
            resume_source.step = 9
            CheckpointCallback(run_ckpt_dir, checkpoint_every=1).save(resume_source)

            init_source = self._make_toy_trainer(training)
            torch.save(init_source.state_dict(), seed_ckpt)

            restored = self._make_toy_trainer(training)
            CheckpointCallback(
                run_ckpt_dir,
                1,
                resume="auto",
                init_from=str(seed_ckpt),
            ).on_train_start(restored)

            self.assertEqual(restored.epoch, 5)
            self.assertEqual(restored.step, 9)

    def test_explicit_resume_path_still_loads_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "source" / "checkpoints"
            data = TensorDataset(torch.randn(4, 2))
            loader = DataLoader(data, batch_size=2, shuffle=False)
            training = TrainingConfig(
                epochs=1,
                batch_size=2,
                eval_every=0,
                checkpoint_every=1,
            )
            trainer = self._make_toy_trainer(training)
            trainer.fit(loader, loader, callbacks=[CheckpointCallback(source_dir, 1)])

            restored = self._make_toy_trainer(training)
            CheckpointCallback(
                Path(tmp) / "dest" / "checkpoints",
                1,
                resume=str(source_dir / "latest.pt"),
            ).on_train_start(restored)

            self.assertEqual(restored.epoch, 1)
            self.assertEqual(restored.step, 2)

    def test_checkpoint_save_removes_atomic_temp_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp)
            cb = CheckpointCallback(ckpt_dir, checkpoint_every=1)
            cb.save(FakeCheckpointTrainer())

            self.assertTrue((ckpt_dir / "latest.pt").exists())
            self.assertEqual(list(ckpt_dir.glob(".latest.pt.*.tmp")), [])

    def test_checkpoint_save_preserves_existing_latest_on_temp_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp)
            cb = CheckpointCallback(ckpt_dir, checkpoint_every=1)
            cb.save(FakeCheckpointTrainer())
            latest = ckpt_dir / "latest.pt"
            before = latest.read_bytes()

            with patch("callbacks.torch.save", side_effect=RuntimeError("boom")):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    cb.save(FakeCheckpointTrainer())

            self.assertEqual(latest.read_bytes(), before)
            self.assertEqual(list(ckpt_dir.glob(".latest.pt.*.tmp")), [])

    def test_tensorboard_writer_uses_purge_step_after_auto_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "stable"
            ckpt_dir = run_dir / "checkpoints"
            training = TrainingConfig(
                epochs=1,
                batch_size=2,
                eval_every=0,
                checkpoint_every=1,
                resume="auto",
                run_dir=str(run_dir),
            )
            source = self._make_toy_trainer(training)
            source.epoch = 1
            source.step = 7
            CheckpointCallback(ckpt_dir, checkpoint_every=1).save(source)
            restored = self._make_toy_trainer(training)
            cfg = OmegaConf.create(
                {
                    "training": {
                        "checkpoint_every": 1,
                        "resume": "auto",
                        "run_dir": str(run_dir),
                        "epochs": 1,
                        "batch_size": 2,
                        "log_every": 1,
                        "eval_every": 0,
                        "p_uncond": None,
                    },
                    "sample_logger": None,
                }
            )
            created = []

            class FakeSummaryWriter:
                def __init__(self, *args, **kwargs):
                    created.append((args, kwargs))

                def add_scalar(self, *args, **kwargs):
                    pass

                def close(self):
                    pass

            loader = DataLoader(TensorDataset(torch.randn(2, 2)), batch_size=2)
            with patch("callbacks.SummaryWriter", FakeSummaryWriter):
                run_dir_cb = RunDirCallback(
                    runs_dir=str(Path(tmp) / "ignored"),
                    run_prefix="ignored",
                    cfg=cfg,
                    run_dir=str(run_dir),
                )
                callbacks = _build_callbacks(cfg, run_dir_cb)
                restored.fit(loader, loader, callbacks=callbacks)

            self.assertEqual(restored.step, 7)
            self.assertEqual(created[-1][1].get("purge_step"), 7)

    def test_max_steps_stops_before_full_epoch(self):
        data = TensorDataset(torch.randn(20, 2))
        loader = DataLoader(data, batch_size=2, shuffle=False)
        training = TrainingConfig(
            epochs=10,
            batch_size=2,
            max_steps=3,
            eval_every=0,
            checkpoint_every=0,
        )
        trainer = Trainer(
            model=MLP(hidden_dim=8, num_layers=1, time_dim=8),
            flow=CondOT(),
            training=training,
            device=torch.device("cpu"),
        )

        trainer.fit(loader, loader, callbacks=[])

        self.assertEqual(trainer.step, 3)
        self.assertEqual(trainer.train_loss_steps, 3)
        self.assertLess(trainer.epoch, training.epochs)

    def test_max_steps_must_be_positive_or_null(self):
        training = TrainingConfig(max_steps=0)
        with self.assertRaises(ValueError):
            Trainer(
                model=MLP(hidden_dim=8, num_layers=1, time_dim=8),
                flow=CondOT(),
                training=training,
                device=torch.device("cpu"),
            )

    def test_interrupted_fit_skips_train_end_and_runs_cleanup(self):
        data = TensorDataset(torch.randn(4, 2))
        loader = DataLoader(data, batch_size=2, shuffle=False)
        training = TrainingConfig(
            epochs=1,
            batch_size=2,
            eval_every=0,
            checkpoint_every=0,
        )
        trainer = Trainer(
            model=MLP(hidden_dim=8, num_layers=1, time_dim=8),
            flow=CondOT(),
            training=training,
            device=torch.device("cpu"),
        )
        recorder = LifecycleRecorder()
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = CheckpointCallback(Path(tmp), checkpoint_every=0)
            with self.assertRaises(SystemExit):
                trainer.fit(
                    loader,
                    loader,
                    callbacks=[recorder, ckpt, InterruptAfterStep()],
                )

            self.assertEqual(recorder.train_end_calls, 0)
            self.assertEqual(recorder.cleanup_calls, 1)
            self.assertFalse((Path(tmp) / "latest.pt").exists())

    def test_normal_train_end_saves_latest(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = CheckpointCallback(Path(tmp), checkpoint_every=1)
            cb.on_train_end(FakeCheckpointTrainer())

            self.assertTrue((Path(tmp) / "latest.pt").exists())

    def test_warmup_scheduler_steps_per_optimizer_step(self):
        data = TensorDataset(torch.randn(20, 2))
        loader = DataLoader(data, batch_size=2, shuffle=False)
        training = TrainingConfig(
            epochs=2,
            batch_size=2,
            lr=1e-3,
            warmup_epochs=1,
            max_steps=3,
            eval_every=0,
            checkpoint_every=0,
        )
        trainer = Trainer(
            model=MLP(hidden_dim=8, num_layers=1, time_dim=8),
            flow=CondOT(),
            training=training,
            device=torch.device("cpu"),
        )
        recorder = LRRecorder()

        trainer.fit(loader, loader, callbacks=[recorder])

        self.assertEqual(trainer.step, 3)
        self.assertEqual(len(recorder.values), 3)
        self.assertLess(recorder.values[0], recorder.values[-1])
        self.assertLess(recorder.values[-1], training.lr)


if __name__ == "__main__":
    unittest.main()
