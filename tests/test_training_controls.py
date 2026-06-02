import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from callbacks import CheckpointCallback
from config import TrainingConfig
from flow import CondOT
from models import MLP
from train import Trainer


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


class TrainingControlsTest(unittest.TestCase):
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
