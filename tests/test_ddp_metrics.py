import multiprocessing as mp
import os
import socket
import time
import traceback
import unittest
from types import SimpleNamespace

import torch


class FakeWriter:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, float(value), int(step)))


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _metric_worker(rank, world_size, port, result_queue):
    try:
        os.environ.update(
            {
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(port),
                "WORLD_SIZE": str(world_size),
                "RANK": str(rank),
                "LOCAL_RANK": str(rank),
            }
        )
        import torch.distributed as dist

        from callbacks import EpochSummaryCallback

        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
        writer = FakeWriter() if rank == 0 else None
        cb = EpochSummaryCallback(writer=writer, total_epochs=1, batch_size=4)
        trainer = SimpleNamespace(
            epoch=1,
            device=torch.device("cpu"),
            train_loss_sum=2.0 if rank == 0 else 6.0,
            train_loss_samples=2 if rank == 0 else 3,
            train_loss_steps=1,
            val_loss_sum=4.0 if rank == 0 else 8.0,
            val_loss_samples=2 if rank == 0 else 2,
            val_loss_steps=1,
            losses=[],
        )
        cb.on_train_start(trainer)
        cb._epoch_start = time.perf_counter() - 1.0
        cb.on_train_epoch_end(trainer)
        cb.on_eval_epoch_end(trainer)
        if rank == 0:
            result_queue.put((trainer.losses, writer.scalars, cb.global_batch_size))
        dist.destroy_process_group()
    except Exception:
        result_queue.put(("error", rank, traceback.format_exc()))
        raise


class DDPMetricAggregationTest(unittest.TestCase):
    def test_epoch_summary_aggregates_loss_by_sample_count(self):
        ctx = mp.get_context("spawn")
        world_size = 2
        port = _free_port()
        result_queue = ctx.Queue()
        processes = [
            ctx.Process(target=_metric_worker, args=(rank, world_size, port, result_queue))
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=20)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
                self.fail("metric worker timed out")
        self.assertEqual([process.exitcode for process in processes], [0] * world_size)

        results = []
        while not result_queue.empty():
            item = result_queue.get()
            if item[0] == "error":
                self.fail(item[2])
            results.append(item)
        self.assertEqual(len(results), 1)
        losses, scalars, global_batch_size = results[0]
        self.assertEqual(global_batch_size, 8)
        self.assertAlmostEqual(losses[0], 8.0 / 5.0)
        scalar_map = {tag: value for tag, value, _step in scalars}
        self.assertAlmostEqual(scalar_map["train/effective_batch_size"], 8.0)
        self.assertAlmostEqual(scalar_map["train/epoch_loss"], 8.0 / 5.0)
        self.assertAlmostEqual(scalar_map["val/loss"], 12.0 / 4.0)


if __name__ == "__main__":
    unittest.main()
