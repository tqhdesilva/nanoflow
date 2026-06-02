import errno
import unittest

from retry import Retryer
from train import Trainer


class EndCallback:
    def __init__(self, failures):
        self.failures = list(failures)
        self.calls = 0

    def on_train_end(self, trainer):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)


class RetryerTest(unittest.TestCase):
    def make_retryer(self, max_retries=2):
        return Retryer(
            max_retries=max_retries,
            base_delay_seconds=0,
            max_delay_seconds=0,
            jitter_seconds=0,
            retryable_os_errnos=["EIO", "ESTALE"],
            retryable_runtime_error_substrings=[
                "PytorchStreamWriter",
                "file write failed",
                "unexpected pos",
            ],
        )

    def test_is_retryable_allowlists_os_errno(self):
        retryer = self.make_retryer()

        self.assertTrue(retryer.is_retryable(OSError(errno.EIO, "io error")))
        self.assertTrue(retryer.is_retryable(OSError(errno.ESTALE, "stale handle")))
        self.assertFalse(retryer.is_retryable(OSError(errno.ENOSPC, "no space")))
        self.assertFalse(retryer.is_retryable(OSError(errno.EACCES, "denied")))

    def test_is_retryable_allowlists_runtime_error_message(self):
        retryer = self.make_retryer()

        self.assertTrue(
            retryer.is_retryable(
                RuntimeError(
                    "PytorchStreamWriter failed writing file data/280: "
                    "file write failed"
                )
            )
        )
        self.assertTrue(
            retryer.is_retryable(RuntimeError("unexpected pos 251048896 vs 251048784"))
        )
        self.assertFalse(retryer.is_retryable(RuntimeError("model shape mismatch")))

    def test_run_retries_retryable_failure(self):
        retryer = self.make_retryer()
        calls = 0

        def flaky():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError(errno.EIO, "io error")
            return "ok"

        self.assertEqual(retryer.run(flaky), "ok")
        self.assertEqual(calls, 2)

    def test_run_fails_fast_for_non_retryable_failure(self):
        retryer = self.make_retryer(max_retries=3)
        calls = 0

        def broken():
            nonlocal calls
            calls += 1
            raise ValueError("code bug")

        with self.assertRaises(ValueError):
            retryer.run(broken)
        self.assertEqual(calls, 1)

    def test_fit_retries_train_end_callback(self):
        retryer = self.make_retryer()
        trainer = Trainer.__new__(Trainer)
        trainer.retryer = retryer
        callback = EndCallback(
            [RuntimeError("PytorchStreamWriter failed writing file: file write failed")]
        )

        trainer._call_callback(callback, "on_train_end")

        self.assertEqual(callback.calls, 2)

    def test_fit_fails_fast_on_non_retryable_train_end_callback(self):
        retryer = self.make_retryer(max_retries=3)
        trainer = Trainer.__new__(Trainer)
        trainer.retryer = retryer
        callback = EndCallback([ValueError("bad callback state")])

        with self.assertRaises(ValueError):
            trainer._call_callback(callback, "on_train_end")

        self.assertEqual(callback.calls, 1)


if __name__ == "__main__":
    unittest.main()
