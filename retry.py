"""Small allowlist-based retry helper for transient infrastructure failures."""

from __future__ import annotations

import errno
import random
import time
from collections.abc import Callable
from typing import Any, Optional


class Retryer:
    """Retry callables only when the exception matches an allowlist."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay_seconds: float = 1.0,
        max_delay_seconds: float = 30.0,
        jitter_seconds: float = 0.25,
        retryable_os_errnos: Optional[list[str | int]] = None,
        retryable_runtime_error_substrings: Optional[list[str]] = None,
    ):
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if base_delay_seconds < 0:
            raise ValueError("base_delay_seconds must be >= 0")
        if max_delay_seconds < 0:
            raise ValueError("max_delay_seconds must be >= 0")
        if jitter_seconds < 0:
            raise ValueError("jitter_seconds must be >= 0")
        self.max_retries = max_retries
        self.base_delay_seconds = base_delay_seconds
        self.max_delay_seconds = max_delay_seconds
        self.jitter_seconds = jitter_seconds
        self.retryable_os_errnos = {
            self._resolve_errno(value) for value in (retryable_os_errnos or [])
        }
        self.retryable_runtime_error_substrings = [
            value.lower() for value in (retryable_runtime_error_substrings or [])
        ]

    @staticmethod
    def _resolve_errno(value: str | int) -> int:
        if isinstance(value, int):
            return value
        if not hasattr(errno, value):
            raise ValueError(f"Unknown errno name: {value}")
        return int(getattr(errno, value))

    def is_retryable(self, exc: Exception) -> bool:
        """Return true only for configured transient IO signatures."""
        if isinstance(exc, OSError):
            return exc.errno in self.retryable_os_errnos
        if isinstance(exc, RuntimeError):
            message = str(exc).lower()
            return any(
                substring in message
                for substring in self.retryable_runtime_error_substrings
            )
        return False

    def run(self, fn: Callable[[], Any], description: str = "operation") -> Any:
        """Run fn with bounded retries for allowlisted transient failures."""
        for attempt in range(self.max_retries + 1):
            try:
                return fn()
            except Exception as exc:
                if attempt >= self.max_retries or not self.is_retryable(exc):
                    raise
                delay = min(
                    self.base_delay_seconds * (2**attempt),
                    self.max_delay_seconds,
                )
                if self.jitter_seconds > 0:
                    delay += random.uniform(0, self.jitter_seconds)
                print(
                    f"Retryable failure during {description} "
                    f"(attempt {attempt + 1}/{self.max_retries + 1}): {exc}. "
                    f"Retrying in {delay:.1f}s."
                )
                time.sleep(delay)
        raise RuntimeError("unreachable retry loop state")
