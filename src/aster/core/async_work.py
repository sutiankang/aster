"""Cancellation barriers that retain ownership until non-interruptible worker operations settle."""

import asyncio
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SettledWork:
    value: Any = None
    cancelled: bool = False
    error: Exception | None = None

    def unwrap(self):
        if self.error is not None:
            raise self.error
        return self.value


async def settle_thread(function, *args, **kwargs):
    """Wait for a worker without abandoning ownership on repeated cancellation.

    asyncio cancellation cannot kill the underlying thread or native operation.
    Every wait is shielded until it settles. The returned value keeps errors and
    caller cancellation separate so the caller can persist receipts or reclaim
    resources before propagating cancellation."""
    operation = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    cancelled = False
    while True:
        try:
            return SettledWork(await asyncio.shield(operation), cancelled)
        except asyncio.CancelledError:
            if operation.cancelled():
                raise
            cancelled = True
        except Exception as error:
            return SettledWork(cancelled=cancelled, error=error)
