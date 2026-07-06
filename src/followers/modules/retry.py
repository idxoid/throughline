"""Retry middleware: exponential backoff around flaky steps (LLM calls, HTTP)."""

from __future__ import annotations

import time
from fnmatch import fnmatch

from ..context import RunContext
from ..errors import EarlyReturn
from ..middleware import Middleware
from ..step import Step


class Retry(Middleware):
    """Retry failing steps.

    Args:
        attempts: total tries per step (1 = no retry).
        backoff:  initial sleep between tries, seconds.
        factor:   backoff multiplier per attempt.
        retry_on: exception type(s) that qualify for retry.
        step:     fnmatch pattern of step names to protect (None = all steps).
    """

    name = "retry"

    def __init__(self, attempts: int = 3, backoff: float = 0.1, factor: float = 2.0,
                 retry_on: type[BaseException] | tuple = (Exception,),
                 step: str | None = None):
        self.attempts = max(1, int(attempts))
        self.backoff = backoff
        self.factor = factor
        self.retry_on = retry_on if isinstance(retry_on, tuple) else (retry_on,)
        self.step = step

    def wrap_step(self, invoke, ctx: RunContext, step: Step):
        if self.step and not fnmatch(step.name, self.step):
            return invoke

        def retried(payload):
            delay = self.backoff
            for attempt in range(1, self.attempts + 1):
                try:
                    return invoke(payload)
                except EarlyReturn:
                    raise  # control flow — never retried
                except self.retry_on as exc:
                    if attempt == self.attempts:
                        raise
                    ctx.metric("retries")
                    ctx.emit("step_retry", step=step.name, attempt=attempt,
                             error=repr(exc), sleep=delay)
                    time.sleep(delay)
                    delay *= self.factor
        return retried
