"""Exception hierarchy. Everything raised by followers derives from FollowersError."""

from __future__ import annotations


class FollowersError(Exception):
    """Base class for all followers errors."""


class FlowError(FollowersError):
    """A step failed while a flow was running.

    Attributes:
        step: name of the failing step (or None for flow-level failures).
        ctx:  the RunContext at the moment of failure — metrics, events and
              lineage collected so far remain inspectable on it.
    """

    def __init__(self, message: str, *, step: str | None = None, ctx=None):
        super().__init__(message)
        self.step = step
        self.ctx = ctx


class EarlyReturn(Exception):
    """Control flow, not an error: finish the run NOW with ``output``.

    Raise from a step, on_run_start or on_step_* middleware hooks to skip the
    remaining steps. Formal semantics (full spec in Flow.run):

      * skipped: remaining on_run_start hooks, remaining steps, and — when
        raised mid-step — that step's on_step_end hooks;
      * bypassed: on_step_error hooks, Retry, error counters (this is not
        an error);
      * still runs: EVERY middleware's on_run_end, in reverse order — a
        finalizer sweep, regardless of whether its on_run_start ran;
        ``ctx.short_circuited`` is True there.

    Used by Cache (run-level hits) and Quota (on_exceed="return"), and
    available to user code:

        def maybe_skip(payload, ctx):
            if payload in known:
                raise EarlyReturn(known[payload])
            return payload

    Deliberately NOT a FollowersError subclass so that generic error handlers
    don't swallow it.
    """

    def __init__(self, output=None):
        super().__init__("early return")
        self.output = output


class QuotaExceeded(FollowersError):
    """A Quota budget was exhausted (raised when on_exceed="raise").

    Attributes:
        budget: which budget tripped ("llm.calls", "cost", "seconds", "steps", ...).
        spent / limit: the numbers at the moment of the check.
        scope: "run" (this run's consumption) or "global" (the middleware
               instance's lifetime consumption across runs).
    """

    def __init__(self, message: str, *, budget: str, spent: float, limit: float,
                 scope: str = "run"):
        super().__init__(message)
        self.budget = budget
        self.spent = spent
        self.limit = limit
        self.scope = scope


class ValidationError(FollowersError):
    """A validator rejected a payload.

    Attributes:
        violations: list of human-readable violation messages.
    """

    def __init__(self, message: str, *, violations: list[str] | None = None, step: str | None = None):
        super().__init__(message)
        self.violations = violations or [message]
        self.step = step


class RegistryError(FollowersError):
    """A component reference could not be resolved."""


class PresetError(FollowersError):
    """A preset file is missing or malformed."""


class WrapError(FollowersError):
    """A foreign object could not be adapted into a Step.

    Attributes:
        tried: method names probed, in priority order.
        found: public attributes actually present on the object.
    """

    def __init__(self, message: str, *, tried: tuple = (), found: list | None = None):
        super().__init__(message)
        self.tried = tried
        self.found = found or []


class StoreError(FollowersError):
    """An artifact store operation failed."""


class ArtifactExpired(StoreError):
    """A handle points at an artifact that is gone (TTL / eviction / session end).

    A normal condition, not a bug: handles are leases. If the producing flow
    is replayable under the same inputs/config/artifact sources, a re-run
    re-creates the artifact; otherwise the caller must handle expiration
    explicitly (longer TTL, persist the output, or accept the loss).
    """

    def __init__(self, message: str, *, artifact_id: str = ""):
        super().__init__(message)
        self.artifact_id = artifact_id
