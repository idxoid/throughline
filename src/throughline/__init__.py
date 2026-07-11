"""throughline — a framework-neutral control plane for LLM/RAG/agent pipelines.

Not another orchestrator competing for your steps: a zero-dependency kernel
that runs the pipeline and owns the cross-cutting concerns — validation,
metrics, observability, lineage, budgets — around components from any
framework (or none).

Five concepts:
    Step        anything callable (or wrappable) that transforms a payload
    Flow        an ordered chain of steps
    Middleware  pluggable pre/post-processing, validation, metrics,
                observability, lineage, retry — wrapped around every step
    Preset      a TOML file describing steps + middleware + config
    Context     carried through the run; collects events and artifacts

Quickstart:

    import throughline as tl

    flow = tl.Flow(
        [str.strip, tl.adapters.llm.FakeLLM().answer_step()],
        middleware=[tl.modules.MetricsMiddleware(), tl.modules.LineageMiddleware()],
    )
    result = flow.run("  what is throughline?  ")
    print(result.output, result.metrics, result.lineage.render_blame(), sep="\n")
"""

from .context import EventBus, Result, RunContext
from .errors import (ArtifactExpired, EarlyReturn, FlowError, ThroughlineError,
                     ManifestVerifyError, PolicyError, PresetError, QuotaExceeded,
                     RegistryError, StoreError, ValidationError, WrapError)
from .flow import Flow
from .middleware import Handled, Middleware
from .presets import build_flow, find_preset, list_presets, load_preset, load_preset_config
from .registry import (KINDS, available, check_kind, entries, load_plugins,
                       register, register_kind, resolve, unavailable)
from .step import Step, as_step, branch, map_step, parallel, step
from .store import ArtifactRef, MemoryArtifactStore
from .adapters import explain, wrap
from . import adapters, manifest, modules

__version__ = "0.1.0"

__all__ = [
    "Flow", "Step", "step", "as_step", "map_step", "parallel", "branch",
    "Middleware", "Handled",
    "RunContext", "Result", "EventBus",
    "register", "register_kind", "resolve", "available", "entries",
    "unavailable", "load_plugins",
    "KINDS", "check_kind",
    "wrap", "explain",
    "ArtifactRef", "MemoryArtifactStore",
    "load_preset", "load_preset_config", "build_flow", "list_presets", "find_preset",
    "ThroughlineError", "FlowError", "ValidationError", "RegistryError", "PresetError",
    "WrapError", "StoreError", "ArtifactExpired",
    "EarlyReturn", "QuotaExceeded", "PolicyError", "ManifestVerifyError",
    "adapters", "modules", "manifest",
    "__version__",
]
