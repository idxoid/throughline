"""Cache middleware: short-circuit repeated requests before the heavy steps.

Two modes on one class:

  * run-level (default, ``step=None``): the incoming payload is looked up in
    ``on_run_start``; a hit raises EarlyReturn — the whole pipeline is
    skipped. Place Cache AFTER the observers (Observe, MetricsMiddleware)
    and BEFORE everything else: a hit short-circuits every layer inside it,
    so an outermost Cache would fire before the observers attach and the hit
    would be invisible (no cache.hits metric, no cache_hit event).
  * step-level (``step="llm*"``): only matching steps are memoized
    (payload -> output); the rest of the pipeline still runs.

Exact matching is the zero-dependency default: the payload is canonicalized
(sorted JSON) and hashed. Pass ``embedder`` (any ``text -> vector`` callable —
an API client, sentence-transformers, whatever) to get a *semantic* cache:
lookups match on cosine similarity >= ``threshold``. ``SemanticCache`` is the
explicit spelling of that mode.

The store lives on the middleware instance, so it is shared across runs of
the same Flow — that is the point. Pass ``store=`` to plug an external
backend (anything with ``get(namespace, text, default)`` / ``set(namespace,
text, value)``).

Purity guard: a cached hit SKIPS the step (or the whole flow) — any side
effect inside it (db write, email, webhook) silently does not happen. Purity
is undecidable statically, so the contract is declarative: steps state their
effects (``@tl.step("save", effects="db.write")``, ``as_step(fn,
effects=...)``), and Cache enforces the declaration via ``on_effects``:

  * "skip" (default): declared-effectful steps are never served from or
    written to the cache. Step-level: that step runs uncached. Run-level:
    the run is not stored (and hence never hit) — a "cache_effects_bypass"
    event + "cache.effects_bypass" metric say why.
  * "raise": treat the combination as a config error and fail the run —
    for pipelines where an accidentally cached side effect is unacceptable.
  * "allow": cache anyway (e.g. idempotent writes) — an explicit opt-in.

Undeclared steps (``effects=None``) are treated as cacheable, as before:
the guard trusts declarations rather than guessing.

Preset usage (embedder/key resolvable by import path):

    [middleware.cache]
    step = "llm*"
    max_size = 1024
    ttl = 600
    # embedder = "my_pkg.embeddings:embed"   # -> semantic mode
    # key = "question"                       # cache key = payload["question"]
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import threading
import time
from collections import OrderedDict
from fnmatch import fnmatch
from typing import Any, Callable

from ..context import RunContext
from ..errors import EarlyReturn, FlowError
from ..middleware import Middleware
from ..step import Step

_MISS = object()


class LRUCache:
    """Thread-safe in-memory LRU with optional TTL (seconds)."""

    def __init__(self, max_size: int = 512, ttl: float | None = None):
        self.max_size = max_size
        self.ttl = ttl
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def _digest(self, namespace: str, text: str) -> str:
        return hashlib.sha256(f"{namespace}\x00{text}".encode()).hexdigest()

    def get(self, namespace: str, text: str, default: Any = _MISS) -> Any:
        key = self._digest(namespace, text)
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return default
            stamp, value = item
            if self.ttl is not None and time.monotonic() - stamp > self.ttl:
                del self._data[key]
                return default
            self._data.move_to_end(key)
            return value

    def set(self, namespace: str, text: str, value: Any) -> None:
        key = self._digest(namespace, text)
        with self._lock:
            self._data[key] = (time.monotonic(), value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_size:
                self._data.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


class SemanticStore:
    """Embedding-based store: exact text hit first, then cosine >= threshold."""

    def __init__(self, embedder: Callable[[str], Any], threshold: float = 0.92,
                 max_size: int = 512, ttl: float | None = None):
        self.embedder = embedder
        self.threshold = threshold
        self.max_size = max_size
        self.ttl = ttl
        # entries: (namespace, text, vector, norm, stamp, value)
        self._entries: list[tuple[str, str, tuple, float, float, Any]] = []
        self._lock = threading.Lock()

    @staticmethod
    def _norm(vector: tuple) -> float:
        return math.sqrt(sum(x * x for x in vector)) or 1.0

    def _embed(self, text: str) -> tuple[tuple, float]:
        vector = tuple(float(x) for x in self.embedder(text))
        return vector, self._norm(vector)

    def get(self, namespace: str, text: str, default: Any = _MISS) -> Any:
        vector, norm = self._embed(text)
        now = time.monotonic()
        with self._lock:
            if self.ttl is not None:
                self._entries = [e for e in self._entries if now - e[4] <= self.ttl]
            best, best_similarity = None, self.threshold
            for entry in self._entries:
                if entry[0] != namespace:
                    continue
                if entry[1] == text:
                    return entry[5]  # exact match wins immediately
                similarity = sum(a * b for a, b in zip(vector, entry[2])) / (norm * entry[3])
                if similarity >= best_similarity:
                    best, best_similarity = entry, similarity
            return best[5] if best is not None else default

    def set(self, namespace: str, text: str, value: Any) -> None:
        vector, norm = self._embed(text)
        with self._lock:
            self._entries.append((namespace, text, vector, norm, time.monotonic(), value))
            if len(self._entries) > self.max_size:
                self._entries = self._entries[-self.max_size:]

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


class Cache(Middleware):
    """Exact or semantic caching for whole runs or individual steps.

    Args:
        step:      fnmatch pattern of step names to memoize; None (default)
                   caches at run level and short-circuits the whole flow.
        key:       what identifies a request. A callable ``payload -> Any``,
                   or a string dict key (like lineage's extract). Default:
                   the whole payload, canonicalized.
        embedder:  ``text -> vector`` callable (or "pkg.mod:fn" import path);
                   turns the cache semantic.
        threshold: cosine similarity for a semantic hit (default 0.92).
        max_size / ttl: store capacity and entry lifetime (seconds).
        version:   salt — bump to invalidate everything after a logic change.
        copy:      deep-copy values on set/get so cached payloads can't be
                   mutated by callers (default True).
        store:     external backend with get/set(namespace, text, ...).
        on_effects: what to do when a step declares side effects (see module
                   docstring): "skip" (default) | "raise" | "allow".
    """

    name = "cache"

    def __init__(self, step: str | None = None, key: Callable | str | None = None,
                 embedder: Callable | str | None = None, threshold: float = 0.92,
                 max_size: int = 512, ttl: float | None = None, version: str = "",
                 copy: bool = True, store: Any = None, on_effects: str = "skip"):
        if on_effects not in ("skip", "raise", "allow"):
            raise ValueError(
                f"on_effects must be 'skip', 'raise' or 'allow', got {on_effects!r}")
        if isinstance(embedder, str):
            from ..registry import resolve
            embedder = resolve(embedder)
        if isinstance(key, str):
            field = key
            key = lambda p: p.get(field, p) if isinstance(p, dict) else p  # noqa: E731
        self.step = step
        self.key_fn = key
        self.version = version
        self.copy = copy
        self.on_effects = on_effects
        # per-run pending state goes under a per-instance artifact key so
        # stacked run-level caches (e.g. exact over semantic) each store
        self._pending_key = f"cache_pending:{id(self)}"
        self.semantic = embedder is not None
        if store is not None:
            self._store = store
        elif embedder is not None:
            self._store = SemanticStore(embedder, threshold, max_size, ttl)
        else:
            self._store = LRUCache(max_size, ttl)

    # -- key & value handling -------------------------------------------------
    def _text(self, payload: Any) -> str:
        base = self.key_fn(payload) if self.key_fn else payload
        if isinstance(base, str):
            text = base
        else:
            text = json.dumps(base, sort_keys=True, ensure_ascii=False, default=str)
        return f"{self.version}\x00{text}" if self.version else text

    def _copy(self, value: Any) -> Any:
        if not self.copy:
            return value
        try:
            return copy.deepcopy(value)
        except Exception:
            return value

    # -- purity guard -----------------------------------------------------------
    def _guard(self, ctx: RunContext, step: Step, scope: str) -> bool:
        """True = caching for this step/run must be bypassed."""
        effects = step.meta.get("effects")  # None/() are fine; labels are not
        if not effects or self.on_effects == "allow":
            return False
        if self.on_effects == "raise":
            raise FlowError(
                f"cache: step {step.name!r} declares side effects "
                f"{list(effects)} — a {scope}-level cache hit would silently "
                f"skip them. Exclude the step from caching, or pass "
                f"on_effects='allow' if the effects are idempotent.")
        ctx.metric("cache.effects_bypass")
        ctx.emit("cache_effects_bypass", scope=scope, step=step.name,
                 effects=list(effects))
        return True

    # -- step-level -------------------------------------------------------------
    def wrap_step(self, invoke, ctx: RunContext, step: Step):
        if self.step is None or not fnmatch(step.name, self.step):
            return invoke
        if self._guard(ctx, step, "step"):
            return invoke  # run uncached: side effects must happen every time

        def cached(payload):
            namespace = f"step:{ctx.flow}:{step.name}"
            text = self._text(payload)
            value = self._store.get(namespace, text, _MISS)
            if value is not _MISS:
                ctx.metric("cache.hits")
                ctx.emit("cache_hit", step=step.name, semantic=self.semantic)
                return self._copy(value)
            ctx.metric("cache.misses")
            output = invoke(payload)
            self._store.set(namespace, text, self._copy(output))
            return output
        return cached

    # -- run-level ----------------------------------------------------------------
    def on_run_start(self, ctx: RunContext, payload):
        if self.step is not None:
            return payload
        namespace = f"run:{ctx.flow}"
        text = self._text(payload)
        value = self._store.get(namespace, text, _MISS)
        if value is not _MISS:
            ctx.metric("cache.hits")
            ctx.emit("cache_hit", scope="run", semantic=self.semantic)
            raise EarlyReturn(self._copy(value))
        ctx.metric("cache.misses")
        ctx.artifacts[self._pending_key] = (namespace, text)
        return payload

    def on_step_start(self, ctx: RunContext, step: Step, payload):
        # run-level purity guard: steps are only visible once the run is under
        # way, so the check happens here. Dropping the pending entry means a
        # run that contains a declared-effectful step is never stored — and a
        # run that is never stored can never be served as a hit.
        if self.step is not None or self._pending_key not in ctx.artifacts:
            return payload
        if self._guard(ctx, step, "run"):
            ctx.artifacts.pop(self._pending_key, None)
        return payload

    def on_run_end(self, ctx: RunContext, output):
        pending = ctx.artifacts.pop(self._pending_key, None)
        if pending is not None:
            self._store.set(pending[0], pending[1], self._copy(output))
        return output


class SemanticCache(Cache):
    """Cache with a required embedder — the explicit semantic spelling.

        SemanticCache(embedder=my_embed, threshold=0.9, step="llm*")
    """

    name = "semantic-cache"

    def __init__(self, embedder: Callable | str, threshold: float = 0.92, **kwargs: Any):
        super().__init__(embedder=embedder, threshold=threshold, **kwargs)
