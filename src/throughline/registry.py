"""Component registry: typed catalog of steps, middleware, stores, ...

The core knows a small closed set of component *kinds* (contracts), never
concrete implementations. Three ways a component becomes resolvable:

  1. ``@throughline.register("name", kind="step")`` in user code,
  2. an import path reference "package.module:attr" (no registration needed),
  3. a pip-installed package exposing the "throughline.plugins" entry-point
     group (discovered lazily on first miss).

A plugin entry point may load to a *manifest* — a dict exporting many typed
components at once:

    COMPONENTS = {
        "requires": "throughline>=0.1",       # optional compatibility gate
        "step:clean": clean,
        "middleware:audit": Audit,
        "store.cache:redis": RedisCache,    # subkind pins the protocol
        "store:memory": MemoryStore,        # umbrella kind also accepted
    }

Plain (un-prefixed) keys default to kind="step" for backward compatibility.
Broken or incompatible plugins never break the host: they are skipped and
reported as unavailable (see ``throughline components``).

The kind taxonomy is open where it matters and strict where it matters:
the core enforces its closed set of *built-in slots* (KINDS) at every point
of use, while plugins may introduce namespaced kinds of their own —
"acme.reranker:fast" in a manifest, ``register_kind("acme.reranker",
check=..., shape=...)`` to optionally give the kind a protocol. Undeclared
custom kinds are catalog-only: resolvable, listable, never enforced.

Name collisions resolve deterministically: builtin < plugin < local
registration; an explicit import path in a preset always wins by construction.
"""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Callable

from .errors import RegistryError

#: The closed set of *built-in runtime slots* the core understands. Each maps
#: to a structural check applied where the component is *used* (preset slots,
#: doctor) — duck-typing stays, but kind mismatches fail loudly at build time.
#:
#: The taxonomy itself is NOT closed: plugins may define namespaced kinds
#: ("acme.reranker") for their own components. The core catalogs them
#: (resolve/entries/`throughline components`) but enforces nothing — unless the
#: kind's author declared a protocol via ``register_kind``. Strict at the
#: point of use, open at the point of definition.
#:
#: Bare names outside this tuple are rejected for custom kinds, which makes
#: them reservable: "policy" spent its first release reserved exactly this
#: way, so no plugin could squat the name before the core defined its
#: protocol (a policy rule: callable (checkpoint, value, ctx) -> verdict).
#:
#: "store" is deliberately an UMBRELLA kind: two distinct protocols hide
#: behind one word — cache stores (get/set keyed by namespace+text) and
#: artifact stores (put -> ref, get(ref)). The umbrella accepts either; the
#: subkinds "store.cache" / "store.artifact" pin the exact protocol. Register
#: under the subkind when you know which one you implement (a Redis cache is
#: not an artifact store); the consuming slot enforces the specific protocol
#: at the point of use either way. Namespaces of built-in kinds ("store.*",
#: "step.*", ...) are reserved for the core — register_kind rejects them.
KINDS = ("step", "middleware", "store", "store.cache", "store.artifact",
         "embedder", "llm", "retriever", "sink", "verifier", "policy")

_HOOKS = ("on_run_start", "on_step_start", "wrap_step",
          "on_step_end", "on_step_error", "on_run_end")
_STEPPABLE = ("invoke", "query", "retrieve", "get_relevant_documents",
              "search", "run", "complete", "generate")


def _check_step(obj: Any) -> bool:
    return callable(obj) or any(callable(getattr(obj, m, None)) for m in _STEPPABLE)


def _check_middleware(obj: Any) -> bool:
    return any(callable(getattr(obj, hook, None)) for hook in _HOOKS)


def _check_cache_store(obj: Any) -> bool:
    return (callable(getattr(obj, "get", None)) and
            callable(getattr(obj, "set", None)))


def _check_artifact_store(obj: Any) -> bool:
    return (callable(getattr(obj, "get", None)) and
            callable(getattr(obj, "put", None)))


def _check_store(obj: Any) -> bool:
    # umbrella: either protocol qualifies (see the KINDS docstring)
    return _check_cache_store(obj) or _check_artifact_store(obj)


def _check_retriever(obj: Any) -> bool:
    return callable(obj) or any(callable(getattr(obj, m, None))
                                for m in ("retrieve", "get_relevant_documents",
                                          "search", "invoke", "query"))


def _check_llm(obj: Any) -> bool:
    return callable(obj) or any(callable(getattr(obj, m, None))
                                for m in ("complete", "generate", "invoke"))


_KIND_CHECKS: dict[str, Callable[[Any], bool]] = {
    "step": _check_step,
    "middleware": _check_middleware,
    "store": _check_store,
    "store.cache": _check_cache_store,
    "store.artifact": _check_artifact_store,
    "embedder": callable,
    "llm": _check_llm,
    "retriever": _check_retriever,
    "sink": callable,
    "verifier": callable,
    "policy": callable,
}

_KIND_SHAPES = {
    "step": "callable, or an object with one of "
            "invoke/query/retrieve/run/complete/generate",
    "middleware": "an object implementing Middleware hooks "
                  "(on_run_start/on_step_end/...)",
    "store": "umbrella: get+set (cache store) OR put+get (artifact store); "
             "use store.cache / store.artifact to pin the protocol",
    "store.cache": "get(namespace, text, default) + set(namespace, text, value)",
    "store.artifact": "put(value, ...) -> ArtifactRef-like + get(ref)",
    "embedder": "callable text -> vector",
    "llm": "callable prompt -> text, or complete()/generate()",
    "retriever": "callable, or retrieve/search/get_relevant_documents; "
                 "may return EvidenceChunk objects (the evidence contract) "
                 "instead of framework docs",
    "sink": "callable event -> None",
    "verifier": "callable (claim, evidence) -> score",
    "policy": "callable (checkpoint, value, ctx) -> verdict "
              "(None=abstain, Allow/Deny/Transform/Flag)",
}


#: Protocols declared by plugins for their namespaced kinds:
#: kind -> (check, shape description)
_CUSTOM_KINDS: dict[str, tuple[Callable[[Any], bool] | None, str]] = {}


def _is_namespaced(kind: str) -> bool:
    namespace, _, local = kind.partition(".")
    return bool(namespace and local)


def register_kind(kind: str, check: Callable[[Any], bool] | None = None,
                  shape: str = "") -> None:
    """Declare a plugin-defined kind, optionally with a structural protocol.

    Custom kinds must be namespaced ("acme.reranker") so they can never
    collide with current or future built-in slots. Without ``check`` the
    kind is catalog-only: components register and resolve under it, but
    ``check_kind`` enforces nothing. With ``check`` the kind gets the same
    build-time enforcement built-in kinds have; ``shape`` is the human
    description used in error messages.
    """
    if kind in KINDS:
        raise RegistryError(f"kind {kind!r} is built-in and cannot be redeclared")
    if not _is_namespaced(kind):
        raise RegistryError(
            f"custom kind {kind!r} must be namespaced ('yourpkg.{kind}') — "
            f"bare names are reserved for built-in slots")
    namespace = kind.partition(".")[0]
    if namespace in KINDS:
        raise RegistryError(
            f"namespace {namespace!r} belongs to the built-in kind taxonomy; "
            f"use your package name instead ('yourpkg.{kind.partition('.')[2]}')")
    _CUSTOM_KINDS[kind] = (check, shape)


def _valid_kind(kind: str) -> bool:
    if kind in KINDS:
        return True
    # custom kinds: namespaced, and never inside a built-in kind's namespace
    return _is_namespaced(kind) and kind.partition(".")[0] not in KINDS


def check_kind(obj: Any, kind: str) -> str | None:
    """Structural contract check. Returns an error message or None if OK.

    Built-in kinds are always enforced. Namespaced kinds are enforced only
    if their author declared a protocol (``register_kind(check=...)``);
    otherwise they pass — the core stays strict at its own slots without
    policing the ecosystem's taxonomy.
    """
    if kind in _KIND_CHECKS:
        if _KIND_CHECKS[kind](obj):
            return None
        return (f"{type(obj).__name__} does not satisfy the {kind!r} contract "
                f"(expected: {_KIND_SHAPES[kind]})")
    if _is_namespaced(kind) and kind.partition(".")[0] in KINDS:
        return (f"unknown kind {kind!r}: the {kind.partition('.')[0]!r} "
                f"namespace is reserved for built-in subkinds "
                f"({', '.join(k for k in KINDS if '.' in k)})")
    if _is_namespaced(kind):
        check, shape = _CUSTOM_KINDS.get(kind, (None, ""))
        if check is None or check(obj):
            return None
        return (f"{type(obj).__name__} does not satisfy the {kind!r} contract"
                + (f" (expected: {shape})" if shape else ""))
    return (f"unknown kind {kind!r}; built-in: {', '.join(KINDS)} "
            f"(custom kinds must be namespaced, e.g. 'yourpkg.{kind}')")


@dataclass
class RegistryEntry:
    name: str
    kind: str
    obj: Any
    source: str = "local"   # "local" | "plugin:<entry-point>" | "builtin"


_REGISTRY: dict[tuple[str, str], RegistryEntry] = {}   # (kind, name) -> entry
_UNAVAILABLE: dict[str, str] = {}                      # plugin name -> reason
_PLUGINS_LOADED = False
_SOURCE_RANK = {"builtin": 0, "plugin": 1, "local": 2}


def _rank(source: str) -> int:
    return _SOURCE_RANK.get(source.partition(":")[0], 2)


def _store(name: str, obj: Any, kind: str, source: str) -> None:
    if not _valid_kind(kind):
        raise RegistryError(
            f"unknown kind {kind!r}; built-in: {', '.join(KINDS)}. "
            f"Plugin-defined kinds must be namespaced with your package name "
            f"('yourpkg.{kind.rpartition('.')[2]}'); built-in namespaces "
            f"(step.*, store.*, ...) are reserved for the core")
    existing = _REGISTRY.get((kind, name))
    if existing is not None and _rank(existing.source) > _rank(source):
        return  # lower-precedence source never shadows a higher one
    _REGISTRY[(kind, name)] = RegistryEntry(name=name, kind=kind, obj=obj, source=source)


def register(name: str | None = None, obj: Any = None, *, kind: str = "step"):
    """Register a component under a kind. Decorator or plain call.

        @register("clean")                        # kind defaults to "step"
        def clean(text): ...

        register("redis", RedisCache(), kind="store.cache")
    """
    if obj is not None:
        _store(name or getattr(obj, "__name__", None) or repr(obj), obj, kind, "local")
        return obj

    def decorate(target: Any) -> Any:
        _store(name or getattr(target, "__name__", "component"), target, kind, "local")
        return target
    return decorate


def resolve(ref: str | Any, kind: str | None = None) -> Any:
    """Resolve a reference to a component.

    - non-str: returned as is
    - "pkg.mod:attr": imported (attr may be dotted for nested access)
    - anything else: registry lookup; ``kind`` narrows the search and makes
      cross-kind mistakes loud ("registered as store, not step").
    """
    if not isinstance(ref, str):
        return ref
    if ":" in ref:
        module_name, _, attr_path = ref.partition(":")
        try:
            target = importlib.import_module(module_name)
        except ImportError as exc:
            raise RegistryError(f"cannot import {module_name!r} for {ref!r}: {exc}") from exc
        for attr in attr_path.split("."):
            try:
                target = getattr(target, attr)
            except AttributeError as exc:
                raise RegistryError(f"{module_name!r} has no attribute {attr!r} ({ref!r})") from exc
        return target

    if not any(name == ref for _, name in _REGISTRY):
        load_plugins()

    if kind is not None:
        entry = _REGISTRY.get((kind, ref))
        if entry is not None:
            return entry.obj
        elsewhere = [e for (k, n), e in _REGISTRY.items() if n == ref]
        if elsewhere:
            kinds = ", ".join(sorted(e.kind for e in elsewhere))
            raise RegistryError(
                f"component {ref!r} is registered as {kinds}, not {kind!r}")
    else:
        matches = [e for (k, n), e in _REGISTRY.items() if n == ref]
        if len(matches) == 1:
            return matches[0].obj
        if matches:  # same name under several kinds: prefer built-in slots...
            for preferred in KINDS:
                for e in matches:
                    if e.kind == preferred:
                        return e.obj
            # ...then custom kinds, deterministically by kind name
            return sorted(matches, key=lambda e: e.kind)[0].obj

    known = ", ".join(sorted({n for _, n in _REGISTRY})) or "(none)"
    raise RegistryError(
        f"unknown component {ref!r}; registered: {known}. "
        f"Use 'package.module:attr' to reference unregistered code."
    ) from None


def available(kind: str | None = None) -> dict[str, Any]:
    """Snapshot {name: component} of registered components (plugins included)."""
    load_plugins()
    return {e.name: e.obj for e in _REGISTRY.values()
            if kind is None or e.kind == kind}


def entries() -> list[RegistryEntry]:
    """Full typed catalog — what ``throughline components`` renders."""
    load_plugins()
    return sorted(_REGISTRY.values(), key=lambda e: (e.kind, e.name))


def unavailable() -> dict[str, str]:
    """Plugins that failed to load or were incompatible, with reasons."""
    load_plugins()
    return dict(_UNAVAILABLE)


def _compatible(requirement: str) -> str | None:
    """Minimal 'throughline>=X.Y' gate. Returns a reason string if incompatible."""
    from . import __version__
    match = re.fullmatch(r"\s*throughline\s*>=\s*([\d.]+)\s*", requirement)
    if match is None:
        return f"unsupported requires spec {requirement!r} (only 'throughline>=X.Y')"
    def parts(version: str) -> tuple:
        return tuple(int(p) for p in version.split(".") if p.isdigit())
    if parts(__version__) < parts(match.group(1)):
        return f"requires throughline>={match.group(1)}, installed {__version__}"
    return None


def _register_manifest(manifest: dict, source: str) -> list[str]:
    loaded = []
    for key, component in manifest.items():
        if key == "requires":
            continue
        kind, _, name = key.partition(":")
        if not name:                       # un-prefixed key: legacy flat dict
            kind, name = "step", key
        _store(name, component, kind, source)
        loaded.append(name)
    return loaded


def load_plugins(group: str = "throughline.plugins") -> list[str]:
    """Discover pip-installed plugins via entry points (idempotent)."""
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return []
    _PLUGINS_LOADED = True
    loaded: list[str] = []
    try:
        entry_points = metadata.entry_points(group=group)
    except Exception:
        return loaded
    for entry_point in entry_points:
        source = f"plugin:{entry_point.name}"
        try:
            obj = entry_point.load()
            if callable(obj) and getattr(obj, "__name__", "") in ("register", "register_components"):
                obj = obj()
            if isinstance(obj, dict):
                requirement = obj.get("requires")
                if isinstance(requirement, str):
                    reason = _compatible(requirement)
                    if reason is not None:
                        _UNAVAILABLE[entry_point.name] = reason
                        continue
                loaded.extend(_register_manifest(obj, source))
            else:
                _store(entry_point.name, obj, "step", source)
                loaded.append(entry_point.name)
        except Exception as exc:  # a broken plugin must not break the host
            _UNAVAILABLE[entry_point.name] = f"failed to load: {exc!r}"
            continue
    return loaded


def _reset_for_tests() -> None:
    global _PLUGINS_LOADED
    _REGISTRY.clear()
    _UNAVAILABLE.clear()
    _CUSTOM_KINDS.clear()
    _PLUGINS_LOADED = False
