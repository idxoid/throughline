"""Presets: declarative TOML flows.

    name = "rag-qa"
    extends = "base"              # optional: preset name or path

    [config]                      # arbitrary run config, deep-merged
    top_k = 3

    [[steps]]
    uses = "my_pkg.rag:make_retriever"   # import path or registered name
    name = "retrieve"                    # optional label
    effects = "pure"                     # optional: purity/side-effect declaration
    [steps.with]                         # kwargs -> uses(**kwargs) is the step
    top_k = 3

    [[steps]]
    uses = "answer"               # no [steps.with] -> used directly as a step

    [middleware.metrics]          # presence enables a module; table = kwargs
    [middleware.lineage]
    [middleware.validate]
    on_fail = "warn"
    [middleware.custom]
    uses = "my_pkg.mw:AuditTrail" # third-party middleware by import path

Search order for `load_preset("name")`: explicit path (contains / or .toml)
-> ./presets/ -> $THROUGHLINE_PRESETS (os.pathsep-separated dirs) -> throughline'
builtin presets.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from .errors import PresetError
from .flow import Flow
from .middleware import Middleware
from .registry import KINDS, check_kind, resolve
from .step import as_step

_BUILTIN_DIR = Path(__file__).parent / "presets"

_BUILTIN_MIDDLEWARE = {
    "metrics": "throughline.modules.metrics:MetricsMiddleware",
    "observe": "throughline.modules.observe:Observe",
    "lineage": "throughline.modules.lineage:LineageMiddleware",
    "validate": "throughline.modules.validate:Validate",
    "retry": "throughline.modules.retry:Retry",
    "cache": "throughline.modules.cache:Cache",
    "quota": "throughline.modules.quota:Quota",
}


def _search_dirs() -> list[Path]:
    dirs = [Path.cwd() / "presets"]
    env = os.environ.get("THROUGHLINE_PRESETS", "")
    dirs += [Path(p) for p in env.split(os.pathsep) if p]
    dirs.append(_BUILTIN_DIR)
    return dirs


def find_preset(ref: str) -> Path:
    """Locate a preset by path or by name across the search directories."""
    candidate = Path(ref)
    if candidate.suffix == ".toml" or os.sep in ref:
        if candidate.exists():
            return candidate
        raise PresetError(f"preset file not found: {ref}")
    for directory in _search_dirs():
        for filename in (f"{ref}.toml", ref):
            path = directory / filename
            if path.is_file():
                return path
    searched = ", ".join(str(d) for d in _search_dirs())
    raise PresetError(f"preset {ref!r} not found (searched: {searched})")


def list_presets() -> dict[str, Path]:
    """All discoverable presets, first hit per name wins."""
    found: dict[str, Path] = {}
    for directory in _search_dirs():
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.toml")):
            found.setdefault(path.stem, path)
    return found


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_preset_config(ref: str, _seen: frozenset = frozenset()) -> dict:
    """Load + resolve `extends` chain into one merged preset dict."""
    path = find_preset(ref)
    if str(path) in _seen:
        raise PresetError(f"circular 'extends' involving {path}")
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise PresetError(f"{path}: invalid TOML: {exc}") from exc

    parent_ref = data.pop("extends", None)
    if parent_ref:
        parent = load_preset_config(parent_ref, _seen | {str(path)})
        # config & middleware deep-merge; steps replace wholesale if present
        merged = _deep_merge(parent, {k: v for k, v in data.items() if k != "steps"})
        if "steps" in data:
            merged["steps"] = data["steps"]
        # the child keeps its own identity unless it names itself explicitly
        merged["name"] = data.get("name", path.stem)
        data = merged
    data.setdefault("name", path.stem)
    return data


def _kinds_satisfied(obj) -> list[str]:
    return [kind for kind in KINDS if check_kind(obj, kind) is None]


def _slot_check(obj, slot_kind: str, where: str) -> None:
    """Contract check at the point of use — kind mismatches fail at build time."""
    problem = check_kind(obj, slot_kind)
    if problem is None:
        return
    looks_like = [k for k in _kinds_satisfied(obj) if k != slot_kind]
    hint = f" It satisfies: {', '.join(looks_like)}." if looks_like else ""
    raise PresetError(f"{where}: {problem}.{hint}")


def build_flow(config: dict) -> Flow:
    """Materialize a merged preset dict into a Flow."""
    preset_name = config.get("name", "preset")
    steps = []
    for index, spec in enumerate(config.get("steps", [])):
        if not isinstance(spec, dict) or "uses" not in spec:
            raise PresetError(f"each [[steps]] needs a 'uses' key, got: {spec!r}")
        target = resolve(spec["uses"])
        kwargs = spec.get("with")  # a [steps.with] table (even empty) => factory call
        if kwargs is not None:
            try:
                target = target(**kwargs)
            except TypeError as exc:
                raise PresetError(
                    f"step {spec['uses']!r}: factory call failed with {kwargs}: {exc}"
                ) from exc
        _slot_check(target, "step",
                    f"preset {preset_name!r}, [[steps]] #{index + 1} ({spec['uses']!r})")
        steps.append(as_step(target, spec.get("name") or str(spec["uses"]),
                             effects=spec.get("effects")))

    middleware: list[Middleware] = []
    for name, options in config.get("middleware", {}).items():
        options = dict(options) if isinstance(options, dict) else {}
        if not options.pop("enabled", True):
            continue
        uses = options.pop("uses", None) or _BUILTIN_MIDDLEWARE.get(name)
        if uses is None:
            known = ", ".join(sorted(_BUILTIN_MIDDLEWARE))
            raise PresetError(
                f"unknown middleware {name!r} (builtin: {known}); "
                f"add 'uses = \"pkg.mod:Class\"' for custom middleware"
            )
        factory = resolve(uses)
        try:
            instance = factory(**options) if (options or isinstance(factory, type)) else factory
        except TypeError as exc:
            raise PresetError(f"middleware {name!r}: {uses} rejected {options}: {exc}") from exc
        _slot_check(instance, "middleware",
                    f"preset {preset_name!r}, [middleware.{name}] ({uses})")
        middleware.append(instance)

    return Flow(steps, middleware=middleware,
                name=preset_name,
                config=config.get("config", {}))


def load_preset(ref: str) -> Flow:
    """One-liner: preset name/path -> ready-to-run Flow."""
    return build_flow(load_preset_config(ref))


def inspect_preset(ref: str) -> dict:
    """Dry-check a preset without running it: resolve every slot, run the
    wrap detection and kind checks, and report per-slot results. This is the
    engine behind `throughline doctor`.
    """
    from .adapters import render_explain
    config = load_preset_config(ref)
    report: dict = {"name": config.get("name", ref), "steps": [], "middleware": []}

    for index, spec in enumerate(config.get("steps", [])):
        row = {"slot": f"[[steps]] #{index + 1}",
               "uses": spec.get("uses") if isinstance(spec, dict) else repr(spec)}
        try:
            if not isinstance(spec, dict) or "uses" not in spec:
                raise PresetError(f"each [[steps]] needs a 'uses' key, got: {spec!r}")
            target = resolve(spec["uses"])
            kwargs = spec.get("with")
            if kwargs is not None:
                target = target(**kwargs)
                row["factory"] = f"called with {kwargs}"
            problem = check_kind(target, "step")
            if problem:
                raise PresetError(problem)
            import inspect as _inspect
            from .step import Step
            if isinstance(target, Step):
                meta = ", ".join(f"{k}={v}" for k, v in target.meta.items() if v)
                row["detail"] = f"Step {target.name!r}" + (f" ({meta})" if meta else "")
            elif _inspect.isroutine(target) or isinstance(target, type(lambda: 0)):
                row["detail"] = "plain callable"
            else:
                row["detail"] = render_explain(target)
            row["status"] = "ok"
        except Exception as exc:
            row["status"] = "error"
            row["detail"] = str(exc)
        report["steps"].append(row)

    for name, options in config.get("middleware", {}).items():
        options = dict(options) if isinstance(options, dict) else {}
        row = {"slot": f"[middleware.{name}]"}
        if not options.pop("enabled", True):
            row.update(status="disabled", detail="enabled = false")
            report["middleware"].append(row)
            continue
        try:
            uses = options.pop("uses", None) or _BUILTIN_MIDDLEWARE.get(name)
            if uses is None:
                raise PresetError(f"unknown middleware {name!r} and no 'uses'")
            row["uses"] = uses
            factory = resolve(uses)
            instance = factory(**options) if (options or isinstance(factory, type)) else factory
            problem = check_kind(instance, "middleware")
            if problem:
                raise PresetError(problem)
            row["status"] = "ok"
            row["detail"] = type(instance).__name__
        except Exception as exc:
            row["status"] = "error"
            row["detail"] = str(exc)
        report["middleware"].append(row)

    report["ok"] = all(r["status"] in ("ok", "disabled")
                       for r in report["steps"] + report["middleware"])
    return report
