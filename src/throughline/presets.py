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

Slots — a preset may declare holes the user must fill (or ship defaults):

    [slots.retriever]
    kind = "step"                        # optional: contract of what fills it
    description = "your corpus retriever"
    default = "examples.rag_docs:retriever"   # optional: makes the slot optional

    [[steps]]
    uses = "@retriever"                  # slot reference
    [steps.with]
    top_k = 4

`@name` works wherever a component reference does: step `uses`, middleware
`uses`, composite inner refs, and whole-string values inside `[steps.with]` /
middleware options (substituted with the *resolved* object, so factories get
live components). `@@x` escapes a literal leading `@`. Fill precedence:
``load_preset(..., fill={...})`` > the `[fill]` table (deep-merged through
`extends`) > the slot's `default`. A referenced slot with no fill fails at
build time, listing every missing slot at once; `doctor` reports slot status
without failing fast.

    [fill]
    retriever = "my_pkg.rag:make_retriever"

Composites — a [[steps]] entry takes exactly one of `uses`/`map`/`parallel`/
`branch`:

    [[steps]]
    map = "my_pkg.report:section_step"   # inner ref (or "@slot")
    workers = 4                          # optional fan-out
    [steps.with]                         # optional: INNER factory kwargs

    [[steps]]
    name = "gather"
    workers = 2
    [steps.parallel]                     # same payload to every entry
    summary = "my_pkg:summarize"
    stats = "my_pkg:stats"

    [[steps]]
    [steps.branch]
    selector = "lang"                    # payload key; "pkg.mod:fn" imports
    default = "my_pkg:en_step"           # optional fallback route
    [steps.branch.routes]
    ru = "my_pkg:ru_step"
    en = "my_pkg:en_step"

Composite inner refs are used directly (no per-route factory call): point
them at ready steps or wrap factories in your module.

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
from .step import Step, as_step, branch, map_step, parallel

_BUILTIN_DIR = Path(__file__).parent / "presets"

_BUILTIN_MIDDLEWARE = {
    "metrics": "throughline.modules.metrics:MetricsMiddleware",
    "observe": "throughline.modules.observe:Observe",
    "lineage": "throughline.modules.lineage:LineageMiddleware",
    "validate": "throughline.modules.validate:Validate",
    "retry": "throughline.modules.retry:Retry",
    "cache": "throughline.modules.cache:Cache",
    "quota": "throughline.modules.quota:Quota",
    "policy": "throughline.modules.policy:Policy",
}

_STEP_FORMS = ("uses", "map", "parallel", "branch")


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
        # config, middleware, slots & fill deep-merge; steps replace wholesale
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


# ---------------------------------------------------------------------------
# Slots
# ---------------------------------------------------------------------------

def _is_slot_ref(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("@") and not value.startswith("@@")


def _referenced_slots(config: dict) -> set[str]:
    """Every `@name` mentioned anywhere in steps or middleware."""
    found: set[str] = set()

    def walk(node: Any) -> None:
        if _is_slot_ref(node):
            found.add(node[1:])
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(config.get("steps", []))
    walk(config.get("middleware", {}))
    return found


class _Slots:
    """Slot declarations + fills for one build: resolution and bookkeeping.

    Fill precedence: explicit ``fill=`` argument > the preset's ``[fill]``
    table > the slot's ``default``. Fill values are component references
    (import path / registered name) or, via the Python argument, live objects.
    """

    def __init__(self, config: dict, fill: dict | None = None):
        self.declared: dict[str, dict] = {}
        for name, spec in (config.get("slots") or {}).items():
            if not isinstance(spec, dict):
                raise PresetError(f"[slots.{name}] must be a table, got {spec!r}")
            self.declared[name] = spec
        self.fills: dict[str, Any] = {}
        self.sources: dict[str, str] = {}
        for name, spec in self.declared.items():
            if "default" in spec:
                self.fills[name] = spec["default"]
                self.sources[name] = "default"
        for source, table in (("[fill]", config.get("fill") or {}),
                              ("fill=", fill or {})):
            for name, value in table.items():
                if name not in self.declared:
                    known = ", ".join(sorted(self.declared)) or "(none declared)"
                    raise PresetError(
                        f"{source} names unknown slot {name!r}; declared slots: {known}")
                self.fills[name] = value
                self.sources[name] = source
        self.referenced = _referenced_slots(config)
        self._resolved: dict[str, Any] = {}

    def missing(self) -> list[str]:
        return sorted(name for name in self.referenced
                      if name in self.declared and name not in self.fills)

    def require_filled(self, preset_name: str) -> None:
        missing = self.missing()
        if not missing:
            return
        lines = []
        for name in missing:
            description = self.declared[name].get("description", "")
            kind = self.declared[name].get("kind")
            lines.append(f"  @{name}" + (f" (kind={kind})" if kind else "")
                         + (f" — {description}" if description else ""))
        raise PresetError(
            f"preset {preset_name!r} has unfilled slots:\n" + "\n".join(lines)
            + "\nFill via [fill] in an extending preset, load_preset(fill=...) "
              "or `throughline run ... --fill name=pkg.mod:obj`.")

    def resolve_ref(self, ref: Any, where: str) -> tuple[Any, str | None]:
        """Resolve a component reference; `@name` goes through the slot fill.

        Returns (object, slot_name-or-None) so the caller can run the slot's
        kind check on whatever finally occupies the position.
        """
        if not isinstance(ref, str):
            return ref, None
        if ref.startswith("@@"):
            return resolve(ref[1:]), None
        if not ref.startswith("@"):
            return resolve(ref), None
        name = ref[1:]
        if name not in self.declared:
            known = ", ".join(f"@{n}" for n in sorted(self.declared)) or "(none)"
            raise PresetError(f"{where}: reference to undeclared slot {ref!r}; "
                              f"declared: {known}")
        if name not in self.fills:
            spec = self.declared[name]
            description = spec.get("description", "")
            raise PresetError(f"{where}: slot {ref!r} is unfilled"
                              + (f" — {description}" if description else ""))
        if name not in self._resolved:
            self._resolved[name] = resolve(self.fills[name])
        return self._resolved[name], name

    def check_kind(self, slot_name: str, obj: Any, where: str) -> None:
        kind = self.declared[slot_name].get("kind")
        if kind:
            _slot_check(obj, kind, f"{where} (slot @{slot_name})")

    def substitute(self, options: Any, where: str) -> Any:
        """Replace whole-string `@name` values (recursively) with resolved
        fill objects, running the slot's kind check in place."""
        if _is_slot_ref(options):
            obj, slot = self.resolve_ref(options, where)
            if slot:
                self.check_kind(slot, obj, where)
            return obj
        if isinstance(options, str) and options.startswith("@@"):
            return options[1:]
        if isinstance(options, dict):
            return {k: self.substitute(v, where) for k, v in options.items()}
        if isinstance(options, list):
            return [self.substitute(v, where) for v in options]
        return options

    def report(self) -> list[dict]:
        rows = []
        for name, spec in self.declared.items():
            row = {"slot": f"[slots.{name}]", "kind": spec.get("kind"),
                   "description": spec.get("description", "")}
            if name in self.fills:
                row["fill"] = (self.fills[name] if isinstance(self.fills[name], str)
                               else type(self.fills[name]).__name__)
                row["source"] = self.sources[name]
                row["status"] = "ok"
            elif name in self.referenced:
                row["status"] = "missing"
                row["detail"] = "referenced but unfilled"
            else:
                row["status"] = "unreferenced"
                row["detail"] = "declared but never referenced"
            rows.append(row)
        return rows


# ---------------------------------------------------------------------------
# Steps (plain and composite)
# ---------------------------------------------------------------------------

def _selector_fn(selector: Any, slots: _Slots, where: str):
    """Branch selector: import path / @slot -> callable, plain string -> payload key."""
    if callable(selector):
        return selector
    if isinstance(selector, str) and (":" in selector or _is_slot_ref(selector)):
        fn, slot = slots.resolve_ref(selector, where)
        if slot:
            slots.check_kind(slot, fn, where)
        if not callable(fn):
            raise PresetError(f"{where}: selector {selector!r} is not callable")
        return fn
    if isinstance(selector, str):
        key = selector
        return lambda payload: payload[key]
    raise PresetError(f"{where}: selector must be a payload key or import path, "
                      f"got {selector!r}")


def _materialize_step(spec: dict, index: int, slots: _Slots,
                      preset_name: str) -> tuple[Step, dict]:
    """One [[steps]] entry -> (Step, doctor info). Shared by build and doctor."""
    where = f"preset {preset_name!r}, [[steps]] #{index + 1}"
    if not isinstance(spec, dict):
        raise PresetError(f"{where}: each [[steps]] must be a table, got: {spec!r}")
    forms = [form for form in _STEP_FORMS if form in spec]
    if len(forms) != 1:
        raise PresetError(f"{where}: needs exactly one of "
                          f"{'/'.join(_STEP_FORMS)}, got: {forms or spec!r}")
    form = forms[0]
    name = spec.get("name")
    info: dict = {}

    def apply_factory(target: Any) -> Any:
        kwargs = spec.get("with")
        if kwargs is None:
            return target
        kwargs = slots.substitute(kwargs, where)
        info["factory"] = f"called with {kwargs}"
        try:
            return target(**kwargs)
        except TypeError as exc:
            raise PresetError(
                f"{where}: factory call failed with {kwargs}: {exc}") from exc

    if form == "uses":
        info["uses"] = spec["uses"] if isinstance(spec["uses"], str) else repr(spec["uses"])
        target, slot = slots.resolve_ref(spec["uses"], where)
        target = apply_factory(target)
        if slot:
            slots.check_kind(slot, target, where)
        _slot_check(target, "step", f"{where} ({info['uses']!r})")
        info["target"] = target  # pre-wrap object, for doctor's explain detail
        return as_step(target, name or info["uses"], effects=spec.get("effects")), info

    workers = int(spec.get("workers", 1))
    if form == "map":
        info["uses"] = f"map {spec['map']}" if isinstance(spec["map"], str) else "map"
        inner, slot = slots.resolve_ref(spec["map"], where)
        inner = apply_factory(inner)
        if slot:
            slots.check_kind(slot, inner, where)
        composed = map_step(inner, name=name, workers=workers)
    elif form == "parallel":
        table = spec["parallel"]
        if not isinstance(table, dict) or not table:
            raise PresetError(f"{where}: [steps.parallel] must be a non-empty "
                              f"table of name = ref entries")
        info["uses"] = f"parallel({','.join(table)})"
        resolved = {}
        for key, ref in table.items():
            obj, slot = slots.resolve_ref(ref, f"{where} parallel.{key}")
            if slot:
                slots.check_kind(slot, obj, f"{where} parallel.{key}")
            resolved[key] = obj
        composed = parallel(resolved, name=name,
                            workers=workers if workers > 1 else None)
    else:  # branch
        table = spec["branch"]
        if not isinstance(table, dict):
            raise PresetError(f"{where}: [steps.branch] must be a table")
        routes = table.get("routes")
        if not isinstance(routes, dict) or not routes:
            raise PresetError(f"{where}: [steps.branch.routes] must be a "
                              f"non-empty table of value = ref entries")
        if "selector" not in table:
            raise PresetError(f"{where}: [steps.branch] needs a 'selector' "
                              f"(payload key or import path)")
        info["uses"] = f"branch on {table['selector']!r} ({','.join(map(str, routes))})"
        selector = _selector_fn(table["selector"], slots, where)
        resolved = {}
        for key, ref in routes.items():
            obj, slot = slots.resolve_ref(ref, f"{where} routes.{key}")
            if slot:
                slots.check_kind(slot, obj, f"{where} routes.{key}")
            resolved[key] = obj
        default = None
        if "default" in table:
            default, slot = slots.resolve_ref(table["default"], f"{where} default")
            if slot:
                slots.check_kind(slot, default, f"{where} default")
        composed = branch(selector, resolved, default=default, name=name)

    return as_step(composed, name, effects=spec.get("effects")), info


def _step_detail(target: Step) -> str:
    kind = target.meta.get("kind")
    if kind in ("map", "parallel", "branch"):
        extra = f", workers={target.meta['workers']}" if target.meta.get("workers") else ""
        return f"composite {kind} step {target.name!r}{extra}"
    meta = ", ".join(f"{k}={v}" for k, v in target.meta.items() if v)
    return f"Step {target.name!r}" + (f" ({meta})" if meta else "")


# ---------------------------------------------------------------------------
# Building & inspecting
# ---------------------------------------------------------------------------

def build_flow(config: dict, fill: dict | None = None) -> Flow:
    """Materialize a merged preset dict into a Flow."""
    preset_name = config.get("name", "preset")
    slots = _Slots(config, fill)
    slots.require_filled(preset_name)

    steps = []
    for index, spec in enumerate(config.get("steps", [])):
        step, _info = _materialize_step(spec, index, slots, preset_name)
        steps.append(step)

    middleware: list[Middleware] = []
    for name, options in config.get("middleware", {}).items():
        options = dict(options) if isinstance(options, dict) else {}
        if not options.pop("enabled", True):
            continue
        where = f"preset {preset_name!r}, [middleware.{name}]"
        uses = options.pop("uses", None) or _BUILTIN_MIDDLEWARE.get(name)
        if uses is None:
            known = ", ".join(sorted(_BUILTIN_MIDDLEWARE))
            raise PresetError(
                f"unknown middleware {name!r} (builtin: {known}); "
                f"add 'uses = \"pkg.mod:Class\"' for custom middleware"
            )
        factory, slot = slots.resolve_ref(uses, where)
        options = slots.substitute(options, where)
        try:
            instance = factory(**options) if (options or isinstance(factory, type)) else factory
        except TypeError as exc:
            raise PresetError(f"middleware {name!r}: {uses} rejected {options}: {exc}") from exc
        if slot:
            slots.check_kind(slot, instance, where)
        _slot_check(instance, "middleware", f"{where} ({uses})")
        middleware.append(instance)

    return Flow(steps, middleware=middleware,
                name=preset_name,
                config=config.get("config", {}))


def load_preset(ref: str, fill: dict | None = None) -> Flow:
    """One-liner: preset name/path -> ready-to-run Flow.

    ``fill`` fills declared slots: {"slot_name": object_or_import_path}.
    """
    return build_flow(load_preset_config(ref), fill=fill)


def inspect_preset(ref: str, fill: dict | None = None) -> dict:
    """Dry-check a preset without running it: resolve every slot, run the
    wrap detection and kind checks, and report per-slot results. This is the
    engine behind `throughline doctor`. Unlike ``build_flow`` it does not
    fail fast on unfilled slots — it reports them.
    """
    import inspect as _inspect

    from .adapters import render_explain
    config = load_preset_config(ref)
    preset_name = config.get("name", ref)
    report: dict = {"name": preset_name, "slots": [], "steps": [], "middleware": []}

    try:
        slots = _Slots(config, fill)
    except PresetError as exc:
        report["slots"].append({"slot": "[slots]", "status": "error", "detail": str(exc)})
        slots = _Slots({k: v for k, v in config.items() if k not in ("fill",)}, None)
    report["slots"] += slots.report()

    for index, spec in enumerate(config.get("steps", [])):
        row = {"slot": f"[[steps]] #{index + 1}"}
        if isinstance(spec, dict):
            form = next((f for f in _STEP_FORMS if f in spec), None)
            if form == "uses" and isinstance(spec.get("uses"), str):
                row["uses"] = spec["uses"]
        try:
            step, info = _materialize_step(spec, index, slots, preset_name)
            target = info.pop("target", step)
            row.update(info)
            if step.meta.get("kind") in ("map", "parallel", "branch"):
                row["detail"] = _step_detail(step)
            elif isinstance(target, Step):
                row["detail"] = _step_detail(target)
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
            where = f"[middleware.{name}]"
            factory, slot = slots.resolve_ref(uses, where)
            options = slots.substitute(options, where)
            instance = factory(**options) if (options or isinstance(factory, type)) else factory
            if slot:
                slots.check_kind(slot, instance, where)
            problem = check_kind(instance, "middleware")
            if problem:
                raise PresetError(problem)
            row["status"] = "ok"
            row["detail"] = type(instance).__name__
        except Exception as exc:
            row["status"] = "error"
            row["detail"] = str(exc)
        report["middleware"].append(row)

    report["ok"] = all(r["status"] in ("ok", "disabled", "unreferenced")
                       for r in report["slots"] + report["steps"] + report["middleware"])
    return report
