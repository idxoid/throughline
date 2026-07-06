"""Example components for the `data-qa` preset.

The shape is intentionally boring: CSV profiling and rule checks are pure,
deterministic Python; only the final explanation pretends to be an LLM step.
That makes the example useful for showing how deterministic validation and an
LLM-facing report can live in the same instrumented flow.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from throughline.context import RunContext
from throughline.registry import register, resolve
from throughline.step import Step
from throughline.store import ArtifactRef, MemoryArtifactStore

DEFAULT_DATASET = "examples/data/customers.csv"
STORE = MemoryArtifactStore(default_ttl=None)

DEFAULT_RULES = [
    {"id": "customer-id-required", "type": "required", "column": "customer_id"},
    {"id": "customer-id-unique", "type": "unique", "column": "customer_id"},
    {"id": "email-required", "type": "required", "column": "email"},
    {"id": "email-format", "type": "regex", "column": "email",
     "pattern": r"^[^@\s]+@[^@\s]+\.[^@\s]+$"},
    {"id": "age-range", "type": "range", "column": "age", "min": 18, "max": 120},
    {"id": "plan-values", "type": "allowed", "column": "plan",
     "values": ["free", "pro", "enterprise"]},
    {"id": "signup-date", "type": "date", "column": "signup_date",
     "format": "%Y-%m-%d"},
]


def seed_dataset(path: str = DEFAULT_DATASET, *, session: str = "examples") -> ArtifactRef:
    """Put the demo CSV into the example artifact store and return a ref."""
    dataset_path = resolve_path(path)
    text = dataset_path.read_text(encoding="utf-8")
    return STORE.put(text, session=session, key=dataset_path.name,
                     meta={"path": str(dataset_path)})


def load_profile(payload, ctx: RunContext) -> dict:
    """Materialize a dataset ArtifactRef/path, parse CSV, and profile it."""
    if not isinstance(payload, dict):
        payload = {"dataset": payload}
    if "dataset" not in payload:
        payload = {**payload, "dataset": DEFAULT_DATASET}

    rules = normalize_rules(payload.get("rules"))
    raw, source = materialize_dataset(payload["dataset"])
    rows, columns = parse_csv(raw)
    profile = profile_rows(rows, columns)
    dataset_hash = stable_digest({"columns": columns, "rows": rows})
    dataset_id = f"{source}#{dataset_hash[:10]}"
    rules_cache_key = f"{dataset_id}:{stable_digest(rules)[:10]}"

    ctx.metric("data.rows", profile["row_count"])
    ctx.metric("data.columns", len(columns))
    return {**payload, "dataset_id": dataset_id, "rows": rows,
            "profile": profile, "rules": rules,
            "rules_cache_key": rules_cache_key}


def rule_checks(verifier: str = "data-qa-rules") -> Step:
    """Factory that resolves a registry verifier and wraps it as a step."""
    register("data-qa-rules", data_quality_verifier, kind="verifier")
    verifier_fn = resolve(verifier, kind="verifier")

    def fn(payload, ctx: RunContext) -> dict:
        violations = verifier_fn(payload["profile"], payload["rules"], payload["rows"])
        status = "fail" if violations else "pass"
        ctx.metric("data.rules", len(payload["rules"]))
        ctx.metric("data.violations", len(violations))
        if status == "fail":
            ctx.metric("data.failed")
        return {**payload, "violations": violations, "status": status}

    return Step(fn=fn, name="check-rules", meta={"adapter": "verifier"})


def llm_summary(payload, ctx: RunContext) -> dict:
    """Fake LLM explanation over deterministic rule-check output."""
    profile = payload["profile"]
    violations = payload["violations"]
    if violations:
        top = sorted(violations, key=lambda item: (-item["count"], item["rule"]))[:3]
        details = "; ".join(
            f"{item['rule']} affected {item['count']} row(s)" for item in top
        )
        summary = (f"Data QA failed {len(violations)} rule check(s) across "
                   f"{profile['row_count']} row(s). Highest impact: {details}.")
    else:
        summary = (f"Data QA passed {len(payload['rules'])} rule check(s) across "
                   f"{profile['row_count']} row(s).")

    report = {
        "dataset": payload["dataset_id"],
        "status": payload["status"],
        "summary": summary,
        "profile": profile,
        "violations": violations,
    }
    prompt_tokens = len(json.dumps({"profile": profile, "violations": violations},
                                   ensure_ascii=False, default=str).split())
    ctx.metric("llm.calls")
    ctx.metric("llm.input_tokens", prompt_tokens)
    ctx.metric("llm.output_tokens", len(summary.split()))
    slim_payload = {key: value for key, value in payload.items() if key != "rows"}
    return {**slim_payload, "summary": summary, "report": report}


@register("data-qa-rules", kind="verifier")
def data_quality_verifier(profile: dict, rules: list[dict],
                          rows: list[dict]) -> list[dict]:
    """Deterministic verifier-kind component used by the example preset."""
    violations: list[dict] = []
    columns = set(profile["columns"])
    for rule in rules:
        rule_id = str(rule.get("id") or f"{rule.get('type')}:{rule.get('column')}")
        kind = rule.get("type")
        column = rule.get("column")
        if column and column not in columns:
            violations.append(violation(rule_id, "error", column,
                                        f"column {column!r} is missing", []))
            continue
        if kind == "required":
            bad_rows = [row["_row"] for row in rows if is_blank(row.get(column, ""))]
            add_if_any(violations, rule_id, column, "required value is blank", bad_rows)
        elif kind == "unique":
            seen: dict[str, list[int]] = {}
            for row in rows:
                value = str(row.get(column, "")).strip()
                if value:
                    seen.setdefault(value, []).append(row["_row"])
            bad_rows = sorted({line for lines in seen.values() if len(lines) > 1
                               for line in lines})
            add_if_any(violations, rule_id, column, "duplicate values found", bad_rows)
        elif kind == "range":
            low, high = float(rule["min"]), float(rule["max"])
            bad_rows = []
            for row in rows:
                value = row.get(column, "")
                if is_blank(value):
                    continue
                try:
                    number = float(value)
                except ValueError:
                    bad_rows.append(row["_row"])
                    continue
                if number < low or number > high:
                    bad_rows.append(row["_row"])
            add_if_any(violations, rule_id, column,
                       f"value must be between {low:g} and {high:g}", bad_rows)
        elif kind == "allowed":
            allowed = {str(value) for value in rule.get("values", [])}
            bad_rows = [row["_row"] for row in rows
                        if str(row.get(column, "")).strip() not in allowed]
            add_if_any(violations, rule_id, column,
                       f"value must be one of {sorted(allowed)}", bad_rows)
        elif kind == "regex":
            pattern = re.compile(str(rule["pattern"]))
            bad_rows = [row["_row"] for row in rows
                        if not is_blank(row.get(column, ""))
                        and not pattern.fullmatch(str(row.get(column, "")).strip())]
            add_if_any(violations, rule_id, column,
                       "value does not match expected pattern", bad_rows)
        elif kind == "date":
            fmt = str(rule.get("format", "%Y-%m-%d"))
            bad_rows = []
            for row in rows:
                value = str(row.get(column, "")).strip()
                if is_blank(value):
                    continue
                try:
                    datetime.strptime(value, fmt)
                except ValueError:
                    bad_rows.append(row["_row"])
            add_if_any(violations, rule_id, column,
                       f"value must match date format {fmt}", bad_rows)
        else:
            violations.append(violation(rule_id, "warning", column,
                                        f"unknown rule type {kind!r}", []))
    return violations


def normalize_rules(rules: Any) -> list[dict]:
    if rules is None:
        return [dict(rule) for rule in DEFAULT_RULES]
    if isinstance(rules, dict):
        rules = rules.get("rules", [])
    return [dict(rule) for rule in rules]


def materialize_dataset(dataset: Any) -> tuple[str, str]:
    if isinstance(dataset, ArtifactRef):
        return as_text(STORE.get(dataset)), dataset.id
    if isinstance(dataset, dict) and "$artifact" in dataset:
        ref = ArtifactRef.from_dict(dataset)
        return as_text(STORE.get(ref)), ref.id
    if isinstance(dataset, str):
        path = resolve_path(dataset)
        if path.is_file():
            return path.read_text(encoding="utf-8"), str(path)
        return dataset, "inline-csv"
    return as_text(dataset), type(dataset).__name__


def parse_csv(raw: str) -> tuple[list[dict], list[str]]:
    reader = csv.DictReader(io.StringIO(raw))
    columns = list(reader.fieldnames or [])
    rows: list[dict] = []
    for line_number, row in enumerate(reader, start=2):
        clean = {
            key: (value.strip() if isinstance(value, str) else value)
            for key, value in row.items()
        }
        clean["_row"] = line_number
        rows.append(clean)
    return rows, columns


def profile_rows(rows: list[dict], columns: list[str]) -> dict:
    null_counts = {
        column: sum(1 for row in rows if is_blank(row.get(column, "")))
        for column in columns
    }
    distinct_counts = {
        column: len({str(row.get(column, "")).strip()
                     for row in rows if not is_blank(row.get(column, ""))})
        for column in columns
    }
    numeric: dict[str, dict] = {}
    for column in columns:
        values = []
        for row in rows:
            value = row.get(column, "")
            if is_blank(value):
                continue
            try:
                values.append(float(value))
            except ValueError:
                values = []
                break
        if values:
            numeric[column] = {"min": min(values), "max": max(values)}
    return {"row_count": len(rows), "columns": columns,
            "null_counts": null_counts, "distinct_counts": distinct_counts,
            "numeric": numeric}


def violation(rule_id: str, severity: str, column: str | None,
              message: str, rows: list[int]) -> dict:
    return {"rule": rule_id, "severity": severity, "column": column,
            "message": message, "rows": rows[:20], "count": len(rows)}


def add_if_any(violations: list[dict], rule_id: str, column: str,
               message: str, rows: list[int]) -> None:
    if rows:
        violations.append(violation(rule_id, "error", column, message, rows))


def is_blank(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    if isinstance(value, Iterable):
        output = io.StringIO()
        writer = None
        for row in value:
            if not isinstance(row, dict):
                return str(value)
            if writer is None:
                writer = csv.DictWriter(output, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
        return output.getvalue()
    return str(value)


def stable_digest(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    repo_root = Path(__file__).resolve().parent.parent
    for base in (Path.cwd(), repo_root):
        resolved = (base / candidate).resolve()
        if resolved.exists():
            return resolved
    return (Path.cwd() / candidate).resolve()
