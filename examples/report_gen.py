"""Example components for the `report-gen` preset.

The data artifact stays out of the control plane. The default fetcher reads a
CSV ArtifactRef/path, publishes aggregates, and drops raw rows immediately.
Real teams replace that fetcher slot with their warehouse/query step.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any

from throughline.context import RunContext
from throughline.step import Step
from throughline.store import ArtifactRef, MemoryArtifactStore

DEFAULT_DATA = "examples/data/sales.csv"
STORE = MemoryArtifactStore(default_ttl=None)


def seed_data(path: str = DEFAULT_DATA, *, session: str = "examples") -> ArtifactRef:
    """Put the demo sales CSV into the example artifact store and return a ref."""
    data_path = resolve_path(path)
    text = data_path.read_text(encoding="utf-8")
    return STORE.put(text, session=session, key=data_path.name,
                     meta={"path": str(data_path), "kind": "sales-csv"})


def fetch_data(payload, ctx: RunContext) -> dict:
    """Default user-replaceable fetcher: data ArtifactRef/path -> aggregates."""
    if not isinstance(payload, dict):
        payload = {"data": payload}
    spec = str(payload.get("spec", "Sales performance")).strip()
    period = str(payload.get("period", "current period")).strip()
    data = payload.get("data", DEFAULT_DATA)
    raw, source = materialize_data(data)
    rows = parse_csv(raw)
    facts = aggregate_sales(rows)
    ctx.metric("data.rows", facts["row_count"])
    ctx.metric("data.revenue", facts["total_revenue"])
    return {**payload, "spec": spec, "period": period,
            "data_source": source, "facts": facts}


def plan_sections(payload, ctx: RunContext) -> list[dict]:
    """Plan report sections; the raw data remains behind payload.data."""
    meta = {
        "spec": payload["spec"],
        "period": payload["period"],
        "data_source": payload["data_source"],
    }
    ctx.state["report_meta"] = meta
    sections = [
        ("executive", "Executive Summary"),
        ("regional", "Regional Performance"),
        ("product", "Product Mix"),
        ("risk", "Risks and Next Steps"),
    ]
    ctx.metric("report.sections", len(sections))
    return [
        {"id": section_id, "title": title, **meta, "facts": payload["facts"]}
        for section_id, title in sections
    ]


def write_section(fail_first: bool = True) -> Step:
    """Factory for map_step: one section payload -> one markdown line."""
    writer = SectionWriter(fail_first=fail_first)

    def fn(section, ctx: RunContext) -> str:
        return writer(section, ctx)

    return Step(fn=fn, name="section-llm", meta={"adapter": "fake-section-llm"})


class SectionWriter:
    """Tiny fake LLM. It can fail once to demonstrate Retry around map.

    This demo object is stateful for the lifetime of one Flow instance and is
    intentionally used with workers=1. Raising workers would share `calls`
    across threads and turn the teaching failure into a race.
    """

    def __init__(self, fail_first: bool = True):
        self.fail_first = fail_first
        self.calls = 0

    def __call__(self, section: dict, ctx: RunContext) -> str:
        self.calls += 1
        ctx.metric("llm.calls")
        ctx.metric("llm.input_tokens", len(json.dumps(section, default=str).split()))
        if self.fail_first and self.calls == 1:
            raise RuntimeError("transient section generation failure")
        line = render_section_line(section)
        ctx.metric("llm.output_tokens", len(line.split()))
        return line


def assemble_report(section_lines: list[str], ctx: RunContext) -> dict:
    """Assemble section lines into a markdown report."""
    meta = ctx.state.get("report_meta", {})
    title = meta.get("spec", "Report").strip().title()
    period = meta.get("period", "current period")
    body = "\n".join(section_lines)
    report = f"# {title}\n\nPeriod: {period}\n\n{body}"
    return {"spec": meta.get("spec", title), "period": period,
            "data_source": meta.get("data_source", ""),
            "section_count": len(section_lines), "report": report}


def render_report(format: str = "md") -> Step:
    """Render markdown/html and store the large artifact."""
    if format not in ("md", "html"):
        raise ValueError("format must be 'md' or 'html'")

    def fn(payload, ctx: RunContext) -> dict:
        report = payload["report"]
        rendered = report if format == "md" else markdown_to_html(report)
        digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:12]
        ref = STORE.put(rendered, session="reports",
                        key=f"{ctx.run_id}-{digest}.{format}",
                        meta={"format": format, "period": payload.get("period")})
        ctx.metric("report.bytes", len(rendered.encode("utf-8")))
        # Keep `report` in the control-plane output for lineage and local CLI
        # demos; MCP projection will artifact oversized outputs separately.
        return {**payload, "format": format, "report": rendered,
                "report_ref": ref.to_dict()}

    return Step(fn=fn, name="render-report", meta={"adapter": "renderer", "format": format})


def render_section_line(section: dict) -> str:
    facts = section["facts"]
    title = section["title"]
    if section["id"] == "executive":
        return (f"## {title}: revenue was ${facts['total_revenue']:,.0f} "
                f"across {facts['row_count']} orders; average order value was "
                f"${facts['average_order_value']:,.0f}.")
    if section["id"] == "regional":
        region, value = facts["top_region"]
        return (f"## {title}: {region} led revenue with ${value:,.0f}; "
                f"regional split was {format_pairs(facts['by_region'])}.")
    if section["id"] == "product":
        product, value = facts["top_product"]
        return (f"## {title}: {product} was the top product at ${value:,.0f}; "
                f"product split was {format_pairs(facts['by_product'])}.")
    return (f"## {title}: watch {facts['lowest_region'][0]} "
            f"(${facts['lowest_region'][1]:,.0f}) and expand the strongest "
            f"{facts['top_product'][0]} motion next period.")


def aggregate_sales(rows: list[dict]) -> dict:
    by_region: dict[str, float] = {}
    by_product: dict[str, float] = {}
    customers: set[str] = set()
    total = 0.0
    for row in rows:
        revenue = float(row.get("revenue", 0) or 0)
        total += revenue
        by_region[row.get("region", "unknown")] = by_region.get(row.get("region", "unknown"), 0) + revenue
        by_product[row.get("product", "unknown")] = by_product.get(row.get("product", "unknown"), 0) + revenue
        if row.get("customer"):
            customers.add(row["customer"])
    top_region = max(by_region.items(), key=lambda item: item[1])
    low_region = min(by_region.items(), key=lambda item: item[1])
    top_product = max(by_product.items(), key=lambda item: item[1])
    return {
        "row_count": len(rows),
        "customer_count": len(customers),
        "total_revenue": total,
        "average_order_value": total / len(rows) if rows else 0.0,
        "by_region": round_values(by_region),
        "by_product": round_values(by_product),
        "top_region": (top_region[0], round(top_region[1], 2)),
        "lowest_region": (low_region[0], round(low_region[1], 2)),
        "top_product": (top_product[0], round(top_product[1], 2)),
    }


def parse_csv(raw: str) -> list[dict]:
    return [dict(row) for row in csv.DictReader(io.StringIO(raw))]


def materialize_data(data: Any) -> tuple[str, str]:
    if isinstance(data, ArtifactRef):
        return as_text(STORE.get(data)), data.id
    if isinstance(data, dict) and "$artifact" in data:
        ref = ArtifactRef.from_dict(data)
        return as_text(STORE.get(ref)), ref.id
    if isinstance(data, str):
        path = resolve_path(data)
        if path.is_file():
            return path.read_text(encoding="utf-8"), str(path)
        return data, "inline-csv"
    return as_text(data), type(data).__name__


def as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value if isinstance(value, str) else str(value)


def round_values(values: dict[str, float]) -> dict[str, float]:
    return {key: round(value, 2) for key, value in values.items()}


def format_pairs(values: dict[str, float]) -> str:
    return ", ".join(f"{key} ${value:,.0f}"
                     for key, value in sorted(values.items()))


def markdown_to_html(markdown: str) -> str:
    lines = []
    for line in markdown.splitlines():
        if line.startswith("# "):
            lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            lines.append(f"<h2>{line[3:]}</h2>")
        elif line.strip():
            lines.append(f"<p>{line}</p>")
    return "\n".join(lines)


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
