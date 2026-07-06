"""throughline <-> surgical_context: two retrievers and a local LLM step.

surgical_context is a local-first context engine for code understanding —
a Neo4j call graph + LanceDB vectors behind an Ask / Inspect / Impact API.
This adapter plugs its two read paths into throughline's evidence contract:

  SurgicalRetriever   ask path: question -> ranked, graph-expanded code
                      chunks (embedding intent + graph walk)
  ImpactRetriever     impact path: symbol -> exact on-disk source of its
                      downstream dependents (pure Neo4j walk, no embeddings)

Both return ``EvidenceChunk`` with real file:line spans, so a downstream
``citations_step`` links every answer line back to code deterministically.

Prototype status: this reaches below ``AxisEngine.ask`` because the public
``AskResult`` rows drop the code body. A production plugin wants a public
``ask_rows(with_code=True)`` upstream instead of the private helpers.
"""

from __future__ import annotations

import importlib.util
import json
import os
import urllib.request
from pathlib import Path

# surgical_context resolves its LanceDB path relative to the CURRENT
# DIRECTORY (LANCEDB_PATH, default "./data/lancedb") — run from anywhere else
# and vector search silently connects to an empty store (0 candidates, no
# warning). Anchor it to the checkout before lancedb_client reads the env.
_spec = importlib.util.find_spec("context_engine")
if _spec and _spec.origin:
    _sc_root = Path(_spec.origin).resolve().parent.parent
    os.environ.setdefault("LANCEDB_PATH", str(_sc_root / "data" / "lancedb"))

from throughline.adapters.llm import from_callable
from throughline.modules import EvidenceChunk
from throughline.step import Step

from context_engine.axis.intent_classifier import ROLE_INTENT_DESCRIPTIONS, IntentMatch
from context_engine.axis.pipeline import AxisRetrievalConfig, run_axis_retrieval
from surgical_context_mcp.config import resolve_workspace_id
from surgical_context_mcp.engine import AxisEngine, _collect_deduped_bundle_symbols

WORKSPACE = os.environ.get("SURGICAL_WORKSPACE", "qa_repo/surgical_context@main")


class SurgicalRetriever:
    """Axis ask path as an evidence retriever.

    ``roles=[...]`` forces intent and skips the embedding classifier — useful
    when the classifier lands just under its 0.20 threshold and returns
    nothing (preview candidates with ``AxisEngine.classify_intent``).

    Known limitation: chunk text is the packer-trimmed render (often just the
    docstring) while the span covers the full symbol — the honest fix is
    re-reading the span via ``read_symbol``, as ``ImpactRetriever`` does.
    """

    def __init__(self, workspace: str = WORKSPACE, token_budget: int = 12_000,
                 roles: list[str] | None = None):
        self.engine = AxisEngine()
        self._workspace = resolve_workspace_id(workspace)
        self._budget = token_budget
        self._roles = roles

    def retrieve(self, question: str) -> list[EvidenceChunk]:
        intent_override = None
        if self._roles:
            intent_override = [
                IntentMatch(role=r, similarity=1.0,
                            description=ROLE_INTENT_DESCRIPTIONS.get(r, ""))
                for r in self._roles
            ]
        with self.engine._lock:
            self.engine._ensure()
            raw = run_axis_retrieval(
                question,
                workspace_id=self._workspace,
                db=self.engine._db,
                lance=self.engine._lance,
                config=AxisRetrievalConfig(base_token_budget=self._budget,
                                           intent_override=intent_override),
            )
        symbols, _files = _collect_deduped_bundle_symbols(raw)
        return [
            EvidenceChunk(
                text=s.code,
                span=(s.start_line, s.end_line),
                score=round(float(s.utility_score), 4),
                source={"path": s.file_path, "uid": s.uid, "role": s.role,
                        "depth": s.distance_from_seed},
            )
            for s in symbols
            if s.code and s.start_line
        ]


class ImpactRetriever:
    """Blast radius as evidence: ``impact()`` walks the graph, ``read_symbol()``
    slices each dependent's exact source off disk — spans are honest by
    construction. The window is centered on the call site: evidence that does
    not SHOW the call makes the model (rightly) deny the dependency.
    """

    def __init__(self, workspace: str = WORKSPACE, max_depth: int = 3,
                 top: int = 8, window: int = 20):
        self.engine = AxisEngine()
        self._workspace = resolve_workspace_id(workspace)
        self._max_depth = max_depth
        self._top = top
        self._window = window

    def retrieve(self, symbol: str) -> list[EvidenceChunk]:
        res = self.engine.impact(symbol, self._workspace, max_depth=self._max_depth)
        if not res.found:
            raise LookupError(f"symbol {symbol!r} not found in workspace "
                              f"{self._workspace!r}")
        rows = sorted(res.affected_symbols,
                      key=lambda r: (r.get("depth", 99), r.get("file_path", "")))
        chunks: list[EvidenceChunk] = []
        seen: set[tuple] = set()
        for row in rows:
            key = (row.get("file_path"), row.get("name"))
            if key in seen or not row.get("name"):
                continue
            seen.add(key)
            src = self.engine.read_symbol(row["name"], self._workspace,
                                          file_path=row.get("file_path"))
            if not (src.found and src.code):
                continue
            lines = src.code.splitlines()
            hit = next((i for i, ln in enumerate(lines) if symbol in ln), 0)
            lo = max(0, hit - 4)
            hi = min(len(lines), lo + self._window)
            chunks.append(EvidenceChunk(
                text="\n".join(lines[lo:hi]),
                span=(src.start_line + lo, src.start_line + hi - 1),
                source={"path": src.file_path, "uid": src.uid,
                        "depth": row.get("depth"),
                        "relation": row.get("edge_type")},
            ))
            if len(chunks) >= self._top:
                break
        return chunks


def ollama_llm(model: str | None = None, num_ctx: int = 16384,
               num_predict: int = 300, prefill: str = "- ") -> Step:
    """Local Ollama chat as a prompt -> text step (stdlib HTTP only).

    Two hard-won knobs for small thinking-tuned models (qwen3 & co):
      * the assistant ``prefill`` makes the model continue the answer
        directly, skipping the thinking phase entirely (``think: false``
        alone does not stop the rambling, and ``think: true`` burns the
        whole ``num_predict`` budget before any content appears);
      * a tight ``num_predict`` cuts the "Wait, let me re-read..." tail —
        whatever slips through is caught by ``citations_step`` anyway.
    """
    # TL_-prefixed on purpose: importing context_engine loads surgical's own
    # .env, which exports OLLAMA_MODEL for its bridge and would override us.
    model = model or os.environ.get("TL_OLLAMA_MODEL", "qwen3:4b")
    url = os.environ.get("TL_OLLAMA_URL", "http://localhost:11434") + "/api/chat"

    def complete(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        if prefill:
            messages.append({"role": "assistant", "content": prefill})
        body = {"model": model, "messages": messages, "stream": False,
                "think": False,
                "options": {"num_ctx": num_ctx, "num_predict": num_predict,
                            "temperature": 0}}
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
        return (prefill + data["message"]["content"].lstrip()).strip()

    return from_callable(complete, name=f"llm-{model}")


def default_llm(**ollama_kwargs) -> Step:
    """REAL_LLM=1 -> Anthropic (needs credits); otherwise local Ollama."""
    if os.environ.get("REAL_LLM") == "1":
        from throughline.adapters.llm import anthropic_chat
        return anthropic_chat(model="claude-sonnet-5", max_tokens=800)
    return ollama_llm(**ollama_kwargs)


def print_claim_report(result, strip_prefix: str | None = None) -> None:
    """The payoff, printed: per answer line — status, citations, and the real
    file:line span the cited evidence came from."""
    evidence = result.ctx.artifacts["evidence"]
    claims = result.ctx.artifacts["claims"]

    print("\n=== claim status counts ===")
    print(claims.status_counts())
    print("violations:", result.ctx.artifacts.get("violations", []))

    print("\n=== join_blame: line -> evidence -> file:lines ===")
    for entry in claims.join_blame(lineage=result.lineage, evidence=evidence):
        if not entry["text"].strip():
            continue
        cites = ",".join(entry["evidence"]) or "-"
        locations = "; ".join(
            (src["source"]["path"].removeprefix(strip_prefix or "")
             + f":{src['span'][0]}-{src['span'][1]}"
             + (f" (d{src['source']['depth']})"
                if src["source"].get("depth") is not None else ""))
            for src in entry.get("sources", [])
        )
        print(f"[{entry.get('status') or 'exempt':13}] {cites:9} {entry['text'][:76]}")
        if locations:
            print(f"{'':15}-> {locations}")

    print("\n=== metrics ===")
    for key, value in sorted(result.metrics["counters"].items()):
        print(f"  {key:28} {value:g}")
