# Lineage

Evidence lineage records which retrieved chunks were placed into context.
When the prompt step is called with `cite="context"`, it renders each context
item with an `[eN]` marker and keeps the evidence ledger synchronized.

Claim lineage is created by `citations_step`. The step parses `[eN]` markers
from the answer, validates that each marker exists in the evidence ledger, and
then strips the markers from the final answer text.

Line-level lineage can be enabled with `LineageMiddleware(extract="answer")`.
The result can join answer lines, editing steps, claims, and sources into a
single report for audits.
