# Guardrails

The production middleware stack should observe runs, collect metrics, retry
only flaky or expensive steps such as the LLM, validate the final payload, and
track lineage. Quota should include a max cost so a bad prompt or loop cannot
spend without a bound.

`citations_step(require="warn")` is useful while a team is tuning prompts. It
records uncited answer lines as violations without breaking the user workflow,
which makes missing citations visible during evaluation.
