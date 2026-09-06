# `eval/` — Benchmarks and `eval.py`

Thirty labelled scenarios (15 in-scope, 8 crisis, 7 out-of-scope). Gold labels: `is_crisis`, `is_out_of_scope`, expected `service_id`, expected tier.

`eval.py` reports **two** configurations on the same gold set:

1. Unconstrained `gpt-4o-mini` (no graph) — baseline
2. Navigator chatbot (regex + graph + `gpt-4o-mini` for intent/extract only)

KPIs: Scope Adherence, Safety Hand-off Success, Navigation Accuracy, Trace Transparency. Those percentages are **eval targets**. Model JSON `confidence` is uncalibrated and is not a crisis cutoff. Policy: [project plan](../docs/project-plan.md#confidence-scores-policy).

See [Steps 8–10](../docs/project-plan.md#phase-2-evaluation-ux-polish-and-presentation) and [KPIs](../docs/project-plan.md#5-evaluation-strategy-and-kpis).
