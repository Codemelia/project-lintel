# `tests/` — pytest

Unit and payload tests for graph nodes: crisis regex, out-of-scope refusals, `/chat/invoke` shapes, `get_node_model()` returning the OpenAI client, and the guarantee that the crisis regex path skips conversational generation (no OpenAI call, no generator).

See [Step 3](../docs/project-plan.md#step-3--state-schema-and-intent-gate), [Step 5](../docs/project-plan.md#step-5--fastapi-backend-integration), and [eval seams](../docs/system-design.md#11-test-and-eval-seams).
