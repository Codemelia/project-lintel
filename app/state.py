from typing import Any, NotRequired, TypedDict

# Extracted parameters from the user query
class ExtractedParameters(TypedDict, total=False):
    age: int | None
    cost_model: str
    urgency_level: str
    category: str
    confidence: float

# Reasoning trace for the graph nodes and their confidence scores
class ReasoningTrace(TypedDict):
    path: list[str]
    egress: bool
    model_backend: NotRequired[str]
    intent_confidence: NotRequired[float]
    fallback_reason: NotRequired[str | None]
    note: NotRequired[str]


class GraphState(TypedDict):
    user_query: str
    is_crisis: bool
    is_out_of_scope: bool
    extracted_parameters: ExtractedParameters
    calculated_tier: str
    retrieved_services: list[dict[str, Any]]
    reasoning_trace: ReasoningTrace
    final_response: str

# Empty graph state for the initial node
def empty_graph_state(user_query: str) -> GraphState:
    return {
        "user_query": user_query,
        "is_crisis": False,
        "is_out_of_scope": False,
        "extracted_parameters": {},
        "calculated_tier": "",
        "retrieved_services": [],
        "reasoning_trace": {"path": [], "egress": False},
        "final_response": "",
    }
