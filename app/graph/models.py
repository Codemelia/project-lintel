import json
import os
from pathlib import Path
from typing import Any, Literal, Protocol

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from app.safety import crisis_regex_match

_REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT / ".env")

TaskType = Literal["intent", "extract"]
Backend = Literal["cloud"]
IntentLabel = Literal["in_scope", "crisis", "out_of_scope"]


INTENT_SYSTEM = """You are the Intent Gate for a Singapore non-clinical service navigator chatbot.
Return JSON only. Do not give therapy, diagnosis, or CBT.
Labels:
- crisis: active suicidal ideation, intent to die, or immediate self-harm
- out_of_scope: asks you to diagnose, be a therapist, or deliver CBT/unconstrained therapy
- in_scope: navigation (age, cost, which service to contact)
Examples:
User: I want to end it all. -> {"label":"crisis","is_crisis":true,"is_out_of_scope":false,"confidence":0.99}
User: Diagnose my depression and be my therapist. -> {"label":"out_of_scope","is_crisis":false,"is_out_of_scope":true,"confidence":0.95}
User: I am 18, no income, overwhelmed. -> {"label":"in_scope","is_crisis":false,"is_out_of_scope":false,"confidence":0.9}
"""

EXTRACT_SYSTEM = """Extract navigation constraints for Singapore mental health service routing.
Return JSON only. Do not invent phone numbers.
cost_model must be one of: Free, Subsidized, Private, Variable (or null if unknown).
urgency_level must be one of: routine, sub-acute, acute, emergency (or null).
category must be one of:
Crisis Support & Hotlines; Youth Assessment & Navigation;
Community Support & Assessment (CREST); Community Counseling & Therapy (COMIT);
Primary Care & Polyclinics; Specialist & Acute Care (IMH/Hospitals); or null.
Example: I am 18, have no income, overwhelmed.
-> {"age":18,"cost_model":"Free","urgency_level":"routine","category":"Youth Assessment & Navigation","confidence":0.85}
"""


class IntentResult(BaseModel):
    label: IntentLabel = "in_scope"
    is_crisis: bool = False
    is_out_of_scope: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ExtractResult(BaseModel):
    age: int | None = Field(default=None, ge=0, le=120)
    cost_model: Literal["Free", "Subsidized", "Private", "Variable"] | None = None
    urgency_level: Literal["routine", "sub-acute", "acute", "emergency"] | None = None
    category: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class StructuredClient(Protocol):
    backend: Backend

    def complete(self, text: str) -> dict[str, Any]: ...


def _system_for(task_type: TaskType) -> str:
    return INTENT_SYSTEM if task_type == "intent" else EXTRACT_SYSTEM


def _parse(task_type: TaskType, payload: dict[str, Any]) -> dict[str, Any]:
    if task_type == "intent":
        return IntentResult.model_validate(payload).model_dump()
    return ExtractResult.model_validate(payload).model_dump()


def _chat_json(*, api_key: str, model: str, system: str, user: str, base_url: str | None) -> dict[str, Any]:
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = response.choices[0].message.content or "{}"
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("model did not return a JSON object")
    return parsed


class OpenAIStructuredClient:
    """Structured JSON for Nodes 1–2. Optional OPENAI_BASE_URL for a private/regional endpoint."""

    backend: Backend = "cloud"

    def __init__(self, model: str | None = None, task_type: TaskType = "intent"):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.task_type: TaskType = task_type

    def complete(self, text: str) -> dict[str, Any]:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for intent and parameter extraction")
        raw = _chat_json(
            api_key=api_key,
            model=self.model,
            system=_system_for(self.task_type),
            user=text,
            base_url=os.getenv("OPENAI_BASE_URL") or None,
        )
        return _parse(self.task_type, raw)


def get_node_model(task_type: TaskType) -> StructuredClient:
    """Return the structured-output client for Node 1 (intent) or Node 2 (extract)."""
    return OpenAIStructuredClient(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        task_type=task_type,
    )


def classify_intent(text: str) -> dict[str, Any]:
    """Node 1: first-person regex, else one structured OpenAI call. Regex never egresses."""
    if crisis_regex_match(text):
        return {
            "is_crisis": True,
            "is_out_of_scope": False,
            "label": "crisis",
            "confidence": 1.0,
            "model_backend": "regex",
            "egress": False,
        }

    client = get_node_model("intent")
    parsed = IntentResult.model_validate(client.complete(text))
    return {
        **parsed.model_dump(),
        "model_backend": client.backend,
        "egress": True,
    }


def extract_parameters(text: str) -> dict[str, Any]:
    """Node 2: structured extraction via OpenAI. Always egresses the utterance."""
    client = get_node_model("extract")
    parsed = ExtractResult.model_validate(client.complete(text))
    return {**parsed.model_dump(), "model_backend": client.backend, "egress": True}
