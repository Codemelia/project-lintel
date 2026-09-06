from app.graph.models import classify_intent, extract_parameters, get_node_model
from app.safety import crisis_regex_match
from app.state import empty_graph_state


def test_empty_graph_state_matches_schema() -> None:
    state = empty_graph_state("I am 18 and overwhelmed")
    assert state["user_query"] == "I am 18 and overwhelmed"
    assert state["is_crisis"] is False
    assert state["is_out_of_scope"] is False
    assert state["extracted_parameters"] == {}
    assert state["retrieved_services"] == []
    assert state["reasoning_trace"]["egress"] is False


def test_factory_returns_cloud_client() -> None:
    client = get_node_model("intent")
    assert client.backend == "cloud"
    assert get_node_model("extract").backend == "cloud"


def test_crisis_regex_true_positives() -> None:
    assert crisis_regex_match("I want to die")
    assert crisis_regex_match("I am going to kill myself")
    assert crisis_regex_match("I will end it all")
    assert crisis_regex_match("I want to end my life")


def test_crisis_regex_benign_false_positives() -> None:
    assert not crisis_regex_match("I am 18, no income, overwhelmed")
    assert not crisis_regex_match("Looking for a diet plan for anxiety")


def test_crisis_regex_helper_not_hard_stop() -> None:
    assert not crisis_regex_match(
        "How do I help my friend who is suicidal? Which number should I give them?"
    )
    assert not crisis_regex_match("thinking about suicide resources for a classmate")


def test_classify_intent_regex_skips_models(monkeypatch) -> None:
    def boom(self, text: str):
        raise AssertionError("model must not run after regex crisis")

    monkeypatch.setattr("app.graph.models.OpenAIStructuredClient.complete", boom)
    out = classify_intent("I want to kill myself")
    assert out["is_crisis"] is True
    assert out["egress"] is False
    assert out["model_backend"] == "regex"


def test_classify_intent_cloud_egresses(monkeypatch) -> None:
    def cloud_complete(self, text: str):
        return {
            "label": "out_of_scope",
            "is_crisis": False,
            "is_out_of_scope": True,
            "confidence": 0.91,
        }

    monkeypatch.setattr("app.graph.models.OpenAIStructuredClient.complete", cloud_complete)
    out = classify_intent("Be my therapist and run CBT")
    assert out["is_out_of_scope"] is True
    assert out["egress"] is True
    assert out["model_backend"] == "cloud"


def test_extract_parameters_cloud_egresses(monkeypatch) -> None:
    def cloud_complete(self, text: str):
        return {
            "age": 18,
            "cost_model": "Free",
            "urgency_level": "routine",
            "category": "Youth Assessment & Navigation",
            "confidence": 0.85,
        }

    monkeypatch.setattr("app.graph.models.OpenAIStructuredClient.complete", cloud_complete)
    out = extract_parameters("I am 18, no income, overwhelmed")
    assert out["age"] == 18
    assert out["cost_model"] == "Free"
    assert out["egress"] is True
    assert out["model_backend"] == "cloud"
