"""Unit tests for deterministic Ask routing helpers."""

from src.router import detect_qualitative_intent


def test_detect_qualitative_intent_true_examples():
    assert detect_qualitative_intent("What does forecast quality mean?") is True
    assert detect_qualitative_intent("How should management interpret at-risk pipeline?") is True
    assert detect_qualitative_intent("What should management focus on this week?") is True


def test_detect_qualitative_intent_false_for_metric_and_chart_examples():
    assert detect_qualitative_intent("What is the total pipeline value?") is False
    assert detect_qualitative_intent("Show revenue by region.") is False
    assert detect_qualitative_intent("Create a chart of deals by stage.") is False
