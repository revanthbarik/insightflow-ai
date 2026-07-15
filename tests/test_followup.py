"""Unit tests for controlled follow-up chat behavior."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from src import followup


def _sample_leads() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Lead_ID": ["L001", "L002", "L003", "L004"],
            "Company_Name": ["Alpha Solar", "Beta Solar", "Gamma Storage", "Delta Grid"],
            "Lead_Source": ["Website Inquiry", "Partner Channel", "Partner Channel", "Referral"],
            "Industry": ["Solar EPC Contractor", "Commercial & Industrial", "Battery Storage", "Utilities"],
            "Estimated_Value": [500_000, 300_000, 1_200_000, 250_000],
            "Status": ["Qualified", "Contacted", "Qualified", "New"],
            "Created_Date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]),
            "Sales_Rep": ["Aisha Khan", "Omar Ali", "Omar Ali", "Sara Noor"],
            "Customer_Type": ["EPC", "Distributor", "Installer", "Government Client"],
        }
    )


def _sample_deals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Deal_ID": ["D001", "D002", "D003", "D004"],
            "Lead_ID": ["L001", "L001", "L002", "L002"],
            "Company_Name": ["Alpha Solar", "Alpha Solar", "Beta Solar", "Beta Solar"],
            "Deal_Stage": ["Closed Won", "Closed Lost", "Negotiation", "Proposal"],
            "Amount": [1_000_000, 500_000, 200_000, 700_000],
            "Closing_Date": pd.to_datetime(["2024-06-01", "2024-07-01", "2024-08-01", "2024-09-01"]),
            "Forecast_Category": ["Closed", "Omitted", "Pipeline", "At Risk"],
            "Product_Category": ["Panels", "Panels", "Batteries", "Batteries"],
        }
    )


def _last_answer_context() -> dict:
    return {
        "original_user_question": "Show monthly pipeline trend based on closing date.",
        "answer_title": "Monthly Pipeline Trend by Closing Date",
        "answer_source": "time trend chart",
        "computed_answer": "This chart shows pipeline deal amount grouped by closing month.",
        "recommended_action": "Review months with high expected pipeline value and confirm deal readiness.",
        "data_records": [
            {"Closing_Month": "Jun 2024", "Total_Amount": 250000.0},
            {"Closing_Month": "Jul 2024", "Total_Amount": 0.0},
            {"Closing_Month": "Aug 2024", "Total_Amount": 400000.0},
        ],
        "retrieved_rag_context": None,
        "data_version": ("demo",),
        "timestamp_utc": "2026-06-19T00:00:00+00:00",
    }


def test_followup_prompt_includes_last_computed_context():
    prompt = followup.build_followup_prompt(
        "Why does this matter?",
        _last_answer_context(),
    )
    assert "Monthly Pipeline Trend by Closing Date" in prompt
    assert "This chart shows pipeline deal amount grouped by closing month." in prompt


def test_followup_prompt_includes_do_not_calculate_and_use_aed_only():
    prompt = followup.build_followup_prompt(
        "Explain this in simple terms.",
        _last_answer_context(),
    )
    assert "Do not calculate." in prompt
    assert "Use AED only." in prompt
    assert "Pandas has already computed the result." in prompt
    assert "Do not invent numbers." in prompt


def test_build_computed_context_returns_grounded_fields():
    result = {
        "title": "High Value but Low Conversion Lead Source",
        "answer_source": "lead conversion analysis",
        "answer": "Partner Channel has high estimated value but low conversion.",
        "recommended_action": "Review qualification quality.",
        "data": pd.DataFrame(
            [
                {
                    "Lead_Source": "Partner Channel",
                    "Total_Leads": 2,
                    "Converted_Leads": 1,
                    "Conversion_Rate": 0.5,
                    "Estimated_Value": 1_500_000,
                }
            ]
        ),
    }
    context = followup.build_computed_context(
        result=result,
        user_question="Which lead source generates the highest estimated value but low conversion into deals?",
    )
    assert context["answer_title"] == "High Value but Low Conversion Lead Source"
    assert context["answer_source"] == "lead conversion analysis"
    assert context["computed_answer"] == "Partner Channel has high estimated value but low conversion."
    assert context["recommended_action"] == "Review qualification quality."
    assert context["currency"] == "AED"
    assert isinstance(context["key_observations"], list)
    assert isinstance(context["known_limitations"], list)


def test_monthly_trend_context_has_monthly_pipeline_metadata():
    result = {
        "title": "Monthly Pipeline Trend by Closing Date",
        "answer_source": "time trend chart",
        "answer": "This chart shows pipeline deal amount grouped by closing month.",
        "recommended_action": "Review months with high expected pipeline value.",
        "chart_key": "monthly_pipeline_trend",
        "chart_type": "line",
        "data": pd.DataFrame(
            [
                {"Closing_Month": "Jan 2025", "Total_Amount": 1000000.0},
                {"Closing_Month": "Feb 2025", "Total_Amount": 0.0},
                {"Closing_Month": "Mar 2025", "Total_Amount": 2760000.0},
            ]
        ),
    }
    context = followup.build_computed_context(
        result=result,
        user_question="Show monthly pipeline trend based on closing date.",
    )
    assert context["metric_type"] == "monthly_pipeline_trend"
    assert context["chart_type"] == "line_chart"
    assert context["chart_title"] == "Monthly Pipeline Trend by Closing Date"
    assert context["x_axis"] == "Closing Month"
    assert context["y_axis"] == "Total Amount (AED)"


def test_forecast_quality_context_mentions_weak_categories():
    result = {
        "title": "Weakest Forecast Quality by Product Category",
        "answer_source": "forecast quality analysis",
        "answer": "Batteries has the weakest forecast quality.",
        "recommended_action": "Review weak forecast deals.",
    }
    context = followup.build_computed_context(
        result=result,
        user_question="Which product category has the weakest forecast quality?",
    )
    assert any("Weak forecast = At Risk + Omitted + Pipeline." == item for item in context["key_observations"])


def test_followup_prompt_uses_compact_json_not_indented_json():
    prompt = followup.build_followup_prompt(
        "Explain this in simple terms.",
        _last_answer_context(),
    )
    assert '"answer_title":"Monthly Pipeline Trend by Closing Date"' in prompt
    assert '"answer_title": "Monthly Pipeline Trend by Closing Date"' not in prompt


def test_action_followup_prompt_requires_practical_structured_response():
    prompt = followup.build_followup_prompt(
        "What can we do to address the closed lost deals?",
        _last_answer_context(),
        intent="business_recommendation",
    )

    assert "**What this indicates**" in prompt
    assert "**Likely causes to investigate**" in prompt
    assert "**Practical next steps**" in prompt
    assert "**Recommended follow-up analysis**" in prompt
    assert "Do not merely restate the chart or table." in prompt
    assert '"categories_and_values"' in prompt
    assert followup.FOLLOWUP_PROMPT_VERSION in prompt


def test_calculation_style_followup_routes_away_from_ollama():
    result = followup.answer_followup_question(
        "What is the total amount?",
        _last_answer_context(),
        _sample_leads(),
        _sample_deals(),
    )
    assert result["answer_source"] != "Ollama follow-up explanation"


def test_explanation_style_followup_uses_ollama_path(monkeypatch):
    monkeypatch.setattr(followup, "check_ollama_available", lambda: True)
    monkeypatch.setattr(followup, "check_ollama_model_available", lambda model: True)
    monkeypatch.setattr(
        followup,
        "generate_fast_explanation",
        lambda prompt, model=None: {
            "ok": True,
            "content": "Management should focus on the stronger closing months and review gaps.",
            "error": None,
            "model": model,
        },
    )
    result = followup.answer_followup_question(
        "What should management do next?",
        _last_answer_context(),
        _sample_leads(),
        _sample_deals(),
    )
    assert result["answer_source"] == "Ollama follow-up explanation"
    assert result["followup_intent"] == "business_recommendation"


def test_missing_last_answer_context_returns_safe_message():
    result = followup.answer_followup_question(
        "Explain this result.",
        None,
        _sample_leads(),
        _sample_deals(),
    )
    assert "Please ask a main question first" in result["answer"]


def test_ollama_unavailable_does_not_crash():
    with patch("src.followup.check_ollama_available", return_value=True), patch(
        "src.followup.check_ollama_model_available",
        return_value=True,
    ), patch(
        "src.followup.generate_fast_explanation",
        side_effect=Exception("Ollama unavailable"),
    ):
        result = followup.answer_followup_question(
            "Why does this matter?",
            _last_answer_context(),
            _sample_leads(),
            _sample_deals(),
        )

    assert "Ollama is unavailable" in result["answer"]


def test_ollama_timeout_returns_timeout_safe_message(monkeypatch):
    monkeypatch.setattr(followup, "check_ollama_available", lambda: True)
    monkeypatch.setattr(followup, "check_ollama_model_available", lambda model: True)
    monkeypatch.setattr(
        followup,
        "generate_fast_explanation",
        lambda prompt, model=None: {
            "ok": False,
            "content": "Ollama is taking too long. The computed Pandas answer is still available.",
            "error": "timeout",
            "model": model,
        },
    )
    result = followup.answer_followup_question(
        "Explain this in simple terms.",
        _last_answer_context(),
        _sample_leads(),
        _sample_deals(),
    )
    assert "taking too long" in result["answer"]


def test_repeated_same_explanation_uses_cache(monkeypatch):
    followup.EXPLANATION_CACHE.clear()
    call_counter = {"count": 0}
    monkeypatch.setattr(followup, "check_ollama_available", lambda: True)
    monkeypatch.setattr(followup, "check_ollama_model_available", lambda model: True)

    def _fake_fast_explanation(prompt, model=None):
        call_counter["count"] += 1
        return {
            "ok": True,
            "content": "Management should focus on the strongest months first.",
            "error": None,
            "model": model,
        }

    monkeypatch.setattr(followup, "generate_fast_explanation", _fake_fast_explanation)

    first = followup.answer_followup_question(
        "Why does this matter?",
        _last_answer_context(),
        _sample_leads(),
        _sample_deals(),
    )
    second = followup.answer_followup_question(
        "Why does this matter?",
        _last_answer_context(),
        _sample_leads(),
        _sample_deals(),
    )

    assert call_counter["count"] == 1
    assert first["from_cache"] is False
    assert second["from_cache"] is True


def test_followup_prompt_does_not_include_raw_dataframe_rows():
    result = {
        "title": "Monthly Pipeline Trend by Closing Date",
        "answer_source": "time trend chart",
        "answer": "This chart shows pipeline deal amount grouped by closing month.",
        "recommended_action": "Review months with high expected pipeline value.",
        "chart_key": "monthly_pipeline_trend",
        "data": _sample_deals(),
    }
    context = followup.build_computed_context(
        result=result,
        user_question="Show monthly pipeline trend based on closing date.",
    )
    prompt = followup.build_followup_prompt("Explain this in simple terms.", context)
    assert "Alpha Solar" not in prompt


def test_classify_followup_intent_examples():
    context = _last_answer_context()
    assert followup.classify_followup_intent("what does this chart actually tell us", context) == "explanation"
    assert followup.classify_followup_intent("what are some changes i can make on this chart", context) == "chart_improvement"
    assert followup.classify_followup_intent("show this as a chart", context) == "analytics"
    assert followup.classify_followup_intent("create a chart by region", context) == "analytics"
    assert followup.classify_followup_intent("compare this with other sales reps", context) == "analytics"
    assert followup.classify_followup_intent("what should management do", context) == "business_recommendation"
    assert followup.classify_followup_intent("what is forecast quality", context) == "clarification"
    assert followup.classify_followup_intent("tell me a joke", context) == "unrelated"
    assert followup.classify_followup_intent("chart", context) == "ambiguous"


def test_chart_explanation_followup_uses_ollama_not_chart_generation(monkeypatch):
    monkeypatch.setattr(followup, "check_ollama_available", lambda: True)
    monkeypatch.setattr(followup, "check_ollama_model_available", lambda model: True)
    monkeypatch.setattr(
        followup,
        "generate_fast_explanation",
        lambda prompt, model=None: {
            "ok": True,
            "content": "The chart shows where pipeline is concentrated and where the gaps are.",
            "error": None,
            "model": model,
        },
    )

    result = followup.answer_followup_question(
        "what does this chart actually tell us",
        _last_answer_context(),
        _sample_leads(),
        _sample_deals(),
    )

    assert result["answer_source"] == "Ollama follow-up explanation"
    assert result["followup_intent"] == "explanation"


def test_chart_improvement_followup_uses_ollama(monkeypatch):
    monkeypatch.setattr(followup, "check_ollama_available", lambda: True)
    monkeypatch.setattr(followup, "check_ollama_model_available", lambda model: True)
    monkeypatch.setattr(
        followup,
        "generate_fast_explanation",
        lambda prompt, model=None: {
            "ok": True,
            "content": "Add AED value labels, sort key categories, and add a management takeaway subtitle.",
            "error": None,
            "model": model,
        },
    )

    result = followup.answer_followup_question(
        "what are some changes i can make on this chart",
        _last_answer_context(),
        _sample_leads(),
        _sample_deals(),
    )

    assert result["answer_source"] == "Ollama follow-up explanation"
    assert result["followup_intent"] == "chart_improvement"


def test_show_this_as_chart_routes_to_analytics(monkeypatch):
    monkeypatch.setattr(
        followup,
        "answer_business_question",
        lambda question, leads_df=None, deals_df=None, active_context=None: {
            "answer_type": "chart",
            "title": "Pipeline by Forecast Category",
            "answer": "Showing pipeline grouped by forecast category.",
            "recommended_action": "Review the largest categories first.",
            "data": pd.DataFrame([{"Forecast_Category": "Pipeline", "Total_Amount": 1200000.0}]),
            "x_axis": "Forecast_Category",
            "y_axis": "Total_Amount",
            "chart_type": "bar",
            "answer_source": "semantic BI calculation",
        },
    )

    result = followup.answer_followup_question(
        "show this as a chart",
        _last_answer_context(),
        _sample_leads(),
        _sample_deals(),
    )

    assert result["answer_type"] == "chart"
    assert result["answer_source"] == "semantic BI calculation"


def test_unrelated_followup_returns_polite_rejection():
    result = followup.answer_followup_question(
        "tell me a joke",
        _last_answer_context(),
        _sample_leads(),
        _sample_deals(),
    )
    assert "only supports questions related to the current InsightFlow result" in result["answer"]


def test_ambiguous_chart_word_alone_does_not_force_chart_generation():
    result = followup.answer_followup_question(
        "chart",
        _last_answer_context(),
        _sample_leads(),
        _sample_deals(),
    )
    assert result["title"] == "Clarify Follow-up"
    assert "one direction" in result["answer"]


def test_contextual_followup_analytics_reuses_active_metric(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_answer_business_question(question, leads_df=None, deals_df=None, active_context=None):
        captured["question"] = question
        captured["active_context"] = active_context
        return {
            "answer_type": "table",
            "title": "Bottom 3 Product Categories by Revenue",
            "answer": "Bottom performers:\nBatteries — AED 900.00K",
            "recommended_action": "Review demand, pricing, sales coverage, or forecast quality for these product categories.",
            "answer_source": "semantic BI calculation",
            "metric_type": "revenue",
        }

    monkeypatch.setattr(followup, "answer_business_question", _fake_answer_business_question)

    result = followup.answer_followup_question(
        "give me 3 product categories that are doing the worst",
        _last_answer_context(),
        _sample_leads(),
        _sample_deals(),
        active_business_context={
            "metric": "revenue",
            "dataset": "deals",
            "dimension": "Region",
            "aggregation": "sum",
            "filters": {},
            "chart_type": "bar",
        },
    )

    assert result["answer_type"] != "unsupported"
    assert result["answer_source"] == "semantic BI calculation"
    assert captured["active_context"]["metric"] == "revenue"


def test_contextual_followup_missing_column_returns_specific_message():
    result = followup.answer_followup_question(
        "give me 3 product categories that are doing the worst",
        _last_answer_context(),
        _sample_leads(),
        _sample_deals().drop(columns=["Product_Category"]),
        active_business_context={
            "metric": "revenue",
            "dataset": "deals",
            "dimension": "Region",
            "aggregation": "sum",
            "filters": {},
        },
    )
    assert result["answer_type"] == "unsupported"
    assert "required column/table is missing" in result["answer"]


def test_management_priority_analytics_followup_stays_on_query_engine(monkeypatch):
    monkeypatch.setattr(
        followup,
        "answer_business_question",
        lambda question, leads_df=None, deals_df=None, active_context=None: {
            "answer_type": "chart",
            "title": "Customer Type Performance",
            "answer": "EPC deserves the highest management priority: AED 1.00M total revenue, AED 1.00M average deal size, and 1 deals.",
            "recommended_action": "Prioritize EPC because it has the strongest revenue contribution and a strong average deal value.",
            "answer_source": "semantic BI calculation",
            "metric_type": "revenue",
            "metrics": ["revenue", "average_deal_size"],
        },
    )

    result = followup.answer_followup_question(
        "which customer segment deserves the highest management priority",
        _last_answer_context(),
        _sample_leads(),
        _sample_deals(),
        active_business_context={
            "metrics": ["revenue", "average_deal_size"],
            "metric": "revenue",
            "dataset": "deals_joined_leads",
            "dimension": "Customer_Type",
            "aggregation": "sum",
            "filters": {},
        },
    )

    assert result["answer_source"] == "semantic BI calculation"
    assert result["followup_intent"] == "analytics"


def test_relevance_score_accepts_indirect_grounded_action_followup():
    context = _last_answer_context()
    assert followup.score_followup_relevance("How should we address this?", context) >= 2
    assert followup.is_grounded_qualitative_followup("How should we address this?", context)


def test_indirect_grounded_action_followup_uses_ollama(monkeypatch):
    monkeypatch.setattr(followup, "check_ollama_available", lambda: True)
    monkeypatch.setattr(followup, "check_ollama_model_available", lambda model: True)
    monkeypatch.setattr(
        followup,
        "generate_fast_explanation",
        lambda prompt, model=None: {
            "ok": True,
            "content": "Review the pipeline gaps and assign clear owners for the weaker months.",
            "error": None,
            "model": model,
        },
    )

    result = followup.answer_followup_question(
        "How should we address this?",
        _last_answer_context(),
        _sample_leads(),
        _sample_deals(),
    )

    assert result["answer_source"] == "Ollama follow-up explanation"
    assert result["followup_intent_bucket"] == "qualitative_contextual"


def test_unrelated_followup_is_blocked_with_contextual_suggestions():
    result = followup.answer_followup_question(
        "What is machine learning?",
        _last_answer_context(),
        _sample_leads(),
        _sample_deals(),
    )

    assert result["answer_type"] == "unsupported"
    assert "not clearly related" in result["answer"]
    assert "Break this down by sales rep" in result["recommended_action"]


def test_stage_entity_resolution_is_conservative():
    context = {
        **_last_answer_context(),
        "answer_title": "Deals by Stage",
        "main_entity": "deal stage",
        "table_preview": [{"Deal_Stage": "Closed Lost", "Total_Amount": 500000.0}],
    }
    resolved = followup.resolve_followup_entities_from_context(
        "Show the lost deals",
        context,
    )

    assert "Closed Lost deal stage" in resolved


def test_closed_lost_action_followup_routes_to_grounded_ollama(monkeypatch):
    monkeypatch.setattr(followup, "check_ollama_available", lambda: True)
    monkeypatch.setattr(followup, "check_ollama_model_available", lambda model: True)
    monkeypatch.setattr(
        followup,
        "generate_fast_explanation",
        lambda prompt, model=None: {
            "ok": True,
            "content": "Review loss reasons, improve qualification, and assign owners to recovery actions.",
            "error": None,
            "model": model,
        },
    )
    stage_result = {
        "answer_type": "table",
        "title": "Deals by Stage",
        "answer": "Deals grouped by stage with counts and total amount.",
        "recommended_action": "Focus on stages with the highest total amount.",
        "answer_source": "table aggregation",
        "data": pd.DataFrame(
            [{"Deal_Stage": "Closed Lost", "Deal_Count": 2, "Total_Amount": 500000.0}]
        ),
    }
    context = followup.build_last_answer_context(
        "Show deals by stage.",
        stage_result,
        _sample_leads(),
        _sample_deals(),
    )

    response = followup.answer_followup_question(
        "what can i do to address the closed lost deals",
        context,
        _sample_leads(),
        _sample_deals(),
    )

    assert response["answer_source"] == "Ollama follow-up explanation"
    assert response["followup_intent_bucket"] == "qualitative_contextual"
    assert response["followup_relevance_score"] >= 2


def test_action_wording_takes_priority_over_deal_entity_metric():
    context = _last_answer_context()
    assert (
        followup.classify_followup_intent(
            "what are the suitable steps we can take for the closed lost deals",
            context,
        )
        == "business_recommendation"
    )


def test_unsupported_pandas_action_route_falls_back_to_grounded_ollama(monkeypatch):
    monkeypatch.setattr(followup, "classify_followup_intent", lambda question, context: "analytics")
    monkeypatch.setattr(
        followup,
        "answer_business_question",
        lambda *args, **kwargs: {
            "answer_type": "unsupported",
            "answer_source": "unsupported",
            "answer": "Pandas cannot determine a requested calculation.",
        },
    )
    monkeypatch.setattr(followup, "check_ollama_available", lambda: True)
    monkeypatch.setattr(followup, "check_ollama_model_available", lambda model: True)
    monkeypatch.setattr(
        followup,
        "generate_fast_explanation",
        lambda prompt, model=None: {
            "ok": True,
            "content": "Review loss reasons and improve early-stage qualification.",
            "error": None,
        },
    )

    response = followup.answer_followup_question(
        "what can we do to address this?",
        _last_answer_context(),
        _sample_leads(),
        _sample_deals(),
    )

    assert response["answer_source"] == "Ollama follow-up explanation"
    assert response["followup_intent_bucket"] == "qualitative_contextual"


def test_explanation_cache_key_isolated_by_provider_and_model(monkeypatch):
    context = _last_answer_context()
    monkeypatch.setattr(followup, "llm_provider_identity", lambda: ("ollama", "llama3.1"))
    ollama_key = followup.make_explanation_cache_key(
        "what does this mean?",
        context,
        "qualitative_contextual",
    )
    monkeypatch.setattr(followup, "llm_provider_identity", lambda: ("openai", "gpt-4o-mini"))
    openai_key = followup.make_explanation_cache_key(
        "what does this mean?",
        context,
        "qualitative_contextual",
    )

    assert ollama_key != openai_key
