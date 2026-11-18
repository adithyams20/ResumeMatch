"""
tests/test_llm_suggestions.py — Unit tests for llm_suggestions module.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from llm_suggestions import (
    GROUNDING_RULE,
    BulletRewriteItem,
    BulletRewritesResult,
    CoverLetterDraftResult,
    GapClosingPlanResult,
    GapClosingStep,
    InterviewQuestionsResult,
    LLMInvalidResponseError,
    LLMUnavailableError,
    MissingSkillExplanation,
    MissingSkillExplanationsResult,
    TailoredSummaryResult,
    clear_suggestions_cache,
    compute_cache_key,
    generate_all_suggestions,
    get_bullet_rewrites,
    get_cover_letter_draft,
    get_gap_closing_plan,
    get_interview_questions,
    get_missing_skill_explanations,
    get_tailored_summary,
)


@pytest.fixture(autouse=True)
def reset_cache():
    """Ensure cache is clean before and after every test."""
    clear_suggestions_cache()
    yield
    clear_suggestions_cache()


# ---------------------------------------------------------------------------
# 1. Schema Validation Tests
# ---------------------------------------------------------------------------

class TestSchemas:
    def test_bullet_rewrites_valid(self):
        data = {
            "rewrites": [
                {
                    "original": "Built APIs",
                    "rewritten": "Designed and deployed REST APIs serving [X] requests daily",
                    "improvement_reason": "Adds scope and metric placeholder",
                }
            ]
        }
        res = BulletRewritesResult.model_validate(data)
        assert len(res.rewrites) == 1
        assert res.rewrites[0].original == "Built APIs"
        assert "[X]" in res.rewrites[0].rewritten

    def test_bullet_rewrites_invalid(self):
        with pytest.raises(ValidationError):
            BulletRewritesResult.model_validate({"rewrites": [{"original": "only one field"}]})

    def test_missing_skills_valid(self):
        data = {
            "explanations": [
                {
                    "skill": "Docker",
                    "why_it_matters": "Essential for containerizing microservices in production.",
                    "related_experience": "No related experience found in resume.",
                }
            ]
        }
        res = MissingSkillExplanationsResult.model_validate(data)
        assert len(res.explanations) == 1
        assert res.explanations[0].skill == "Docker"

    def test_tailored_summary_valid(self):
        data = {"summary": "Experienced Python Engineer with 4 years building scalable web services."}
        res = TailoredSummaryResult.model_validate(data)
        assert "Python" in res.summary

    def test_gap_closing_plan_valid(self):
        data = {
            "steps": [
                {
                    "gap": "Kubernetes",
                    "suggested_step": "Deploy a multi-service app on local Minikube with Helm charts.",
                    "timeframe": "1-2 weeks",
                }
            ]
        }
        res = GapClosingPlanResult.model_validate(data)
        assert len(res.steps) == 1
        assert res.steps[0].timeframe == "1-2 weeks"

    def test_interview_questions_valid(self):
        data = {
            "technical_questions": [f"Tech Q{i}" for i in range(1, 6)],
            "behavioral_questions": [f"Beh Q{i}" for i in range(1, 4)],
        }
        res = InterviewQuestionsResult.model_validate(data)
        assert len(res.technical_questions) == 5
        assert len(res.behavioral_questions) == 3

    def test_cover_letter_draft_valid(self):
        data = {
            "draft": "Dear Hiring Team,\n\nI am writing to express my interest in the Software Engineer role...",
            "word_count": 120,
            "notice": "Draft to edit",
        }
        res = CoverLetterDraftResult.model_validate(data)
        assert "Hiring Team" in res.draft
        assert res.word_count == 120


# ---------------------------------------------------------------------------
# 2. Cache Key and Caching Behavior Tests
# ---------------------------------------------------------------------------

class TestCaching:
    def test_cache_key_deterministic(self):
        k1 = compute_cache_key("Resume A", "JD B", "bullet_rewrites")
        k2 = compute_cache_key("Resume A", "JD B", "bullet_rewrites")
        k3 = compute_cache_key("Resume A", "JD B", "tailored_summary")
        assert k1 == k2
        assert k1 != k3

    @patch("llm_suggestions._call_llm_with_retry")
    def test_cache_prevents_duplicate_api_calls(self, mock_call):
        mock_call.return_value = json.dumps({
            "summary": "Software engineer with experience in Python and Flask."
        })

        resume = "Python Flask developer"
        jd = "Looking for Flask developer"

        # First call hits mock
        res1 = get_tailored_summary(resume, jd, provider="openai")
        assert mock_call.call_count == 1
        assert "Flask" in res1.summary

        # Second call with same inputs should be served from cache
        res2 = get_tailored_summary(resume, jd, provider="openai")
        assert mock_call.call_count == 1  # Not called again
        assert res1.summary == res2.summary


# ---------------------------------------------------------------------------
# 3. Grounding Rule in Prompts
# ---------------------------------------------------------------------------

class TestGroundingRule:
    def test_grounding_rule_present_in_all_prompts(self):
        """Verify that every prompt sent to the LLM contains the exact required grounding rule."""
        captured_prompts = []

        def mock_call(prompt, **kwargs):
            captured_prompts.append(prompt)
            return json.dumps({})

        with patch("llm_suggestions._call_llm_with_retry", side_effect=mock_call):
            for fn, args in [
                (get_bullet_rewrites, ("resume", "jd")),
                (get_missing_skill_explanations, ("resume", "jd", ["Docker"])),
                (get_tailored_summary, ("resume", "jd")),
                (get_gap_closing_plan, ("resume", "jd", ["Docker"])),
                (get_interview_questions, ("resume", "jd", ["Python"], ["Docker"])),
                (get_cover_letter_draft, ("resume", "jd")),
            ]:
                try:
                    fn(*args, bypass_cache=True)
                except Exception:
                    pass

        assert len(captured_prompts) == 6
        for p in captured_prompts:
            assert GROUNDING_RULE in p


# ---------------------------------------------------------------------------
# 4. Error Handling Tests
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_missing_api_key_raises_llm_unavailable(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=True):
            with patch("config.OPENAI_API_KEY", None):
                with pytest.raises(LLMUnavailableError):
                    get_tailored_summary("resume", "jd", provider="openai", bypass_cache=True)

    @patch("llm_suggestions._call_llm_with_retry")
    def test_invalid_json_raises_llm_invalid_response(self, mock_call):
        mock_call.return_value = "This is not JSON at all!"
        with pytest.raises(LLMInvalidResponseError):
            get_tailored_summary("resume", "jd", provider="openai", bypass_cache=True)

    def test_generate_all_suggestions_missing_key_graceful(self):
        """generate_all_suggestions should not crash if API key is missing."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=True):
            with patch("config.OPENAI_API_KEY", None):
                bundle = generate_all_suggestions("resume", "jd", provider="openai")
                assert bundle.overall_warning is not None
                assert "OPENAI_API_KEY" in bundle.overall_warning


# ---------------------------------------------------------------------------
# 5. End-to-End Mocked Feature Generation
# ---------------------------------------------------------------------------

class TestFeatureGeneration:
    @patch("llm_suggestions._call_llm_with_retry")
    def test_bullet_rewrites_generation(self, mock_call):
        mock_call.return_value = json.dumps({
            "rewrites": [
                {
                    "original": "Worked on backend",
                    "rewritten": "Architected Python backend handling [X] daily transactions",
                    "improvement_reason": "Specific impact",
                }
            ]
        })
        res = get_bullet_rewrites("resume", "jd")
        assert len(res.rewrites) == 1
        assert "[X]" in res.rewrites[0].rewritten

    @patch("llm_suggestions._call_llm_with_retry")
    def test_missing_skills_generation(self, mock_call):
        mock_call.return_value = json.dumps({
            "explanations": [
                {
                    "skill": "PostgreSQL",
                    "why_it_matters": "Primary relational database for this team.",
                    "related_experience": "Resume shows SQLite experience.",
                }
            ]
        })
        res = get_missing_skill_explanations("resume", "jd", ["PostgreSQL"])
        assert len(res.explanations) == 1
        assert res.explanations[0].skill == "PostgreSQL"

    @patch("llm_suggestions._call_llm_with_retry")
    def test_interview_questions_generation(self, mock_call):
        mock_call.return_value = json.dumps({
            "technical_questions": [f"Tech {i}" for i in range(1, 6)],
            "behavioral_questions": [f"Beh {i}" for i in range(1, 4)],
        })
        res = get_interview_questions("resume", "jd", ["Python"], ["Docker"])
        assert len(res.technical_questions) == 5
        assert len(res.behavioral_questions) == 3

    @patch("llm_suggestions._call_llm_with_retry")
    def test_cover_letter_generation(self, mock_call):
        mock_call.return_value = json.dumps({
            "draft": "I am writing to express my strong enthusiasm for this engineering role.",
            "word_count": 14,
            "notice": "Draft to edit — review and customize before submitting.",
        })
        res = get_cover_letter_draft("resume", "jd")
        assert "Draft to edit" in res.notice
        assert res.word_count > 0

