"""
tests/test_scorer.py — Unit tests for matcher/scorer.py.
"""

from __future__ import annotations

import pytest

from matcher.scorer import _experience_score, _semantic_score, _skill_score, compute_scores
from rag.retriever import RequirementMatch, RetrievedChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_match(similarity: float) -> RequirementMatch:
    """Create a RequirementMatch with a single chunk at the given similarity."""
    chunk = RetrievedChunk(
        chunk_id="0",
        text="test chunk",
        section="experience",
        similarity=similarity,
    )
    return RequirementMatch(requirement="test requirement", matches=[chunk])


def _make_empty_match() -> RequirementMatch:
    """Create a RequirementMatch with no retrieved chunks."""
    return RequirementMatch(requirement="empty requirement", matches=[])


# ---------------------------------------------------------------------------
# Sub-score unit tests
# ---------------------------------------------------------------------------

class TestSkillScore:
    def test_all_matched(self) -> None:
        assert _skill_score(["Python", "Docker"], ["Python", "Docker"]) == 1.0

    def test_none_matched(self) -> None:
        assert _skill_score([], ["Python", "Docker"]) == 0.0

    def test_partial(self) -> None:
        score = _skill_score(["Python"], ["Python", "Docker"])
        assert score == pytest.approx(0.5)

    def test_empty_jd_skills(self) -> None:
        # No JD skills → default to perfect match
        assert _skill_score([], []) == 1.0

    def test_score_in_range(self) -> None:
        score = _skill_score(["A"], ["A", "B", "C"])
        assert 0.0 <= score <= 1.0


class TestSemanticScore:
    def test_perfect_similarity(self) -> None:
        matches = [_make_match(1.0), _make_match(1.0)]
        assert _semantic_score(matches) == pytest.approx(1.0)

    def test_zero_similarity(self) -> None:
        matches = [_make_match(0.0), _make_match(0.0)]
        assert _semantic_score(matches) == pytest.approx(0.0)

    def test_mean_of_best_scores(self) -> None:
        matches = [_make_match(0.8), _make_match(0.4)]
        score = _semantic_score(matches)
        assert score == pytest.approx(0.6)

    def test_empty_matches_list(self) -> None:
        assert _semantic_score([]) == 0.0

    def test_match_with_no_chunks(self) -> None:
        matches = [_make_empty_match()]
        assert _semantic_score(matches) == 0.0

    def test_score_in_range(self) -> None:
        matches = [_make_match(0.75)]
        score = _semantic_score(matches)
        assert 0.0 <= score <= 1.0


class TestExperienceScore:
    def test_exceeds_requirement(self) -> None:
        assert _experience_score(resume_years=7.0, required_years=5.0) == 1.0

    def test_exactly_meets_requirement(self) -> None:
        assert _experience_score(resume_years=5.0, required_years=5.0) == 1.0

    def test_below_requirement(self) -> None:
        score = _experience_score(resume_years=3.0, required_years=5.0)
        assert score == pytest.approx(0.6)

    def test_zero_resume_years(self) -> None:
        assert _experience_score(resume_years=0.0, required_years=5.0) == 0.0

    def test_no_requirement_defaults_to_one(self) -> None:
        assert _experience_score(resume_years=0.0, required_years=0.0) == 1.0

    def test_score_in_range(self) -> None:
        score = _experience_score(resume_years=2.0, required_years=4.0)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Integration: compute_scores
# ---------------------------------------------------------------------------

class TestComputeScores:
    def test_returns_expected_keys(self) -> None:
        result = compute_scores(
            resume_skills=["Python"],
            jd_skills=["Python"],
            requirement_matches=[_make_match(0.8)],
            resume_years=3.0,
            required_years=2.0,
            weight_skill=0.4,
            weight_semantic=0.4,
            weight_experience=0.2,
        )
        expected_keys = {
            "skill_match", "semantic_match", "experience_match",
            "final_score", "matched_skills", "missing_skills",
        }
        assert expected_keys.issubset(result.keys())

    def test_final_score_range(self) -> None:
        result = compute_scores(
            resume_skills=["Python", "Docker"],
            jd_skills=["Python", "Docker", "Kubernetes"],
            requirement_matches=[_make_match(0.6), _make_match(0.4)],
            resume_years=4.0,
            required_years=5.0,
            weight_skill=0.4,
            weight_semantic=0.4,
            weight_experience=0.2,
        )
        assert 0.0 <= result["final_score"] <= 100.0

    def test_perfect_score(self) -> None:
        result = compute_scores(
            resume_skills=["Python"],
            jd_skills=["Python"],
            requirement_matches=[_make_match(1.0)],
            resume_years=5.0,
            required_years=5.0,
            weight_skill=0.4,
            weight_semantic=0.4,
            weight_experience=0.2,
        )
        assert result["final_score"] == pytest.approx(100.0)

    def test_weights_normalised(self) -> None:
        # Even with weights not summing to 1, the score should be in range
        result = compute_scores(
            resume_skills=["Python"],
            jd_skills=["Python"],
            requirement_matches=[_make_match(0.5)],
            resume_years=3.0,
            required_years=3.0,
            weight_skill=2.0,   # non-standard weights
            weight_semantic=2.0,
            weight_experience=1.0,
        )
        assert 0.0 <= result["final_score"] <= 100.0

    def test_missing_skills_list(self) -> None:
        result = compute_scores(
            resume_skills=["Python"],
            jd_skills=["Python", "Docker"],
            requirement_matches=[],
            resume_years=0.0,
            required_years=0.0,
            weight_skill=0.4,
            weight_semantic=0.4,
            weight_experience=0.2,
        )
        assert "Docker" in result["missing_skills"]
        assert "Python" in result["matched_skills"]

