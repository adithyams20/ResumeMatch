"""
matcher/scorer.py — Deterministic scoring of resume vs job description.

Computes three sub-scores and combines them with configurable weights:
  - skill_match      (exact + fuzzy skill overlap)
  - semantic_match   (mean best cosine similarity from RAG retrieval)
  - experience_match (resume YOE vs required YOE)
"""

from __future__ import annotations

import logging

from nlp.skill_extractor import match_skills
from rag.retriever import RequirementMatch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_scores(
    resume_skills: list[str],
    jd_skills: list[str],
    requirement_matches: list[RequirementMatch],
    resume_years: float,
    required_years: float,
    weight_skill: float,
    weight_semantic: float,
    weight_experience: float,
) -> dict[str, float]:
    """Compute sub-scores and the weighted final score.

    Args:
        resume_skills: Skills extracted from the resume.
        jd_skills: Skills extracted from the job description.
        requirement_matches: RAG retrieval results per JD requirement.
        resume_years: Estimated years of experience from the resume.
        required_years: Required years of experience from the JD.
        weight_skill: Weight for the skill-match component (0–1).
        weight_semantic: Weight for the semantic-match component (0–1).
        weight_experience: Weight for the experience-match component (0–1).

    Returns:
        Dict with keys:
          - ``"skill_match"``      — float in [0, 1]
          - ``"semantic_match"``   — float in [0, 1]
          - ``"experience_match"`` — float in [0, 1]
          - ``"final_score"``      — float in [0, 100] (percentage)
          - ``"matched_skills"``   — list[str] (stored for the report)
          - ``"missing_skills"``   — list[str] (stored for the report)
    """
    # --- Skill match ---
    matched_skills, missing_skills = match_skills(resume_skills, jd_skills)
    skill_match = _skill_score(matched_skills, jd_skills)

    # --- Semantic match ---
    semantic_match = _semantic_score(requirement_matches)

    # --- Experience match ---
    experience_match = _experience_score(resume_years, required_years)

    # --- Normalise weights (in case the user chose non-summing sliders) ---
    total_weight = weight_skill + weight_semantic + weight_experience
    if total_weight == 0:
        total_weight = 1.0  # prevent division by zero
    w_skill = weight_skill / total_weight
    w_sem = weight_semantic / total_weight
    w_exp = weight_experience / total_weight

    final = (
        w_skill * skill_match
        + w_sem * semantic_match
        + w_exp * experience_match
    ) * 100  # scale to 0-100

    logger.info(
        "Scores — skill: %.3f, semantic: %.3f, experience: %.3f, final: %.1f",
        skill_match,
        semantic_match,
        experience_match,
        final,
    )

    return {
        "skill_match": round(skill_match, 4),
        "semantic_match": round(semantic_match, 4),
        "experience_match": round(experience_match, 4),
        "final_score": round(final, 2),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
    }


# ---------------------------------------------------------------------------
# Sub-score helpers
# ---------------------------------------------------------------------------

def _skill_score(matched: list[str], jd_skills: list[str]) -> float:
    """Compute skill match ratio.

    Args:
        matched: JD skills found in the resume.
        jd_skills: All JD skills.

    Returns:
        Float in [0, 1].
    """
    if not jd_skills:
        return 1.0  # No skills listed in JD → perfect match
    return len(matched) / len(jd_skills)


def _semantic_score(requirement_matches: list[RequirementMatch]) -> float:
    """Compute mean of best cosine similarity across JD requirements.

    Args:
        requirement_matches: List of :class:`RequirementMatch` objects.

    Returns:
        Float in [0, 1].
    """
    if not requirement_matches:
        return 0.0
    best_scores = [rm.best_similarity for rm in requirement_matches]
    return sum(best_scores) / len(best_scores)


def _experience_score(resume_years: float, required_years: float) -> float:
    """Compute experience match ratio.

    Args:
        resume_years: Years of experience found in the resume.
        required_years: Years of experience required by the JD.

    Returns:
        Float in [0, 1].  Returns 1.0 if no requirement is specified.
    """
    if required_years <= 0:
        return 1.0  # JD doesn't specify — assume match
    if resume_years >= required_years:
        return 1.0
    return resume_years / required_years

