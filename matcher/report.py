"""
matcher/report.py — Assemble the final MatchReport dataclass.

Orchestrates the full pipeline (NLP → RAG → scoring → LLM) and returns
a single structured result object that app.py can render.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from llm.client import LLMAnalysis, LLMInvalidResponseError, LLMUnavailableError
from matcher.scorer import compute_scores
from nlp.preprocess import (
    clean_text,
    extract_required_experience,
    extract_resume_experience,
    split_sentences,
)
from nlp.section_splitter import split_sections
from nlp.skill_extractor import extract_skills
from rag.indexer import build_index
from rag.retriever import RequirementMatch, retrieve_for_requirements

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class MatchReport:
    """Full result of a resume-vs-JD analysis."""

    # Scores
    final_score: float
    skill_match: float
    semantic_match: float
    experience_match: float

    # Skill breakdown
    matched_skills: list[str]
    missing_skills: list[str]
    jd_skills: list[str]
    resume_skills: list[str]

    # Experience
    resume_years: float
    required_years: float

    # RAG evidence
    requirement_matches: list[RequirementMatch]

    # LLM analysis (optional — None if LLM step was skipped or failed)
    llm_analysis: LLMAnalysis | None = None
    llm_warning: str | None = None   # set when LLM step failed gracefully

    # Section breakdown
    sections: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def build_report(
    resume_text: str,
    jd_text: str,
    nlp: Any,
    embedding_model: Any,
    weight_skill: float,
    weight_semantic: float,
    weight_experience: float,
    run_llm: bool = False,
    llm_provider: str = "openai",
) -> MatchReport:
    """Run the full ResumeMatch pipeline and return a :class:`MatchReport`.

    Args:
        resume_text: Plain-text resume content.
        jd_text: Plain-text job description.
        nlp: Loaded spaCy language model.
        embedding_model: SentenceTransformer model instance.
        weight_skill: Skill-match weight (0–1).
        weight_semantic: Semantic-match weight (0–1).
        weight_experience: Experience-match weight (0–1).
        run_llm: Whether to call the LLM for gap analysis.
        llm_provider: ``"openai"`` or ``"anthropic"``.

    Returns:
        Populated :class:`MatchReport`.
    """
    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # Stage 1 — Clean text
    # ------------------------------------------------------------------
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(jd_text)
    logger.info("Stage 1 (clean text): %.2fs", time.perf_counter() - t0)

    # ------------------------------------------------------------------
    # Stage 2 — NLP: section split, skill extraction, experience
    # ------------------------------------------------------------------
    t1 = time.perf_counter()
    sections = split_sections(resume_clean)
    resume_skills = extract_skills(resume_clean, nlp=nlp)
    jd_skills = extract_skills(jd_clean, nlp=nlp)
    resume_years = extract_resume_experience(resume_clean)
    required_years = extract_required_experience(jd_clean)
    logger.info(
        "Stage 2 (NLP): %.2fs — resume skills: %d, JD skills: %d, "
        "resume YOE: %.1f, required YOE: %.1f",
        time.perf_counter() - t1,
        len(resume_skills),
        len(jd_skills),
        resume_years,
        required_years,
    )

    # ------------------------------------------------------------------
    # Stage 3 — RAG: index resume chunks, retrieve per JD requirement
    # ------------------------------------------------------------------
    t2 = time.perf_counter()
    collection = build_index(sections, embedding_model)
    jd_sentences = split_sentences(jd_clean, nlp=nlp)
    # Filter to likely requirement sentences (non-trivial length)
    requirements = [s for s in jd_sentences if len(s.split()) >= 5][:40]
    requirement_matches = retrieve_for_requirements(requirements, collection, embedding_model)
    logger.info(
        "Stage 3 (RAG): %.2fs — %d requirements, %d chunks indexed",
        time.perf_counter() - t2,
        len(requirements),
        collection.count(),
    )

    # ------------------------------------------------------------------
    # Stage 4 — Scoring
    # ------------------------------------------------------------------
    t3 = time.perf_counter()
    score_data = compute_scores(
        resume_skills=resume_skills,
        jd_skills=jd_skills,
        requirement_matches=requirement_matches,
        resume_years=resume_years,
        required_years=required_years,
        weight_skill=weight_skill,
        weight_semantic=weight_semantic,
        weight_experience=weight_experience,
    )
    logger.info("Stage 4 (scoring): %.2fs", time.perf_counter() - t3)

    # ------------------------------------------------------------------
    # Stage 5 — LLM (optional)
    # ------------------------------------------------------------------
    llm_analysis: LLMAnalysis | None = None
    llm_warning: str | None = None

    if run_llm:
        t4 = time.perf_counter()
        evidence_chunks = [
            {
                "requirement": rm.requirement,
                "chunk": rm.matches[0].text if rm.matches else "",
                "score": rm.best_similarity,
            }
            for rm in requirement_matches
            if rm.matches
        ]
        try:
            from llm.client import run_analysis
            llm_analysis = run_analysis(
                job_description=jd_clean,
                missing_skills=score_data["missing_skills"],
                evidence_chunks=evidence_chunks,
                provider=llm_provider,
            )
            logger.info("Stage 5 (LLM): %.2fs", time.perf_counter() - t4)
        except (LLMUnavailableError, LLMInvalidResponseError) as exc:
            llm_warning = str(exc)
            logger.warning("LLM step skipped: %s", exc)

    total = time.perf_counter() - t0
    logger.info("Total pipeline time: %.2fs", total)

    return MatchReport(
        final_score=score_data["final_score"],
        skill_match=score_data["skill_match"],
        semantic_match=score_data["semantic_match"],
        experience_match=score_data["experience_match"],
        matched_skills=score_data["matched_skills"],
        missing_skills=score_data["missing_skills"],
        jd_skills=jd_skills,
        resume_skills=resume_skills,
        resume_years=resume_years,
        required_years=required_years,
        requirement_matches=requirement_matches,
        llm_analysis=llm_analysis,
        llm_warning=llm_warning,
        sections=sections,
    )

