"""
matcher/report.py — Assemble the final MatchReport dataclass.

Orchestrates the full pipeline (NLP → RAG → scoring → LLM) and returns
a single structured result object that app.py can render.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

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

class MatchReport(BaseModel):
    """Validated result of a resume-vs-JD analysis."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    final_score: float
    skill_match: float
    semantic_match: float
    experience_match: float
    matched_skills: list[str]
    missing_skills: list[str]
    jd_skills: list[str]
    resume_skills: list[str]
    resume_years: float
    required_years: float
    requirement_matches: list[RequirementMatch]
    llm_analysis: LLMAnalysis | None = None
    llm_warning: str | None = None
    sections: dict[str, str] = Field(default_factory=dict)
    resume_text: str = ""
    jd_text: str = ""
    run_llm: bool = False
    llm_provider: str = "openai"

    @property
    def formula_explanation(self) -> str:
        return (
            "Final Score = (0.4 * skill_match) + (0.4 * semantic_match) + "
            "(0.2 * experience_match), scaled to 0-100."
        )


# ---------------------------------------------------------------------------
# JD requirement extraction helpers
# ---------------------------------------------------------------------------

# Header / label lines that should not be treated as requirements.
_HEADER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*(what\s+we'?re?\s+(looking\s+for|offer)|requirements?|"
               r"qualifications?|responsibilities|skills?|eligibility|"
               r"nice\s+to\s+have|good\s+to\s+have|preferred|benefits?|"
               r"about\s+(us|the\s+(role|team|company))|"
               r"job\s+(description|summary|overview)|duties|"
               r"who\s+you\s+are|you\s+will|we\s+(are|offer)|"
               r"minimum|education|degree|experience)\s*:?\s*$",
               re.IGNORECASE),
    # Eligibility degree lines like "Degree: B.E. / B.Tech..."
    re.compile(r"^\s*(eligibility|degree|qualification)\s*:", re.IGNORECASE),
    # Single-word or very short lines (< 3 words) that are just labels
    re.compile(r"^\s*\w{1,30}\s*:\s*$"),
]

# Patterns that indicate the line is a real requirement (has a verb or skill keyword)
_SKIP_IF_NO_VERB = re.compile(
    r"\b(experience|knowledge|proficiency|familiarity|understanding|"
    r"ability|skill|work|develop|build|design|implement|write|manage|"
    r"strong|excellent|good|comfortable|passionate|eager|willing|"
    r"collaborate|communicate|learn)\b",
    re.IGNORECASE,
)


def _is_header_line(line: str) -> bool:
    """Return True if *line* looks like a section header, not a requirement."""
    stripped = line.strip()
    if not stripped:
        return True
    for pat in _HEADER_PATTERNS:
        if pat.match(stripped):
            return True
    # Pure bullet with a single word (e.g. "* Python")
    bullet_text = re.sub(r"^[\s\-\*\•\·]+", "", stripped)
    if len(bullet_text.split()) <= 1:
        return True  # too short to be a real requirement
    return False


# Sentences that describe a real requirement rather than company boilerplate.
_REQUIREMENT_HINTS: list[re.Pattern[str]] = [
    re.compile(r"\b(requirements?|qualifications?|responsibilities|duties|"
               r"skills?|tools|technolog|experience|knowledge|proficiency|"
               r"familiarity|understanding|ability|develop|build|design|"
               r"implement|write|manage|work|collaborate|communicate|"
               r"qualify|eligible|degree|certificate|certification|"
               r"role|position|candidate|applicant)\b", re.IGNORECASE),
]

# Company boilerplate that should never be evaluated as a requirement.
_BOILERPLATE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(about\s+us|who\s+we\s+are|our\s+company|our\s+team|"
               r"headcount|hiring|recruiting|equal\s+opportunity|"
               r"diversity|inclusion|benefits?|perks|compensation|salary|"
               r"culture|mission|vision|values?|work\s+life|flexible|remote|"
               r"join\s+us|apply\s+now|we\s+are\s+hiring|now\s+hiring|"
               r"location|based\s+in|country|countries|state|city|"
               r"EOE|EEO|affirmative\s+action)\b", re.IGNORECASE),
    re.compile(r"\b(equity|health\s+insurance|401k|retirement|vacation|"
               r"PTO|paid\s+time|dental|vision|life\s+insurance|"
               r"wellness|gym|free\s+lunch|snacks|breakroom|kitchen|"
               r"parking|transit|commute|shuttle|relocation|"
               r"mentorship|growth|learning\s+and\s+development|L&D)\b", re.IGNORECASE),
]


def _is_requirement_sentence(sent: str) -> bool:
    """Return True if *sent* looks like a real requirement rather than boilerplate."""
    stripped = sent.strip()
    if not stripped:
        return False
    if len(stripped.split()) < 6:
        return False
    for pat in _BOILERPLATE_PATTERNS:
        if pat.search(stripped):
            return False
    for pat in _REQUIREMENT_HINTS:
        if pat.search(stripped):
            return True
    return False


def _extract_jd_requirements(jd_text: str, nlp: Any | None = None) -> list[str]:
    """Extract meaningful requirement phrases from a job description.

    Strategy:
    1. Split on newlines first to preserve bullet-point structure.
    2. Further split long lines on ". " or "; " sentence boundaries.
    3. Drop header lines and lines shorter than 5 words.
    4. Deduplicate while preserving order.

    Args:
        jd_text: Cleaned job description text.
        nlp: Optional spaCy model (unused here but kept for signature compat).

    Returns:
        List of requirement strings, deduplicated, capped at 40.
    """
    requirements: list[str] = []
    seen: set[str] = set()

    for raw_line in jd_text.splitlines():
        # Strip leading bullet characters
        line = re.sub(r"^[\s\-\*\•\·\–\—]+", "", raw_line).strip()
        if not line:
            continue

        # Split on sentence boundaries within a single line
        sub_sentences = re.split(r"(?<=[.!?;])\s+", line)

        requirements: list[str] = []
    seen: set[str] = set()
    skipped = 0

    for raw_line in jd_text.splitlines():
        line = re.sub(r"^[\s\-\*\•\·\–\—]+", "", raw_line).strip()
        if not line:
            continue
        sub_sentences = re.split(r"(?<=[.!?;])\s+", line)
        for sent in sub_sentences:
            sent = sent.strip().rstrip(".")
            if not sent:
                continue
            if _is_header_line(sent):
                continue
            if not _is_requirement_sentence(sent):
                skipped += 1
                continue
            key = sent.lower()
            if key in seen:
                continue
            seen.add(key)
            requirements.append(sent)

    return requirements[:40], skipped


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

class MatchReport(BaseModel):
    """Validated result of a resume-vs-JD analysis."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    final_score: float
    skill_match: float
    semantic_match: float
    experience_match: float
    matched_skills: list[str]
    missing_skills: list[str]
    jd_skills: list[str]
    resume_skills: list[str]
    resume_years: float
    required_years: float
    requirement_matches: list[RequirementMatch]
    llm_analysis: LLMAnalysis | None = None
    llm_warning: str | None = None
    sections: dict[str, str] = Field(default_factory=dict)
    resume_text: str = ""
    jd_text: str = ""
    run_llm: bool = False
    llm_provider: str = "openai"
    jd_requirement_count: int = 0
    jd_skipped_count: int = 0

    @property
    def formula_explanation(self) -> str:
        return (
            "Final Score = (0.4 * skill_match) + (0.4 * semantic_match) + "
            "(0.2 * experience_match), scaled to 0-100."
        )


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
    requirements, skipped_count = _extract_jd_requirements(jd_clean, nlp=nlp)
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
        resume_text=resume_clean,
        jd_text=jd_clean,
        run_llm=run_llm,
        llm_provider=llm_provider,
        jd_requirement_count=len(requirements),
        jd_skipped_count=skipped_count,
    )

