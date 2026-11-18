"""
llm_suggestions.py — Dedicated LLM suggestions module for ResumeMatch.

Provides 6 structured, validated AI-powered resume enhancement features:
a. Bullet rewrites (up to 5 bullets matching job language with [X] metrics)
b. Missing-skill explanations (why each matters and related experience)
c. Tailored summary (2-3 lines grounded strictly in resume facts)
d. Gap-closing plan (1-2 week small projects or learning steps)
e. Interview questions (5 technical and 3 behavioral questions)
f. Cover letter draft (under 250 words, marked as draft)

All prompts enforce:
"Use only facts present in the resume. If something is missing, say so instead of inventing it."
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from typing import Any, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

import config

logger = logging.getLogger(__name__)

# Ensure .env is loaded
load_dotenv(config.BASE_DIR / ".env", override=False)

# ---------------------------------------------------------------------------
# Strict Grounding Rule (Required in every prompt)
# ---------------------------------------------------------------------------
GROUNDING_RULE: str = (
    "Use only facts present in the resume. If something is missing, say so instead of inventing it."
)

SYSTEM_PROMPT: str = f"""\
You are an expert recruitment advisor and ATS resume specialist.
CRITICAL MANDATORY RULE:
{GROUNDING_RULE}
Never invent technologies, employers, dates, metrics, or credentials.
You must output ONLY valid, parseable JSON conforming strictly to the requested schema.
Do not include any introductory or concluding conversational prose.
"""


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class LLMUnavailableError(RuntimeError):
    """Raised when an API key is missing or the LLM service is unreachable."""
    pass


class LLMInvalidResponseError(ValueError):
    """Raised when the LLM response cannot be parsed as valid JSON matching the schema."""
    pass


# ---------------------------------------------------------------------------
# Pydantic Schemas for the 6 Features
# ---------------------------------------------------------------------------

# a. Bullet rewrites
class BulletRewriteItem(BaseModel):
    original: str = Field(description="The original weak resume bullet point")
    rewritten: str = Field(description="The rewritten bullet aligned with job language, using [X] for missing metrics")
    improvement_reason: str = Field(description="Brief explanation of why this rewrite is stronger")


class BulletRewritesResult(BaseModel):
    rewrites: list[BulletRewriteItem] = Field(default_factory=list, description="Up to 5 rewritten bullet points")


# b. Missing-skill explanations
class MissingSkillExplanation(BaseModel):
    skill: str = Field(description="Name of the missing skill")
    why_it_matters: str = Field(description="1-2 sentences on why this skill matters for the specific role")
    related_experience: str = Field(
        description="Whether the resume shows related or adjacent experience; if none, explicitly says so"
    )


class MissingSkillExplanationsResult(BaseModel):
    explanations: list[MissingSkillExplanation] = Field(default_factory=list)


# c. Tailored summary
class TailoredSummaryResult(BaseModel):
    summary: str = Field(description="A 2-3 line professional summary using only facts from the resume")


# d. Gap-closing plan
class GapClosingStep(BaseModel):
    gap: str = Field(description="The specific skill or qualification gap identified")
    suggested_step: str = Field(description="One small practical project or focused learning step to close the gap")
    timeframe: str = Field(default="1-2 weeks", description="Estimated completion timeframe")


class GapClosingPlanResult(BaseModel):
    steps: list[GapClosingStep] = Field(default_factory=list)


# e. Interview questions
class InterviewQuestionsResult(BaseModel):
    technical_questions: list[str] = Field(
        default_factory=list, description="Exactly 5 technical questions based on matched and missing skills"
    )
    behavioral_questions: list[str] = Field(
        default_factory=list, description="Exactly 3 behavioral questions based on matched and missing skills"
    )


# f. Cover letter draft
class CoverLetterDraftResult(BaseModel):
    draft: str = Field(description="First draft cover letter under 250 words, grounded strictly in resume facts")
    word_count: int = Field(default=0, description="Approximate word count of the draft")
    notice: str = Field(default="Draft to edit — review and customize before submitting.")


# Bundle containing all results
class SuggestionsBundle(BaseModel):
    bullet_rewrites: BulletRewritesResult | None = None
    bullet_rewrites_error: str | None = None

    missing_skills: MissingSkillExplanationsResult | None = None
    missing_skills_error: str | None = None

    summary: TailoredSummaryResult | None = None
    summary_error: str | None = None

    gap_plan: GapClosingPlanResult | None = None
    gap_plan_error: str | None = None

    interview_questions: InterviewQuestionsResult | None = None
    interview_questions_error: str | None = None

    cover_letter: CoverLetterDraftResult | None = None
    cover_letter_error: str | None = None

    overall_warning: str | None = None


# ---------------------------------------------------------------------------
# In-Memory Cache (hashed by resume text + job description + feature name)
# ---------------------------------------------------------------------------

_SUGGESTIONS_CACHE: dict[str, Any] = {}


def compute_cache_key(resume_text: str, jd_text: str, feature_name: str) -> str:
    """Generate a SHA-256 cache key from resume text, job description, and feature name."""
    raw = f"{resume_text.strip()}\n---\n{jd_text.strip()}\n---\n{feature_name.strip()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def clear_suggestions_cache() -> None:
    """Clear all cached LLM suggestion results."""
    _SUGGESTIONS_CACHE.clear()


# ---------------------------------------------------------------------------
# Provider Dispatch & Low-Level API Calling
# ---------------------------------------------------------------------------

T = TypeVar("T", bound=BaseModel)


def _get_api_key(provider: str) -> str:
    """Retrieve the API key for the chosen provider from environment variables."""
    # Re-read environment variables in case .env changed or was set at runtime
    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY") or config.OPENAI_API_KEY
        if not key:
            raise LLMUnavailableError(
                "OPENAI_API_KEY is not set. Add it to your .env file to enable AI suggestions."
            )
        return key
    elif provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY") or config.ANTHROPIC_API_KEY
        if not key:
            raise LLMUnavailableError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env file to enable AI suggestions."
            )
        return key
    else:
        raise LLMUnavailableError(
            f"Unsupported LLM provider: '{provider}'. Supported providers are 'openai' and 'anthropic'."
        )


def _call_llm_with_retry(
    prompt: str,
    provider: str = "openai",
    timeout: int = config.LLM_TIMEOUT,
    max_retries: int = config.LLM_MAX_RETRIES,
) -> str:
    """Execute a prompt with retry logic and timeout.

    Args:
        prompt: User turn prompt string.
        provider: 'openai' or 'anthropic'.
        timeout: Timeout in seconds.
        max_retries: Maximum retry attempts for transient errors.

    Returns:
        Raw text response from the model.
    """
    api_key = _get_api_key(provider)
    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Calling LLM (attempt %d/%d, provider=%s)...", attempt, max_retries, provider)
            if provider == "openai":
                from openai import OpenAI
                client = OpenAI(api_key=api_key, timeout=timeout)
                model_name = os.getenv("OPENAI_MODEL", config.OPENAI_MODEL)
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=config.LLM_TEMPERATURE,
                    response_format={"type": "json_object"} if "gpt" in model_name else None,
                )
                return completion.choices[0].message.content or ""

            elif provider == "anthropic":
                import anthropic
                client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
                model_name = os.getenv("ANTHROPIC_MODEL", config.ANTHROPIC_MODEL)
                message = client.messages.create(
                    model=model_name,
                    max_tokens=2048,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=config.LLM_TEMPERATURE,
                )
                return message.content[0].text if message.content else ""

        except LLMUnavailableError:
            raise
        except Exception as exc:
            last_exc = exc
            logger.warning("LLM call attempt %d failed: %s", attempt, exc)
            if attempt < max_retries:
                time.sleep(1.5 ** attempt)

    raise LLMUnavailableError(f"LLM request failed after {max_retries} attempts: {last_exc}") from last_exc


def _parse_pydantic_json(raw_text: str, pydantic_cls: type[T]) -> T:
    """Extract and validate JSON against a Pydantic schema."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0].strip()

    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        text = text[start_idx : end_idx + 1]

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMInvalidResponseError(
            f"Failed to parse LLM response as JSON: {exc}. Content snippet: {raw_text[:200]}"
        ) from exc

    try:
        return pydantic_cls.model_validate(data)
    except ValidationError as exc:
        raise LLMInvalidResponseError(
            f"LLM response failed {pydantic_cls.__name__} schema validation: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Feature a: Bullet Rewrites
# ---------------------------------------------------------------------------

def get_bullet_rewrites(
    resume_text: str,
    jd_text: str,
    provider: str = "openai",
    bypass_cache: bool = False,
) -> BulletRewritesResult:
    """Take up to 5 weak resume bullets and rewrite each to match the job's language.

    Uses [X] placeholders for metrics the resume doesn't contain.
    Never invents technologies, employers, dates, or numbers.
    """
    cache_key = compute_cache_key(resume_text, jd_text, "bullet_rewrites")
    if not bypass_cache and cache_key in _SUGGESTIONS_CACHE:
        return _SUGGESTIONS_CACHE[cache_key]

    prompt = f"""\
### STRICT RULE
{GROUNDING_RULE}

### CONTEXT
Job Description (excerpt):
{jd_text[:2500]}

Candidate Resume:
{resume_text[:3500]}

### INSTRUCTIONS
Identify up to 5 weak or generic bullet points in the candidate's resume.
Rewrite each bullet point to closely match the terminology, action verbs, and tone of the Job Description.
If a metric or quantification is needed to strengthen the impact, use a placeholder such as "[X]%" or "[X] users" or "[X] hours" — NEVER invent numbers, dates, tools, or projects that are not in the resume.

### SCHEMA
Return a JSON object with this exact structure:
{{
  "rewrites": [
    {{
      "original": "original bullet text from resume",
      "rewritten": "rewritten bullet with [X] for missing metrics and aligned keywords",
      "improvement_reason": "concise explanation of why this is more effective for this role"
    }}
  ]
}}
"""
    raw = _call_llm_with_retry(prompt, provider=provider)
    result = _parse_pydantic_json(raw, BulletRewritesResult)
    _SUGGESTIONS_CACHE[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Feature b: Missing-Skill Explanations
# ---------------------------------------------------------------------------

def get_missing_skill_explanations(
    resume_text: str,
    jd_text: str,
    missing_skills: list[str] | None = None,
    provider: str = "openai",
    bypass_cache: bool = False,
) -> MissingSkillExplanationsResult:
    """For each missing skill, give 1-2 sentences on why it matters and related experience."""
    cache_key = compute_cache_key(resume_text, jd_text, "missing_skills")
    if not bypass_cache and cache_key in _SUGGESTIONS_CACHE:
        return _SUGGESTIONS_CACHE[cache_key]

    missing_list_str = ", ".join(missing_skills) if missing_skills else "No missing skills identified."

    prompt = f"""\
### STRICT RULE
{GROUNDING_RULE}

### CONTEXT
Job Description:
{jd_text[:2500]}

Missing Skills from Job Description:
{missing_list_str}

Candidate Resume:
{resume_text[:3500]}

### INSTRUCTIONS
For each missing skill listed above:
1. Explain in 1-2 sentences why this skill matters for this specific role based on the Job Description.
2. Check the resume for any related, adjacent, or foundational experience that could demonstrate transferability. If there is NO related experience, plainly state: "No related experience found in resume."

### SCHEMA
Return a JSON object with this exact structure:
{{
  "explanations": [
    {{
      "skill": "skill name",
      "why_it_matters": "1-2 sentences on importance in this job",
      "related_experience": "analysis of adjacent skills or 'No related experience found in resume.'"
    }}
  ]
}}
"""
    raw = _call_llm_with_retry(prompt, provider=provider)
    result = _parse_pydantic_json(raw, MissingSkillExplanationsResult)
    _SUGGESTIONS_CACHE[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Feature c: Tailored Summary
# ---------------------------------------------------------------------------

def get_tailored_summary(
    resume_text: str,
    jd_text: str,
    provider: str = "openai",
    bypass_cache: bool = False,
) -> TailoredSummaryResult:
    """Generate a 2-3 line professional summary using only facts from the resume."""
    cache_key = compute_cache_key(resume_text, jd_text, "tailored_summary")
    if not bypass_cache and cache_key in _SUGGESTIONS_CACHE:
        return _SUGGESTIONS_CACHE[cache_key]

    prompt = f"""\
### STRICT RULE
{GROUNDING_RULE}

### CONTEXT
Target Job Description:
{jd_text[:2500]}

Candidate Resume:
{resume_text[:3500]}

### INSTRUCTIONS
Write a compelling, professional 2-3 line summary highlighting the candidate's genuine background as it relates to this role.
Use ONLY facts present in the resume. Do not claim years of experience or technologies not mentioned in the resume.

### SCHEMA
Return a JSON object with this exact structure:
{{
  "summary": "2-3 line professional summary"
}}
"""
    raw = _call_llm_with_retry(prompt, provider=provider)
    result = _parse_pydantic_json(raw, TailoredSummaryResult)
    _SUGGESTIONS_CACHE[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Feature d: Gap-Closing Plan
# ---------------------------------------------------------------------------

def get_gap_closing_plan(
    resume_text: str,
    jd_text: str,
    missing_skills: list[str] | None = None,
    provider: str = "openai",
    bypass_cache: bool = False,
) -> GapClosingPlanResult:
    """Suggest one small project or learning step (1-2 weeks) for each real gap."""
    cache_key = compute_cache_key(resume_text, jd_text, "gap_closing_plan")
    if not bypass_cache and cache_key in _SUGGESTIONS_CACHE:
        return _SUGGESTIONS_CACHE[cache_key]

    missing_str = ", ".join(missing_skills) if missing_skills else "General alignment gaps"

    prompt = f"""\
### STRICT RULE
{GROUNDING_RULE}

### CONTEXT
Job Description Requirements:
{jd_text[:2500]}

Candidate Gaps / Missing Skills:
{missing_str}

Candidate Resume:
{resume_text[:3500]}

### INSTRUCTIONS
For each genuine qualification or skill gap:
Suggest ONE realistic, small project or focused learning step that the candidate could realistically complete in 1-2 weeks to demonstrate capability.

### SCHEMA
Return a JSON object with this exact structure:
{{
  "steps": [
    {{
      "gap": "name of gap or missing skill",
      "suggested_step": "concrete project or actionable learning step",
      "timeframe": "1-2 weeks"
    }}
  ]
}}
"""
    raw = _call_llm_with_retry(prompt, provider=provider)
    result = _parse_pydantic_json(raw, GapClosingPlanResult)
    _SUGGESTIONS_CACHE[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Feature e: Interview Questions
# ---------------------------------------------------------------------------

def get_interview_questions(
    resume_text: str,
    jd_text: str,
    matched_skills: list[str] | None = None,
    missing_skills: list[str] | None = None,
    provider: str = "openai",
    bypass_cache: bool = False,
) -> InterviewQuestionsResult:
    """Generate 5 technical and 3 behavioral questions based on matched and missing skills."""
    cache_key = compute_cache_key(resume_text, jd_text, "interview_questions")
    if not bypass_cache and cache_key in _SUGGESTIONS_CACHE:
        return _SUGGESTIONS_CACHE[cache_key]

    matched_str = ", ".join(matched_skills) if matched_skills else "None specified"
    missing_str = ", ".join(missing_skills) if missing_skills else "None specified"

    prompt = f"""\
### STRICT RULE
{GROUNDING_RULE}

### CONTEXT
Job Description:
{jd_text[:2500]}

Candidate Matched Skills: {matched_str}
Candidate Missing Skills: {missing_str}

Candidate Resume:
{resume_text[:3500]}

### INSTRUCTIONS
Generate interview questions tailored to this candidate and role:
1. Exactly 5 technical questions testing both the candidate's claimed strengths (matched skills) and assessing knowledge in the required areas (missing skills).
2. Exactly 3 behavioral questions evaluating how they adapt to new technologies, collaborate, and overcome technical gaps.

### SCHEMA
Return a JSON object with this exact structure:
{{
  "technical_questions": [
    "Technical question 1",
    "Technical question 2",
    "Technical question 3",
    "Technical question 4",
    "Technical question 5"
  ],
  "behavioral_questions": [
    "Behavioral question 1",
    "Behavioral question 2",
    "Behavioral question 3"
  ]
}}
"""
    raw = _call_llm_with_retry(prompt, provider=provider)
    result = _parse_pydantic_json(raw, InterviewQuestionsResult)
    _SUGGESTIONS_CACHE[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Feature f: Cover Letter Draft
# ---------------------------------------------------------------------------

def get_cover_letter_draft(
    resume_text: str,
    jd_text: str,
    provider: str = "openai",
    bypass_cache: bool = False,
) -> CoverLetterDraftResult:
    """Draft a cover letter under 250 words, grounded in the resume and marked as a draft."""
    cache_key = compute_cache_key(resume_text, jd_text, "cover_letter_draft")
    if not bypass_cache and cache_key in _SUGGESTIONS_CACHE:
        return _SUGGESTIONS_CACHE[cache_key]

    prompt = f"""\
### STRICT RULE
{GROUNDING_RULE}

### CONTEXT
Job Description:
{jd_text[:2500]}

Candidate Resume:
{resume_text[:3500]}

### INSTRUCTIONS
Write a concise, compelling first-draft cover letter:
- Word count MUST be strictly under 250 words.
- Highlight the candidate's actual projects, roles, and skills from the resume that best align with the Job Description.
- NEVER invent facts or past employers.
- Clearly structured as a draft for the candidate to review and edit.

### SCHEMA
Return a JSON object with this exact structure:
{{
  "draft": "Full text of the cover letter under 250 words",
  "word_count": 210,
  "notice": "Draft to edit — review and customize before submitting."
}}
"""
    raw = _call_llm_with_retry(prompt, provider=provider)
    result = _parse_pydantic_json(raw, CoverLetterDraftResult)
    # Ensure word count is computed accurately if model's is off
    actual_words = len(re.findall(r"\b\w+\b", result.draft))
    result.word_count = actual_words
    _SUGGESTIONS_CACHE[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Master Orchestrator: Generate All Suggestions
# ---------------------------------------------------------------------------

def generate_all_suggestions(
    resume_text: str,
    jd_text: str,
    matched_skills: list[str] | None = None,
    missing_skills: list[str] | None = None,
    provider: str = "openai",
) -> SuggestionsBundle:
    """Generate all 6 LLM suggestion features with individual fault tolerance.

    If any individual feature fails (e.g. timeout or schema issue), the other
    features continue to function and the error is cleanly recorded.
    """
    bundle = SuggestionsBundle()

    # Pre-check API key availability to give an upfront friendly warning if missing
    try:
        _get_api_key(provider)
    except LLMUnavailableError as exc:
        bundle.overall_warning = str(exc)
        return bundle

    # 1. Bullet rewrites
    try:
        bundle.bullet_rewrites = get_bullet_rewrites(resume_text, jd_text, provider=provider)
    except Exception as exc:
        logger.warning("Bullet rewrites generation failed: %s", exc)
        bundle.bullet_rewrites_error = str(exc)

    # 2. Missing skills explanation
    try:
        bundle.missing_skills = get_missing_skill_explanations(
            resume_text, jd_text, missing_skills=missing_skills, provider=provider
        )
    except Exception as exc:
        logger.warning("Missing skills explanation failed: %s", exc)
        bundle.missing_skills_error = str(exc)

    # 3. Tailored summary
    try:
        bundle.summary = get_tailored_summary(resume_text, jd_text, provider=provider)
    except Exception as exc:
        logger.warning("Tailored summary generation failed: %s", exc)
        bundle.summary_error = str(exc)

    # 4. Gap-closing plan
    try:
        bundle.gap_plan = get_gap_closing_plan(
            resume_text, jd_text, missing_skills=missing_skills, provider=provider
        )
    except Exception as exc:
        logger.warning("Gap-closing plan generation failed: %s", exc)
        bundle.gap_plan_error = str(exc)

    # 5. Interview questions
    try:
        bundle.interview_questions = get_interview_questions(
            resume_text,
            jd_text,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            provider=provider,
        )
    except Exception as exc:
        logger.warning("Interview questions generation failed: %s", exc)
        bundle.interview_questions_error = str(exc)

    # 6. Cover letter draft
    try:
        bundle.cover_letter = get_cover_letter_draft(resume_text, jd_text, provider=provider)
    except Exception as exc:
        logger.warning("Cover letter draft generation failed: %s", exc)
        bundle.cover_letter_error = str(exc)

    return bundle

