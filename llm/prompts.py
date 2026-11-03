"""
llm/prompts.py — Prompt templates for the LLM analysis step.

All prompts enforce strict JSON output so responses can be validated
by the Pydantic LLMAnalysis model.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = """\
You are an expert career coach and resume analyst.
Your job is to review a candidate's resume evidence against a job description
and provide honest, actionable feedback.

CRITICAL RULES:
- Never invent experience or skills the candidate does not have.
- Be specific and concrete — reference actual job requirements.
- Return ONLY valid JSON matching the schema exactly. No markdown fences, no prose.
"""


# ---------------------------------------------------------------------------
# Analysis prompt template
# ---------------------------------------------------------------------------

ANALYSIS_PROMPT_TEMPLATE: str = """\
### JOB DESCRIPTION
{job_description}

### MISSING SKILLS (not found in resume)
{missing_skills}

### RESUME EVIDENCE (top matching chunks per JD requirement)
{evidence_chunks}

### TASK
Analyse the gaps between the job description and the resume evidence.
Return a JSON object with exactly these three keys:

{{
  "gaps": [
    // List of strings: JD requirements NOT supported by the evidence.
    // Be specific; quote the JD requirement.
  ],
  "suggestions": [
    // List of 3–5 strings: concrete, honest edits the candidate can make.
    // Example: "Quantify your Python experience with project names and impact metrics."
    // NEVER suggest adding experience the candidate does not have.
  ],
  "keywords_to_add": [
    // List of strings: keywords from the JD the candidate could truthfully add
    // if they have that experience but haven't mentioned it explicitly.
  ]
}}

Return ONLY the JSON object. No explanation before or after.
"""


def build_analysis_prompt(
    job_description: str,
    missing_skills: list[str],
    evidence_chunks: list[dict[str, str]],
) -> str:
    """Build the user-turn analysis prompt.

    Args:
        job_description: Full JD text (truncated to ~2000 chars for safety).
        missing_skills: Skills in the JD not found in the resume.
        evidence_chunks: List of dicts with keys "requirement", "chunk", "score".

    Returns:
        Formatted prompt string.
    """
    jd_truncated = job_description[:2000] + ("…" if len(job_description) > 2000 else "")

    missing_str = (
        ", ".join(missing_skills) if missing_skills else "None — all skills matched!"
    )

    evidence_parts: list[str] = []
    for item in evidence_chunks[:15]:  # cap to avoid token overflow
        evidence_parts.append(
            f"- REQ: {item['requirement']}\n"
            f"  MATCH: {item['chunk']} (score: {item['score']:.2f})"
        )
    evidence_str = "\n".join(evidence_parts) if evidence_parts else "No evidence retrieved."

    return ANALYSIS_PROMPT_TEMPLATE.format(
        job_description=jd_truncated,
        missing_skills=missing_str,
        evidence_chunks=evidence_str,
    )

