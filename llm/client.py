"""
llm/client.py — Provider-agnostic LLM wrapper with retries and timeouts.

Supports OpenAI and Anthropic. The provider is selected via the
LLM_PROVIDER environment variable ("openai" or "anthropic").

If no API key is available, the call raises LLMUnavailableError so
callers can degrade gracefully.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from pydantic import BaseModel, ValidationError

from config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    LLM_MAX_RETRIES,
    LLM_MAX_TOKENS,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)
from llm.prompts import SYSTEM_PROMPT, build_analysis_prompt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response schema (Pydantic v2)
# ---------------------------------------------------------------------------

class LLMAnalysis(BaseModel):
    """Validated output from the LLM analysis step."""

    gaps: list[str]
    suggestions: list[str]
    keywords_to_add: list[str]


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class LLMUnavailableError(RuntimeError):
    """Raised when the LLM cannot be called (missing key, network error, etc.)."""


class LLMInvalidResponseError(ValueError):
    """Raised when the LLM returns text that cannot be parsed as valid JSON."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_analysis(
    job_description: str,
    missing_skills: list[str],
    evidence_chunks: list[dict[str, str]],
    provider: str = LLM_PROVIDER,
) -> LLMAnalysis:
    """Call the LLM and return a validated :class:`LLMAnalysis`.

    Args:
        job_description: Full JD text.
        missing_skills: Skills not found in the resume.
        evidence_chunks: Retrieved evidence dicts (keys: requirement, chunk, score).
        provider: ``"openai"`` or ``"anthropic"``.

    Returns:
        :class:`LLMAnalysis` with gaps, suggestions, and keywords_to_add.

    Raises:
        LLMUnavailableError: No API key or provider not reachable.
        LLMInvalidResponseError: Response could not be parsed / validated.
    """
    prompt = build_analysis_prompt(job_description, missing_skills, evidence_chunks)
    raw_response = _call_with_retry(prompt, provider)
    return _parse_response(raw_response)


# ---------------------------------------------------------------------------
# Provider dispatch
# ---------------------------------------------------------------------------

def _call_with_retry(prompt: str, provider: str) -> str:
    """Call the selected LLM with exponential back-off retries.

    Args:
        prompt: The user-turn prompt text.
        provider: ``"openai"`` or ``"anthropic"``.

    Returns:
        Raw text response from the LLM.

    Raises:
        LLMUnavailableError: After all retries are exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            logger.info(
                "LLM call attempt %d/%d (provider=%s) …", attempt, LLM_MAX_RETRIES, provider
            )
            if provider == "openai":
                return _call_openai(prompt)
            elif provider == "anthropic":
                return _call_anthropic(prompt)
            else:
                raise LLMUnavailableError(
                    f"Unknown LLM provider '{provider}'. "
                    "Set LLM_PROVIDER to 'openai' or 'anthropic'."
                )
        except LLMUnavailableError:
            raise  # Don't retry on config errors
        except Exception as exc:
            last_exc = exc
            wait = 2 ** attempt
            logger.warning("LLM call failed (attempt %d): %s. Retrying in %ds …", attempt, exc, wait)
            time.sleep(wait)

    raise LLMUnavailableError(
        f"LLM call failed after {LLM_MAX_RETRIES} attempts: {last_exc}"
    ) from last_exc


def _call_openai(prompt: str) -> str:
    """Call the OpenAI chat completions API.

    Args:
        prompt: User-turn prompt.

    Returns:
        Response text.

    Raises:
        LLMUnavailableError: If the API key is missing.
    """
    if not OPENAI_API_KEY:
        raise LLMUnavailableError(
            "OPENAI_API_KEY is not set. Add it to your .env file to enable LLM features."
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise LLMUnavailableError("openai package is not installed.") from exc

    client = OpenAI(api_key=OPENAI_API_KEY, timeout=LLM_TIMEOUT)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )
    return response.choices[0].message.content or ""


def _call_anthropic(prompt: str) -> str:
    """Call the Anthropic messages API.

    Args:
        prompt: User-turn prompt.

    Returns:
        Response text.

    Raises:
        LLMUnavailableError: If the API key is missing.
    """
    if not ANTHROPIC_API_KEY:
        raise LLMUnavailableError(
            "ANTHROPIC_API_KEY is not set. Add it to your .env file to enable LLM features."
        )
    try:
        import anthropic
    except ImportError as exc:
        raise LLMUnavailableError("anthropic package is not installed.") from exc

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=LLM_TIMEOUT)
    message = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text if message.content else ""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_response(raw: str) -> LLMAnalysis:
    """Parse and validate the raw LLM response string.

    Strips markdown fences if present before JSON parsing.

    Args:
        raw: Raw text returned by the LLM.

    Returns:
        Validated :class:`LLMAnalysis`.

    Raises:
        LLMInvalidResponseError: If the text is not valid JSON or fails schema.
    """
    # Strip markdown code fences
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMInvalidResponseError(
            f"LLM returned invalid JSON: {exc}\n\nRaw response:\n{raw[:500]}"
        ) from exc

    try:
        return LLMAnalysis(**data)
    except (ValidationError, TypeError) as exc:
        raise LLMInvalidResponseError(
            f"LLM JSON did not match expected schema: {exc}"
        ) from exc

