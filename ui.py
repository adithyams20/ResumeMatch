"""
ui.py -- Presentation layer for ResumeMatch.

All functions are pure rendering helpers. They build HTML strings with no
leading indentation so Streamlit treats them as raw HTML (not code blocks).
Only render_inputs() returns data; everything else returns None.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------


def load_styles() -> None:
    css = (Path(__file__).parent / "styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _score_ring_svg(score: float) -> str:
    """Return an SVG circular progress ring for score (0-100). Larger (180px)."""
    radius = 72
    circ = 2 * 3.14159265 * radius
    pct = max(0.0, min(100.0, score))
    offset = circ * (1.0 - pct / 100.0)
    if pct >= 70:
        colour = "#E85D1E"
    elif pct >= 45:
        colour = "#C98A2A"
    else:
        colour = "#B8423A"
    return (
        f'<svg width="180" height="180" viewBox="0 0 180 180" aria-label="Score {pct:.0f} out of 100">'
        f'<circle cx="90" cy="90" r="{radius}" stroke="#E3E5EA" stroke-width="12" fill="none"/>'
        f'<circle cx="90" cy="90" r="{radius}" stroke="{colour}" stroke-width="12" fill="none"'
        f' stroke-linecap="round" stroke-dasharray="{circ:.2f}" stroke-dashoffset="{offset:.2f}"'
        f' transform="rotate(-90 90 90)"/>'
        f'<text x="90" y="82" text-anchor="middle" font-size="28" font-weight="800" fill="{colour}"'
        f' font-family="system-ui, sans-serif">{pct:.0f}</text>'
        f'<text x="90" y="100" text-anchor="middle" font-size="12" font-weight="500" fill="#9AA0AA"'
        f' font-family="system-ui, sans-serif">out of 100</text>'
        f'</svg>'
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


def render_header() -> None:
    """Render the top header with logo and brand name."""
    st.markdown(
        '<div class="rm-header">'
        '<div class="rm-header-logo">'
        '<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 40 40" role="img" aria-label="ResumeMatch logo">'
        '<title>ResumeMatch logo</title>'
        '<rect x="2" y="2" width="36" height="36" rx="10" fill="#1B2A4A"/>'
        '<path d="M13 21 L19 27 L28 16" fill="none" stroke="#E85D1E" stroke-width="3.4" stroke-linecap="round" stroke-linejoin="round"/>'
        '<path d="M13 21 L13 13 M13 13 C13 9 21 9 21 13 C21 15.5 18.5 17 15 17" fill="none" stroke="#FFFFFF" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>'
        '<path d="M15 17 L23 17" fill="none" stroke="#FFFFFF" stroke-width="2.6" stroke-linecap="round"/>'
        '</svg>'
        '</div>'
        '<div>'
        '<div class="rm-brand-name">ResumeMatch</div>'
        '<div class="rm-brand-tagline">Understand your resume\'s fit for every opportunity.</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Input panels + status line
# ---------------------------------------------------------------------------


def render_status(has_resume: bool, has_jd: bool) -> None:
    """Render the quiet inline status line below the input panels."""
    if has_resume and has_jd:
        cls, label = "is-ok", "Ready to analyze"
    elif has_resume:
        cls, label = "is-warn", "Add a job description"
    elif has_jd:
        cls, label = "is-warn", "Upload your resume"
    else:
        cls, label = "", "Upload a resume and paste a job description"
    st.markdown(
        f'<div class="rm-status {cls}">'
        f'<span class="rm-dot"></span>'
        f'<span>{label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_inputs() -> tuple[Any | None, str]:
    """Render two matching input panels. Return (uploaded_file, jd_text)."""
    col_resume, col_jd = st.columns(2, gap="large")

    with col_resume:
        with st.container(border=True):
            st.markdown('<div class="rm-input-eyebrow">Your Resume</div>', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Upload resume",
                type=["pdf", "docx", "txt"],
                help="Text-based PDFs, Word documents, or plain text. Scanned images are not supported.",
                label_visibility="collapsed",
            )

    with col_jd:
        with st.container(border=True):
            st.markdown('<div class="rm-input-eyebrow">Job Description</div>', unsafe_allow_html=True)
            jd_text = st.text_area(
                "Job description",
                height=210,
                placeholder="Paste the full job description here...",
                label_visibility="collapsed",
            )

    return uploaded_file, jd_text


# ---------------------------------------------------------------------------
# Score area: ring + verdict text
# ---------------------------------------------------------------------------


def render_score(report: Any) -> None:
    """Render the score ring and verdict sentence."""
    score = report.final_score

    # Verdict text
    if score >= 70:
        verdict = "Strong match - your profile closely aligns with this role."
    elif score >= 55:
        verdict = "Moderate match - there are some areas to strengthen."
    elif score >= 35:
        verdict = "Partial match - several gaps are visible."
    else:
        verdict = "Low match - significant gaps exist between your resume and this role."

    st.markdown(
        '<div class="rm-score">'
        '<div class="rm-score-ring">'
        f'{_score_ring_svg(score)}'
        '</div>'
        '<div class="rm-score-text">'
        f'<div class="rm-verdict">{verdict}</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Breakdown: three connected sub-scores with explanations
# ---------------------------------------------------------------------------


def render_breakdown(report: Any) -> None:
    """Render the three sub-scores as a connected strip with thin bars and explanations."""
    subs = [
        ("Skills Match", report.skill_match * 100,
         "Measures overlap between resume and JD skills."),
        ("Content Alignment", report.semantic_match * 100,
         "Semantic similarity of resume content to job requirements."),
        ("Experience", report.experience_match * 100,
         "Years of experience relative to what the role demands."),
    ]

    parts = []
    for i, (name, val, explanation) in enumerate(subs):
        border = ' style="border-left:1px solid var(--border);"' if i > 0 else ""
        # Color bar based on value
        if val >= 75:
            bar_cls = "green"
        elif val >= 50:
            bar_cls = "amber"
        else:
            bar_cls = "red"
        parts.append(
            f'<div class="rm-sub"{border}>'
            f'<div class="rm-sub-head">'
            f'<span class="rm-sub-name">{name}</span>'
            f'<span class="rm-sub-value">{val:.0f}%</span>'
            f'</div>'
            f'<div class="rm-sub-bar"><i class="{bar_cls}" style="width:{val:.0f}%"></i></div>'
            f'<div class="rm-sub-explanation">{explanation}</div>'
            f'</div>'
        )

    st.markdown(
        '<div class="rm-subscores">' + "".join(parts) + '</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Skills: matched tags (no pill, no x) + missing with unique reasons
# ---------------------------------------------------------------------------


def render_skills(report: Any, missing_reasons: dict[str, str]) -> None:
    """Render matched skill tags and missing skill list with short notes."""
    matched = report.matched_skills
    missing = report.missing_skills
    jd_skills = report.jd_skills

    # Matched
    if matched:
        tag_html = "".join(
            f'<span class="rm-skill-tag">{s}</span>'
            for s in sorted(matched)
        )
        count_html = (
            f'<div style="font-size:0.75rem;color:var(--text-muted);margin-top:0.5rem;">'
            f'{len(matched)} of {len(jd_skills)} required skills detected</div>'
        )
        matched_html = tag_html + count_html
    else:
        matched_html = (
            '<div style="font-size:0.85rem;color:var(--text-muted);">'
            'No matching skills found. Consider aligning your terminology with the job description.'
            '</div>'
        )

    # Missing with individual reasons
    if missing:
        miss_html = "".join(
            f'<div class="rm-missing-item">'
            f'<span class="rm-missing-name">{s}</span>'
            f'<span class="rm-missing-note">{missing_reasons.get(s, "Not found in your resume text.")}</span>'
            f'</div>'
            for s in sorted(missing)
        )
    else:
        miss_html = (
            '<div style="font-size:0.85rem;color:var(--match-green);">'
            'All required skills from the job description were found in your resume.'
            '</div>'
        )

    st.markdown(
        '<div class="rm-two-col">'
        '<div class="rm-input-panel">'
        '<div class="rm-input-eyebrow">Matching Strengths (' + str(len(matched)) + ')</div>'
        f'{matched_html}'
        '</div>'
        '<div class="rm-input-panel">'
        '<div class="rm-input-eyebrow">Areas to Improve (' + str(len(missing)) + ')</div>'
        f'{miss_html}'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Method: collapsible "How this score is calculated"
# ---------------------------------------------------------------------------


def render_method() -> None:
    """Render the collapsible explanation of the scoring formula."""
    st.markdown(
        '<details class="rm-method">'
        '<summary>'
        'How this score is calculated'
        '<svg class="rm-method-chevron" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">'
        '<polyline points="6 9 12 15 18 9"></polyline>'
        '</svg>'
        '</summary>'
        '<div class="rm-method-body">'
        '<p style="font-size:0.85rem;color:var(--text-muted);margin:0 0 0.75rem 0;">'
        'The overall match score combines three weighted signals, each computed locally without external API calls.'
        '</p>'
        '<div class="rm-formula">'
        'Final Score = (0.40 x Skills Match) + (0.40 x Content Alignment) + (0.20 x Experience)'
        '</div>'
        '<div class="rm-legend">'
        '<div><b>Skills Match</b> -- fuzzy overlap between resume skills and JD skills (rapidfuzz threshold 90). Uses word-boundary matching so single-letter skills like "C" match only as whole words.</div>'
        '<div><b>Content Alignment</b> -- semantic similarity of resume sections to each JD requirement (MiniLM embeddings, cosine similarity). Requirements identified by requirement keywords (skills, experience, tools, etc.), filtering out company boilerplate.</div>'
        '<div><b>Experience</b> -- resume years vs. required years; full credit at or above requirement, linear decay below.</div>'
        '</div>'
        '</div>'
        '</details>',
        unsafe_allow_html=True,
    )