"""

app.py – ResumeMatch Streamlit UI.



All business logic lives in the pipeline modules; this file is pure UI wiring.

Run with:  streamlit run app.py

"""



from __future__ import annotations



import logging

import sys

from pathlib import Path



import streamlit as st



# Ensure the project root is on sys.path when running via `streamlit run app.py`

sys.path.insert(0, str(Path(__file__).parent))



import config

from matcher.report import MatchReport, build_report

from parsers.resume_parser import parse_resume_bytes



# ---------------------------------------------------------------------------

# Logging

# ---------------------------------------------------------------------------

logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",

)

logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------

# Page config

# ---------------------------------------------------------------------------

st.set_page_config(

    page_title="ResumeMatch",

    page_icon="🎯",

    layout="wide",

    initial_sidebar_state="collapsed",

)



# ---------------------------------------------------------------------------

# Design tokens (mirrors the spec; kept as constants for easy tweaking)

# ---------------------------------------------------------------------------

_BG          = "#F7FAFA"

_CARD        = "#FFFFFF"

_PRIMARY     = "#159B8E"

_PRIMARY_DRK = "#0f7a6f"

_TEXT        = "#203238"

_TEXT2       = "#73858A"

_BORDER      = "#E4ECEB"

_SUCCESS_BG  = "#E3F6EC"

_SUCCESS_FG  = "#1a6640"

_WARN_BG     = "#FFF3D9"

_WARN_FG     = "#7a5c00"

_ERR_BG      = "#FCE9E9"

_ERR_FG      = "#9b1c1c"



# ---------------------------------------------------------------------------

# CSS

# ---------------------------------------------------------------------------



_CSS = f"""

<style>

/* ── Tokens & base ───────────────────────────────────────────────────────── */

:root {{

  --bg:         {_BG};

  --card:       {_CARD};

  --primary:    {_PRIMARY};

  --primary-dk: {_PRIMARY_DRK};

  --text:       {_TEXT};

  --text2:      {_TEXT2};

  --border:     {_BORDER};

}}



html, body,

[data-testid="stAppViewContainer"],

[data-testid="stMain"],

section.main > div {{

  background-color: var(--bg) !important;

  color: var(--text) !important;

}}



/* ── Sidebar ─────────────────────────────────────────────────────────────── */

[data-testid="stSidebar"] {{

  background-color: var(--card) !important;

  border-right: 1px solid var(--border) !important;

}}

[data-testid="stSidebar"] * {{

  color: var(--text) !important;

}}

[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"] {{

  color: var(--primary) !important;

}}



/* ── Header bar ──────────────────────────────────────────────────────────── */

.rm-header {{

  display: flex;

  align-items: center;

  gap: 0.75rem;

  padding: 1.5rem 0 1rem 0;

  border-bottom: 1px solid var(--border);

  margin-bottom: 2rem;

}}

.rm-logo {{

  width: 38px; height: 38px;

  background: var(--primary);

  border-radius: 10px;

  display: flex; align-items: center; justify-content: center;

  color: #fff; font-size: 1.25rem; font-weight: 800; flex-shrink: 0;

}}

.rm-brand {{

  line-height: 1.2;

}}

.rm-brand-name {{

  font-size: 1.2rem; font-weight: 700; color: var(--text);

}}

.rm-brand-tagline {{

  font-size: 0.8rem; color: var(--text2);

}}



/* ── Input section label ─────────────────────────────────────────────────── */

.input-label {{

  font-size: 0.7rem;

  font-weight: 700;

  letter-spacing: 0.08em;

  text-transform: uppercase;

  color: var(--text2);

  margin-bottom: 0.35rem;

}}



/* ── Card ────────────────────────────────────────────────────────────────── */

.rm-card {{

  background: var(--card);

  border: 1px solid var(--border);

  border-radius: 14px;

  padding: 1.5rem 1.6rem;

  box-shadow: 0 1px 5px rgba(0,0,0,0.05);

  margin-bottom: 1.1rem;

}}

.rm-card-title {{

  font-size: 0.7rem;

  font-weight: 700;

  letter-spacing: 0.08em;

  text-transform: uppercase;

  color: var(--text2);

  margin-bottom: 0.8rem;

}}

.rm-card-body {{

  font-size: 0.95rem;

  color: var(--text);

}}



/* ── Score ring ──────────────────────────────────────────────────────────── */

.score-outer {{

  display: flex;

  flex-direction: column;

  align-items: center;

  padding: 1rem 0 0.5rem 0;

}}

.score-label {{

  font-size: 0.72rem;

  font-weight: 600;

  letter-spacing: 0.07em;

  text-transform: uppercase;

  color: var(--text2);

  margin-top: 0.5rem;

}}

.score-interpretation {{

  font-size: 0.82rem;

  font-weight: 600;

  margin-top: 0.2rem;

  padding: 0.2rem 0.7rem;

  border-radius: 999px;

}}



/* ── Metric tile ─────────────────────────────────────────────────────────── */

.metric-tile {{

  background: var(--card);

  border: 1px solid var(--border);

  border-radius: 12px;

  padding: 1.1rem 1rem 1rem 1rem;

  text-align: center;

  box-shadow: 0 1px 4px rgba(0,0,0,0.04);

}}

.metric-tile-label {{

  font-size: 0.68rem;

  font-weight: 700;

  letter-spacing: 0.07em;

  text-transform: uppercase;

  color: var(--text2);

  margin-bottom: 0.3rem;

}}

.metric-tile-value {{

  font-size: 1.75rem;

  font-weight: 800;

  color: var(--primary);

  line-height: 1.1;

}}

.metric-tile-sub {{

  font-size: 0.72rem;

  color: var(--text2);

  margin-top: 0.15rem;

}}



/* ── Summary card with accent left-border ────────────────────────────────── */

.summary-card {{

  background: var(--card);

  border: 1px solid var(--border);

  border-left: 4px solid var(--primary);

  border-radius: 12px;

  padding: 1rem 1.25rem;

  box-shadow: 0 1px 4px rgba(0,0,0,0.04);

  font-size: 0.95rem;

  color: var(--text);

  line-height: 1.6;

  margin: 0;

}}



/* ── Badge pills ─────────────────────────────────────────────────────────── */

.badge-wrap {{

  display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.5rem;

}}

.badge {{

  display: inline-block;

  padding: 0.22rem 0.65rem;

  border-radius: 999px;

  font-size: 0.75rem;

  font-weight: 600;

  line-height: 1.4;

}}

.badge-green  {{ background: {_SUCCESS_BG}; color: {_SUCCESS_FG}; }}

.badge-amber  {{ background: {_WARN_BG};    color: {_WARN_FG};    }}

.badge-red    {{ background: {_ERR_BG};     color: {_ERR_FG};     }}

.badge-teal   {{ background: #d1f4f0;       color: #0b6b63;       }}

.badge-grey   {{ background: #edf2f2;       color: var(--text2);  }}



/* ── Req status rows ─────────────────────────────────────────────────────── */

.req-row {{

  display: flex;

  align-items: flex-start;

  gap: 0.75rem;

  padding: 0.65rem 0;

  border-bottom: 1px solid var(--border);

}}

.req-row:last-child {{ border-bottom: none; }}

.req-dot {{

  width: 9px; height: 9px;

  border-radius: 50%;

  flex-shrink: 0;

  margin-top: 0.35rem;

}}

.req-dot-green  {{ background: {_SUCCESS_FG}; }}

.req-dot-amber  {{ background: #c97b00; }}

.req-dot-red    {{ background: {_ERR_FG}; }}

.req-text       {{ font-size: 0.88rem; color: var(--text); flex: 1; }}

.req-status     {{

  font-size: 0.7rem; font-weight: 600;

  padding: 0.15rem 0.55rem; border-radius: 999px; flex-shrink: 0;

  white-space: nowrap;

}}

.req-status-green {{ background: {_SUCCESS_BG}; color: {_SUCCESS_FG}; }}

.req-status-amber {{ background: {_WARN_BG};    color: {_WARN_FG};    }}

.req-status-red   {{ background: {_ERR_BG};     color: {_ERR_FG};     }}



/* ── Recommendation item ─────────────────────────────────────────────────── */

.rec-item {{

  display: flex; gap: 0.75rem; align-items: flex-start;

  padding: 0.7rem 0;

  border-bottom: 1px solid var(--border);

}}

.rec-item:last-child {{ border-bottom: none; }}

.rec-num {{

  width: 22px; height: 22px; border-radius: 50%;

  background: var(--primary); color: #fff;

  font-size: 0.7rem; font-weight: 800;

  display: flex; align-items: center; justify-content: center;

  flex-shrink: 0; margin-top: 0.1rem;

}}

.rec-text {{ font-size: 0.9rem; color: var(--text); line-height: 1.55; }}



/* ── Section heading ─────────────────────────────────────────────────────── */

.section-head {{

  font-size: 0.7rem;

  font-weight: 700;

  letter-spacing: 0.08em;

  text-transform: uppercase;

  color: var(--text2);

  margin: 1.8rem 0 0.75rem 0;

}}



/* ── Primary button ──────────────────────────────────────────────────────── */

[data-testid="stButton"] button[kind="primary"] {{

  background: var(--primary) !important;

  border: none !important;

  color: #fff !important;

  font-weight: 700 !important;

  font-size: 0.95rem !important;

  padding: 0.6rem 2rem !important;

  border-radius: 10px !important;

  letter-spacing: 0.01em !important;

  transition: background 0.15s ease !important;

}}

[data-testid="stButton"] button[kind="primary"]:hover {{

  background: var(--primary-dk) !important;

}}



/* ── File uploader, textarea ─────────────────────────────────────────────── */

[data-testid="stFileUploader"] {{

  border-radius: 10px !important;

}}

textarea {{

  background: var(--card) !important;

  border-radius: 10px !important;

  color: var(--text) !important;

  border: 1px solid var(--border) !important;

}}

textarea::placeholder {{ color: var(--text2) !important; }}



/* ── Expander ────────────────────────────────────────────────────────────── */

[data-testid="stExpander"] {{

  border: 1px solid var(--border) !important;

  border-radius: 10px !important;

  background: var(--card) !important;

  box-shadow: 0 1px 3px rgba(0,0,0,0.03) !important;

  margin-bottom: 0.4rem !important;

}}

[data-testid="stExpander"] summary {{

  font-size: 0.85rem !important;

  font-weight: 600 !important;

  color: var(--text) !important;

}}



/* ── Alerts ──────────────────────────────────────────────────────────────── */

[data-testid="stAlert"] {{

  border-radius: 10px !important;

  font-size: 0.88rem !important;

}}



/* ── Info box (quiet) ────────────────────────────────────────────────────── */

.quiet-info {{

  background: #f0f8f7;

  border: 1px solid #c6e8e5;

  border-radius: 10px;

  padding: 0.65rem 1rem;

  font-size: 0.83rem;

  color: var(--text2);

  line-height: 1.5;

}}



/* ── Dividers and spacing ────────────────────────────────────────────────── */

hr {{ border-color: var(--border) !important; margin: 1.25rem 0 !important; }}



/* ── Hide Streamlit chrome ───────────────────────────────────────────────── */

#MainMenu, footer {{ visibility: hidden; }}

header[data-testid="stHeader"] {{ background: transparent !important; }}

</style>

"""





# ---------------------------------------------------------------------------

# Cached model loaders

# ---------------------------------------------------------------------------



@st.cache_resource(show_spinner="Loading NLP model…")

def _load_spacy():

    """Load and cache the spaCy language model."""

    from nlp.preprocess import load_spacy_model

    return load_spacy_model(config.SPACY_MODEL)





@st.cache_resource(show_spinner="Loading embedding model…")

def _load_embedding_model():

    """Load and cache the SentenceTransformer embedding model."""

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL)





# ---------------------------------------------------------------------------

# Sidebar settings

# ---------------------------------------------------------------------------



def _render_sidebar() -> dict:

    """Render the settings sidebar and return user-selected settings."""

    with st.sidebar:

        st.markdown("### Settings")



        st.markdown("**AI Analysis**")

        provider = st.selectbox(

            "LLM provider",

            options=["openai", "anthropic"],

            index=0 if config.LLM_PROVIDER == "openai" else 1,

            help="Set LLM_PROVIDER in .env to change the default.",

        )

        run_llm = st.toggle(

            "Enable AI suggestions",

            value=False,

            help="Requires an API key in .env.",

        )



        st.divider()

        st.markdown("**Score Weights**")

        st.caption("Weights are auto-normalised.")

        w_skill = st.slider(

            "Skills weight", 0.0, 1.0, config.WEIGHT_SKILL, 0.05,

            help="Influence of skill match on the overall score.",

        )

        w_semantic = st.slider(

            "Semantic weight", 0.0, 1.0, config.WEIGHT_SEMANTIC, 0.05,

            help="Influence of semantic alignment on the overall score.",

        )

        w_experience = st.slider(

            "Experience weight", 0.0, 1.0, config.WEIGHT_EXPERIENCE, 0.05,

            help="Influence of experience match on the overall score.",

        )



        st.divider()

        st.caption(

            "Scores are computed locally with NLP + embeddings. "

            "The AI step is optional and requires an API key."

        )



    return {

        "provider": provider,

        "run_llm": run_llm,

        "w_skill": w_skill,

        "w_semantic": w_semantic,

        "w_experience": w_experience,

    }





# ---------------------------------------------------------------------------

# UI helpers

# ---------------------------------------------------------------------------



def _badge_list(items: list[str], css_class: str) -> str:

    """Return an HTML string of pill badges for *items*."""

    if not items:

        return ""

    badges = "".join(

        f'<span class="badge {css_class}">{item}</span>'

        for item in sorted(items)

    )

    return f'<div class="badge-wrap">{badges}</div>'





def _score_ring(score: float) -> str:
    """Return an SVG circular progress ring for *score* (0–100)."""
    radius = 64
    circ = 2 * 3.14159265 * radius
    pct = max(0.0, min(100.0, score))
    offset = circ * (1.0 - pct / 100.0)
    if pct >= 70:
        colour = _PRIMARY
    elif pct >= 45:
        colour = "#E6A817"
    else:
        colour = "#D94F4F"
    return (
        f'<svg width="160" height="160" viewBox="0 0 160 160" aria-label="Score {pct:.0f} out of 100">'
        f'<circle cx="80" cy="80" r="{radius}" stroke="{_BORDER}" stroke-width="12" fill="none"/>'
        f'<circle cx="80" cy="80" r="{radius}" stroke="{colour}" stroke-width="12" fill="none"'
        f' stroke-linecap="round" stroke-dasharray="{circ:.2f}" stroke-dashoffset="{offset:.2f}"'
        f' transform="rotate(-90 80 80)"/>'
        f'<text x="80" y="76" text-anchor="middle" font-size="26" font-weight="800" fill="{colour}"'
        f' font-family="system-ui, sans-serif">{pct:.0f}</text>'
        f'<text x="80" y="96" text-anchor="middle" font-size="11" font-weight="500" fill="{_TEXT2}"'
        f' font-family="system-ui, sans-serif">out of 100</text>'
        f'</svg>'
    )





def _interpretation(score: float) -> tuple[str, str, str]:

    """Return (label, background, foreground) for the score interpretation chip."""

    if score >= 75:

        return "Strong match", _SUCCESS_BG, _SUCCESS_FG

    if score >= 55:

        return "Moderate match", _WARN_BG, _WARN_FG

    if score >= 35:

        return "Partial match", _WARN_BG, _WARN_FG

    return "Low match", _ERR_BG, _ERR_FG





def _summary_text(report: MatchReport) -> str:

    """Generate a short, cautious candidate-facing summary from actual results."""

    score = report.final_score

    n_matched = len(report.matched_skills)

    n_missing = len(report.missing_skills)

    exp_ok = (

        report.resume_years >= report.required_years

        if report.required_years

        else None

    )



    parts: list[str] = []



    if n_matched > 0:

        parts.append(

            f"Your resume demonstrates {n_matched} skill"

            f"{'s' if n_matched != 1 else ''} relevant to this role."

        )

    else:

        parts.append("Few explicit skills from the job description were detected in your resume.")



    if n_missing > 0:

        parts.append(

            f"{n_missing} skill{'s' if n_missing != 1 else ''} mentioned in the job description "

            f"{'were' if n_missing != 1 else 'was'} not clearly found — "

            "consider adding them if they genuinely apply to you."

        )



    if exp_ok is True:

        parts.append("Your estimated experience meets or exceeds the stated requirement.")

    elif exp_ok is False:

        parts.append(

            f"The role appears to require around {report.required_years:.0f} years of experience; "

            "your resume suggests less — this estimate may not be fully accurate."

        )



    if score >= 70:

        parts.append("Overall, your profile looks like a strong candidate for this opportunity.")

    elif score >= 45:

        parts.append(

            "There is a reasonable degree of alignment, with some gaps that are worth addressing."

        )

    else:

        parts.append(

            "There are meaningful gaps between your resume and this job description. "

            "The suggestions below may help you close them."

        )



    return " ".join(parts)





def _req_status(similarity: float) -> tuple[str, str, str, str]:

    """Map a similarity score to a candidate-facing status label and dot/badge colours."""

    if similarity >= 0.60:

        return "Demonstrated", "req-dot-green", "req-status-green", "demonstrated"

    if similarity >= 0.35:

        return "Partially demonstrated", "req-dot-amber", "req-status-amber", "partial"

    return "Not clearly demonstrated", "req-dot-red", "req-status-red", "absent"





def _metric_tile(label: str, value: str, sub: str = "") -> str:

    sub_html = f'<div class="metric-tile-sub">{sub}</div>' if sub else ""

    return f"""

<div class="metric-tile">

  <div class="metric-tile-label">{label}</div>

  <div class="metric-tile-value">{value}</div>

  {sub_html}

</div>"""





# ---------------------------------------------------------------------------

# Results rendering

# ---------------------------------------------------------------------------



def _render_results(report: MatchReport) -> None:

    """Render the full analysis results in a polished candidate-facing layout."""



    score = report.final_score

    interp_label, interp_bg, interp_fg = _interpretation(score)



    # ── Results overview ────────────────────────────────────────────────────

    st.markdown('<div class="section-head">Results Overview</div>', unsafe_allow_html=True)



    ring_col, summary_col = st.columns([1, 2], gap="large")



    with ring_col:

        chip_style = (

            f"background:{interp_bg}; color:{interp_fg}; "

            "font-size:0.75rem; font-weight:700; padding:0.25rem 0.8rem; "

            "border-radius:999px; display:inline-block;"

        )

        st.markdown(

            f'<div class="score-outer">'

            f'{_score_ring(score)}'

            f'<div class="score-label">Overall Match Score</div>'

            f'<span style="{chip_style}">{interp_label}</span>'

            f'</div>',

            unsafe_allow_html=True,

        )



    with summary_col:

        st.markdown('<div class="rm-card-title">Summary</div>', unsafe_allow_html=True)

        st.markdown(

            f'<div class="summary-card">{_summary_text(report)}</div>',

            unsafe_allow_html=True,

        )



        st.markdown("<br>", unsafe_allow_html=True)



        mc1, mc2, mc3 = st.columns(3)

        mc1.markdown(

            _metric_tile("Skills Match", f"{report.skill_match * 100:.0f}%"),

            unsafe_allow_html=True,

        )

        mc2.markdown(

            _metric_tile("Content Alignment", f"{report.semantic_match * 100:.0f}%"),

            unsafe_allow_html=True,

        )

        # Experience tile — show years if available, else just the score

        if report.required_years:

            exp_sub = (

                f"{report.resume_years:.1f} yrs vs {report.required_years:.1f} yrs required"

            )

        else:

            exp_sub = f"{report.resume_years:.1f} yrs detected"

        mc3.markdown(

            _metric_tile("Experience", f"{report.experience_match * 100:.0f}%", exp_sub),

            unsafe_allow_html=True,

        )



    # ── Strengths & improvement areas ──────────────────────────────────────

    st.markdown(

        '<div class="section-head">Strengths &amp; Improvement Areas</div>',

        unsafe_allow_html=True,

    )



    str_col, imp_col = st.columns(2, gap="medium")



    with str_col:

        st.markdown(

            '<div class="rm-card">'

            '<div class="rm-card-title">Matching Strengths</div>',

            unsafe_allow_html=True,

        )

        if report.matched_skills:

            st.markdown(

                _badge_list(report.matched_skills, "badge-green"),

                unsafe_allow_html=True,

            )

            total = len(report.jd_skills)

            matched = len(report.matched_skills)

            if total:

                st.markdown(

                    f'<div style="font-size:0.78rem;color:{_TEXT2};margin-top:0.6rem;">'

                    f'{matched} of {total} required skills detected in your resume</div>',

                    unsafe_allow_html=True,

                )

        else:

            st.markdown(

                '<div class="rm-card-body" style="color:#73858A;">No explicitly required '

                "skills from the job description were detected in your resume. "

                "This may be because your resume uses different phrasing — "

                "consider aligning your terminology with the job description.</div>",

                unsafe_allow_html=True,

            )

        st.markdown("</div>", unsafe_allow_html=True)



    with imp_col:

        st.markdown(

            '<div class="rm-card">'

            '<div class="rm-card-title">Areas to Improve</div>',

            unsafe_allow_html=True,

        )

        if report.missing_skills:

            st.markdown(

                _badge_list(report.missing_skills, "badge-amber"),

                unsafe_allow_html=True,

            )

            st.markdown(

                f'<div style="font-size:0.78rem;color:{_TEXT2};margin-top:0.6rem;">'

                "These skills appear in the job description but were not found in "

                "your resume. Add them if they genuinely apply to your background.</div>",

                unsafe_allow_html=True,

            )

        else:

            st.markdown(

                f'<div class="rm-card-body" style="color:{_SUCCESS_FG};">'

                "All required skills from the job description were found in your resume.</div>",

                unsafe_allow_html=True,

            )

        st.markdown("</div>", unsafe_allow_html=True)



    # ── How to improve your resume ──────────────────────────────────────────

    st.markdown(

        '<div class="section-head">How to Improve Your Resume</div>',

        unsafe_allow_html=True,

    )



    if report.llm_warning:

        st.markdown(

            f'<div class="quiet-info">AI suggestions were unavailable: {report.llm_warning}</div>',

            unsafe_allow_html=True,

        )

    elif report.llm_analysis is None:

        st.markdown(

            '<div class="quiet-info">'

            "Personalised AI suggestions are not enabled. "

            "Turn on <strong>AI suggestions</strong> in the sidebar to receive "

            "tailored recommendations based on this job description."

            "</div>",

            unsafe_allow_html=True,

        )

    else:

        analysis = report.llm_analysis

        suggestions: list[str] = list(analysis.suggestions)



        # Surface keyword gaps as actionable recommendations

        if analysis.gaps:

            for gap in analysis.gaps[:3]:

                suggestions.insert(0, gap)



        if suggestions:

            items_html = "".join(

                f'<div class="rec-item">'

                f'<div class="rec-num">{i}</div>'

                f'<div class="rec-text">{s}</div>'

                f'</div>'

                for i, s in enumerate(suggestions[:6], 1)

            )

            st.markdown(

                f'<div class="rm-card">{items_html}</div>',

                unsafe_allow_html=True,

            )

        else:

            st.markdown(

                f'<div class="quiet-info" style="color:{_SUCCESS_FG};">'

                "No significant gaps identified — your resume appears well-aligned with this role."

                "</div>",

                unsafe_allow_html=True,

            )



        # Keywords to add — shown only when they exist

        if analysis.keywords_to_add:

            st.markdown(

                '<div class="rm-card-title" style="margin-top:0.5rem;">Keywords to consider adding</div>',

                unsafe_allow_html=True,

            )

            st.markdown(

                _badge_list(analysis.keywords_to_add, "badge-teal"),

                unsafe_allow_html=True,

            )

            st.markdown(

                f'<div style="font-size:0.75rem;color:{_TEXT2};margin-top:0.4rem;">'

                "Only add keywords that genuinely reflect your experience.</div>",

                unsafe_allow_html=True,

            )



    # ── Job requirements analysis ───────────────────────────────────────────

    st.markdown(

        '<div class="section-head">Job Requirements Analysis</div>',

        unsafe_allow_html=True,

    )

    st.markdown(

        f'<div style="font-size:0.82rem;color:{_TEXT2};margin-bottom:0.75rem;">'

        "How well your resume addresses each key requirement from the job description.</div>",

        unsafe_allow_html=True,

    )



    visible_reqs = [rm for rm in report.requirement_matches if rm.matches][:20]



    if visible_reqs:

        rows_html = ""

        for rm in visible_reqs:

            best = rm.matches[0]

            label, dot_cls, status_cls, _ = _req_status(best.similarity)

            req_short = rm.requirement[:140] + ("…" if len(rm.requirement) > 140 else "")

            rows_html += (

                f'<div class="req-row">'

                f'  <div class="req-dot {dot_cls}"></div>'

                f'  <div class="req-text">{req_short}</div>'

                f'  <span class="req-status {status_cls}">{label}</span>'

                f'</div>'

            )

        st.markdown(

            f'<div class="rm-card">{rows_html}</div>',

            unsafe_allow_html=True,

        )

    else:

        st.markdown(

            '<div class="quiet-info">No requirement-level evidence was available.</div>',

            unsafe_allow_html=True,

        )



    # ── Technical details (collapsed) ───────────────────────────────────────

    with st.expander("Technical Details — for advanced users"):

        st.caption(

            "This section contains raw pipeline output. "

            "It is intended for debugging, not for evaluating your candidacy."

        )



        st.markdown("**Requirement–chunk similarity scores**")

        for rm in report.requirement_matches:

            if not rm.matches:

                continue

            best = rm.matches[0]

            sim_pct = best.similarity * 100

            req_label = rm.requirement[:90] + ("…" if len(rm.requirement) > 90 else "")

            with st.expander(f"{req_label}  ·  {sim_pct:.1f}%"):

                st.markdown(f"**Requirement:** {rm.requirement}")

                st.markdown(f"**Matched section:** `{best.section}`")

                st.markdown(f"**Similarity:** {sim_pct:.2f}%  |  Chunk ID: `{best.chunk_id}`")

                st.info(best.text)

                if len(rm.matches) > 1:

                    st.markdown("**Other retrieved chunks:**")

                    for chunk in rm.matches[1:]:

                        st.write(

                            f"- [{chunk.section}] {chunk.text[:100]}… "

                            f"({chunk.similarity * 100:.1f}%)"

                        )



        st.divider()

        st.markdown("**Detected resume sections**")

        for section_name, section_text in report.sections.items():

            with st.expander(f"{section_name.title()}  ({len(section_text.split())} words)"):

                st.text(section_text[:600] + ("…" if len(section_text) > 600 else ""))



        st.divider()

        st.markdown("**Raw skill extraction**")

        st.markdown(

            f"Resume skills detected: `{', '.join(sorted(report.resume_skills)) or 'none'}`"

        )

        st.markdown(

            f"JD skills detected: `{', '.join(sorted(report.jd_skills)) or 'none'}`"

        )

        st.markdown(

            f"Experience — resume: **{report.resume_years:.1f} yrs** | "

            f"required: **{report.required_years:.1f} yrs**"

        )





# ---------------------------------------------------------------------------

# Main

# ---------------------------------------------------------------------------



def main() -> None:

    """Entry point — renders the full Streamlit app."""

    st.markdown(_CSS, unsafe_allow_html=True)



    settings = _render_sidebar()



    # ── Header ──────────────────────────────────────────────────────────────

    st.markdown(

        '<div class="rm-header">'

        '  <div class="rm-logo">R</div>'

        '  <div class="rm-brand">'

        '    <div class="rm-brand-name">ResumeMatch</div>'

        "    <div class=\"rm-brand-tagline\">Understand your resume's fit for every opportunity.</div>"

        '  </div>'

        '</div>',

        unsafe_allow_html=True,

    )



    # Load models (cached after first load)

    try:

        nlp = _load_spacy()

        embedding_model = _load_embedding_model()

    except OSError as exc:

        st.error(str(exc))

        st.stop()



    # ── Input section ────────────────────────────────────────────────────────

    col_resume, col_jd = st.columns(2, gap="large")



    with col_resume:

        st.markdown('<div class="input-label">Your Resume</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(

            "Upload resume",

            type=["pdf", "docx", "txt"],

            help="Text-based PDFs, Word documents, or plain text. Scanned images are not supported.",

            label_visibility="collapsed",

        )

        if uploaded_file is not None:

            st.success(f"Uploaded: {uploaded_file.name}")



    with col_jd:

        st.markdown('<div class="input-label">Job Description</div>', unsafe_allow_html=True)

        jd_text = st.text_area(

            "Job description",

            height=210,

            placeholder="Paste the full job description here…",

            label_visibility="collapsed",

        )



    # ── Analyse button ───────────────────────────────────────────────────────

    st.markdown("<br>", unsafe_allow_html=True)

    _, btn_col, _ = st.columns([1, 2, 1])

    with btn_col:

        analyze_btn = st.button(

            "Analyze My Match",

            type="primary",

            use_container_width=True,

        )



    if analyze_btn:

        # Validate

        if uploaded_file is None:

            st.error("Please upload your resume before analysing.")

            st.stop()

        if not jd_text.strip():

            st.error("Please paste a job description before analysing.")

            st.stop()



        # Parse resume

        with st.spinner("Reading your resume…"):

            try:

                resume_text = parse_resume_bytes(uploaded_file.read(), uploaded_file.name)

            except ValueError as exc:

                st.error(f"Could not read the resume: {exc}")

                st.stop()

            except Exception as exc:

                st.error(f"Unexpected error while reading the resume: {exc}")

                st.stop()



        # Run pipeline

        with st.spinner("Analysing your resume against the job description…"):

            try:

                report = build_report(

                    resume_text=resume_text,

                    jd_text=jd_text,

                    nlp=nlp,

                    embedding_model=embedding_model,

                    weight_skill=settings["w_skill"],

                    weight_semantic=settings["w_semantic"],

                    weight_experience=settings["w_experience"],

                    run_llm=settings["run_llm"],

                    llm_provider=settings["provider"],

                )

            except Exception as exc:

                logger.exception("Pipeline error")

                st.error(f"Analysis failed: {exc}")

                st.stop()



        _render_results(report)





if __name__ == "__main__":

    main()
