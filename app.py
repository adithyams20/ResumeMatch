"""
app.py - ResumeMatch Streamlit UI.

All business logic lives in the pipeline modules; this file is pure UI wiring.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import html
import logging
import sys
from pathlib import Path

import streamlit as st

# Ensure the project root is on sys.path when running via `streamlit run app.py`
sys.path.insert(0, str(Path(__file__).parent))

import config
import ui
from llm_suggestions import generate_all_suggestions
from matcher.report import MatchReport, build_report
from nlp.skill_extractor import match_skills
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

# Load CSS before any rendering
ui.load_styles()

# ---------------------------------------------------------------------------
# Missing skill reasons
# ---------------------------------------------------------------------------

_MISSING_REASONS = {
    "Python": "Listed as a required skill in the job description.",
    "JavaScript": "Listed as a required skill in the job description.",
    "TypeScript": "Listed as a required skill in the job description.",
    "Java": "Listed as a required skill in the job description.",
    "C++": "Listed as a required skill in the job description.",
    "C": "Listed as a required skill in the job description.",
    "C#": "Listed as a required skill in the job description.",
    "Go": "Listed as a required skill in the job description.",
    "Rust": "Listed as a required skill in the job description.",
    "Swift": "Listed as a required skill in the job description.",
    "Kotlin": "Listed as a required skill in the job description.",
    "Ruby": "Listed as a required skill in the job description.",
    "PHP": "Listed as a required skill in the job description.",
    "Scala": "Listed as a required skill in the job description.",
    "R": "Listed as a required skill in the job description.",
    "MATLAB": "Listed as a required skill in the job description.",
    "Perl": "Listed as a required skill in the job description.",
    "Haskell": "Listed as a required skill in the job description.",
    "Elixir": "Listed as a required skill in the job description.",
    "Clojure": "Listed as a required skill in the job description.",
    "Lua": "Listed as a required skill in the job description.",
    "Dart": "Listed as a required skill in the job description.",
    "Julia": "Listed as a required skill in the job description.",
    "F#": "Listed as a required skill in the job description.",
    "Groovy": "Listed as a required skill in the job description.",
    "Shell": "Listed as a required skill in the job description.",
    "Bash": "Listed as a required skill in the job description.",
    "PowerShell": "Listed as a required skill in the job description.",
    "SQL": "Listed as a required skill in the job description.",
    "HTML": "Listed as a required skill in the job description.",
    "CSS": "Listed as a required skill in the job description.",
    "React": "Listed as a required skill in the job description.",
    "Angular": "Listed as a required skill in the job description.",
    "Vue": "Listed as a required skill in the job description.",
    "Next.js": "Listed as a required skill in the job description.",
    "Nuxt.js": "Listed as a required skill in the job description.",
    "Svelte": "Listed as a required skill in the job description.",
    "Remix": "Listed as a required skill in the job description.",
    "Gatsby": "Listed as a required skill in the job description.",
    "Redux": "Listed as a required skill in the job description.",
    "MobX": "Listed as a required skill in the job description.",
    "GraphQL": "Listed as a required skill in the job description.",
    "REST": "Listed as a required skill in the job description.",
    "gRPC": "Listed as a required skill in the job description.",
    "WebSocket": "Listed as a required skill in the job description.",
    "Node.js": "Listed as a required skill in the job description.",
    "Express": "Listed as a required skill in the job description.",
    "FastAPI": "Listed as a required skill in the job description.",
    "Django": "Listed as a required skill in the job description.",
    "Flask": "Listed as a required skill in the job description.",
    "Docker": "Listed as a required skill in the job description.",
    "Kubernetes": "Listed as a required skill in the job description.",
    "Terraform": "Listed as a required skill in the job description.",
    "AWS": "Listed as a required skill in the job description.",
    "GCP": "Listed as a required skill in the job description.",
    "Azure": "Listed as a required skill in the job description.",
    "CI/CD": "Listed as a required skill in the job description.",
    "scikit-learn": "Listed as a required skill in the job description.",
    "Hugging Face": "Listed as a required skill in the job description.",
}

# ---------------------------------------------------------------------------
# Cached model loaders
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading NLP model...")
def _load_spacy():
    """Load and cache the spaCy language model."""
    from nlp.preprocess import load_spacy_model
    return load_spacy_model(config.SPACY_MODEL)


@st.cache_resource(show_spinner="Loading embedding model...")
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


def _get_missing_reason(skill: str) -> str:
    """Return a reason for why a skill is missing."""
    return _MISSING_REASONS.get(skill, "Not found in your resume text.")


def _req_status(similarity: float) -> tuple[str, str]:
    """Return (label, css_class) for requirement status."""
    if similarity >= 0.60:
        return "Found in resume", "found"
    if similarity >= 0.35:
        return "Partly found", "partly"
    return "Not found", "not"


def _badge_list(items: list[str], css_class: str) -> str:
    """Return an HTML string of pill badges for *items*."""
    if not items:
        return ""
    badges = "".join(
        f'<span class="badge {css_class}">{html.escape(item)}</span>'
        for item in sorted(items)
    )
    return f'<div class="badge-wrap">{badges}</div>'


# ---------------------------------------------------------------------------
# Results rendering
# ---------------------------------------------------------------------------


def _render_results(report: MatchReport) -> None:
    """Render the full analysis results using the new UI layer."""

    # Build missing skill reasons
    missing_reasons = {s: _get_missing_reason(s) for s in report.missing_skills}

    # New UI components: score ring, breakdown, skills, method
    ui.render_score(report)
    ui.render_breakdown(report)
    ui.render_skills(report, missing_reasons)
    ui.render_method()

    # -- How to improve your resume (AI suggestions) --------------------------
    st.markdown(
        '<div class="section-head">How to Improve Your Resume</div>',
        unsafe_allow_html=True,
    )

    if not report.run_llm:
        st.markdown(
            '<div class="quiet-info">'
            "Personalised AI suggestions are not enabled. "
            "Turn on <strong>AI suggestions</strong> in the sidebar to receive "
            "tailored recommendations based on this job description."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        with st.spinner("Generating AI suggestions (bullet rewrites, interview questions, cover letter...)..."):
            bundle = generate_all_suggestions(
                resume_text=report.resume_text,
                jd_text=report.jd_text,
                matched_skills=report.matched_skills,
                missing_skills=report.missing_skills,
                provider=report.llm_provider,
            )

        if bundle.overall_warning:
            st.markdown(
                f'<div class="quiet-info" style="color:#7a5c00; background:#FFF3D9;">'
                f'AI suggestions were unavailable: {html.escape(bundle.overall_warning)}'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            # a. Bullet rewrites expander
            with st.expander("Targeted Bullet Rewrites", expanded=True):
                if bundle.bullet_rewrites_error:
                    st.warning(f"Could not generate bullet rewrites: {bundle.bullet_rewrites_error}")
                elif bundle.bullet_rewrites and bundle.bullet_rewrites.rewrites:
                    st.caption("Rewritten to match job terminology. Placeholders like **[X]** indicate metrics to fill with your real numbers.")
                    for item in bundle.bullet_rewrites.rewrites:
                        st.markdown(
                            f'<div class="rm-card" style="padding: 1rem 1.25rem; margin-bottom: 0.75rem;">'
                            f'<div style="font-size:0.75rem; font-weight:700; color:var(--text-muted); text-transform:uppercase; margin-bottom:0.25rem;">Original Bullet</div>'
                            f'<div style="font-size:0.9rem; color:var(--text); margin-bottom:0.6rem;">{html.escape(item.original)}</div>'
                            f'<div style="font-size:0.75rem; font-weight:700; color:var(--accent); text-transform:uppercase; margin-bottom:0.25rem;">Tailored Rewrite</div>'
                            f'<div style="font-size:0.95rem; font-weight:600; color:var(--text); margin-bottom:0.4rem;">{html.escape(item.rewritten)}</div>'
                            f'<div style="font-size:0.8rem; color:var(--text-muted); font-style:italic;">{html.escape(item.improvement_reason)}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("No bullet rewrites could be generated for this resume.")

            # b. Missing-skill explanations expander
            with st.expander("Missing Skill Explanations & Relevance", expanded=False):
                if bundle.missing_skills_error:
                    st.warning(f"Could not generate missing skill explanations: {bundle.missing_skills_error}")
                elif bundle.missing_skills and bundle.missing_skills.explanations:
                    for item in bundle.missing_skills.explanations:
                        st.markdown(
                            f'<div class="rm-card" style="padding: 0.9rem 1.25rem; margin-bottom: 0.6rem;">'
                            f'<div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.4rem;">'
                            f'  <span class="badge badge-amber">{html.escape(item.skill)}</span>'
                            f'</div>'
                            f'<div style="font-size:0.88rem; color:var(--text); margin-bottom:0.35rem;"><strong>Why it matters:</strong> {html.escape(item.why_it_matters)}</div>'
                            f'<div style="font-size:0.84rem; color:var(--text-muted);"><strong>Related experience:</strong> {html.escape(item.related_experience)}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("No missing skills to explain.")

            # c. Tailored summary expander
            with st.expander("Tailored Professional Summary", expanded=False):
                if bundle.summary_error:
                    st.warning(f"Could not generate tailored summary: {bundle.summary_error}")
                elif bundle.summary and bundle.summary.summary:
                    st.markdown(
                        f'<div class="summary-card" style="font-size:0.95rem; line-height:1.6; margin-bottom:0.5rem;">'
                        f'{html.escape(bundle.summary.summary)}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption("2-3 line summary grounded only in facts from your resume.")
                else:
                    st.info("No tailored summary available.")

            # d. Gap-closing plan expander
            with st.expander("1-2 Week Gap-Closing Action Plan", expanded=False):
                if bundle.gap_plan_error:
                    st.warning(f"Could not generate gap-closing plan: {bundle.gap_plan_error}")
                elif bundle.gap_plan and bundle.gap_plan.steps:
                    st.caption("Focused small projects or learning steps to address key gaps.")
                    for idx, step in enumerate(bundle.gap_plan.steps, 1):
                        st.markdown(
                            f'<div class="rm-card" style="padding: 0.9rem 1.25rem; margin-bottom: 0.6rem;">'
                            f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.3rem;">'
                            f'  <strong style="font-size:0.92rem; color:var(--text);">{idx}. {html.escape(step.gap)}</strong>'
                            f'  <span class="badge badge-teal">{html.escape(step.timeframe)}</span>'
                            f'</div>'
                            f'<div style="font-size:0.88rem; color:var(--text); line-height:1.5;">{html.escape(step.suggested_step)}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("No gap-closing steps identified.")

            # e. Interview questions expander
            with st.expander("Targeted Interview Questions (5 Tech + 3 Behavioral)", expanded=False):
                if bundle.interview_questions_error:
                    st.warning(f"Could not generate interview questions: {bundle.interview_questions_error}")
                elif bundle.interview_questions:
                    iq = bundle.interview_questions
                    if iq.technical_questions:
                        st.markdown('<div class="rm-card-title" style="margin-top:0.3rem;">5 Technical Questions</div>', unsafe_allow_html=True)
                        for idx, q in enumerate(iq.technical_questions[:5], 1):
                            st.markdown(
                                f'<div class="rec-item">'
                                f'<div class="rec-num">{idx}</div>'
                                f'<div class="rec-text"><strong>{html.escape(q)}</strong></div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                    if iq.behavioral_questions:
                        st.markdown('<div class="rm-card-title" style="margin-top:1rem;">3 Behavioral Questions</div>', unsafe_allow_html=True)
                        for idx, q in enumerate(iq.behavioral_questions[:3], 1):
                            st.markdown(
                                f'<div class="rec-item">'
                                f'<div class="rec-num" style="background:#4b5563;">{idx}</div>'
                                f'<div class="rec-text"><strong>{html.escape(q)}</strong></div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                else:
                    st.info("No interview questions available.")

            # f. Cover letter draft expander
            with st.expander("Tailored Cover Letter Draft (<250 words)", expanded=False):
                if bundle.cover_letter_error:
                    st.warning(f"Could not generate cover letter draft: {bundle.cover_letter_error}")
                elif bundle.cover_letter and bundle.cover_letter.draft:
                    cl = bundle.cover_letter
                    st.markdown(
                        f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">'
                        f'  <span class="badge badge-amber">{html.escape(cl.notice)}</span>'
                        f'  <span style="font-size:0.75rem; color:var(--text-muted); font-weight:600;">{cl.word_count} words</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="rm-card" style="padding: 1.25rem 1.5rem; white-space: pre-wrap; font-size:0.9rem; line-height:1.6; color:var(--text);">'
                        f'{html.escape(cl.draft)}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("No cover letter draft available.")

    # -- Job requirements analysis (new UI) ------------------------------------
    st.markdown(
        '<div class="section-head">Job Requirements Analysis</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div style="font-size:0.82rem;color:var(--text-muted);margin-bottom:0.75rem;">'
        f'Evaluated {report.jd_requirement_count} requirements from the job description.'
        f' {report.jd_skipped_count} sentences were skipped (company information or too short).'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Get top 8 most relevant requirements by similarity
    sorted_matches = sorted(
        [rm for rm in report.requirement_matches if rm.matches],
        key=lambda rm: rm.best_similarity,
        reverse=True
    )[:8]

    show_all = len([rm for rm in report.requirement_matches if rm.matches]) > 8

    if sorted_matches:
        for rm in sorted_matches:
            best = rm.matches[0]
            sim = best.similarity
            label, tag_cls = _req_status(sim)
            evidence = best.text[:120] + "..." if len(best.text) > 120 else best.text
            req_display = rm.requirement[:120] + "..." if len(rm.requirement) > 120 else rm.requirement
            st.markdown(
                f'<div class="rm-req-row">'
                f'<div class="rm-req-head">'
                f'<div class="rm-req-text">{html.escape(req_display)}</div>'
                f'<span class="rm-req-tag {tag_cls}">{html.escape(label)}</span>'
                f'</div>'
                f'<div class="rm-req-evidence">{html.escape(evidence)}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        if show_all:
            other_matches = [rm for rm in report.requirement_matches if rm.matches and rm not in sorted_matches]
            with st.expander("Show all evaluated lines"):
                for rm in other_matches:
                    best = rm.matches[0]
                    sim = best.similarity
                    label, tag_cls = _req_status(sim)
                    evidence = best.text[:120] + "..." if len(best.text) > 120 else best.text
                    req_display = rm.requirement[:120] + "..." if len(rm.requirement) > 120 else rm.requirement
                    st.markdown(
                        f'<div class="rm-req-row">'
                        f'<div class="rm-req-head">'
                        f'<div class="rm-req-text">{html.escape(req_display)}</div>'
                        f'<span class="rm-req-tag {tag_cls}">{html.escape(label)}</span>'
                        f'</div>'
                        f'<div class="rm-req-evidence">{html.escape(evidence)}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
    else:
        st.markdown(
            '<div class="quiet-info">No requirement-level evidence was available.</div>',
            unsafe_allow_html=True,
        )

    # -- Technical details (collapsed) -----------------------------------------
    with st.expander("Technical Details -- for advanced users"):
        st.caption(
            "This section contains raw pipeline output. "
            "It is intended for debugging, not for evaluating your candidacy."
        )

        st.markdown("**Requirement-chunk similarity scores**")
        for rm in report.requirement_matches:
            if not rm.matches:
                continue
            best = rm.matches[0]
            sim_pct = best.similarity * 100
            req_label = rm.requirement[:90] + ("..." if len(rm.requirement) > 90 else "")
            with st.expander(f"{req_label} - {sim_pct:.1f}%"):
                st.markdown(f"**Requirement:** {rm.requirement}")
                st.markdown(f"**Matched section:** `{best.section}`")
                st.markdown(f"**Similarity:** {sim_pct:.2f}%  |  Chunk ID: `{best.chunk_id}`")
                st.info(best.text)
                if len(rm.matches) > 1:
                    st.markdown("**Other retrieved chunks:**")
                    for chunk in rm.matches[1:]:
                        st.write(
                            f"- [{chunk.section}] {chunk.text[:100]}... "
                            f"({chunk.similarity * 100:.1f}%)"
                        )

        st.divider()
        st.markdown("**Detected resume sections**")
        for section_name, section_text in report.sections.items():
            with st.expander(f"{section_name.title()}  ({len(section_text.split())} words)"):
                st.text(section_text[:600] + ("..." if len(section_text) > 600 else ""))

        st.divider()
        st.markdown("**Raw skill extraction**")
        st.markdown(
            f"Resume skills detected: `{', '.join(sorted(report.resume_skills)) or 'none'}`"
        )
        st.markdown(
            f"JD skills detected: `{', '.join(sorted(report.jd_skills)) or 'none'}`"
        )
        st.markdown(
            f"Experience -- resume: **{report.resume_years:.1f} yrs** | "
            f"required: **{report.required_years:.1f} yrs**"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point -- renders the full Streamlit app."""
    settings = _render_sidebar()

    # -- Header --------------------------------------------------------------
    ui.render_header()

    # Load models (cached after first load)
    try:
        nlp = _load_spacy()
        embedding_model = _load_embedding_model()
    except OSError as exc:
        st.error(str(exc))
        st.stop()

    # -- Input section -------------------------------------------------------
    uploaded_file, jd_text = ui.render_inputs()

    # Unescape HTML entities for text that will go into analysis (not for display)
    jd_text_unescaped = html.unescape(jd_text) if jd_text else ""

    # Status line
    has_resume = uploaded_file is not None
    has_jd = bool(jd_text_unescaped.strip())
    ui.render_status(has_resume, has_jd)

    # -- Analyse button ------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        analyze_btn = st.button(
            "Analyze My Match",
            type="primary",
            use_container_width=True,
            disabled=not (has_resume and has_jd),
        )

    if analyze_btn:
        # Validate (redundant but safe)
        if not has_resume:
            st.error("Please upload your resume before analysing.")
            st.stop()
        if not has_jd:
            st.error("Please paste a job description before analysing.")
            st.stop()

        # Parse resume
        with st.spinner("Parsing resume..."):
            try:
                resume_bytes = uploaded_file.read()
                resume_text = parse_resume_bytes(resume_bytes, uploaded_file.name)
            except ValueError as exc:
                st.error(f"Could not read the resume: {exc}")
                st.stop()
            except Exception as exc:
                st.error(f"Unexpected error while reading the resume: {exc}")
                st.stop()

        # Unescape HTML entities from resume text for analysis
        resume_text_unescaped = html.unescape(resume_text)

        # Run pipeline
        with st.spinner("Extracting skills..."):
            pass  # skill extraction happens inside build_report

        with st.spinner("Scoring..."):
            try:
                report = build_report(
                    resume_text=resume_text_unescaped,
                    jd_text=jd_text_unescaped,
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