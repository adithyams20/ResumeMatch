# 🎯 ResumeMatch

An **AI-powered Resume vs Job Description matcher** that uses NLP, vector embeddings, RAG,
and an optional LLM to give you a transparent, deterministic match score — with honest,
actionable improvement suggestions.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                          app.py  (Streamlit UI)                        │
│   Sidebar: provider, weights       Main: upload, paste, Analyze btn    │
└────────────────────┬──────────────────────────────────────┬────────────┘
                     │  resume text + JD text                │ MatchReport
                     ▼                                        ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    matcher/report.py  (Pipeline orchestrator)          │
│                                                                        │
│  Stage 1 ─ parsers/resume_parser.py                                    │
│             PDF (pdfplumber) / DOCX (python-docx) / TXT                │
│                                                                        │
│  Stage 2 ─ nlp/                                                        │
│             preprocess.py    clean text, split sentences, extract YOE  │
│             section_splitter.py  experience│skills│education│projects   │
│             skill_extractor.py   seed list + spaCy noun chunks +        │
│                                  rapidfuzz fuzzy match (threshold 90)   │
│                                                                        │
│  Stage 3 ─ rag/                                                        │
│             indexer.py    chunk (≤150 words) → embed (MiniLM) → Chroma │
│             retriever.py  per JD requirement → top-3 resume chunks     │
│                                                                        │
│  Stage 4 ─ matcher/scorer.py  (deterministic)                          │
│             skill_match + semantic_match + experience_match → final    │
│                                                                        │
│  Stage 5 ─ llm/  (optional, requires API key)                          │
│             client.py    OpenAI or Anthropic wrapper (retries+timeout)  │
│             prompts.py   → strict JSON {gaps, suggestions, keywords}    │
│             Validated by Pydantic LLMAnalysis model                    │
└────────────────────────────────────────────────────────────────────────┘
         │                                  │
    sentence-transformers            ChromaDB (local, persistent)
    (all-MiniLM-L6-v2)              .chromadb/
```

---

## Score Formula

$$\text{final\_score} = 0.4 \times \text{skill\_match} + 0.4 \times \text{semantic\_match} + 0.2 \times \text{experience\_match}$$

| Component | How it's computed |
|-----------|------------------|
| **skill_match** | `matched_JD_skills / total_JD_skills` (exact + fuzzy ≥ 90 via rapidfuzz) |
| **semantic_match** | Mean of best cosine similarity per JD requirement (from ChromaDB retrieval) |
| **experience_match** | `1.0` if `resume_years ≥ required_years`, else `resume_years / required_years` |
| **final_score** | Weighted sum × 100 (displayed as 0–100) |

Weights are configurable via sliders in the sidebar (auto-normalised so they don't need to sum to 1). Defaults are stored in `config.py`.

---

## Project Structure

```
resumematch/
├── app.py                      # Streamlit UI (pure wiring, no business logic)
├── config.py                   # All knobs: weights, model names, paths, thresholds
├── .env.example                # Environment variable template
├── requirements.txt
├── README.md
├── data/
│   ├── skills.txt              # ~300 tech skill seeds
│   └── samples/
│       ├── sample_resume.txt   # Example resume for testing
│       └── sample_jd.txt       # Example job description for testing
├── parsers/
│   └── resume_parser.py        # PDF / DOCX / TXT → clean text
├── nlp/
│   ├── preprocess.py           # Cleaning, sentence split, YOE extraction
│   ├── skill_extractor.py      # Skill extraction & fuzzy matching
│   └── section_splitter.py     # Heuristic section detection
├── rag/
│   ├── indexer.py              # Chunk → embed → ChromaDB
│   └── retriever.py            # Per-requirement top-k retrieval
├── llm/
│   ├── client.py               # Provider-agnostic LLM wrapper
│   └── prompts.py              # Prompt templates (strict JSON output)
├── matcher/
│   ├── scorer.py               # Deterministic sub-scores
│   └── report.py               # Pipeline orchestrator, MatchReport
└── tests/
    ├── test_parser.py
    ├── test_skill_extractor.py
    ├── test_section_splitter.py
    └── test_scorer.py
```

---

## Setup

### 1. Clone & create virtual environment

```bash
git clone <repo-url>
cd resumematch
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the spaCy model

```bash
python -m spacy download en_core_web_sm
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your API key(s)
```

`.env` example:
```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

> **The app works without any API key.** The LLM step (gap analysis) is opt-in via the sidebar toggle.

### 5. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Testing the App with Sample Data

1. Start the app (`streamlit run app.py`).
2. Upload `data/samples/sample_resume.txt` as the resume.
3. Copy the contents of `data/samples/sample_jd.txt` into the Job Description text area.
4. Click **Analyze**.

---

## Running Tests

```bash
pytest tests/ -v
```

With coverage:

```bash
pytest tests/ -v --cov=. --cov-report=term-missing
```

Individual test files:

```bash
pytest tests/test_parser.py -v
pytest tests/test_skill_extractor.py -v
pytest tests/test_section_splitter.py -v
pytest tests/test_scorer.py -v
```

> Tests run fully offline — no API key or network access required.

---

## LLM Provider Configuration

| Variable | Default | Options |
|----------|---------|---------|
| `LLM_PROVIDER` | `openai` | `openai`, `anthropic` |
| `OPENAI_API_KEY` | — | Your OpenAI key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Any OpenAI chat model |
| `ANTHROPIC_API_KEY` | — | Your Anthropic key |
| `ANTHROPIC_MODEL` | `claude-3-haiku-20240307` | Any Anthropic messages model |

The provider can also be changed at runtime via the sidebar selector.

---

## Graceful Degradation

If the LLM step fails (missing key, network error, invalid JSON), the app:
- Shows all deterministic results (score, skill table, RAG evidence) unchanged.
- Displays a yellow warning banner explaining the failure.
- **Never crashes** — the `run_llm` toggle is off by default.

---

## Key Design Decisions

- **No business logic in `app.py`** — only UI wiring; all computation is in the pipeline modules.
- **Per-session ChromaDB reset** — avoids stale chunks from a previous resume polluting results.
- **Pydantic validation everywhere** — `MatchReport` for pipeline output, `LLMAnalysis` for LLM responses.
- **Cached models** — spaCy and SentenceTransformer are loaded once with `st.cache_resource`.
- **Deterministic first** — scoring works fully offline without any external API.

