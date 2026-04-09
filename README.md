# DR-Agent

A small deep-research agent built on the OpenAI Agents SDK. It asks the model to clarify intent, create a search plan, execute web/paper/local lookups, and synthesize a final answer.


## Installation

```bash
uv venv
source .venv/bin/activate
uv sync
```

## Environment

Create `.env` from `.env.example` and fill the values you need:

```env
# ── Inference Backend ──────────────────────────────────────────────
MODEL_NAME_AT_ENDPOINT=
BASE_KEY=
BASE_URL=

# ── Search APIs (at least configure TAVILY_API_KEY) ──────────────
TAVILY_API_KEY=
S2_API_KEY=
LOCAL_DOCS_DIR=

# ── Experiment Tracking ───────────────────────────────────────────
WANDB_API_KEY=
WEAVE_PROJECT=

# ── Teacher Model (for data collection) ──────────────────────────
OPENAI_API_KEY=sk-xxxxxxxxxxxx      # Only for collecting training data

# Download Datasets.(Note: You may need to obtain authorization on the dataset page.)
HF_TOKEN=

# Local_docs Path
LOCAL_DOCS_DIR=

```

Notes:

- `MODEL_NAME_AT_ENDPOINT`, `BASE_KEY`, and `BASE_URL` are required for the LLM backend.
- `S2_API_KEY` is strongly recommended if you want stable Semantic Scholar access.

## Run

```bash
uv run agent.py
```

## Structure

```text
dr_agent/
├── agent.py        # Main interactive loop
├── models.py       # SearchObjective / ResearchPlan
├── prompt.py       # Agent prompts
└── tools/          # Active tool implementations
```

## Tools

- `web_search`: Tavily candidate search.
- `fetch_webpage`: Fetch one webpage and extract readable content.
- `paper_search`: Mode-aware academic search with fallback across sources.
- `local_docs_lookup`: Local file and directory lookup.