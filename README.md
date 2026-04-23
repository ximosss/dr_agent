# DR-Agent

A small deep-research agent built on the OpenAI Agents SDK. It asks the model to clarify intent, create a search plan, execute web/paper/local lookups, and synthesize a final answer.

![workflow](./imgs/dr_agent.png)


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
uv run run_agent.py
```

Non-interactive search:

```bash
uv run run_agent.py --search "your question"
```

Benchmark evaluation(SimpleQA, GAIA, Frames):

```bash
uv run run_agent.py --eval -b simpleqa -n 50
```

Search/interact reports are written to `results/` as Markdown and are not printed to the terminal. Evaluation outputs go to `eval_outputs/`, and logs to `logs/`.


## Training

Training is driven from the scripts under `training/scripts/`.

```bash
# 1. Collect benchmark-train trajectories with the teacher model.
bash training/scripts/run_teacher.sh

# 2. Download Weave traces, extract trajectories, run augmentation,
#    build ShareGPT SFT data, and launch LLaMA-Factory training.
bash training/scripts/run_training.sh

# 3. Merge the LoRA adapter and launch a local vLLM server.
bash training/scripts/run_merge.sh

# 4. Evaluate the merged model on the reserved benchmark test splits.
bash training/scripts/run_eval.sh
```

Notes:

- `run_training.sh` now performs `uv sync --group training --inexact` internally; `training/scripts/setup_env.sh` is no longer needed.
- Augmentation is enabled by default inside `run_training.sh` and uses the teacher endpoint at `http://localhost:8001/v1`. Override with `AUGMENT_BASE_URL`, `AUGMENT_MODEL_NAME`, `AUGMENT_NUM_INTENT`, `AUGMENT_NUM_ANSWER`, and `AUGMENT_NUM_REUSE` when needed.
- `run_eval.sh` targets the merged model on `http://localhost:8000/v1` and lets you adjust benchmark sizes with `FRAMES_EVAL_N`, `SIMPLEQA_EVAL_N`, and `GAIA_EVAL_N`.


## Tools

- `web_search`: Tavily candidate search.
- `fetch_webpage`: Fetch one webpage and extract readable content.
- `paper_search`: Mode-aware academic search with fallback across sources.
- `local_docs_lookup`: Local file and directory lookup.
