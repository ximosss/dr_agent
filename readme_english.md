## English

DR-Agent is a lightweight research agent built on the OpenAI Agents SDK for multi-step information gathering and answer synthesis. Instead of sending a user question directly to a model and returning a single-shot response, it breaks the process into explicit stages: clarify the user intent, produce a search plan, query web, paper, and local-document sources when needed, and finally synthesize a structured answer grounded in collected evidence.

This repository covers both inference-time agent orchestration and the training-time data loop behind it. On the runtime side, the focus is a controllable research workflow. On the training side, the focus is converting real search trajectories into supervision data that can be reused for SFT. In practice, that makes the repo useful both as a runnable deep-research agent and as an experimental scaffold for tool-using or retrieval-oriented model training.

Key features:

- Multi-stage reasoning and tool orchestration with the OpenAI Agents SDK.
- Support for web search, academic paper retrieval, and local document lookup.
- Markdown reports for search and interaction traces to simplify inspection.
- A built-in training pipeline covering teacher sampling, Weave export, augmentation, ShareGPT conversion, LoRA SFT, merge, and evaluation.

## Installation

```bash
uv venv
source .venv/bin/activate
uv sync
```

## Environment

Create `.env` from `.env.example`, then fill in the values you need:

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

- `MODEL_NAME_AT_ENDPOINT`, `BASE_KEY`, and `BASE_URL` are required for the inference backend.
- `TAVILY_API_KEY` is the core configuration for web search and is usually the minimum required search key.
- `S2_API_KEY` is strongly recommended for more stable Semantic Scholar access.
- `WEAVE_PROJECT` and `WANDB_API_KEY` are mainly used for trajectory logging and experiment tracking during training.

## Run

Interactive mode:

```bash
uv run run_agent.py
```

Non-interactive search:

```bash
uv run run_agent.py --search "your question"
```

Benchmark evaluation (SimpleQA / GAIA / Frames):

```bash
uv run run_agent.py --eval -b simpleqa -n 50
```

Search and interaction reports are written to `results/` as Markdown instead of being printed to the terminal. Evaluation outputs are written to `eval_outputs/`, and logs go to `logs/`.

## Training

The training entrypoints live under `training/scripts/`. The overall pipeline is: collect trajectories, clean/augment data, run SFT, merge for deployment, then evaluate.

```bash
# 1. Collect benchmark training trajectories with the teacher model
bash training/scripts/run_teacher.sh

# 2. Download Weave traces, extract trajectories, run augmentation,
#    build ShareGPT SFT data, and launch LLaMA-Factory training
bash training/scripts/run_training.sh

# 3. Merge the LoRA adapter and launch a local vLLM server
bash training/scripts/run_merge.sh

# 4. Evaluate the merged model on the held-out benchmark test splits
bash training/scripts/run_eval.sh
```

Key techniques used in training:

- Trajectory collection: the teacher model runs the full agent workflow so training data includes real tool-use traces instead of only final answers.
- Experiment tracking: Weave / W&B are used to store calls and traces, which are later exported into local JSONL files.
- Data extraction and construction: Weave traces are split into intent clarification, search planning, search execution, and answer-generation samples.
- Data augmentation: `training/augment_data.py` uses the teacher model to synthesize additional intent, answer, and observation-reuse examples while reusing existing tool observations as much as possible.
- SFT format conversion: the processed dataset is converted into ShareGPT format so it can be consumed directly by LLaMA-Factory.
- Training method: supervised fine-tuning is run with LLaMA-Factory, using LoRA / PEFT for parameter-efficient training; the dependencies also include common training components such as `bitsandbytes`, `accelerate`, and `transformers`.
- Deployment and evaluation: the trained LoRA adapter is merged into standalone model weights, served locally with vLLM, and evaluated on benchmarks such as SimpleQA, GAIA, and Frames.

Additional notes:

- `run_training.sh` enables augmentation by default and uses the teacher endpoint at `http://localhost:8001/v1`. Override it with `AUGMENT_BASE_URL`, `AUGMENT_MODEL_NAME`, `AUGMENT_NUM_INTENT`, `AUGMENT_NUM_ANSWER`, and `AUGMENT_NUM_REUSE` when needed.
- `run_eval.sh` targets the merged model at `http://localhost:8000/v1` and lets you adjust evaluation sizes with `FRAMES_EVAL_N`, `SIMPLEQA_EVAL_N`, and `GAIA_EVAL_N`.

## Tools

- `web_search`: Tavily-backed web candidate retrieval.
- `fetch_webpage`: Fetch a webpage and extract readable content.
- `paper_search`: Academic paper search with multi-source fallback.
- `local_docs_lookup`: Local file and directory lookup.