# DR-Agent

A deep research agent that performs multi-source research by searching the web and academic papers, then summarizing findings. Built with OpenAI Agents SDK and powered by a self-hosted LLM backend (vLLM with Qwen3-8B).

## Features

- **Web Search** - Search the web via DuckDuckGo, crawl pages with crawl4ai, extract main content with trafilatura/readability
- **Paper Search** - Academic paper search with two modes:
  - `precise`: DDG search + Semantic Scholar enrichment for specific papers
  - `broad`: Direct Semantic Scholar API for topic searches
- **Local Docs Lookup** - Read local files/directories, convert to markdown, and summarize
- **Source Summarization** - Generate structured JSON summaries from fetched documents
- **Human-in-the-Loop** - Interactive agent loop with user feedback integration

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd dr-agent

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# or with uv
uv sync
```

## Configuration

Create a `.env` file with the following variables:

```env
MODEL_NAME_AT_ENDPOINT=<model-name>
BASE_KEY=<api-key>
BASE_URL=<vllm-endpoint-url>
```

> Note: The agent uses HTTP/HTTPS proxy on `localhost:8081`.

## Usage

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the agent
python agent.py
```

## Architecture

```
dr-agent/
├── agent.py              # Main agent loop with human-in-the-loop interaction
├── tools.py              # Four function tools exposed to the agent
├── prompt.py             # System prompts and JSON schema instructions
└── utils/
    ├── web_seach_pipeline.py    # Web search, crawling, content extraction
    ├── paper_search_pipeline.py # Paper discovery, PDF download, metadata
    ├── convert_to_md.py         # PDF/text to markdown conversion
    └── helpers.py               # Utility functions
```

### Tools

| Tool | Description |
|------|-------------|
| `web_search` | Web search via DuckDuckGo with content extraction |
| `paper_search` | Academic paper search via DDG/Semantic Scholar |
| `local_docs_lookup` | Local file reading and markdown conversion |
| `summarize_sources` | Structured JSON summary generation |

### Data Models

The codebase uses Pydantic models for structured data:

- `WebDocument` / `WebSearchResult` - Web search outputs
- `PaperDocument` / `PaperSearchResult` - Paper search outputs
- `WebTemplate` / `PaperTemplate` - Structured summaries with overview, main points, evidence, limitations
- `SummarizedSource` - Unified wrapper for both source types

## Key Dependencies

- `agents` - OpenAI Agents SDK for agent orchestration
- `litellm` - LLM gateway (using hosted_vllm provider)
- `crawl4ai` - Async web crawler
- `trafilatura` / `readability-lxml` - Content extraction
- `ddgs` - DuckDuckGo search
- `marker` - PDF to markdown conversion
- `httpx` - Async HTTP client
- `weave` - Observability/tracing

## Requirements

- Python >= 3.11
- vLLM endpoint with Qwen3-8B (or compatible model)

## License

MIT
