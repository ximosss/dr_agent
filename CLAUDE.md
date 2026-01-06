# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

dr-agent is a deep research agent that performs multi-source research by searching the web and academic papers, then summarizing findings. It uses the OpenAI Agents SDK with a self-hosted LLM backend (vLLM with Qwen3-8B).

## Commands

```bash
# Run the main agent
python agent.py

# Activate virtual environment
source .venv/bin/activate
```

## Environment Variables

Required in `.env`:
- `MODEL_NAME_AT_ENDPOINT` - Model name at the vLLM endpoint
- `BASE_KEY` - API key for the LLM endpoint
- `BASE_URL` - Base URL for the vLLM endpoint

The agent uses HTTP/HTTPS proxy on `localhost:8081`.

## Architecture

### Entry Point
- `agent.py` - Main agent loop with human-in-the-loop interaction and agent execution via OpenAI Agents SDK (`agents` package)

### Tools (`tools.py`)
Four function tools exposed to the agent:
1. **web_search** - Web search via DuckDuckGo, crawls pages with crawl4ai, extracts main content with trafilatura/readability
2. **paper_search** - Academic paper search with two modes:
   - `precise`: DDG search + Semantic Scholar enrichment for specific papers
   - `broad`: Direct Semantic Scholar API for topic searches
3. **local_docs_lookup** - Reads local files/directories, converts to markdown, and summarizes
4. **summarize_sources** - Generates structured JSON summaries (WebTemplate/PaperTemplate) from fetched documents

### Pipelines (`utils/`)
- `web_seach_pipeline.py` - DuckDuckGo search, URL canonicalization, concurrent crawling with crawl4ai, content extraction with trafilatura/readability fallback
- `paper_search_pipeline.py` - Paper discovery via DDG/Semantic Scholar, PDF download, metadata enrichment, deduplication
- `convert_to_md.py` - Converts PDFs (via `marker`) and text files to markdown
- `helpers.py` - Utility for stripping `<think>` blocks from model output

### Prompts (`prompt.py`)
Contains system prompts and JSON schema instructions for structured output:
- `LOCAL_FILES_SUMMARY_PROMPT` - For summarizing local project context
- `WEB_TEMPLATE_INSTRUCTIONS` - JSON schema for web page summaries
- `PAPER_TEMPLATE_INSTRUCTIONS` - JSON schema for paper summaries

## Key Dependencies

- `agents` - OpenAI Agents SDK for agent orchestration
- `litellm` - LLM gateway (using hosted_vllm provider)
- `crawl4ai` - Async web crawler
- `trafilatura` / `readability-lxml` - Content extraction
- `ddgs` - DuckDuckGo search
- `marker` - PDF to markdown conversion
- `httpx` - Async HTTP client for paper fetching
- `weave` - Observability/tracing

## Data Models

The codebase uses Pydantic models for structured data:
- `WebDocument` / `WebSearchResult` - Web search outputs
- `PaperDocument` / `PaperSearchResult` - Paper search outputs
- `WebTemplate` / `PaperTemplate` - Structured summaries with overview, main points, evidence, limitations
- `SummarizedSource` - Unified wrapper for both source types
