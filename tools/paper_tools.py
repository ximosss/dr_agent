"""Academic paper search: Semantic Scholar, arXiv, OpenAlex."""

from __future__ import annotations

import os
import arxiv

import requests
from agents import function_tool
from dotenv import load_dotenv

from prompt import PAPER_SEARCH_TOOL_PROMPT

from .utils import truncate_text

load_dotenv()

S2_API_KEY = os.getenv("S2_API_KEY", "")


SOURCE_LABELS = {
    "semantic_scholar": "Semantic Scholar",
    "arxiv": "arXiv",
    "openalex": "OpenAlex",
}
DEFAULT_TIMEOUT = 15


def run_search_papers(
    query: str,
    max_results: int = 5,
    source: str = "semantic_scholar",
    fallback_sources: list[str] | None = None,
) -> str:
    """Run paper search against the selected backend with graceful fallback."""
    if fallback_sources is None:
        default_fallbacks = {
            "semantic_scholar": ["arxiv", "openalex"],
            "openalex": ["arxiv", "semantic_scholar"],
            "arxiv": ["openalex", "semantic_scholar"],
        }
        fallback_sources = default_fallbacks.get(source, [])

    sources = _normalize_sources(source, fallback_sources)
    errors: list[str] = []

    for idx, current_source in enumerate(sources):
        result = _run_source_search(query=query, max_results=max_results, source=current_source)

        if result.startswith(("[NO_RESULTS]", "[RATE_LIMITED]", "[UPSTREAM_ERROR]")):
            errors.append(result)
            continue

        if idx == 0:
            return result

        reasons = "; ".join(errors)
        return (
            f"[PAPER_SEARCH_FALLBACK] Switched to {SOURCE_LABELS.get(current_source, current_source)} "
            f"after upstream issues: {reasons}\n\n{result}"
        )

    if errors:
        return "[PAPER_SEARCH_UNAVAILABLE] " + "; ".join(errors)
    return "[PAPER_SEARCH_UNAVAILABLE] No paper source is configured."


@function_tool(description_override=PAPER_SEARCH_TOOL_PROMPT)
def paper_search(query: str, top_k: int = 5, mode: str = "precise") -> str:
    if mode == "broad":
        primary = "openalex"
        fallbacks = ["arxiv", "semantic_scholar"]
    else:
        primary = "semantic_scholar"
        fallbacks = ["arxiv", "openalex"]

    return run_search_papers(
        query=query,
        max_results=top_k,
        source=primary,
        fallback_sources=fallbacks,
    )

def _normalize_sources(source: str, fallback_sources: list[str] | None) -> list[str]:
    sources = [source]
    if fallback_sources:
        sources.extend(fallback_sources)

    deduped: list[str] = []
    for item in sources:
        if item not in {"semantic_scholar", "arxiv", "openalex"}:
            continue
        if item not in deduped:
            deduped.append(item)
    return deduped


def _run_source_search(query: str, max_results: int, source: str) -> str:
    if source == "arxiv":
        return _arxiv_search(query, max_results)
    if source == "openalex":
        return _openalex_search(query, max_results)
    return _s2_search(query, max_results)


def _request(
    *,
    url: str,
    params: dict,
    headers: dict | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> requests.Response:
    return requests.get(
        url,
        params=params,
        headers=headers,
        timeout=timeout,
    )


def _s2_search(query: str, n: int) -> str:
    response = _request(
        url="https://api.semanticscholar.org/graph/v1/paper/search",
        params={
            "query": query,
            "limit": n,
            "fields": "title,authors,year,abstract,openAccessPdf,citationCount,externalIds",
        },
        headers={"x-api-key": S2_API_KEY} if S2_API_KEY else None,
    )
    if response.status_code == 429:
        return "[RATE_LIMITED] Semantic Scholar returned 429 Too Many Requests."
    if not response.ok:
        return f"[UPSTREAM_ERROR] Semantic Scholar returned status {response.status_code}."

    payload = response.json()
    papers = payload.get("data", [])
    if not papers:
        return "[NO_RESULTS] No papers found."

    out = []
    for p in papers:
        pdf = (p.get("openAccessPdf") or {}).get("url", "N/A")
        authors = ", ".join(a["name"] for a in p.get("authors", [])[:5])
        abstract = truncate_text(p.get("abstract") or "N/A", 400)
        out.append(
            f"**{p['title']}** ({p.get('year', '?')})\n"
            f"Authors: {authors}\n"
            f"Citations: {p.get('citationCount', 0)} | PDF: {pdf}\n"
            f"Abstract: {abstract}\n"
        )
    return "\n---\n".join(out)


def _arxiv_search(query: str, n: int) -> str:
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=n,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    out = []
    for r in client.results(search):
        out.append(
            f"**{r.title}** ({r.published.year})\n"
            f"Authors: {', '.join(str(a) for a in r.authors[:5])}\n"
            f"PDF: {r.pdf_url}\n"
            f"Abstract: {truncate_text(r.summary, 400)}\n"
        )
    return "\n---\n".join(out) if out else "[NO_RESULTS] No arXiv papers found."


def _openalex_abstract_to_text(abstract_index: dict | None) -> str | None:
    if not abstract_index:
        return None

    positions: list[tuple[int, str]] = []
    for token, indices in abstract_index.items():
        for idx in indices:
            positions.append((idx, token))

    if not positions:
        return None

    positions.sort(key=lambda item: item[0])
    return " ".join(token for _, token in positions)


def _openalex_search(query: str, n: int) -> str:
    response = _request(
        url="https://api.openalex.org/works",
        params={
            "search": query,
            "per_page": n,
            "select": "title,authorships,publication_year,abstract_inverted_index,open_access,doi",
        },
        headers={"User-Agent": "DeepResearchAgent/2.0 (mailto:dev@example.com)"},
    )
    if response.status_code == 429:
        return "[RATE_LIMITED] OpenAlex returned 429 Too Many Requests."
    if not response.ok:
        return f"[UPSTREAM_ERROR] OpenAlex returned status {response.status_code}."

    payload = response.json()
    works = payload.get("results", [])
    if not works:
        return "[NO_RESULTS] No papers found on OpenAlex."

    out = []
    for w in works:
        authors = ", ".join(
            a["author"]["display_name"] for a in w.get("authorships", [])[:5]
        )
        pdf = w.get("open_access", {}).get("oa_url", "N/A")
        abstract = truncate_text(
            _openalex_abstract_to_text(w.get("abstract_inverted_index")) or "N/A",
            400,
        )
        out.append(
            f"**{w.get('title', 'N/A')}** ({w.get('publication_year', '?')})\n"
            f"Authors: {authors}\n"
            f"DOI: {w.get('doi', 'N/A')} | PDF: {pdf}\n"
            f"Abstract: {abstract}\n"
        )
    return "\n---\n".join(out)
