"""Web search tool backed by Tavily."""

import os

from agents import function_tool
from tavily import TavilyClient

from dotenv import load_dotenv

from prompt import WEB_SEARCH_TOOL_PROMPT

from .utils import truncate_text

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")


def run_web_search(query: str, max_results: int = 10) -> str:
    """Run a Tavily search and format the results for the agent."""
    if not TAVILY_API_KEY:
        return "[SEARCH_UNAVAILABLE] TAVILY_API_KEY is not configured."

    client = TavilyClient(api_key=TAVILY_API_KEY)
    results = client.search(query, max_results=max_results).get("results", [])

    if not results:
        return "[NO_RESULTS] No results found for this query."

    lines = ["[Search results from Tavily]\n"]
    for i, result in enumerate(results, 1):
        title = result.get("title", result.get("name", "No title"))
        url = result.get("url", result.get("href", result.get("link", "N/A")))
        snippet = truncate_text(
            result.get("content", result.get("snippet", result.get("body", "No snippet"))),
            1000,
        )
        lines.append(f"{i}. **{title}**\n   URL: {url}\n   {snippet}\n")

    return "\n".join(lines)


@function_tool(description_override=WEB_SEARCH_TOOL_PROMPT)
def web_search(
    query: str,
    n_urls: int = 10,
    max_chars_per_doc: int = 5000,
) -> str:
    del max_chars_per_doc
    return run_web_search(query=query, max_results=n_urls)
