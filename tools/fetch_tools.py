"""Webpage fetching and extraction helpers."""

from agents import function_tool
import trafilatura

from prompt import FETCH_WEBPAGE_TOOL_PROMPT

from .utils import clean_whitespace, truncate_text


def run_fetch_webpage(url: str, max_chars: int = 8000) -> str:
    """Fetch a webpage and extract markdown content."""
    if not url.startswith("http"):
        return "[INVALID_URL] URL must start with http:// or https://"

    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        text = trafilatura.extract(
            downloaded,
            output_format="markdown",
            include_tables=True,
            include_images=False,
            favor_precision=True,
            no_fallback=False,
        )
        if text and len(text) > 300:
            cleaned = clean_whitespace(text)
            return f"[Source: {url}]\n\n{truncate_text(cleaned, max_chars)}"

    return f"[FETCH_ERROR] Unable to retrieve content from: {url}"


@function_tool(description_override=FETCH_WEBPAGE_TOOL_PROMPT)
def fetch_webpage(url: str, max_chars: int = 8000) -> str:
    return run_fetch_webpage(url=url, max_chars=max_chars)
