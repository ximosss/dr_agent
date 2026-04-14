"""Webpage fetching and extraction helpers."""

import configparser
import logging
from urllib.parse import urldefrag

import requests
import trafilatura
from trafilatura.settings import use_config
from agents import function_tool

from prompt import FETCH_WEBPAGE_TOOL_PROMPT

from .utils import clean_whitespace, truncate_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Trafilatura config: lower min-output size, custom UA
# ---------------------------------------------------------------------------
_TRAFILATURA_CONFIG: configparser.ConfigParser = use_config()
_TRAFILATURA_CONFIG.set("DEFAULT", "MIN_OUTPUT_SIZE", "50")
_TRAFILATURA_CONFIG.set("DEFAULT", "MIN_EXTRACTED_SIZE", "50")

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

_FETCH_TIMEOUT = 20  # seconds

# ---------------------------------------------------------------------------
# Per-session URL cache — prevents the model from fetching the same page in a
# loop which is the #1 cause of context-window blowup.
# ---------------------------------------------------------------------------
_url_cache: dict[str, str] = {}


def _normalize_url(url: str) -> str:
    """Strip fragment so foo.html#sec1 and foo.html hit the same cache entry."""
    return urldefrag(url)[0]


def run_fetch_webpage(url: str, max_chars: int = 10000) -> str:
    """Fetch a webpage and extract markdown content."""
    if not url.startswith("http"):
        return "[INVALID_URL] URL must start with http:// or https://"

    cache_key = _normalize_url(url)
    if cache_key in _url_cache:
        return _url_cache[cache_key]

    # Attempt 1: trafilatura native fetch
    downloaded = trafilatura.fetch_url(url, config=_TRAFILATURA_CONFIG)

    # Attempt 2: requests fallback
    if not downloaded:
        downloaded = _fetch_with_requests(url)

    if not downloaded:
        result = f"[FETCH_ERROR] Unable to retrieve content from: {url}"
        _url_cache[cache_key] = result
        return result

    text = trafilatura.extract(
        downloaded,
        output_format="markdown",
        include_tables=True,
        include_images=False,
        favor_precision=False,
        no_fallback=False,
        config=_TRAFILATURA_CONFIG,
    )

    if text and text.strip():
        cleaned = clean_whitespace(text)
        result = f"[Source: {url}]\n\n{truncate_text(cleaned, max_chars)}"
    else:
        result = f"[FETCH_ERROR] Unable to extract content from: {url}"

    _url_cache[cache_key] = result
    return result


def clear_fetch_cache() -> None:
    """Reset the URL cache (call between eval examples)."""
    _url_cache.clear()


def _fetch_with_requests(url: str) -> str | None:
    """Fallback fetcher using requests when trafilatura fails."""
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": _USER_AGENT},
            timeout=_FETCH_TIMEOUT,
            allow_redirects=True,
        )
        resp.raise_for_status()
        return resp.text
    except Exception as exc:
        logger.debug("requests fallback failed for %s: %s", url, exc)
        return None


@function_tool(description_override=FETCH_WEBPAGE_TOOL_PROMPT)
def fetch_webpage(url: str, max_chars: int = 8000) -> str:
    return run_fetch_webpage(url=url, max_chars=max_chars)
