import re
import asyncio
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

from ddgs import DDGS
import trafilatura
from readability import Document
from lxml import html as lxml_html

from crawl4ai import AsyncWebCrawler

TRACKING_KEYS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "fbclid", "igshid", "mc_cid", "mc_eid", "ref", "ref_src"
}

def canonicalize_url(url: str) -> str:
    """
    Remove common tracking query params and fragments.
    This reduces duplicates and improves cache hit rates.
    """
    try:
        u = urlparse(url)
        q = [(k, v) for k, v in parse_qsl(u.query, keep_blank_values=True) if k not in TRACKING_KEYS]
        new_query = urlencode(q, doseq=True)
        return urlunparse((u.scheme, u.netloc, u.path, u.params, new_query, ""))
    except Exception:
        return url

def post_clean(text: str) -> str:
    """
    Normalize whitespace: collapse repeated spaces and excessive blank lines.
    The goal is to make the output LLM-friendly pure text.
    """
    text = text or ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def truncate(text: str, max_chars: int = 5000) -> Tuple[str, bool, int]:
    """
    Truncate to a hard char limit to keep tool output bounded.
    Returns (text, truncated?, original_len).
    """
    text = text or ""
    original_len = len(text)
    if original_len <= max_chars:
        return text, False, original_len
    return text[:max_chars], True, original_len

def extract_main_text(html: str, min_chars: int = 200) -> str:
    """
    Extract "main article text" from raw HTML.
    Strategy:
      1) trafilatura (best overall for news/blog/wiki-like pages)
      2) readability-lxml fallback ("reader mode" extraction)
      3) empty string if both fail (better than returning noisy nav/footer)
    """
    # 1) trafilatura extraction
    t = trafilatura.extract(
        html,
        output_format="txt",         # pure text output
        include_comments=False,      # drop comment sections
        include_tables=False,        # often reduces noise from layout tables
        favor_precision=True,        # prefer cleaner (less noisy) text
    )
    if t and len(t.strip()) >= min_chars:
        return post_clean(t)

    # 2) readability fallback
    try:
        doc = Document(html)
        main_html = doc.summary(html_partial=True)  # extracted "article" HTML
        tree = lxml_html.fromstring(main_html)
        t2 = tree.text_content()                    # strip tags -> text
        t2 = " ".join(t2.split())                   # compress whitespace
        if len(t2) >= min_chars:
            return post_clean(t2)
    except Exception:
        pass

    return ""

def search_urls_ddgs(
    query: str,
    total: int = 20,
    per_domain: int = 2,
    fetch_factor: int = 4
) -> List[Dict[str, Any]]:
    """
    Perform keyword search and return up to `total` results.

    Domain diversity:
      - We cap results per domain (per_domain) to avoid a single site dominating.
      - This generally improves breadth and reduces duplicated templates/noise.

    fetch_factor:
      - We'll initially request ~ total*fetch_factor results from ddgs,
        because diversity caps and de-dup may discard many.
    """
    wanted = max(30, total * fetch_factor)

    raw: List[Dict[str, Any]] = []

    with DDGS() as d:
        for r in d.text(query, max_results=wanted):
            href = (r.get("href") or "").strip()
            if not href.startswith("http"):
                continue

            url = canonicalize_url(href)
            domain = urlparse(url).netloc.lower()
            title = (r.get("title") or "").strip()
            snippet = (r.get("body") or "").strip()

            if not domain or not title:
                continue

            raw.append({
                "source": "ddgs",
                "title": title,
                "url": url,
                "snippet": snippet,
                "domain": domain,
            })

            if len(raw) >= wanted:
                break

    dedup: List[Dict[str, Any]] = []
    seen_url = set()
    for item in raw:
        if item["url"] in seen_url:
            continue
        seen_url.add(item["url"])
        dedup.append(item)

    out: List[Dict[str, Any]] = []
    domain_count: Dict[str, int] = {}

    for item in dedup:
        dmn = item["domain"]
        domain_count.setdefault(dmn, 0)
        if domain_count[dmn] >= per_domain:
            continue
        domain_count[dmn] += 1
        out.append(item)
        if len(out) >= total:
            break

    return out

async def crawl_and_extract_text(url: str) -> str:
    """
    Use crawl4ai to fetch HTML, then extract main text.
    We intentionally do NOT fall back to markdown if HTML is missing,
    because markdown conversion often includes nav/footer links (noise).
    """
    async with AsyncWebCrawler() as crawler:
        res = await crawler.arun(url=url)
        html = getattr(res, "html", None) or getattr(res, "cleaned_html", None)
        if not html:
            return ""

        return extract_main_text(html)
        
async def search_and_fetch_contents(
    query: str,
    n_urls: int = 20,
    per_domain: int = 2,
    max_chars_per_doc: int = 5000,
    crawl_concurrency: int = 5,
) -> Dict[str, Any]:
   
    """
    Tool name: research_search_fetch

    Purpose
    -------
    Perform web research retrieval in a single call:
      1) Search the web for relevant URLs (DuckDuckGo via ddgs)
      2) Crawl each URL (via crawl4ai) and extract main-body text only
         (boilerplate removal using trafilatura + readability fallback)
      3) Post-clean whitespace and truncate each document to a maximum length

    Inputs
    ------
    query:
        Search query (English recommended).
    n_urls:
        Number of URLs to retrieve from the search stage (after de-dup and filters).
    per_domain:
        Domain diversity cap. At most this many results will be kept per domain.
        Example: per_domain=2 means no more than 2 URLs from the same domain.
    max_chars_per_doc:
        Hard limit (in characters) for extracted text returned per document.
        This bounds tool output size.
    crawl_concurrency:
        Maximum number of concurrent crawl+extract tasks.
    drop_empty:
        If True, drop documents whose extracted content is empty.

    Behavior & Guarantees
    ---------------------
    - The function returns plain text only. Images/videos are not returned.
    - If a page cannot be fetched or main text cannot be extracted reliably,
      its content may be an empty string (and an error may be recorded).
    - URL canonicalization is applied to reduce duplicates (e.g., remove tracking params).

    Returns (JSON-serializable)
    ---------------------------
    A dict with the following shape:

    {
      "query": str,
      "n_urls_requested": int,
      "per_domain": int,
      "max_chars_per_doc": int,
      "candidates": [
        {"source": "ddgs", "title": str, "url": str, "snippet": str, "domain": str}
      ],
      "documents": [
        {
          "source": "ddgs",
          "title": str,
          "url": str,
          "snippet": str,
          "domain": str,
          "content": str,
          "content_original_len": int,
          "content_truncated": bool,
          "error": str | None
        }
      ],
      "documents_dropped_empty": int
    }

    Notes
    -----
    - Intended to be used as a single Agent Tool call.
    - For best stability, keep n_urls moderate (e.g., 10-30) and use a conservative concurrency.
    """
   
    candidates = search_urls_ddgs(query, total=n_urls, per_domain=per_domain)

    sem = asyncio.Semaphore(crawl_concurrency)

    async def _fetch_one(item: Dict[str, Any]) -> Dict[str, Any]:
        async with sem:
            err = None
            try:
                text = await crawl_and_extract_text(item["url"])
            except Exception as e:
                text = ""
                err = str(e)

            text = post_clean(text)
            text, truncated, original_len = truncate(text, max_chars=max_chars_per_doc)

            return {
                **item,
                "content": text,
                "content_original_len": original_len,
                "content_truncated": truncated,
                "error": err,
            }

    docs = await asyncio.gather(*[_fetch_one(it) for it in candidates])
    docs_nonempty = [d for d in docs if d.get("content")]
    return {
        "query": query,
        "n_urls_requested": n_urls,
        "per_domain": per_domain,
        "max_chars_per_doc": max_chars_per_doc,
        "candidates": candidates,          # what search returned after filters
        "documents": docs_nonempty,        # only successful extracted texts
        "documents_dropped_empty": len(docs) - len(docs_nonempty),
    }

