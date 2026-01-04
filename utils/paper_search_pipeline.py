from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Literal
from urllib.parse import urlparse
import tempfile
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS
from tenacity import retry, stop_after_attempt, wait_exponential

from utils.convert_to_md import to_md


OPEN_ACCESS_DOMAINS = {
    "arxiv.org",
    "pmc.ncbi.nlm.nih.gov",
}

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (paper-search-pipeline/0.1; +https://example.com/bot)"
}

DOI_RE = re.compile(r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.IGNORECASE)
ARXIV_RE = re.compile(r"arxiv\.org/(abs|pdf)/(\d{4}\.\d{4,5})(v\d+)?", re.IGNORECASE)

S2_FIELDS = (
    "title,year,authors,venue,externalIds,url,openAccessPdf,"
    "publicationTypes,journal,referenceCount,citationCount,abstract"
)


@dataclass
class SearchHit:
    rank: int
    title: Optional[str]
    url: str
    snippet: Optional[str] = None
    source: str = "ddg"

@dataclass
class PaperCandidate:
    source_hit: SearchHit
    landing_page_url: str
    title_guess: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    pdf_url_hint: Optional[str] = None
    open_access_domain: bool = False
    debug: dict[str, Any] = field(default_factory=dict)

@dataclass
class PaperResult:
    title: Optional[str] = None
    year: Optional[int] = None
    authors: list[str] = field(default_factory=list)
    venue: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    abstract: Optional[str] = None  
    landing_page_url: Optional[str] = None
    pdf_url: Optional[str] = None
    sources: dict[str, Any] = field(default_factory=dict)
    downstream_payload: dict[str, Any] = field(default_factory=dict)


def _host(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""
    
def normalize_title(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def jaccard(a: str, b: str) -> float:
    sa = set(normalize_title(a).split())
    sb = set(normalize_title(b).split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def ddgs_search(
    query: str,
    max_results: int = 15,
    backend: Literal["bing", "google", "duckduckgo"] = "duckduckgo",
) -> list[SearchHit]:
    hits: list[SearchHit] = []
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results, backend=backend))

    for i, r in enumerate(results):
        url = r.get("href")
        if not url:
            continue
        hits.append(
            SearchHit(
                rank=i + 1,
                title=r.get("title"),
                url=url,
                snippet=r.get("body"),
                source="ddg",
            )
        )
    return hits

def extract_ids_from_url(url: str) -> tuple[Optional[str], Optional[str]]:
    doi = None
    arxiv = None

    m = ARXIV_RE.search(url)
    if m:
        arxiv = m.group(2) + (m.group(3) or "")

    if "doi.org/" in url.lower():
        m2 = DOI_RE.search(url)
        if m2:
            doi = m2.group(1)

    return doi, arxiv

def arxiv_pdf_url_from_id(arxiv_id: str) -> str:
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"

async def fetch_html(
    client: httpx.AsyncClient,
    url: str,
    timeout: float = 15.0,
) -> Optional[str]:
    r = await client.get(url, timeout=timeout, headers=DEFAULT_HEADERS, follow_redirects=True)
    ct = (r.headers.get("content-type") or "").lower()
    if "text/html" not in ct:
        return None
    return r.text

def extract_meta_from_html(html: str) -> dict[str, Optional[str]]:
    soup = BeautifulSoup(html, "html.parser")

    def meta(name: str) -> Optional[str]:
        tag = soup.find("meta", attrs={"name": name})
        return tag.get("content") if tag and tag.get("content") else None

    title = meta("citation_title")
    doi = meta("citation_doi")
    pdf_url = meta("citation_pdf_url")

    if not doi:
        m = DOI_RE.search(str(soup))
        if m:
            doi = m.group(1)

    if not title:
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

    return {"title": title, "doi": doi, "pdf_url": pdf_url}

async def build_candidate(client: httpx.AsyncClient, hit: SearchHit) -> PaperCandidate:
    url = hit.url
    host = _host(url)
    open_domain = host in OPEN_ACCESS_DOMAINS

    doi, arxiv_id = extract_ids_from_url(url)

    cand = PaperCandidate(
        source_hit=hit,
        landing_page_url=url,
        title_guess=hit.title,
        doi=doi,
        arxiv_id=arxiv_id,
        open_access_domain=open_domain,
    )

    if host == "arxiv.org" and arxiv_id:
        cand.pdf_url_hint = arxiv_pdf_url_from_id(arxiv_id)
        cand.debug["oa_rule"] = "arxiv_pdf_constructed"
        return cand

    if open_domain:
        try:
            html = await fetch_html(client, url)
            if html:
                meta = extract_meta_from_html(html)
                cand.title_guess = meta.get("title") or cand.title_guess
                cand.doi = cand.doi or meta.get("doi")
                cand.pdf_url_hint = cand.pdf_url_hint or meta.get("pdf_url")
                cand.debug["meta_extracted"] = True
        except Exception as e:
            cand.debug["meta_extracted_error"] = str(e)

    return cand


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=0.5, max=4))
async def s2_get_by_id(
    client: httpx.AsyncClient,
    paper_id: str,
) -> Optional[dict[str, Any]]:
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
    r = await client.get(url, params={"fields": S2_FIELDS}, timeout=15.0)
    if r.status_code != 200:
        return None
    return r.json()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=0.5, max=4))
async def s2_search_by_title(
    client: httpx.AsyncClient,
    title: str,
) -> Optional[dict[str, Any]]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    r = await client.get(
        url,
        params={"query": title, "limit": 5, "fields": S2_FIELDS},
        timeout=15.0,
    )
    if r.status_code != 200:
        return None
    data = r.json()
    if not data or "data" not in data:
        return None

    best = None
    best_score = 0.0
    for item in data["data"]:
        score = jaccard(title, item.get("title") or "")
        if score > best_score:
            best_score = score
            best = item
    if best and best_score >= 0.5:
        return best
    return best

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=0.5, max=4))
async def s2_search(
    client: httpx.AsyncClient,
    query: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    r = await client.get(
        url,
        params={"query": query, "limit": limit, "fields": S2_FIELDS},
        timeout=15.0,
    )
    if r.status_code != 200:
        return []
    data = r.json()
    return data.get("data") or []


def pick_pdf_url(
    candidate: PaperCandidate,
    s2: Optional[dict[str, Any]],
) -> Optional[str]:
    if candidate.pdf_url_hint:
        return candidate.pdf_url_hint

    if s2:
        oap = s2.get("openAccessPdf") or {}
        if oap.get("url"):
            return oap["url"]

    return None


def merge_to_result(
    cand: PaperCandidate,
    s2: Optional[dict[str, Any]],
) -> PaperResult:
    res = PaperResult()

    res.doi = cand.doi
    res.arxiv_id = cand.arxiv_id
    res.landing_page_url = cand.landing_page_url

    if s2:
        res.title = s2.get("title") or res.title
        res.year = s2.get("year") or res.year
        res.venue = s2.get("venue") or res.venue
        res.abstract = s2.get("abstract") or res.abstract 

        authors = s2.get("authors") or []
        res.authors = [a.get("name") for a in authors if a.get("name")]

        ext = s2.get("externalIds") or {}
        res.doi = res.doi or ext.get("DOI")
        res.arxiv_id = res.arxiv_id or ext.get("ArXiv")

    res.title = res.title or cand.title_guess
    res.pdf_url = pick_pdf_url(cand, s2)
    res.sources = {
        "candidate": {
            "from": cand.source_hit.source,
            "rank": cand.source_hit.rank,
            "open_access_domain": cand.open_access_domain,
            "debug": cand.debug,
        },
        "semantic_scholar": s2,
    }

    res.downstream_payload = {
        "pdf_url": res.pdf_url,
        "landing_page_url": res.landing_page_url,
        "doi": res.doi,
        "arxiv_id": res.arxiv_id,
        "title": res.title,
        "next_action": (
            "IF pdf_url present: "
            "download_pdf -> pdf_to_markdown -> fanout_to_agents -> aggregate_summary"
        ),
    }

    return res


def dedupe(results: list[PaperResult]) -> list[PaperResult]:
    seen: set[str] = set()
    uniq: list[PaperResult] = []

    for r in results:
        key = None
        if r.doi:
            key = f"doi:{r.doi.lower()}"
        elif r.arxiv_id:
            key = f"arxiv:{r.arxiv_id.lower()}"
        elif r.title:
            key = f"title:{normalize_title(r.title)}"
        else:
            key = f"url:{r.landing_page_url}"

        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)

    return uniq

async def enrich_candidate(
    client: httpx.AsyncClient,
    cand: PaperCandidate,
) -> PaperResult:
    s2: Optional[dict[str, Any]] = None
    if cand.doi:
        s2 = await s2_get_by_id(client, f"DOI:{cand.doi}")
    elif cand.arxiv_id:
        s2 = await s2_get_by_id(client, f"ARXIV:{cand.arxiv_id}")
    elif cand.title_guess:
        s2 = await s2_search_by_title(client, cand.title_guess)

    return merge_to_result(cand, s2)

async def paper_search_precise(
    query: str,
    top_k: int = 10,
    max_serp_per_query: int = 10,
    max_candidates: int = 30,
    concurrency: int = 10,
    backend: Literal["bing", "google", "duckduckgo"] = "duckduckgo",
) -> list[PaperResult]:
    hits: list[SearchHit] = []
    seen_urls: set[str] = set()
    for h in ddgs_search(query, max_results=max_serp_per_query, backend=backend):
        if h.url in seen_urls:
            continue
        seen_urls.add(h.url)
        hits.append(h)

    hits = hits[: max_candidates * 2]

    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    timeout = httpx.Timeout(connect=10.0, read=20.0, write=10.0, pool=10.0)

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        sem = asyncio.Semaphore(concurrency)

        async def _build(h: SearchHit):
            async with sem:
                return await build_candidate(client, h)

        candidates = await asyncio.gather(*[_build(h) for h in hits])
        candidates = [c for c in candidates if c.open_access_domain or c.doi or c.arxiv_id]
        candidates = candidates[:max_candidates]

        async def _enrich(c: PaperCandidate):
            async with sem:
                return await enrich_candidate(client, c)

        enriched = await asyncio.gather(*[_enrich(c) for c in candidates])

    ranked = dedupe(enriched)
    return ranked[:top_k]


def s2_item_to_result(item: dict[str, Any]) -> PaperResult:
    res = PaperResult()
    res.title = item.get("title")
    res.year = item.get("year")
    res.venue = item.get("venue")
    res.abstract = item.get("abstract") 

    authors = item.get("authors") or []
    res.authors = [a.get("name") for a in authors if a.get("name")]

    ext = item.get("externalIds") or {}
    res.doi = ext.get("DOI")
    res.arxiv_id = ext.get("ArXiv")

    res.landing_page_url = item.get("url")
    oap = item.get("openAccessPdf") or {}
    res.pdf_url = oap.get("url")

    res.sources = {
        "semantic_scholar": item,
    }
    res.downstream_payload = {
        "pdf_url": res.pdf_url,
        "landing_page_url": res.landing_page_url,
        "doi": res.doi,
        "arxiv_id": res.arxiv_id,
        "title": res.title,
        "next_action": (
            "IF pdf_url present: "
            "download_pdf -> pdf_to_markdown -> fanout_to_agents -> aggregate_summary"
        ),
    }
    return res

async def semantic_scholar_topic_search(
    query: str,
    top_k: int = 10,
) -> list[PaperResult]:
    async with httpx.AsyncClient(timeout=15.0) as client:
        items = await s2_search(client, query, limit=top_k)

    results = [s2_item_to_result(it) for it in items]
    results = dedupe(results)
    return results[:top_k]


async def run_paper_search(
    query: str,
    top_k: int = 10,
    mode: Literal["precise", "broad"] = "precise",
) -> list[PaperResult]:
    """
    mode="precise": Suitable for specific paper searches (Title/DOI/arXiv).
    mode="broad": Suitable for broad/thematic searches (Surveys/Topics).
    """
    if mode == "precise":
        return await paper_search_precise(
            query=query,
            top_k=top_k,
            # internal paras
            max_serp_per_query=10,
            max_candidates=30,
            concurrency=10,
            backend="google",
        )
    else:
        return await semantic_scholar_topic_search(query=query, top_k=top_k)


async def _download_to_tempfile(
    client: httpx.AsyncClient,
    url: str,
    suffix: str = ".pdf",
    timeout: float = 30.0,
) -> Optional[Path]:
    try:
        r = await client.get(url, timeout=timeout, follow_redirects=True)
        if r.status_code != 200:
            print(f"[download] status {r.status_code} for {url}")
            return None

        tmp_dir = tempfile.mkdtemp(prefix="paper_pdf_")
        tmp_path = Path(tmp_dir) / f"paper{suffix}"
        tmp_path.write_bytes(r.content)
        return tmp_path
    except Exception as e:
        print(f"[download] error downloading {url}: {e}")
        return None
    

async def get_full_paper_text(
    query: str,
    top_k: int = 1,
    mode: Literal["precise", "broad"] = "precise",
) -> list[dict[str, Any]]:
    """
    Finalization Function:

    Retrieves a list of PaperResult objects using paper_search.

    Attempts to download the PDF via pdf_url and extract Markdown text using to_md.

    Falls back to the abstract if the PDF is unavailable or extraction fails.
    
    Returns structured results to facilitate direct use by downstream agents.
    """
    results = await run_paper_search(query=query, top_k=top_k, mode=mode)

    out: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        for r in results:
            text: Optional[str] = None
            text_source: str = "none"

            pdf_path: Optional[Path] = None
            if r.pdf_url:
                pdf_path = await _download_to_tempfile(client, r.pdf_url, suffix=".pdf")
                if pdf_path:
                    try:
                        md_list = to_md(str(pdf_path))
                        if md_list:
                            text = md_list[0]
                            text_source = "pdf_md"
                    except Exception as e:
                        print(f"[get_full_paper_text] to_md error for {pdf_path}: {e}")

            if text is None and getattr(r, "abstract", None):
                text = r.abstract
                text_source = "abstract"

            out.append(
                {
                    "title": r.title,
                    "year": r.year,
                    "venue": r.venue,
                    "authors": r.authors,
                    "doi": r.doi,
                    "arxiv_id": r.arxiv_id,
                    "landing_page_url": r.landing_page_url,
                    "pdf_url": r.pdf_url,
                    "abstract": getattr(r, "abstract", None),
                    "text": text,               
                    "text_source": text_source, 
                    "sources": r.sources,
                }
            )

    return out

# if __name__ == "__main__":
#     async def _demo():
#         q = "SimCLR: A Simple Framework for Contrastive Learning of Visual Representations"
#         full_list = await get_full_paper_text(q, top_k=1, mode="precise")
#         for item in full_list:
#             print("Title:", item["title"])
#             print("Year:", item["year"], "Venue:", item["venue"])
#             print("DOI:", item["doi"], "arXiv:", item["arxiv_id"])
#             print("PDF:", item["pdf_url"])
#             print("Text source:", item["text_source"])
#             snippet = (item["text"] or "")[:5000]
#             print("Text snippet:\n", snippet, "\n...\n")

#     await _demo()

if __name__ == "__main__":
    async def _demo():
        # q1 = "jit Kaiming he"
        # precise_results = await paper_search(q1, top_k=10, mode="precise")
        # print("\n=== Precise mode ===")
        # for i, r in enumerate(precise_results, 1):
        #     print(f"\n[{i}] {r.title}")
        #     print(f"  year: {r.year}  venue: {r.venue}")
        #     print(f"  doi: {r.doi}  arxiv: {r.arxiv_id}")
        #     print(f"  abstract: {r.abstract}")
        #     print(f"  landing: {r.landing_page_url}")
        #     print(f"  pdf: {r.pdf_url}")

        q2 = "jit"
        broad_results = await run_paper_search(q2, top_k=10, mode="broad")
        print("\n=== Broad mode ===")
        for i, r in enumerate(broad_results, 1):
            print(f"\n[{i}] {r.title}")
            print(f"  year: {r.year}  venue: {r.venue}")
            print(f"  doi: {r.doi}  arxiv: {r.arxiv_id}")
            print(f"  abstract: {r.abstract}")
            print(f"  landing: {r.landing_page_url}")
            print(f"  pdf: {r.pdf_url}")

    asyncio.run(_demo())