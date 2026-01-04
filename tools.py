import os
from typing import Literal, List
from pathlib import Path
from agents import function_tool
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

from utils.web_seach_pipeline import search_and_fetch_contents
from utils.paper_search_pipeline import get_full_paper_text
from utils.convert_to_md import to_md  
from prompt import LOCAL_FILES_SUMMARY_PROMPT, WEB_TEMPLATE_INSTRUCTIONS, PAPER_TEMPLATE_INSTRUCTIONS

BASE_URL = os.getenv("BASE_URL")

client = AsyncOpenAI(
    api_key="EMPTY",
    base_url=BASE_URL,
)

class WebDocument(BaseModel):
    id: int = Field(description="Local document id within this search result.")
    title: str = Field(description="Page title, used for citation and further deep search.")
    url: str = Field(description="Canonical URL of the page.")
    content_markdown: str | None = Field(
        default=None,
        description="Main page content converted to markdown. May be truncated."
    )

class WebSearchResult(BaseModel):
    query: str = Field(description="Original user query for this search.")
    n_urls_requested: int = Field(description="Requested number of URLs.")
    max_chars_per_doc: int = Field(description="Max chars per document used for crawling.")
    documents: list[WebDocument] = Field(description="Fetched and processed web documents.")


class PaperDocument(BaseModel):
    id: int = Field(description="Local paper id within this search result.")
    title: str = Field(description="Paper title.")
    authors: list[str] = Field(default_factory=list, description="List of authors.")
    year: int | None = Field(default=None, description="Publication year, if available.")
    venue: str | None = Field(default=None, description="Conference/journal name, if available.")
    abstract: str | None = Field(default=None, description="Abstract text.")
    full_text_markdown: str | None = Field(
        default=None,
        description="Full paper text converted to markdown."
    )

class PaperSearchResult(BaseModel):
    query: str = Field(description="Original query used for this paper search.")
    top_k: int = Field(description="Requested number of paper candidates.")
    mode: Literal["precise", "broad"] = Field(
        description="Search mode used by the backend."
    )
    papers: list[PaperDocument] = Field(description="Retrieved paper candidates.")

class WebTemplate(BaseModel):
    source_type: Literal["web"] = Field("web")
    doc_id: int = Field(description="Original document id within the web search result.")
    title: str = Field(description="Web page title.")
    url: str | None = Field(default=None, description="Web page URL.")
    citation: str | None = Field(
        default=None,
        description="Short human-readable reference, e.g. 'Example.com, accessed 2025-12-28'."
    )

    overview: str = Field(
        description="2–4 sentences summarizing the overall content and main message of the page."
    )
    main_points: list[str] = Field(
        default_factory=list,
        description="Bullet-point list of the key arguments or facts."
    )

    evidence_or_sources: list[str] = Field(
        default_factory=list,
        description="What evidence or sources the page relies on (data, citations, organizations, etc.)."
    )

    limitations_or_biases: list[str] = Field(
        default_factory=list,
        description="Important limitations, potential biases, or reasons to doubt this source."
    )

    # relevance_score: float = Field(
    #     description="Overall relevance to the user question in [0, 1]."
    # )
    # reliability_score: float = Field(
    #     description="Perceived reliability/authority of the source in [0, 1]."
    # )
    # suggested_followup_queries: list[str] = Field(
    #     default_factory=list,
    #     description="Concrete follow-up queries or directions inspired by this page."
    # )

class PaperTemplate(BaseModel):
    source_type: Literal["paper"] = Field("paper")
    doc_id: int = Field(description="Original paper id within the paper search result.")
    title: str = Field(description="Paper title.")
    url: str | None = Field(default=None, description="Landing page or PDF URL if available.")
    citation: str | None = Field(
        default=None,
        description="Short citation string, e.g. 'Author et al., 2023, NeurIPS'."
    )

    introduction: str = Field(
        description="What problem does this paper study, and why is it important?"
    )
    related_work: str = Field(
        description="How this paper situates itself relative to prior work."
    )
    method: str = Field(
        description="Core methods/algorithms/models proposed or used in this paper."
    )
    experiments: str = Field(
        description="Experimental setup, datasets, baselines, and evaluation protocols."
    )
    results: str = Field(
        description="Key results and findings, including any important numbers or trends."
    )
    conclusion: str = Field(
        description="Main takeaway messages and implications."
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Limitations, failure cases, or assumptions that restrict applicability."
    )
    # relevance_score: float = Field(
    #     description="Overall relevance to the user question in [0, 1]."
    # )
    # evidence_strength: float = Field(
    #     description="Strength of evidence (quality of experiments, clarity of results) in [0, 1]."
    # )
    # suggested_followup_queries: list[str] = Field(
    #     default_factory=list,
    #     description="Queries or directions for finding complementary or deeper papers."
    # )

class SummarizedSource(BaseModel):
    """
    A unified wrapper so that the tool returns a single list,
    but each element is either a WebTemplate or a PaperTemplate.
    """
    source_type: Literal["web", "paper"]
    web: WebTemplate | None = None
    paper: PaperTemplate | None = None


@function_tool
async def web_search(
    query: str,
    n_urls: int = 20, # search breadth
    max_chars_per_doc: int = 5000,   # search depth
) -> WebSearchResult:
    
    "one search tool calling, return text in markdown"

    # irrelevance to agent, set by actual running environment
    per_domain = 2
    crawl_concurrency = 5 
    raw = await search_and_fetch_contents(
            query=query,
            n_urls=n_urls,
            max_chars_per_doc=max_chars_per_doc,
            per_domain=per_domain,
            crawl_concurrency=crawl_concurrency,
        )
    
    documents: list[WebDocument] = []

    for i, doc in enumerate(raw.get("documents", [])):
        content_markdown = doc["content"]
        documents.append(
            WebDocument(
                id=i,
                title=doc.get("title") or doc.get("url") or f"Document {i}",
                url=doc.get("url", ""),
                content_markdown=content_markdown,
            )
        )

    return WebSearchResult(
        query=query,
        n_urls_requested=n_urls,
        max_chars_per_doc=max_chars_per_doc,
        documents=documents,
    )


@function_tool
async def paper_search(
    query: str,
    top_k: int = 10,
    mode: Literal["precise", "broad"] = "precise",
) -> PaperSearchResult:
    """
    top_k: control number of paper candidates.

    mode="precise": Suitable for specific paper searches (Title/DOI/arXiv).

    mode="broad": Suitable for broad/thematic searches (Surveys/Topics).
    """
    raw_results = await get_full_paper_text(query=query, top_k=top_k, mode=mode)

    papers: list[PaperDocument] = []

    for i, r in enumerate(raw_results):
        papers.append(
            PaperDocument(
                id=i,
                title=r.get("title") or f"Paper {i}",
                authors=r.get("authors") or [],
                year=r.get("year"),
                venue=r.get("venue"),
                abstract=r.get("abstract"),
                full_text_markdown=r.get("text"),
            )
        )

    return PaperSearchResult(
        query=query,
        top_k=top_k,
        mode=mode,
        papers=papers,
    )


# helper function 
def build_web_input(
    question: str,
    doc: WebDocument,
    extra_prompt: str,
) -> str:
    return f"""
        User question:
        {question}

        You are summarizing the following web page content in markdown:

        TITLE: {doc.title}
        URL: {doc.url}
        DOMAIN: {doc.domain}

        CONTENT:
        {doc.content_markdown}

        Additional instructions:
        {extra_prompt}
    """.strip()

# helper function
def build_paper_input(
    question: str,
    paper: PaperDocument,
    extra_prompt: str,
) -> str:
    return f"""
        User question:
        {question}

        You are summarizing the following research paper content in markdown:

        TITLE: {paper.title}
        URL: {paper.landing_page_url or paper.pdf_url or 'N/A'}

        ABSTRACT:
        {paper.abstract or 'N/A'}

        FULL TEXT:
        {paper.full_text_markdown or 'N/A'}

        Additional instructions:
        {extra_prompt}
    """.strip()


@function_tool
async def summarize_sources(
    question: str,
    web_docs: list[WebDocument] | None = None,
    papers: list[PaperDocument] | None = None,
    max_tokens: int = 2048,
    summary_prompt: str = "",
) -> list[SummarizedSource]:

    results: list[SummarizedSource] = []

    # client = AsyncOpenAI(
    #     api_key="EMPTY",
    #     base_url=BASE_URL,
    # )

    if web_docs:
        web_schema = WebTemplate.model_json_schema()
        for doc in web_docs:
            messages = [
                {
                    "role": "system",
                    "content": WEB_TEMPLATE_INSTRUCTIONS + "\n" + summary_prompt,
                },
                {
                    "role": "user",
                    "content": build_web_input(question, doc),
                },
            ]
            resp = await client.chat.completions.create(
                    model="qwen3-8b",
                    messages=messages,
                    max_tokens=max_tokens,
                    response_format={
                        "type": "json_schema",
                        "json_schema": web_schema
                    }
            )
            raw_json = resp.choices[0].message.content[0].text  
            web_summary = WebTemplate.model_validate_json(raw_json)

            results.append(
                SummarizedSource(source_type="web", web=web_summary, paper=None)
            )

    if papers:
        paper_schema = PaperTemplate.model_json_schema()
        for paper in papers:
            messages = [
                {
                    "role": "system",
                    "content": PAPER_TEMPLATE_INSTRUCTIONS + "\n" + summary_prompt,
                },
                {
                    "role": "user",
                    "content": build_paper_input(question, paper),
                },
            ]
            resp = await client.chat.completions.create(
                model="qwen3-8b",
                messages=messages,
                max_tokens=max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": paper_schema,
                },
            )
            raw_json = resp.choices[0].message.content[0].text
            paper_summary = PaperTemplate.model_validate_json(raw_json)

            results.append(
                SummarizedSource(source_type="paper", web=None, paper=paper_summary)
            )

    return results


@function_tool
def local_files_exist(local_files_path: str) -> bool:
    """
    Check whether the given local_files_path exists.

    - If it's a file, return True.
    - If it's a directory, return True when the directory exists
    - Otherwise return False.
    """
    p = Path(local_files_path).expanduser()
    return p.exists()

@function_tool
async def local_docs_lookup(
    local_files_path: str,
    question: str,
    max_chars: int = 20_000,
    summary_prompt: str = "",
) -> str:
    """
    Usage during the Human-in-the-loop phase:

    If the path exists, use to_md to read the corresponding file/directory and convert it into a Markdown list;
    Concatenate into a single large Markdown string and truncate at max_chars;
    Use a one-time LLM call to generate a compressed summary (free-form, without a template);

    Return this summary to the main agent, rather than the original text.
    If the path does not exist or there is no available content, return an empty string "".
    """

    base_path = Path(local_files_path).expanduser().resolve()
    if not base_path.exists():
        return ""
    
    try:
        md_list: List[str] = to_md(str(base_path))
    except Exception:
        return ""

    if not md_list:
        return ""

    merged_parts: List[str] = []
    total = 0

    for i, md in enumerate(md_list):
        if total >= max_chars:
            break
        remaining = max_chars - total
        snippet = md[:remaining]
        merged_parts.append(f"\n\n--- LOCAL FILE #{i+1} ---\n\n{snippet}")
        total += len(snippet)

    if not merged_parts:
        return ""
    
    merged_md = "".join(merged_parts)

    messages = [
        {
            "role": "system",
            "content": LOCAL_FILES_SUMMARY_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"User question:\n{question}\n\n"
                f"Here are local files in markdown format "
                f"(truncated to at most {max_chars} characters in total):\n"
                f"{merged_md}"
            ),
        },
    ]

    resp = await client.chat.completions.create(
        model="qwen3-8b",
        messages=messages,
    )

    content = resp.choices[0].message.content
    # if isinstance(content, str):
    #     summary = content
    # else:
    #     # 新 SDK: content 是 [ { "type": "text", "text": {...} }, ... ]
    #     summary = "".join(part.text for part in content if hasattr(part, "text"))

    return content

