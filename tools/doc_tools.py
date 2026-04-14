"""Local document tools: PDF reading and local doc search."""

import os
from pathlib import Path
from typing import Optional

from agents import function_tool
import pymupdf4llm

from dotenv import load_dotenv

from prompt import LOCAL_DOCS_LOOKUP_TOOL_PROMPT

from .utils import clean_whitespace, truncate_text

load_dotenv()

LOCAL_DOCS_DIR = Path(os.getenv("LOCAL_DOCS_DIR", "data/local_docs")).expanduser()


def run_read_pdf(file_path: str, max_pages: int = 30) -> str:
    """Extract markdown text from a local PDF file."""

    path = Path(file_path)
    if not path.exists():
        return f"[FILE_NOT_FOUND] {file_path}"
    if path.suffix.lower() != ".pdf":
        return f"[NOT_PDF] File must be a .pdf, got: {path.suffix}"

    md = pymupdf4llm.to_markdown(
        str(path),
        pages=list(range(min(max_pages, 500))),
        show_progress=False,
    )
    result = clean_whitespace(md)
    return f"[PDF: {path.name}]\n\n{truncate_text(result, 12000)}"


def run_query_local_docs(search_term: str, docs_dir: str = str(LOCAL_DOCS_DIR)) -> str:
    """Search local docs for passages matching a term or phrase."""

    docs_path = Path(docs_dir)
    if not docs_path.exists():
        return "[NO_LOCAL_DOCS] Local docs directory not found."

    pdf_files = list(docs_path.glob("*.pdf"))
    md_files = list(docs_path.glob("*.md"))
    txt_files = list(docs_path.glob("*.txt"))
    csv_files = list(docs_path.glob("*.csv"))
    tsv_files = list(docs_path.glob("*.tsv"))
    json_files = list(docs_path.glob("*.json"))

    if not (pdf_files + md_files + txt_files + csv_files + tsv_files + json_files):
        return "[NO_LOCAL_DOCS] No documents found in local docs directory."

    results = []
    term_lower = search_term.lower()

    for pdf in pdf_files:
        content = pymupdf4llm.to_markdown(str(pdf), show_progress=False)
        _extract_matches(content, term_lower, str(pdf), results)

    for txt_file in md_files + txt_files + csv_files + tsv_files + json_files:
        content = txt_file.read_text(encoding="utf-8", errors="ignore")
        _extract_matches(content, term_lower, str(txt_file), results)

    if not results:
        return f"[NO_MATCH] No local documents contain '{search_term}'."

    return "\n\n---\n\n".join(results[:5])


def _read_text_file(path: Path, max_chars: int) -> str:
    content = path.read_text(encoding="utf-8", errors="ignore")
    return f"[Local file: {path.name}]\n\n{truncate_text(clean_whitespace(content), max_chars)}"


def _preview_directory(path: Path, max_chars: int) -> str:
    snippets: list[str] = []
    remaining = max_chars

    for candidate in sorted(path.iterdir()):
        if remaining <= 0 or candidate.is_dir():
            continue

        suffix = candidate.suffix.lower()
        snippet = ""
        if suffix == ".pdf":
            snippet = run_read_pdf(str(candidate), max_pages=10)
        elif suffix in {".md", ".txt", ".csv", ".tsv", ".json"}:
            snippet = _read_text_file(candidate, min(remaining, 4000))

        if not snippet:
            continue

        snippet = truncate_text(snippet, remaining)
        snippets.append(snippet)
        remaining -= len(snippet)

    return "\n\n---\n\n".join(snippets)


def run_local_docs_lookup(
    local_files_path: str,
    question: str,
    max_chars: int = 12000,
) -> str:
    """Backward-compatible local context lookup for the current agent flow."""

    path = Path(local_files_path).expanduser().resolve()
    if not path.exists():
        return ""

    if path.is_file():
        if path.suffix.lower() == ".pdf":
            return truncate_text(run_read_pdf(str(path)), max_chars)
        if path.suffix.lower() in {".md", ".txt", ".csv", ".tsv", ".json"}:
            return _read_text_file(path, max_chars)
        return f"[UNSUPPORTED_FILE] Unsupported local file type: {path.suffix}"

    result = run_query_local_docs(search_term=question, docs_dir=str(path))
    if result.startswith("[NO_MATCH]") or result.startswith("[NO_LOCAL_DOCS]"):
        return _preview_directory(path, max_chars)
    return truncate_text(result, max_chars)


@function_tool(description_override=LOCAL_DOCS_LOOKUP_TOOL_PROMPT)
def local_docs_lookup(
    local_files_path: str,
    question: str,
    max_chars: int = 12000,
) -> str:
    """Look up local files. The file path is provided in your task context."""
    return run_local_docs_lookup(
        local_files_path=local_files_path,
        question=question,
        max_chars=max_chars,
    )

def _extract_matches(content: str, term: str, source: str, results: list[str]) -> None:
    paragraphs = content.split("\n\n")
    for para in paragraphs:
        if term in para.lower() and len(para.strip()) > 50:
            results.append(f"[From: {source}]\n{truncate_text(para.strip(), 600)}")
