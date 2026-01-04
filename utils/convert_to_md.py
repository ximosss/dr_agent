from __future__ import annotations

from pathlib import Path
from typing import List

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered


def _pdf_to_md(pdf_path: Path, converter: PdfConverter) -> str:
    rendered = converter(str(pdf_path))
    text, _, _ = text_from_rendered(rendered)
    return text


def _text_like_to_md(path: Path) -> str:
    """
    plain text to Markdown。
    - Source code add ```code block```
    - pure text directly return
    """
    suffix = path.suffix.lower()
    content = path.read_text(encoding="utf-8", errors="ignore")

    code_lang_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".sh": "bash",
        ".bash": "bash",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".c": "c",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".html": "html",
        ".css": "css",
        ".csv": "",
    }

    if suffix in {".md", ".markdown"}:
        return content

    if suffix in {".txt"}:
        return content

    lang = code_lang_map.get(suffix, "")
    fence_lang = lang if lang else ""
    return f"```{fence_lang}\n{content}\n```"


def _iter_target_files(base_path: Path) -> List[Path]:
    if base_path.is_file():
        return [base_path]

    if base_path.is_dir():
        exts = {
            ".pdf",
            ".md",
            ".markdown",
            ".txt",
            ".py",
            ".json",
            ".yaml",
            ".yml",
            ".csv",
            ".js",
            ".ts",
            ".sh",
            ".bash",
            ".c",
            ".cc",
            ".cpp",
            ".java",
            ".go",
            ".rs",
            ".html",
            ".css",
        }
        files = [
            p
            for p in base_path.rglob("*")
            if p.is_file() and p.suffix.lower() in exts
        ]
        files.sort()
        return files

    raise FileNotFoundError(f"path not found: {base_path}")


def to_md(local_files_path: str) -> List[str]:

    base_path = Path(local_files_path).expanduser().resolve()

    converter = PdfConverter(
        artifact_dict=create_model_dict(),
    )

    target_files = _iter_target_files(base_path)
    md_list: List[str] = []

    for path in target_files:
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            md = _pdf_to_md(path, converter)
        else:
            md = _text_like_to_md(path)

        md_list.append(md)

    return md_list
