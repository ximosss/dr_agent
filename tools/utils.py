"""Shared utility functions for tools."""


def truncate_text(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, breaking at last whitespace."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.8:
        truncated = truncated[:last_space]
    return truncated + "\n\n[... truncated]"


def clean_whitespace(text: str) -> str:
    """Collapse excessive blank lines to at most one."""
    lines = text.split("\n")
    clean_lines = []
    prev_blank = False
    for line in lines:
        is_blank = not line.strip()
        if is_blank and prev_blank:
            continue
        clean_lines.append(line)
        prev_blank = is_blank
    return "\n".join(clean_lines)
