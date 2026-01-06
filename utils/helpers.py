import re

def strip_think_block(text: str) -> str:
    pattern = r"<think>[\s\S]*?</think>\s*"
    return re.sub(pattern, "", text)
