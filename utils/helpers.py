import re
from pathlib import Path
import sys
from loguru import logger


def strip_think_block(text: str) -> str:
    pattern = r"<think>[\s\S]*?</think>\s*"
    return re.sub(pattern, "", text)

def setup_loguru(log_dir: str, level: str = "INFO"):
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger.remove()

    logger.add(sys.stderr, level=level, backtrace=True, diagnose=True)

    logger.add(
        str(Path(log_dir) / "train_{time:YYYYMMDD_HHmmss}.log"),
        level=level,
        rotation="50 MB",
        retention=5,
        compression="zip",
        enqueue=True,
        backtrace=True,
        diagnose=True,
        mode="a",
    )
    return logger
