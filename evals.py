"""Evaluation helpers for benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os
import re

from datasets import concatenate_datasets, get_dataset_config_names, load_dataset
from huggingface_hub import snapshot_download


@dataclass
class EvalExample:
    example_id: str
    question: str
    answer: Optional[str]
    metadata: dict
    file_path: Optional[str] = None
    level: Optional[str] = None


def extract_final_answer(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"final answer\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def normalize_answer(text: str) -> str:
    if text is None:
        return ""
    text = extract_final_answer(text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def score_prediction(prediction: str, gold: Optional[str]) -> Optional[bool]:
    if gold is None:
        return None
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)
    if not pred_norm or not gold_norm:
        return False
    if pred_norm == gold_norm:
        return True
    if pred_norm in gold_norm or gold_norm in pred_norm:
        return True
    return False


def load_simpleqa() -> list[EvalExample]:
    dataset = load_dataset("OpenEvals/SimpleQA", split="test")
    examples: list[EvalExample] = []
    for idx, row in enumerate(dataset):
        examples.append(
            EvalExample(
                example_id=str(row.get("id", idx)),
                question=str(row.get("problem", "")).strip(),
                answer=str(row.get("answer", "")).strip() if row.get("answer") is not None else None,
                metadata={"metadata": row.get("metadata")},
            )
        )
    return examples


def load_frames() -> list[EvalExample]:
    dataset = load_dataset("google/frames-benchmark", split="test")
    examples: list[EvalExample] = []
    for idx, row in enumerate(dataset):
        examples.append(
            EvalExample(
                example_id=str(row.get("id", idx)),
                question=str(row.get("Prompt", "")).strip(),
                answer=str(row.get("Answer", "")).strip() if row.get("Answer") is not None else None,
                metadata={
                    "reasoning_types": row.get("reasoning_types"),
                    "wiki_links": row.get("wiki_links"),
                },
            )
        )
    return examples


def _select_gaia_configs(repo_id: str, data_dir: str) -> list[str]:

    token = os.getenv("HF_TOKEN")
    configs = get_dataset_config_names(repo_id, token=token)
    
    if not configs:
        data_root = Path(data_dir)
        for candidate in ["2023", "2023_level1", "2023_level2", "2023_level3"]:
            if (data_root / candidate).exists():
                configs.append(candidate)

    if "2023" in configs:
        return ["2023"]

    level_configs = [c for c in ["2023_level1", "2023_level2", "2023_level3"] if c in configs]
    if level_configs:
        return level_configs

    return configs[:1]


def _load_gaia_split(data_dir: str, config: str):
    for split in ["validation", "dev", "test"]:
        return load_dataset(data_dir, config, split=split)
    raise RuntimeError(f"No usable split found for GAIA config {config}.")


def load_gaia() -> list[EvalExample]:
    repo_id = "gaia-benchmark/GAIA"
    token = os.getenv("HF_TOKEN")
    data_dir = snapshot_download(repo_id=repo_id, repo_type="dataset", token=token)


    configs = _select_gaia_configs(repo_id, data_dir)
    if not configs:
        raise RuntimeError("Unable to determine GAIA dataset configs.")

    datasets = [_load_gaia_split(data_dir, config) for config in configs]
    dataset = datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)

    examples: list[EvalExample] = []
    for idx, row in enumerate(dataset):
        question = row.get("Question") or row.get("question") or ""
        answer = row.get("Final answer") or row.get("final_answer") or row.get("Answer")
        file_path = row.get("file_path") or row.get("file_name")
        if file_path:
            file_path = str(Path(data_dir) / file_path)
        examples.append(
            EvalExample(
                example_id=str(row.get("task_id", idx)),
                question=str(question).strip(),
                answer=str(answer).strip() if answer is not None else None,
                metadata={"source": row.get("source"), "difficulty": row.get("Level")},
                file_path=file_path,
                level=str(row.get("Level")) if row.get("Level") is not None else None,
            )
        )
    return examples


def load_benchmark(name: str) -> list[EvalExample]:
    name = name.lower().strip()
    if name == "simpleqa":
        return load_simpleqa()
    if name == "frames":
        return load_frames()
    if name == "gaia":
        return load_gaia()
    raise ValueError(f"Unsupported benchmark: {name}")
