"""
Step 7: Merge LoRA adapter into the base model.

Produces a standalone model that can be served by vLLM without any adapter loading.

Usage:
    python training/merge_lora.py [--checkpoint PATH] [--output-dir PATH]

Or via LLaMA-Factory CLI:
    llamafactory-cli export \
        --model_name_or_path /data/qwen3-8b \
        --adapter_name_or_path /data/sft_output/qwen3-8b-dr-agent \
        --template qwen3 \
        --finetuning_type lora \
        --export_dir /data/qwen3-8b-sft-merged \
        --export_size 5 \
        --export_legacy_format false
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


LATEST_ADAPTER_FILE = Path("/home/ubuntu/workspace/dr_agent/training/data/latest_adapter_dir.txt")


def resolve_default_adapter_path() -> str:
    if LATEST_ADAPTER_FILE.exists():
        return LATEST_ADAPTER_FILE.read_text().strip()
    return "/data/sft_output/qwen3-8b-dr-agent"


def merge_lora(
    base_model: str = "/data/qwen3-8b",
    adapter_path: str = resolve_default_adapter_path(),
    output_dir: str = "/data/qwen3-8b-sft-merged",
) -> None:
    """Merge LoRA adapter using llamafactory-cli export."""
    cmd = [
        "llamafactory-cli", "export",
        "--model_name_or_path", base_model,
        "--adapter_name_or_path", adapter_path,
        "--template", "qwen3",
        "--finetuning_type", "lora",
        "--export_dir", output_dir,
        "--export_size", "5",
        "--export_legacy_format", "false",
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    print(f"\nMerged model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA into base model")
    parser.add_argument("--base-model", default="/data/qwen3-8b")
    parser.add_argument("--adapter-path", default=resolve_default_adapter_path())
    parser.add_argument("--output-dir", default="/data/qwen3-8b-sft-merged")
    args = parser.parse_args()
    merge_lora(args.base_model, args.adapter_path, args.output_dir)


if __name__ == "__main__":
    main()
