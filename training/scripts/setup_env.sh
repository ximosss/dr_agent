#!/usr/bin/env bash
set -euo pipefail

echo "=== Step 0: Environment Setup ==="

# 1. Sync the local project venv with training dependencies
echo "[1/2] Syncing .venv with the training dependency group ..."
uv sync --group training --inexact

# 2. Verify
echo "[2/2] Verifying training environment ..."
uv run --group training python -c "
import torch
print(f'torch {torch.__version__}, cuda={torch.cuda.is_available()}, devices={torch.cuda.device_count()}')
import transformers; print(f'transformers {transformers.__version__}')
import peft; print(f'peft {peft.__version__}')
import llamafactory
print(f'llamafactory {llamafactory.__version__}')
"

echo "=== Environment setup complete ==="
