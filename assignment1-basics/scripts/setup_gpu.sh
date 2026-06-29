#!/usr/bin/env bash
set -euo pipefail

N_PROC="${N_PROC:-$(nproc 2>/dev/null || echo 4)}"
RUN_TRAINING=0

if [[ "${1:-}" == "--train" ]]; then
  RUN_TRAINING=1
fi

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it with:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

mkdir -p cs336_basics/data outputs

if [[ ! -f cs336_basics/data/TinyStoriesV2-GPT4-train.txt ]]; then
  echo "Downloading TinyStories train split..."
  if command -v wget >/dev/null 2>&1; then
    wget -P cs336_basics/data https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
  else
    curl -L \
      -o cs336_basics/data/TinyStoriesV2-GPT4-train.txt \
      https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
  fi
fi

echo "Syncing Python dependencies..."
uv sync

if [[ ! -f outputs/output_train_vocab_tinystories.pkl || ! -f outputs/output_train_merges_tinystories.pkl ]]; then
  echo "Training TinyStories BPE tokenizer with ${N_PROC} process(es)..."
  uv run train_bpe_on_dataset.py tinystories train "${N_PROC}"
fi

echo "Checking train.py syntax..."
uv run python -m py_compile cs336_basics/transformer/train.py

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is set for this shell."
else
  echo "WANDB_API_KEY is not set. If W&B asks for auth, run one of:"
  echo "  wandb login"
  echo "  export WANDB_API_KEY=your_key_here"
fi

if [[ "${RUN_TRAINING}" == "1" ]]; then
  echo "Starting training..."
  uv run cs336_basics/transformer/train.py
else
  echo "Setup complete. Start training with:"
  echo "  uv run cs336_basics/transformer/train.py"
  echo "Or rerun this script with:"
  echo "  bash scripts/setup_gpu.sh --train"
fi
