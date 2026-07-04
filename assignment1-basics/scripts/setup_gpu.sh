#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a fresh GPU instance after git clone: dataset, deps, BPE artifacts.
# Idempotent: every step checks whether its output already exists, so it is
# safe to rerun on every session.
# Usage: bash scripts/setup_gpu.sh [--train]

RUN_TRAINING=0
if [[ "${1:-}" == "--train" ]]; then
  RUN_TRAINING=1
elif [[ $# -gt 0 ]]; then
  echo "Usage: bash scripts/setup_gpu.sh [--train]"
  exit 2
fi

TRAIN_ENTRYPOINT="cs336_basics/transformer/runner.py"
DATASET="cs336_basics/data/TinyStoriesV2-GPT4-train.txt"

# Run from the repo root regardless of where the script was invoked from.
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it with:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

mkdir -p cs336_basics/data outputs checkpoints

# The dataset is neither in git nor in the docker image (~2GB), so fetch it
# on first run.
if [[ ! -f "${DATASET}" ]]; then
  echo "Downloading TinyStories train split..."
  curl -L -o "${DATASET}" \
    https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
fi

# --frozen installs exactly what uv.lock pins and fails if the lockfile is
# stale, so the remote env cannot silently drift from the local one.
echo "Syncing Python dependencies from uv.lock..."
uv sync --frozen

# The canonical BPE artifacts are committed in outputs/, so this normally
# never runs; it is only a fallback if they are deleted.
if [[ ! -f outputs/output_train_vocab_tinystories.pkl || ! -f outputs/output_train_merges_tinystories.pkl ]]; then
  echo "Training BPE tokenizer..."
  uv run python "${TRAIN_ENTRYPOINT}" train-bpe train
fi

# Fail fast before training: under nohup a W&B auth prompt does not fail
# visibly, it hangs the run silently.
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is not set."
  if [[ "${RUN_TRAINING}" == "1" ]]; then
    echo "Set it before training so the run does not hang on W&B auth:"
    echo "  export WANDB_API_KEY=your_key_here"
    exit 1
  fi
fi

if [[ "${RUN_TRAINING}" == "1" ]]; then
  echo "Starting training..."
  uv run python "${TRAIN_ENTRYPOINT}" train
else
  echo "Setup complete. Train with:"
  echo "  bash scripts/setup_gpu.sh --train"
fi
