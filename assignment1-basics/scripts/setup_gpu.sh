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
TRAIN_DATASET="cs336_basics/data/TinyStoriesV2-GPT4-train.txt"
VALID_DATASET="cs336_basics/data/TinyStoriesV2-GPT4-valid.txt"
TRAIN_TOKEN_DATASET="cs336_basics/data/ts-train-dataset.npy"
VALID_TOKEN_DATASET="cs336_basics/data/ts-valid-dataset.npy"
HF_CHECKPOINT_REPO="Graffioh/lmfromscratch-checkpoints"

# Run from the repo root regardless of where the script was invoked from.
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it with:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

mkdir -p cs336_basics/data outputs checkpoints

# --frozen installs exactly what uv.lock pins and fails if the lockfile is
# stale, so the remote env cannot silently drift from the local one.
echo "Syncing Python dependencies from uv.lock..."
uv sync --frozen

if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "Logging in to Hugging Face from HF_TOKEN..."
  if ! uv run hf auth login --token "${HF_TOKEN}" --add-to-git-credential; then
    echo "Could not log in with uv run hf; Hugging Face libraries will still read HF_TOKEN directly."
  fi
fi

if [[ ! -f "${TRAIN_TOKEN_DATASET}" || ! -f "${VALID_TOKEN_DATASET}" ]]; then
  echo "Downloading pre-tokenized datasets from Hugging Face..."
  if uv run hf download "${HF_CHECKPOINT_REPO}" \
    ts-train-dataset.npy \
    ts-valid-dataset.npy \
    --local-dir cs336_basics/data; then
    echo "Pre-tokenized datasets are ready."
  else
    echo "Could not download pre-tokenized datasets with uv run hf; will create them locally."
  fi
fi

# The raw text is neither in git nor in the docker image (~2GB). It is only
# needed when pre-tokenized arrays are unavailable or BPE artifacts must be
# regenerated.
NEEDS_RAW_TEXT=0
if [[ ! -f "${TRAIN_TOKEN_DATASET}" || ! -f "${VALID_TOKEN_DATASET}" ]]; then
  NEEDS_RAW_TEXT=1
elif [[ ! -f outputs/output_train_vocab_tinystories.pkl || ! -f outputs/output_train_merges_tinystories.pkl ]]; then
  NEEDS_RAW_TEXT=1
fi

if [[ "${NEEDS_RAW_TEXT}" == "1" && ! -f "${TRAIN_DATASET}" ]]; then
  echo "Downloading TinyStories train split..."
  curl -L -o "${TRAIN_DATASET}" \
    https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
fi

if [[ "${NEEDS_RAW_TEXT}" == "1" && ! -f "${VALID_DATASET}" ]]; then
  echo "Downloading TinyStories validation split..."
  curl -L -o "${VALID_DATASET}" \
    https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
fi

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
