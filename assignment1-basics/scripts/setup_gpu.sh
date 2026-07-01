#!/usr/bin/env bash
set -euo pipefail

N_PROC="${N_PROC:-$(nproc 2>/dev/null || echo 4)}"
RUN_TRAINING=0
SKIP_DOWNLOAD=0
SKIP_TOKENIZER=0
TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-cs336_basics/transformer/core.py}"

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_gpu.sh [--train] [--skip-download] [--skip-tokenizer] [--train-entrypoint PATH]

Options:
  --train                  Run the training entrypoint after setup.
  --skip-download          Do not download TinyStories if it is missing.
  --skip-tokenizer         Do not train the TinyStories BPE artifacts if missing.
  --train-entrypoint PATH  Python file to compile/run for training.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
  --train)
    RUN_TRAINING=1
    shift
    ;;
  --skip-download)
    SKIP_DOWNLOAD=1
    shift
    ;;
  --skip-tokenizer)
    SKIP_TOKENIZER=1
    shift
    ;;
  --train-entrypoint)
    if [[ $# -lt 2 ]]; then
      echo "--train-entrypoint requires a path."
      usage
      exit 2
    fi
    TRAIN_ENTRYPOINT="$2"
    shift 2
    ;;
  -h | --help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown option: $1"
    usage
    exit 2
    ;;
  esac
done

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it with:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

mkdir -p cs336_basics/data outputs

if [[ ! -f cs336_basics/data/TinyStoriesV2-GPT4-train.txt && "${SKIP_DOWNLOAD}" == "0" ]]; then
  echo "Downloading TinyStories train split..."
  if command -v wget >/dev/null 2>&1; then
    wget -P cs336_basics/data https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
  else
    curl -L \
      -o cs336_basics/data/TinyStoriesV2-GPT4-train.txt \
      https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
  fi
fi

if [[ ! -f cs336_basics/data/TinyStoriesV2-GPT4-train.txt && "${SKIP_DOWNLOAD}" == "1" ]]; then
  echo "TinyStories train split is missing and --skip-download was set."
  if [[ "${SKIP_TOKENIZER}" == "0" || "${RUN_TRAINING}" == "1" ]]; then
    echo "Cannot continue because tokenizer setup or training needs the dataset."
    exit 1
  fi
fi

echo "Syncing Python dependencies from uv.lock..."
uv sync --frozen

if [[ "${SKIP_TOKENIZER}" == "0" && \
  (! -f outputs/output_train_vocab_tinystories.pkl || ! -f outputs/output_train_merges_tinystories.pkl) ]]; then
  echo "Training TinyStories BPE tokenizer with ${N_PROC} process(es)..."
  uv run python scripts/train_bpe_on_dataset.py tinystories train "${N_PROC}"
fi

if [[ ! -f "${TRAIN_ENTRYPOINT}" ]]; then
  echo "Training entrypoint not found: ${TRAIN_ENTRYPOINT}"
  echo "Pass --train-entrypoint PATH or set TRAIN_ENTRYPOINT if your training file lives elsewhere."
  exit 1
fi

echo "Checking ${TRAIN_ENTRYPOINT} syntax..."
uv run python -m py_compile "${TRAIN_ENTRYPOINT}"

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is set for this shell."
else
  echo "WANDB_API_KEY is not set. If W&B asks for auth, run one of:"
  echo "  wandb login"
  echo "  export WANDB_API_KEY=your_key_here"
fi

if [[ "${RUN_TRAINING}" == "1" ]]; then
  echo "Starting training..."
  uv run python "${TRAIN_ENTRYPOINT}" --train
else
  echo "Setup complete. Start training with:"
  echo "  uv run python ${TRAIN_ENTRYPOINT} --train"
  echo "Or rerun this script with:"
  echo "  bash scripts/setup_gpu.sh --train"
fi
