#!/usr/bin/env bash
set -euo pipefail

RUN_SETUP=0
RUN_TRAINING=0
PULL_CHECKPOINTS=0
DELETE=1
DRY_RUN=0
SSH_KEY=""

usage() {
  cat <<'EOF'
Usage: bash scripts/remote_sync.sh [options] USER@HOST [REMOTE_DIR]
       bash scripts/remote_sync.sh [options] USER@HOST:/absolute/remote/dir

Sync this repository to a remote GPU with rsync. outputs/ is only for tokenizer
vocab/merge artifacts and is synced to the GPU. checkpoints/ is only for model
checkpoints and is never pushed to the GPU; use --pull-checkpoints to copy it
back from the remote.

Options:
  --setup       After syncing, run bash scripts/setup_gpu.sh on the remote.
  --train       After syncing, run bash scripts/setup_gpu.sh --train on the remote.
  --pull-checkpoints
                Copy remote checkpoints/ into local checkpoints/.
  --ssh-key PATH
                SSH private key to use for ssh and rsync.
  --dry-run     Show what rsync would transfer.
  --no-delete   Do not delete stale remote files.
EOF
}

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
  --setup)
    RUN_SETUP=1
    shift
    ;;
  --train)
    RUN_SETUP=1
    RUN_TRAINING=1
    shift
    ;;
  --pull-checkpoints)
    PULL_CHECKPOINTS=1
    shift
    ;;
  --ssh-key)
    if [[ $# -lt 2 ]]; then
      echo "--ssh-key requires a path."
      usage
      exit 2
    fi
    SSH_KEY="$2"
    shift 2
    ;;
  --dry-run)
    DRY_RUN=1
    shift
    ;;
  --no-delete)
    DELETE=0
    shift
    ;;
  -h | --help)
    usage
    exit 0
    ;;
  --*)
    echo "Unknown option: $1"
    usage
    exit 2
    ;;
  *)
    POSITIONAL+=("$1")
    shift
    ;;
  esac
done

if [[ "${#POSITIONAL[@]}" -lt 1 || "${#POSITIONAL[@]}" -gt 2 ]]; then
  usage
  exit 2
fi

cd "$(dirname "$0")/.."

REMOTE="${POSITIONAL[0]}"
REMOTE_DIR="${POSITIONAL[1]:-/workspace/assignment1-basics}"

if [[ "${REMOTE}" == *:* && "${#POSITIONAL[@]}" == "1" ]]; then
  REMOTE_DIR="${REMOTE#*:}"
  REMOTE="${REMOTE%%:*}"
fi

TARGET="${REMOTE}:${REMOTE_DIR%/}/"
SSH_ARGS=()
RSYNC_SSH_ARGS=()

if [[ -n "${SSH_KEY}" ]]; then
  SSH_ARGS+=(-i "${SSH_KEY}" -o IdentitiesOnly=yes)
  RSYNC_SSH_ARGS=(-e "ssh -i ${SSH_KEY} -o IdentitiesOnly=yes")
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is not installed locally."
  exit 1
fi

if [[ "${PULL_CHECKPOINTS}" == "1" ]]; then
  if [[ "${RUN_SETUP}" == "1" || "${RUN_TRAINING}" == "1" ]]; then
    echo "--pull-checkpoints cannot be combined with --setup or --train."
    exit 2
  fi

  CHECKPOINT_RSYNC_ARGS=(
    -az
    --human-readable
    --stats
    --progress
    --partial
    --exclude='*vocab*'
    --exclude='*merge*'
  )

  if [[ "${#RSYNC_SSH_ARGS[@]}" -gt 0 ]]; then
    CHECKPOINT_RSYNC_ARGS+=("${RSYNC_SSH_ARGS[@]}")
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    CHECKPOINT_RSYNC_ARGS+=(--dry-run)
  else
    mkdir -p checkpoints
  fi

  rsync "${CHECKPOINT_RSYNC_ARGS[@]}" "${REMOTE}:${REMOTE_DIR%/}/checkpoints/" ./checkpoints/
  exit 0
fi

RSYNC_ARGS=(
  -az
  --human-readable
  --stats
  --progress
  --exclude=/.git/
  --exclude=/.venv/
  --exclude=/.ruff_cache/
  --exclude=/.pytest_cache/
  --exclude='__pycache__/'
  --exclude='*.pyc'
  --exclude=/cs336_basics/data/
  --exclude=/checkpoints/
  --include=/outputs/
  --include='/outputs/*vocab*'
  --include='/outputs/*merge*'
  --exclude=/outputs/**
)

if [[ "${DELETE}" == "1" ]]; then
  RSYNC_ARGS+=(--delete)
fi

if [[ "${#RSYNC_SSH_ARGS[@]}" -gt 0 ]]; then
  RSYNC_ARGS+=("${RSYNC_SSH_ARGS[@]}")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  RSYNC_ARGS+=(--dry-run)
else
  printf -v QUOTED_REMOTE_DIR '%q' "${REMOTE_DIR}"
  ssh "${SSH_ARGS[@]}" "${REMOTE}" "mkdir -p ${QUOTED_REMOTE_DIR}"
fi

rsync "${RSYNC_ARGS[@]}" ./ "${TARGET}"

if [[ "${RUN_SETUP}" == "1" ]]; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Dry run: skipping remote setup."
    exit 0
  fi

  printf -v QUOTED_REMOTE_DIR '%q' "${REMOTE_DIR}"
  REMOTE_SETUP_CMD="cd ${QUOTED_REMOTE_DIR} && bash scripts/setup_gpu.sh"
  if [[ "${RUN_TRAINING}" == "1" ]]; then
    REMOTE_SETUP_CMD="${REMOTE_SETUP_CMD} --train"
  fi
  ssh "${SSH_ARGS[@]}" "${REMOTE}" "${REMOTE_SETUP_CMD}"
fi
