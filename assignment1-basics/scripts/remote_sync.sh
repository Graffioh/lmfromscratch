#!/usr/bin/env bash
set -euo pipefail

RUN_SETUP=0
RUN_TRAINING=0
DELETE=1
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: bash scripts/remote_sync.sh [options] USER@HOST [REMOTE_DIR]
       bash scripts/remote_sync.sh [options] USER@HOST:/absolute/remote/dir

Sync this repository to a remote GPU with rsync. Generated data, outputs, caches,
the local virtualenv, and Git metadata are intentionally left on their own side.

Options:
  --setup       After syncing, run bash scripts/setup_gpu.sh on the remote.
  --train       After syncing, run bash scripts/setup_gpu.sh --train on the remote.
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

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is not installed locally."
  exit 1
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
  --exclude=/outputs/
)

if [[ "${DELETE}" == "1" ]]; then
  RSYNC_ARGS+=(--delete)
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  RSYNC_ARGS+=(--dry-run)
else
  printf -v QUOTED_REMOTE_DIR '%q' "${REMOTE_DIR}"
  ssh "${REMOTE}" "mkdir -p ${QUOTED_REMOTE_DIR}"
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
  ssh "${REMOTE}" "${REMOTE_SETUP_CMD}"
fi
