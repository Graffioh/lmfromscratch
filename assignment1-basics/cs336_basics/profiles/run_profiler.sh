#!/usr/bin/env bash
set -euo pipefail

# usage: ./run_profiler.sh {cprofile|viztracer} --file SCRIPT [--output OUTPUT]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROFILER="${1:-}"
OUTPUT="profile"
FILE=""
shift || true

while [[ $# -gt 0 ]]; do
  case "$1" in
  --file)
    FILE="$2"
    shift 2
    ;;
  --output)
    OUTPUT="$2"
    shift 2
    ;;
  *)
    echo "unknown arg: $1"
    exit 1
    ;;
  esac
done

if [[ -z "$FILE" ]]; then
  echo "error: --file SCRIPT is required"
  exit 1
fi

case "$PROFILER" in
cprofile)
  python -m cProfile -s tottime -m pytest "$FILE" >"${SCRIPT_DIR}/${OUTPUT}.txt"
  echo "wrote ${SCRIPT_DIR}/${OUTPUT}.txt"
  ;;

viztracer)
  viztracer --output_file "${SCRIPT_DIR}/${OUTPUT}.json" -m pytest "$FILE"
  ;;

*)
  echo "usage: $0 {cprofile|viztracer} --file SCRIPT [--output OUTPUT]"
  exit 1
  ;;
esac
