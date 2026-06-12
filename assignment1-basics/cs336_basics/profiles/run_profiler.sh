#!/usr/bin/env bash
set -euo pipefail

# usage: ./run_profiler.sh {cprofile|viztracer} --file SCRIPT [--output OUTPUT] [--test]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROFILER="${1:-}"
OUTPUT="profile"
FILE=""
IS_TEST=0
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
  --test)
    IS_TEST=1
    shift
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
  if [[ "$IS_TEST" -eq 1 ]]; then
    python -m cProfile -s tottime -m pytest "$FILE" >"${SCRIPT_DIR}/${OUTPUT}.txt"
  else
    python -m cProfile -s tottime "$FILE" >"${SCRIPT_DIR}/${OUTPUT}.txt"
  fi
  echo "wrote ${SCRIPT_DIR}/${OUTPUT}.txt"
  ;;

viztracer)
  if [[ "$IS_TEST" -eq 1 ]]; then
    viztracer --output_file "${SCRIPT_DIR}/${OUTPUT}.json" -m pytest "$FILE"
  else
    viztracer --output_file "${SCRIPT_DIR}/${OUTPUT}.json" "$FILE"
  fi
  ;;

*)
  echo "usage: $0 {cprofile|viztracer} --file SCRIPT [--output OUTPUT] [--test]"
  exit 1
  ;;
esac
