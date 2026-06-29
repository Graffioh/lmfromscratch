#!/usr/bin/env bash
set -euo pipefail

# usage: ./run_profiler.sh {cprofile|viztracer} --file SCRIPT [--output OUTPUT] [--test] [-- SCRIPT_ARGS...]
#
# examples:
#   # cprofile, test file (--file is a pytest node id; extra pytest flags after --)
#   ./run_profiler.sh cprofile --file "tests/test_train_bpe.py::test_train_bpe" --output bpe_test_cprofile --test -- -s
#
#   # cprofile, non-test script (script args after --)
#   ./run_profiler.sh cprofile --file train_bpe_on_dataset.py --output bpe_train_cprofile -- train 8
#
#   # viztracer, test file
#   ./run_profiler.sh viztracer --file "tests/test_train_bpe.py::test_train_bpe" --output bpe_test_viztrace --test
#
#   # viztracer, non-test script
#   ./run_profiler.sh viztracer --file train_bpe_on_dataset.py --output bpe_train_viztrace -- train 8
#
#   # then open the viztracer result:  vizviewer cs336_basics/profiles/bpe_train_viztrace.json

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILES_DIR="${SCRIPT_DIR}/cs336_basics/profiles"

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
  --)
    shift
    break
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
    python -m cProfile -s tottime -m pytest "$FILE" "$@" >"${PROFILES_DIR}/${OUTPUT}.txt"
  else
    python -m cProfile -s tottime "$FILE" "$@" >"${PROFILES_DIR}/${OUTPUT}.txt"
  fi
  echo "wrote ${PROFILES_DIR}/${OUTPUT}.txt"
  ;;

viztracer)
  if [[ "$IS_TEST" -eq 1 ]]; then
    viztracer --output_file "${PROFILES_DIR}/${OUTPUT}.json" -m pytest "$FILE" "$@"
  else
    viztracer --output_file "${PROFILES_DIR}/${OUTPUT}.json" "$FILE" "$@"
  fi
  ;;

*)
  echo "usage: $0 {cprofile|viztracer} --file SCRIPT [--output OUTPUT] [--test] [-- SCRIPT_ARGS...]"
  exit 1
  ;;
esac
