#!/usr/bin/env bash
set -euo pipefail

# usage: ./profile.sh {cprofile|viztracer} --file SCRIPT [--output OUTPUT]

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
  python -m cProfile -o "${OUTPUT}.prof" "$FILE"
  echo "wrote ${OUTPUT}.prof  ->  python -m pstats ${OUTPUT}.prof  (sort cumulative; stats 20)"
  ;;

viztracer)
  viztracer --output_file "${OUTPUT}.json" -- python "$FILE"
  echo "wrote ${OUTPUT}.json  ->  vizviewer ${OUTPUT}.json"
  ;;

*)
  echo "usage: $0 {cprofile|viztracer} --file SCRIPT [--output OUTPUT]"
  exit 1
  ;;
esac
