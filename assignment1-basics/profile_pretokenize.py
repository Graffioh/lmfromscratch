"""Single-chunk cProfile harness for pretokenize.

Runs the EXISTING pretokenize path (find_chunk_boundaries + pretokenize_worker)
on ONE chunk, in-process, with NO multiprocessing.Pool — so cProfile actually
sees the work (cProfile only profiles the process it is launched in, and Pool
runs the real work in child processes it can't follow).

Pretokenize is pure CPU regex work, so its per-pretoken cost is identical whether
it runs in a worker or here in the main process. One chunk is enough to see the
hot path.

usage:
  # profile chunk 0, sizing chunks the same way a real n_proc=5 run would
  uv run python profile_pretokenize.py train 5
  # a different chunk, more rows
  uv run python profile_pretokenize.py train 5 --chunk 2 --top 60
"""

import argparse
import cProfile
import io
import pstats
from pathlib import Path

from cs336_basics.bpe_tokenizer.bpe_parallel import find_chunk_boundaries, pretokenize_worker

project_root = Path(__file__).resolve().parent
profiles_dir = project_root / "cs336_basics" / "profiles"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("split", choices=["train", "valid"])
    _ = parser.add_argument(
        "n_proc", type=int, help="how many chunks the real run splits into; chunk size = file_size / n_proc"
    )
    _ = parser.add_argument("--chunk", type=int, default=0, help="which chunk index to profile")
    _ = parser.add_argument("--top", type=int, default=40, help="rows to print, sorted by tottime")
    args = parser.parse_args()

    dataset_file = "TinyStoriesV2-GPT4-train.txt" if args.split == "train" else "TinyStoriesV2-GPT4-valid.txt"
    dataset_path = project_root / "cs336_basics" / "data" / dataset_file
    special_tokens = ["<|endoftext|>"]

    with open(dataset_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, args.n_proc, b"<|endoftext|>")

    start, end = boundaries[args.chunk], boundaries[args.chunk + 1]
    print(f"profiling chunk {args.chunk}/{len(boundaries) - 1}: bytes {start}..{end} ({(end - start) / 1e6:.1f} MB)")

    profiler = cProfile.Profile()
    profiler.enable()
    freq_table = pretokenize_worker(str(dataset_path), start, end, special_tokens)
    profiler.disable()

    print(f"unique pretokens in chunk: {len(freq_table)}")

    s = io.StringIO()
    _ = pstats.Stats(profiler, stream=s).sort_stats("tottime").print_stats(args.top)
    print(s.getvalue())

    profiles_dir.mkdir(exist_ok=True)
    out_path = profiles_dir / f"pretokenize_chunk{args.chunk}_cprofile.txt"
    with open(out_path, "w") as fout:
        _ = fout.write(s.getvalue())
    print(f"wrote {out_path}")
