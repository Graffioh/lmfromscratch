"""Single-chunk cProfile harness for pretokenize + merge.

Runs the EXISTING pretokenize + merge path on ONE chunk, in-process, with NO
multiprocessing.Pool — so cProfile actually sees the work (cProfile only profiles
the process it is launched in, and Pool runs the real work in child processes it
can't follow).

Pretokenize is pure CPU regex work, so its per-pretoken cost is identical whether
it runs in a worker or here in the main process. One chunk is enough to see the
hot path. Merge runs on that one chunk's frequency table.

usage:
  # profile chunk 0, sizing chunks the same way a real n_proc=5 run would,
  # then run 1000 merges under the profiler
  uv run python profile_pretokenize.py train 5
  # smaller/faster chunk + fewer merges while iterating on merge()
  uv run python profile_pretokenize.py train 20 --merges 300 --top 60
"""

import argparse
import cProfile
import io
import pstats
from pathlib import Path

from cs336_basics.bpe_tokenizer.bpe_parallel import find_chunk_boundaries, merge, pretokenize_worker

project_root = Path(__file__).resolve().parent
profiles_dir = project_root / "cs336_basics" / "profiles"


def dump_stats(profiler: cProfile.Profile, top: int, out_name: str) -> None:
    s = io.StringIO()
    _ = pstats.Stats(profiler, stream=s).sort_stats("tottime").print_stats(top)
    print(s.getvalue())
    profiles_dir.mkdir(exist_ok=True)
    out_path = profiles_dir / out_name
    with open(out_path, "w") as f:
        _ = f.write(s.getvalue())
    print(f"wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("split", choices=["train", "valid"])
    _ = parser.add_argument(
        "n_proc", type=int, help="how many chunks the real run splits into; chunk size = file_size / n_proc"
    )
    _ = parser.add_argument("--chunk", type=int, default=0, help="which chunk index to profile")
    _ = parser.add_argument("--merges", type=int, default=1000, help="number of BPE merges to run under the profiler")
    _ = parser.add_argument("--top", type=int, default=40, help="rows to print, sorted by tottime")
    args = parser.parse_args()

    dataset_file = "TinyStoriesV2-GPT4-train.txt" if args.split == "train" else "TinyStoriesV2-GPT4-valid.txt"
    dataset_path = project_root / "cs336_basics" / "data" / dataset_file
    special_tokens = ["<|endoftext|>"]

    with open(dataset_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, args.n_proc, b"<|endoftext|>")

    start, end = boundaries[args.chunk], boundaries[args.chunk + 1]
    print(f"profiling chunk {args.chunk}/{len(boundaries) - 1}: bytes {start}..{end} ({(end - start) / 1e6:.1f} MB)")

    # --- stage 1: pretokenize ---
    pre_profiler = cProfile.Profile()
    pre_profiler.enable()
    word_freq = pretokenize_worker(str(dataset_path), start, end, special_tokens)
    pre_profiler.disable()
    print(f"unique pretokens in chunk: {len(word_freq)}")
    dump_stats(pre_profiler, args.top, f"pretokenize_chunk{args.chunk}_cprofile.txt")

    # --- stage 2: merge ---
    # NOTE: feeding the string-keyed freq table straight into merge() keeps this
    # harness pure profiling plumbing (no string -> tuple[bytes, ...] bridge added
    # here — that's yours to add in the real pipeline). For TinyStories (~ASCII) a
    # 1-char key and its 1-byte encoding have equal length, so merge()'s per-iter
    # cost and hot spots match the real byte-level run. The OUTPUT is not the
    # byte-level result; we only care about timing. (Pyright will flag the str vs
    # tuple[bytes,...] key type — expected and harmless for the harness.)
    merge_profiler = cProfile.Profile()
    merge_profiler.enable()
    merges, new_vocab_words = merge(word_freq, args.merges)
    merge_profiler.disable()
    print(f"merges done: {len(merges)} (requested {args.merges})")
    dump_stats(merge_profiler, args.top, f"merge_chunk{args.chunk}_cprofile.txt")
