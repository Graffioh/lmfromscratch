import argparse
import os
import pickle
from pathlib import Path

from cs336_basics.bpe_tokenizer.bpe_parallel import train_bpe

project_root = Path(__file__).resolve().parent
outputs_path = project_root / "outputs"


def train_dataset(
    dataset_path: str | os.PathLike[str], vocab_size: int, special_tokens: list[str], n_proc_from_args: int
):
    vocab, merges = train_bpe(dataset_path, vocab_size, special_tokens, n_proc_from_args)

    print("******************************")
    print(f"len(vocab)={len(vocab)} n_merges_done={len(merges)}")
    print("******************************")

    outputs_path.mkdir(exist_ok=True)

    output_train_vocab_path = outputs_path / "output_train_vocab.pkl"
    with open(output_train_vocab_path, "wb") as f:
        pickle.dump(vocab, f)

    output_train_merges_path = outputs_path / "output_train_merges.pkl"
    with open(output_train_merges_path, "wb") as f:
        pickle.dump(merges, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("split", choices=["train", "valid"])
    _ = parser.add_argument("n_proc", type=int, choices=range(1, (os.cpu_count() or 1) + 1))
    args = parser.parse_args()

    dataset_file = "TinyStoriesV2-GPT4-train.txt" if args.split == "train" else "TinyStoriesV2-GPT4-valid.txt"
    dataset_path = project_root / "cs336_basics" / "data" / dataset_file
    train_dataset(dataset_path, vocab_size=10000, special_tokens=["<|endoftext|>"], n_proc_from_args=args.n_proc)
