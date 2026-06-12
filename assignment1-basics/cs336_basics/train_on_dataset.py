import argparse
import os
import pickle
from pathlib import Path

from cs336_basics.bpe_parallel import train_bpe

parent_path = Path(__file__).resolve().parent


def train_dataset(dataset_path: str | os.PathLike[str], vocab_size: int, special_tokens: list[str]):
    vocab, merges = train_bpe(dataset_path, vocab_size, special_tokens)

    print("******************************")
    print(f"len(vocab)={len(vocab)} n_merges_done={len(merges)}")
    print("******************************")

    output_train_vocab_path = parent_path / "output_train_vocab.pkl"
    with open(output_train_vocab_path, "wb") as f:
        pickle.dump(vocab, f)

    output_train_merges_path = parent_path / "output_train_merges.pkl"
    with open(output_train_merges_path, "wb") as f:
        pickle.dump(merges, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("split", choices=["train", "valid"])
    args = parser.parse_args()

    dataset_file = "TinyStoriesV2-GPT4-train.txt" if args == "train" else "TinyStoriesV2-GPT4-valid.txt"
    dataset_path = parent_path / "data" / dataset_file
    train_dataset(dataset_path, vocab_size=10000, special_tokens=["<|endoftext|>"])
