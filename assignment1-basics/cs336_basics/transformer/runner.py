import argparse
import json
from pathlib import Path

from cs336_basics.transformer.core import (
    CONFIGS_DIR,
    DATA_DIR,
    create_dataset_from_file_txt,
    generate,
    train,
    train_bpe,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BPE_CONFIG_SRC = CONFIGS_DIR / "bpe_config.json"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"


def run_train_command(_: argparse.Namespace) -> None:
    dataset_src = create_dataset_from_file_txt()
    train(
        train_dataset_src=dataset_src,
        valid_dataset_src=dataset_src,
        checkpoint_dst=CHECKPOINTS_DIR,
        training_config_src=CONFIGS_DIR / "training_config.json",
        model_config_src=CONFIGS_DIR / "model_config.json",
    )


def run_decode_command(args: argparse.Namespace) -> None:
    generate(
        args.prompt,
        max_tokens=args.max_tokens,
        model_config_src=CONFIGS_DIR / "model_config.json",
        checkpoint_src=args.checkpoint,
    )


def run_train_bpe_command(args: argparse.Namespace) -> None:
    with open(BPE_CONFIG_SRC) as f:
        bpe_config = json.load(f)

    dataset = bpe_config["dataset"]

    if dataset == "tinystories":
        dataset_file = "TinyStoriesV2-GPT4-train.txt" if args.split == "train" else "TinyStoriesV2-GPT4-valid.txt"
    else:
        dataset_file = "owt_train.txt" if args.split == "train" else "owt_valid.txt"

    train_bpe(
        DATA_DIR / dataset_file,
        vocab_size=bpe_config["vocab_size"],
        special_tokens=bpe_config["special_tokens"],
        n_proc_from_args=bpe_config["n_proc"],
        chosen_dataset=dataset,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train, decode, or train BPE artifacts for the CS336 basics model.")
    subparsers = parser.add_subparsers(dest="command", metavar="command", required=True)

    # TRAIN
    train_parser = subparsers.add_parser("train", help="Run transformer training.")
    train_parser.set_defaults(func=run_train_command)

    # DECODE
    decode_parser = subparsers.add_parser("decode", help="Generate text from a prompt.")
    decode_parser.add_argument("prompt", help="Prompt text to continue.")
    decode_parser.add_argument("--max-tokens", type=int, default=5, help="Maximum number of tokens to generate.")
    decode_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINTS_DIR / "ckp-iteration-10",
        help="Checkpoint file to load.",
    )
    decode_parser.set_defaults(func=run_decode_command)

    # BPE TRAIN
    train_bpe_parser = subparsers.add_parser("train-bpe", help="Train BPE vocab and merges artifacts.")
    train_bpe_parser.add_argument("split", choices=["train", "valid"], help="Dataset split to read.")
    train_bpe_parser.set_defaults(func=run_train_bpe_command)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.func(args)
