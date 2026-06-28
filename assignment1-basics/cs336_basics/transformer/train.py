import json
import os
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from cs336_basics.bpe_tokenizer.tokenizer import Tokenizer
from cs336_basics.transformer.layers import TransformerLM
from cs336_basics.transformer.loader import get_input_target_pairs, save_checkpoint
from cs336_basics.transformer.ops import cross_entropy
from cs336_basics.transformer.optim import AdamW


def create_dataset_from_txt():
    bpe_tknzr = Tokenizer.from_files(
        vocab_filepath="/Users/ubreglia/Desktop/lmfromscratch/assignment1-basics/outputs/output_train_vocab_tinystories.pkl",
        merges_filepath="/Users/ubreglia/Desktop/lmfromscratch/assignment1-basics/outputs/output_train_merges_tinystories.pkl",
        special_tokens=["<|endoftext|>"],
    )
    train_text = ""
    with open(
        "/Users/ubreglia/Desktop/lmfromscratch/assignment1-basics/cs336_basics/data/TinyStoriesV2-GPT4-train.txt"
    ) as f:
        train_text = f.read()
    tokens = bpe_tknzr.encode(train_text[:1600])
    dataset_dst = "/Users/ubreglia/Desktop/lmfromscratch/assignment1-basics/cs336_basics/data/smoke-dataset-train.npy"
    np.save(dataset_dst, tokens)

    return dataset_dst


def train(
    train_dataset_src: str | os.PathLike[str],
    valid_dataset_src: str | os.PathLike[str],
    checkpoint_dst: str | os.PathLike[str],
    training_config_src: str | os.PathLike[str],
    model_config_src: str | os.PathLike[str],
    iterations: int = 10,
    split: Literal["train"] | Literal["valid"] = "train",
    device: torch.device | None = None,
):

    torch.autograd.set_detect_anomaly(True, check_nan=False)

    if not device:
        device = torch.device("mps")

    train_dataset = np.load(train_dataset_src, mmap_mode="r")
    valid_dataset = np.load(valid_dataset_src, mmap_mode="r")

    with open(training_config_src) as f:
        training_config = json.load(f)

    with open(model_config_src) as f:
        model_config = json.load(f)

    ctx_len = model_config["context_length"]
    token_positions = torch.arange(0, ctx_len, device=device)
    model = TransformerLM(
        vocab_size=model_config["vocab_size"],
        d_model=model_config["d_model"],
        num_heads=model_config["num_heads"],
        d_ff=model_config["d_ff"],
        theta=model_config["rope_theta"],
        context_length=ctx_len,
        token_positions=token_positions,
        num_layers=model_config["num_layers"],
    ).to(device)

    optim = AdamW(
        model.parameters(),
        lr=training_config["lr"],
        weight_decay=training_config["weight_decay"],
        betas=(training_config["beta1"], training_config["beta2"]),
        eps=training_config["eps"],
    )

    batch_size = training_config["batch_size"]

    model.train()
    for it in range(iterations):
        input_batch, target = get_input_target_pairs(
            train_dataset if split == "train" else valid_dataset,
            batch_size,
            ctx_len,
            device_str=str(device),
        )

        optim.zero_grad()

        loss = cross_entropy(model(input_batch), target)

        print(f"LOSS it={it} -> ", loss.to(device).item())

        loss.backward()
        optim.step()

        save_checkpoint(model=model, optimizer=optim, iteration=it, out=Path(checkpoint_dst) / f"iteration-{it}")


# throwaway
if __name__ == "__main__":
    dataset_src = create_dataset_from_txt()
    train(
        train_dataset_src=dataset_src,
        valid_dataset_src=dataset_src,
        checkpoint_dst="/Users/ubreglia/Desktop/lmfromscratch/assignment1-basics/outputs/",
        training_config_src="/Users/ubreglia/Desktop/lmfromscratch/assignment1-basics/cs336_basics/transformer/configs/training_config.json",
        model_config_src="/Users/ubreglia/Desktop/lmfromscratch/assignment1-basics/cs336_basics/transformer/configs/model_config.json",
    )
