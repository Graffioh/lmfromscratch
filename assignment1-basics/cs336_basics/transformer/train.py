import json
import os
from pathlib import Path
from typing import Literal

import numpy as np
import torch

import wandb
from cs336_basics.bpe_tokenizer.tokenizer import Tokenizer
from cs336_basics.transformer.layers import TransformerLM
from cs336_basics.transformer.loader import get_input_target_pairs, save_checkpoint
from cs336_basics.transformer.ops import cross_entropy
from cs336_basics.transformer.optim import AdamW


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "cs336_basics" / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs"
CONFIGS_DIR = REPO_ROOT / "cs336_basics" / "transformer" / "configs"


def create_dataset_from_txt():
    bpe_tknzr = Tokenizer.from_files(
        vocab_filepath=str(OUTPUTS_DIR / "output_train_vocab_tinystories.pkl"),
        merges_filepath=str(OUTPUTS_DIR / "output_train_merges_tinystories.pkl"),
        special_tokens=["<|endoftext|>"],
    )
    train_text = ""
    with open(DATA_DIR / "TinyStoriesV2-GPT4-train.txt") as f:
        train_text = f.read()
    tokens = bpe_tknzr.encode(train_text[:10000])
    dataset_dst = DATA_DIR / "smoke-dataset-train.npy"
    np.save(dataset_dst, tokens)

    return dataset_dst


def train(
    train_dataset_src: str | os.PathLike[str],
    valid_dataset_src: str | os.PathLike[str],
    checkpoint_dst: str | os.PathLike[str],
    training_config_src: str | os.PathLike[str],
    model_config_src: str | os.PathLike[str],
    iterations: int = 18,
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

    overall_config = training_config | model_config
    wandb_run = wandb.init(
        entity="logberto-na",
        project="llm-from-scratch-1",
        config=overall_config,
    )

    batch_size = training_config["batch_size"]
    iterations = training_config["iterations"]
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

        wandb_run.log({"loss": loss})

        loss.backward()
        optim.step()

        save_checkpoint(model=model, optimizer=optim, iteration=it, out=Path(checkpoint_dst) / f"iteration-{it}")

    wandb_run.finish()


# throwaway
if __name__ == "__main__":
    dataset_src = create_dataset_from_txt()
    train(
        train_dataset_src=dataset_src,
        valid_dataset_src=dataset_src,
        checkpoint_dst=OUTPUTS_DIR,
        training_config_src=CONFIGS_DIR / "training_config.json",
        model_config_src=CONFIGS_DIR / "model_config.json",
    )
