import json
import os
import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from tqdm import tqdm

import wandb
from cs336_basics.bpe_tokenizer.bpe_parallel import train_bpe as train_bpe_impl
from cs336_basics.bpe_tokenizer.tokenizer import Tokenizer
from cs336_basics.transformer.layers import TransformerLM
from cs336_basics.transformer.loader import get_input_target_pairs, load_checkpoint, save_checkpoint
from cs336_basics.transformer.ops import cross_entropy, softmax
from cs336_basics.transformer.optim import AdamW

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "cs336_basics" / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs"
CONFIGS_DIR = REPO_ROOT / "cs336_basics" / "transformer" / "configs"


def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_dataset_from_file_txt(split: Literal["train"] | Literal["valid"]):
    dataset_dst = DATA_DIR / f"ts-{split}-dataset.npy"
    if dataset_dst.is_file():
        print(f"{split} dataset already exists, skipping dataset creation.")
        return dataset_dst

    bpe_tknzr = Tokenizer.from_files(
        vocab_filepath=str(OUTPUTS_DIR / "output_train_vocab_tinystories.pkl"),
        merges_filepath=str(OUTPUTS_DIR / "output_train_merges_tinystories.pkl"),
        special_tokens=["<|endoftext|>"],
    )
    dataset_text = ""
    with open(
        DATA_DIR / "TinyStoriesV2-GPT4-train.txt" if split == "train" else DATA_DIR / "TinyStoriesV2-GPT4-valid.txt"
    ) as f:
        dataset_text = f.read()
    tokens = bpe_tknzr.encode(dataset_text)
    np.save(dataset_dst, tokens)

    return dataset_dst


def train(
    train_dataset_src: str | os.PathLike[str],
    valid_dataset_src: str | os.PathLike[str],
    checkpoint_dst: str | os.PathLike[str],
    training_config_src: str | os.PathLike[str],
    model_config_src: str | os.PathLike[str],
    device: torch.device | None = None,
):
    # for debugging
    torch.autograd.set_detect_anomaly(True, check_nan=False)

    if not device:
        device = get_default_device()

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
    for it in tqdm(range(iterations)):
        input_batch, target = get_input_target_pairs(
            train_dataset,
            batch_size,
            ctx_len,
            device_str=str(device),
        )

        optim.zero_grad()

        train_loss = cross_entropy(model(input_batch), target)

        train_loss.backward()
        optim.step()

        wandb_run.log(
            {
                "loss/train": train_loss.float(),
            },
            step=it,
        )

        validation_batch_size = training_config["validation_batch_size"]
        eval_interval = training_config["eval_interval"]
        if it % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                valid_losses = []
                for _ in range(validation_batch_size):
                    input_batch, target = get_input_target_pairs(
                        valid_dataset,
                        batch_size,
                        ctx_len,
                        device_str=str(device),
                    )

                    cur_valid_loss = cross_entropy(model(input_batch), target)

                    valid_losses.append(cur_valid_loss.float())

                wandb_run.log(
                    {
                        "loss/valid": sum(valid_losses) / validation_batch_size,
                    },
                    step=it,
                )

            save_checkpoint(
                model=model, optimizer=optim, iteration=it, out=Path(checkpoint_dst) / f"ckp-iteration-{it}"
            )

        model.train()

    wandb_run.finish()


def get_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0:
        return probs

    probs_sort = torch.sort(probs, dim=-1, descending=True)
    values, indices = probs_sort

    cum_sum = values.cumsum(dim=-1)
    stop_condition = cum_sum > p
    when_to_stop = stop_condition.float().argmax(dim=-1)
    top_p_mask = torch.arange(probs.size(-1), device=probs.device) <= when_to_stop.unsqueeze(-1)

    values = torch.masked_fill(values, ~top_p_mask, 0.0)
    normalized_values = values / torch.sum(values, dim=-1, keepdim=True)

    res = torch.zeros(probs.shape, dtype=probs.dtype, device=probs.device)
    return torch.scatter(res, -1, indices, normalized_values)


def generate(
    prompt: str,
    max_tokens: int,
    t: float = 0.7,
    top_p: float = 1.0,
    device: torch.device | None = None,
    model_config_src: str | os.PathLike[str] = "",
    checkpoint_src: str | os.PathLike[str] = "",
):
    if not device:
        device = get_default_device()

    with open(model_config_src) as f:
        model_config = json.load(f)

    ctx_len = model_config["context_length"]

    bpe_tknzr = Tokenizer.from_files(
        vocab_filepath=str(OUTPUTS_DIR / "output_train_vocab_tinystories.pkl"),
        merges_filepath=str(OUTPUTS_DIR / "output_train_merges_tinystories.pkl"),
        special_tokens=["<|endoftext|>"],
    )
    prompt_tokens = bpe_tknzr.encode(prompt)

    token_positions = torch.arange(0, len(prompt_tokens), device=device)
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

    x_in = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)

    load_checkpoint(checkpoint_src, model)
    model.eval()

    i = 0
    prediction_ids = []
    while i < max_tokens:
        logits = model(x_in)
        next_token_logits = logits[:, -1, :]
        next_token_distribution = softmax(next_token_logits, dim=-1, temperature=t)
        next_token_top_p = get_top_p(next_token_distribution, p=top_p)

        next_token_sample = torch.multinomial(next_token_top_p, 1)
        next_token_ids = next_token_sample.flatten(0, -1).tolist()
        next_token_str = bpe_tknzr.decode(next_token_ids)

        print(f"SAMPLED TOKEN {i} -> {next_token_str}")

        prediction_ids.extend(next_token_ids)

        if next_token_str == "<|endoftext|>":
            break

        x_in = torch.cat((x_in, next_token_sample), dim=-1)
        i += 1

    final_txt = prompt_tokens + prediction_ids
    print(f"FINAL TEXT = {bpe_tknzr.decode(final_txt)}")


def train_bpe(
    dataset_path: str | os.PathLike[str],
    vocab_size: int,
    special_tokens: list[str],
    n_proc_from_args: int,
    chosen_dataset: str,
):
    vocab, merges = train_bpe_impl(dataset_path, vocab_size, special_tokens, n_proc_from_args)

    print("******************************")
    print(f"len(vocab)={len(vocab)} n_merges_done={len(merges)}")
    print("******************************")

    OUTPUTS_DIR.mkdir(exist_ok=True)

    output_train_vocab_path = OUTPUTS_DIR / f"output_train_vocab_{chosen_dataset}.pkl"
    with open(output_train_vocab_path, "wb") as f:
        pickle.dump(vocab, f)

    output_train_merges_path = OUTPUTS_DIR / f"output_train_merges_{chosen_dataset}.pkl"
    with open(output_train_merges_path, "wb") as f:
        pickle.dump(merges, f)
