import os
import typing

import numpy as np
import torch


def get_input_target_pairs(
    x: np.ndarray, batch_size: int, context_length: int, device_str: str = "mps"
) -> tuple[torch.Tensor, torch.Tensor]:
    # randomly pick context_length tokens in batches
    # and
    # construct next-tokens batches from the random picked input batches
    x_batches = []
    next_tokens_batches = []
    starts = np.random.randint(0, len(x) - context_length, size=batch_size)
    for i in starts:
        x_batches.append(x[i : context_length + i])
        next_tokens_batches.append(x[i + 1 : context_length + i + 1])

    in_seq = torch.from_numpy(np.stack(x_batches)).to(torch.device(device_str))
    next_tokens = torch.from_numpy(np.stack(next_tokens_batches)).to(torch.device(device_str))

    return (in_seq, next_tokens)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike[str] | typing.BinaryIO | typing.IO[bytes],
):
    model_state = model.state_dict()
    optim_state = optimizer.state_dict()

    out_obj = {"model_state": model_state, "optim_state": optim_state, "iteration": iteration}
    torch.save(out_obj, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optim_state"])
    return checkpoint["iteration"]
