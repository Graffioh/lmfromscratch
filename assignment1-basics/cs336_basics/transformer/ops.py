import math
from collections.abc import Iterable

import einops
import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_x = torch.max(x, dim=dim, keepdim=True)
    return torch.exp(x - max_x[0]) / torch.sum(torch.exp(x - max_x[0]), dim=dim, keepdim=True)


# to simplify and stabilize cross entropy calculation
def log_softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_x = torch.max(x, dim=dim, keepdim=True)
    # must do: add back the max
    # why? we want to keep the original value computation not an altered one
    #    so we must add back what we removed, but since we don't have a division that cancels the max subtraction like in normal softmax
    #      we must add it back in the calculation somewhere (with a bit of math you end up with summing the maximum)
    log_exp = max_x[0] + torch.log(torch.sum(torch.exp(x - max_x[0]), dim=dim, keepdim=True))
    return log_exp - x


def attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    # need to ask: for each query, which key should i take in consideration?
    # q_seq_len = k_seq_len = seq_len (is the same as input, since it's not cross attention)
    attn_score = einops.einsum(Q, K, "... q_seq_len d_k, ... k_seq_len d_k -> ... q_seq_len k_seq_len") / math.sqrt(
        Q.shape[-1]
    )

    # no cheating
    masked_attn_score = torch.masked_fill(attn_score, ~mask, float("-inf")) if mask is not None else attn_score

    # get distribution to understand the keys that matters for that query
    attn_score_sm = softmax(masked_attn_score, dim=-1)

    # ok now we know the keys that matters, we must create the answer to the query with its values
    attn = einops.einsum(attn_score_sm, V, "... q_seq_len k_seq_len, ... k_seq_len d_v -> ... q_seq_len d_v")

    return attn


def cross_entropy(o: torch.Tensor, x_idx: torch.Tensor) -> torch.Tensor:
    neg_log_likelihood = log_softmax(o, dim=-1)
    losses = torch.gather(neg_log_likelihood, -1, x_idx.unsqueeze(-1))
    return torch.mean(losses)


def learning_rate_schedule(t: int, lr_max: float, lr_min: float, warmup_iters: int, cosine_iters: int) -> float:
    lr_t = lr_min
    if t < warmup_iters:
        lr_t = (t / warmup_iters) * lr_max
    elif warmup_iters <= t <= cosine_iters:
        lr_t = lr_min + (1 + math.cos((t - warmup_iters) / (cosine_iters - warmup_iters) * math.pi)) / 2 * (
            lr_max - lr_min
        )

    return lr_t


def gradient_clipping(params: Iterable[torch.nn.Parameter], max_norm: float, eps: float = 1e-6):
    g_squared_sum = 0
    for p in params:
        if p.grad is None:
            continue
        g_squared_sum += torch.sum(p.grad.data**2)

    g_l2_norm = math.sqrt(g_squared_sum)

    if g_l2_norm >= max_norm:
        for p in params:
            if p.grad is None:
                continue
            p.grad.data *= max_norm / (g_l2_norm + eps)
