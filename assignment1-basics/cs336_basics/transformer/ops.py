import math

import einops
import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_x = torch.max(x, dim=dim, keepdim=True)
    return torch.exp(x - max_x[0]) / torch.sum(torch.exp(x - max_x[0]), dim=dim, keepdim=True)


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


def cross_entropy(o: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    neg_log_likelihood = log_softmax(o, dim=-1)
    losses = neg_log_likelihood[torch.arange(0, o.shape[0]), x]
    return torch.mean(losses)
