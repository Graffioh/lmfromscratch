import math

import einops
import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_x = torch.max(x)
    return torch.exp(x - max_x) / torch.sum(torch.exp(x - max_x), dim=dim, keepdim=True)


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
