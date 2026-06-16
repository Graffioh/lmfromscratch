from math import sqrt

import einops
import torch


class Linear(torch.nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        torch.nn.Module.__init__(self)

        variance: float = 2 / (in_features + out_features)
        std: float = sqrt(variance)
        trunc_normal_init_w = torch.nn.init.trunc_normal_(
            torch.empty(out_features, in_features, dtype=dtype, device=device), mean=0, std=std, a=-3 * std, b=3 * std
        )

        self.W: torch.nn.Parameter = torch.nn.Parameter(trunc_normal_init_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einops.einsum(self.W, x, "out_features in_features, ... in_features -> ... out_features")


class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        torch.nn.Module.__init__(self)

        trunc_normal_init_emb = torch.nn.init.trunc_normal_(
            torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=device),
            mean=0,
            std=1,
            a=-3,
            b=3,
        )

        self.We: torch.nn.Parameter = torch.nn.Parameter(trunc_normal_init_emb)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.We[token_ids]
