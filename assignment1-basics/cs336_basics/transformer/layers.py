from math import sqrt
from typing import override

import einops
import torch


class Linear(torch.nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        torch.nn.Module.__init__(self)

        # Xavier-style init
        variance: float = 2 / (in_features + out_features)
        std: float = sqrt(variance)
        trunc_normal_init_w = torch.nn.init.trunc_normal_(
            torch.empty((out_features, in_features), dtype=dtype, device=device), mean=0, std=std, a=-3 * std, b=3 * std
        )

        self.W = torch.nn.Parameter(trunc_normal_init_w)

    @override
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
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype),
            mean=0,
            std=1,
            a=-3,
            b=3,
        )

        self.We = torch.nn.Parameter(trunc_normal_init_emb)

    @override
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.We[token_ids]


class RMSNorm(torch.nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        torch.nn.Module.__init__(self)

        self.G = torch.nn.Parameter(torch.zeros(d_model, device=device, dtype=dtype))
        self.eps = eps

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        _, _, d_model = x.shape

        in_rms = (1 / d_model) * torch.sum(torch.pow(x, 2), dim=-1, keepdim=True) + self.eps
        rms = torch.sqrt(in_rms)
        result = (x / rms) * self.G

        return result.to(in_dtype)


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        torch.nn.Module.__init__(self)

        variance: float = 2 / (d_model + d_ff)
        std: float = sqrt(variance)
        trunc_normal_init_up = torch.nn.init.trunc_normal_(
            torch.empty((d_ff, d_model), dtype=dtype, device=device), mean=0, std=std, a=-3 * std, b=3 * std
        )
        trunc_normal_init_gate = torch.nn.init.trunc_normal_(
            torch.empty((d_ff, d_model), dtype=dtype, device=device), mean=0, std=std, a=-3 * std, b=3 * std
        )
        trunc_normal_init_down = torch.nn.init.trunc_normal_(
            torch.empty((d_model, d_ff), dtype=dtype, device=device), mean=0, std=std, a=-3 * std, b=3 * std
        )

        self.Wu = torch.nn.Parameter(trunc_normal_init_up)
        self.Wg = torch.nn.Parameter(trunc_normal_init_gate)
        self.Wd = torch.nn.Parameter(trunc_normal_init_down)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up_proj = einops.einsum(self.Wu, x, "d_ff d_model, ... d_model -> ... d_ff")

        silu = up_proj * torch.sigmoid(up_proj)

        gate_proj = einops.einsum(self.Wg, x, "d_ff d_model, ... d_model -> ... d_ff")

        glu = silu * gate_proj

        down_proj = einops.einsum(self.Wd, glu, "d_model d_ff, ... d_ff -> ... d_model")

        return down_proj
