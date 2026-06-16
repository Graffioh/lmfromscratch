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
