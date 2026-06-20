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


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        torch.nn.Module.__init__(self)

        positions = [i for i in range(max_seq_len)]
        inv_freq = [(1 / theta ** (2 * k / d_k)) for k in range(d_k // 2)]
        POS = torch.Tensor(positions).to(device)
        IF = torch.Tensor(inv_freq).to(device)

        angles = POS[:, None] * IF[None, :]
        COS = torch.cos(angles)
        SIN = torch.sin(angles)

        # (2, max_seq_len, d_k)
        self.register_buffer("sincos_buf", torch.stack((COS, SIN)), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # split and get pairs along d_k last dimension
        x_pairs = einops.rearrange(x, "... (d_k_pair two) -> ... d_k_pair two", two=2)
        a = x_pairs[..., 0]
        b = x_pairs[..., 1]

        # get cos and sin for rotation
        cos = self.get_buffer("sincos_buf")[0, token_positions, :]
        sin = self.get_buffer("sincos_buf")[1, token_positions, :]

        # rotate pairs
        a_rotated = a * cos - b * sin
        b_rotated = a * sin + b * cos

        # reshape to input shape
        rotated_pairs = torch.stack((a_rotated, b_rotated), dim=-1)
        x_rotated = einops.rearrange(rotated_pairs, "... d_k_pair two -> ... (d_k_pair two)", two=2)

        return x_rotated
