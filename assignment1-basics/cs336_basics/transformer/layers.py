from math import sqrt
from typing import override

import einops
import torch

from .ops import attention


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


class CausalMultiHeadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        token_positions: torch.Tensor | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        torch.nn.Module.__init__(self)

        variance: float = 2 / (d_model + d_model)
        std: float = sqrt(variance)
        trunc_normal_init_q = torch.nn.init.trunc_normal_(
            torch.empty((d_model, d_model), dtype=dtype, device=device), mean=0, std=std, a=-3 * std, b=3 * std
        )
        trunc_normal_init_k = torch.nn.init.trunc_normal_(
            torch.empty((d_model, d_model), dtype=dtype, device=device), mean=0, std=std, a=-3 * std, b=3 * std
        )
        trunc_normal_init_v = torch.nn.init.trunc_normal_(
            torch.empty((d_model, d_model), dtype=dtype, device=device), mean=0, std=std, a=-3 * std, b=3 * std
        )
        trunc_normal_init_o = torch.nn.init.trunc_normal_(
            torch.empty((d_model, d_model), dtype=dtype, device=device), mean=0, std=std, a=-3 * std, b=3 * std
        )

        self.Wq = torch.nn.Parameter(trunc_normal_init_q)
        self.Wk = torch.nn.Parameter(trunc_normal_init_k)
        self.Wv = torch.nn.Parameter(trunc_normal_init_v)

        self.Wo = torch.nn.Parameter(trunc_normal_init_o)
        self.h = num_heads

        self.rope_l = None
        if theta and max_seq_len and token_positions is not None:
            self.token_positions = token_positions
            self.rope_l = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len, device=device)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool)).to(x.device)
        d_k = d_q = d_v = self.Wk.shape[-2] // self.h

        Q = einops.einsum(self.Wq, x, "d_model in_d_model, ... seq_len in_d_model -> ... seq_len d_model")
        Q = einops.rearrange(Q, "... seq_len (num_heads d_q) -> ... num_heads seq_len d_q", num_heads=self.h, d_q=d_q)
        if self.rope_l:
            Q = self.rope_l.forward(Q, self.token_positions)

        K = einops.einsum(self.Wk, x, "d_model in_d_model, ... seq_len in_d_model -> ... seq_len d_model")
        K = einops.rearrange(K, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.h, d_k=d_k)
        if self.rope_l:
            K = self.rope_l.forward(K, self.token_positions)

        V = einops.einsum(self.Wv, x, "d_model in_d_model, ... seq_len in_d_model -> ... seq_len d_model")
        V = einops.rearrange(V, "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v", num_heads=self.h, d_v=d_v)

        multi_head = attention(Q, K, V, mask)

        Wo = einops.rearrange(self.Wo, "d_model (num_heads d_v) -> num_heads d_v d_model", num_heads=self.h, d_v=d_v)
        return einops.einsum(Wo, multi_head, "num_heads d_v d_model, ... num_heads seq_len d_v -> ... seq_len d_model")
