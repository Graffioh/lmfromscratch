import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_x = torch.max(x)
    return torch.exp(x - max_x) / torch.sum(torch.exp(x - max_x), dim=dim, keepdim=True)
