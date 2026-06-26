import math
from collections.abc import Callable

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=1e-1, betas=(0.9, 0.95), eps=1e-8):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr, weight_decay, betas, eps = (
                group["lr"],
                group["weight_decay"],
                group["betas"],
                group["eps"],
            )

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", 0)
                v = state.get("v", 0)

                grad = p.grad.data
                beta1, beta2 = betas

                lr_t = lr * (math.sqrt(1 - beta2**t)) / (1 - beta1**t)
                p.data -= lr * weight_decay * p.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad**2)
                p.data -= lr_t * (m / (torch.sqrt(v) + eps))

                state["m"] = m
                state["v"] = v
                state["t"] = t + 1

        return loss
