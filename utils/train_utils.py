from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np
import torch


class Queue:
    def __init__(self, max_len: int = 100) -> None:
        self.max_len = max_len
        self.data: deque[float] = deque(maxlen=max_len)

    def add(self, value: float) -> None:
        self.data.append(float(value))

    def mean(self) -> float:
        if not self.data:
            return 0.0
        return float(np.mean(self.data))

    def std(self) -> float:
        if len(self.data) <= 1:
            return 0.0
        return float(np.std(self.data))


def get_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    grads = [
        param.grad.detach().norm(2)
        for param in parameters
        if param.grad is not None
    ]
    if not grads:
        return torch.tensor(0.0)
    return torch.norm(torch.stack(grads), 2)
