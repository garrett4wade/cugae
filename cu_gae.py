from typing import Tuple
import torch

import gae_cuda  # isort: skip


def cu_gae_1d_func(
    rewards: torch.FloatTensor,
    values: torch.FloatTensor,
    cu_seqlens: torch.IntTensor,
    bootstrap: torch.BoolTensor,
    gamma: float,
    lam: float,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    return gae_cuda.gae_1d(rewards, values, cu_seqlens, bootstrap, gamma, lam)
