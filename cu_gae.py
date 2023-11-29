from typing import Tuple
import torch

import gae_cuda  # isort: skip


def cu_gae_func(
    rewards: torch.FloatTensor,
    values: torch.FloatTensor,
    cu_seqlens: torch.IntTensor,
    gamma: float,
    lam: float,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    return gae_cuda.gae(rewards, values, cu_seqlens, gamma, lam)
