from typing import Tuple
import torch

import gae_cuda  # isort: skip


def cugae1d_nolp_misalign_func(
    rewards: torch.FloatTensor,
    values: torch.FloatTensor,
    cu_seqlens: torch.IntTensor,
    truncate: torch.BoolTensor,
    gamma: float,
    lam: float,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    return gae_cuda.gae_1d_nolp_misalign(rewards, values, cu_seqlens, truncate, gamma, lam)


def cugae2d_olp_func(
    rewards: torch.FloatTensor,
    values: torch.FloatTensor,
    dones: torch.BoolTensor,
    truncates: torch.BoolTensor,
    gamma: float,
    lam: float,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    done_indices = dones.nonzero()
    num_dones = dones.float().sum(1)
    max_num_dones = int(num_dones.max())
    cu_num_dones = torch.nn.functional.pad(num_dones.cumsum(0), (1, 0), value=0).int()
    is_truncate = truncates[done_indices[:, 0], done_indices[:, 1]]
    return gae_cuda.gae_2d_olp(
        rewards,
        values,
        dones,
        done_indices[:, 1].int(),
        cu_num_dones,
        max_num_dones,
        is_truncate,
        gamma,
        lam,
    )
