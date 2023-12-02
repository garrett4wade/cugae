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


def cugae2d_nolp_func(
    rewards: torch.FloatTensor,
    values: torch.FloatTensor,
    on_reset: torch.BoolTensor,
    trunates: torch.BoolTensor,
    gamma: float,
    lam: float,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    on_reset_indices = on_reset.nonzero()
    num_resets = on_reset.float().sum(1)
    max_num_resets = int(num_resets.max())
    cu_num_resets = torch.nn.functional.pad(num_resets.cumsum(0), (1, 0), value=0).int()
    trunates = torch.cat([torch.zeros_like(trunates[:, 0:1]), trunates[:, :-1]], dim=1)
    bootstrap = trunates[on_reset_indices[:, 0], on_reset_indices[:, 1]]
    return gae_cuda.gae_2d_nolp(
        rewards,
        values,
        on_reset,
        on_reset_indices[:, 1].int(),
        cu_num_resets,
        max_num_resets,
        bootstrap,
        gamma,
        lam,
    )
