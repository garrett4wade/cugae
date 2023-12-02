from typing import Tuple
import torch

import gae_cuda  # isort: skip


def cu_gae_1d_func(
    rewards: torch.FloatTensor,
    values: torch.FloatTensor,
    cu_seqlens: torch.IntTensor,
    truncate: torch.BoolTensor,
    gamma: float,
    lam: float,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    return gae_cuda.gae_1d(rewards, values, cu_seqlens, truncate, gamma, lam)


def cu_gae_2d_func(
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
    return gae_cuda.gae_2d(
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


if __name__ == "__main__":
    from py_gae import pygae2d

    bs = 3
    seqlen = 100
    gamma = 0.9
    lmbda = 0.5
    torch.random.manual_seed(0)
    rewards = torch.randn(bs, seqlen).cuda()
    values = torch.randn(bs, seqlen + 1).cuda()
    dones = torch.randint(0, 2, (bs, seqlen + 1)).bool().cuda()
    truncates = dones.logical_and(torch.randint(0, 2, (bs, seqlen + 1)).bool().cuda())
    adv, ret = cu_gae_2d_func(rewards, values, dones, truncates, gamma, lmbda)
    py_adv, py_ret = pygae2d(rewards, values, dones, truncates, gamma, lmbda)
    assert torch.allclose(adv, py_adv, atol=1e-5), (adv - py_adv).abs().max()
    assert torch.allclose(ret, py_ret, atol=1e-5), (ret - py_ret).abs().max()
