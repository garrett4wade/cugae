from typing import Tuple
import time
import cu_gae
import torch
import pytest


@torch.no_grad()
def get_packed_advantages_and_returns(
    gamma: float,
    lam: float,
    values: torch.FloatTensor,
    rewards: torch.FloatTensor,
    short1cu_seqlens: torch.IntTensor,
    seq_no_eos_mask: torch.FloatTensor,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    cu_seqlens = short1cu_seqlens.clone()
    cu_seqlens[1:] += torch.ones_like(short1cu_seqlens[1:]).cumsum(0)

    bs = short1cu_seqlens.shape[0] - 1
    assert values.shape[0] == rewards.shape[0] + bs
    advantages_reversed = []
    returns_reversed = []
    for i in reversed(range(bs)):
        v_offset = cu_seqlens[i]
        r_offset, r_end = short1cu_seqlens[i], short1cu_seqlens[i + 1]
        assert cu_seqlens[i + 1] - v_offset - 1 == r_end - r_offset
        lastgaelam = 0
        for t in reversed(range(r_end - r_offset)):
            nextvalues = values[v_offset + t + 1]
            if t == r_end - r_offset - 1:
                nextvalues *= seq_no_eos_mask[i]
            delta = rewards[r_offset + t] + gamma * nextvalues - values[v_offset + t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
            returns_reversed.append(lastgaelam + values[v_offset + t])

    advantages = torch.stack(advantages_reversed[::-1])
    returns = torch.stack(returns_reversed[::-1])
    return advantages, returns


@pytest.mark.parametrize("max_seqlen", [32, 128, 512])
@pytest.mark.parametrize("bs", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("gamma", [0.99, 1.0])
@pytest.mark.parametrize("lam", [0.5, 0.9, 0.95])
def test_gae(max_seqlen: int, bs: int, gamma: float, lam: float):
    seqlens = torch.randint(1, max_seqlen, (bs,), dtype=torch.int32, device="cuda")
    rewards = torch.randn(seqlens.sum(), dtype=torch.float32, device="cuda")
    values = torch.randn(seqlens.sum() + bs, dtype=torch.float32, device="cuda")
    bootstrap = torch.ones(bs, dtype=torch.bool, device="cuda")
    cu_seqlens = torch.nn.functional.pad(seqlens.cumsum(0), (1, 0)).int()

    adv, ret = cu_gae.cu_gae_1d_func(rewards, values, cu_seqlens, bootstrap, gamma, lam)
    py_adv, py_ret = get_packed_advantages_and_returns(
        gamma, lam, values, rewards, cu_seqlens, bootstrap.float()
    )

    t1 = time.perf_counter_ns()
    py_adv, py_ret = get_packed_advantages_and_returns(
        gamma, lam, values, rewards, cu_seqlens, bootstrap.float()
    )
    t2 = time.perf_counter_ns()
    adv, ret = cu_gae.cu_gae_1d_func(rewards, values, cu_seqlens, bootstrap, gamma, lam)
    t3 = time.perf_counter_ns()
    assert torch.allclose(adv, py_adv, atol=1e-5), (adv - py_adv).abs().max()
    assert torch.allclose(ret, py_ret, atol=1e-5), (ret - py_ret).abs().max()

    print(f"max_seqlen={max_seqlen},bs={bs}, CUDA acceleration ratio", (t2 - t1) / (t3 - t2))
