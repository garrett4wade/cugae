from typing import Tuple
import time
import cu_gae
import torch
import pytest


def py_gae(
    rewards: torch.FloatTensor,
    values: torch.FloatTensor,
    cu_seqlens: torch.IntTensor,
    gamma: float,
    lam: float,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    input_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    v_cu_seqlens = torch.nn.functional.pad((input_lens + 1).cumsum(0), (1, 0)).int()

    bs = cu_seqlens.shape[0] - 1
    assert values.shape[0] == rewards.shape[0] + bs
    advantages_reversed = []
    returns_reversed = []
    for i in reversed(range(bs)):
        v_offset = v_cu_seqlens[i]
        r_offset, r_end = cu_seqlens[i], cu_seqlens[i + 1]
        assert v_cu_seqlens[i + 1] - v_offset - 1 == r_end - r_offset
        lastgaelam = 0
        for t in reversed(range(r_end - r_offset)):
            nextvalues = values[v_offset + t + 1]
            delta = rewards[r_offset + t] + gamma * nextvalues - values[v_offset + t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
            returns_reversed.append(lastgaelam + values[v_offset + t])

    advantages = torch.stack(advantages_reversed[::-1])
    returns = torch.stack(returns_reversed[::-1])
    return advantages, returns


@pytest.mark.parametrize("max_seqlen", [32, 128, 512])
@pytest.mark.parametrize("bs", [1, 2, 4, 8, 16])
def test_gae(max_seqlen: int, bs: int):
    seqlens = torch.randint(0, max_seqlen, (bs,), dtype=torch.int32, device="cuda")
    rewards = torch.randn(seqlens.sum(), dtype=torch.float32, device="cuda")
    values = torch.randn(seqlens.sum() + bs, dtype=torch.float32, device="cuda")
    cu_seqlens = torch.nn.functional.pad(seqlens.cumsum(0), (1, 0)).int()

    adv, ret = cu_gae.cu_gae_func(rewards, values, cu_seqlens, 1.0, 1.0)
    py_adv, py_ret = py_gae(rewards, values, cu_seqlens, 1.0, 1.0)

    t1 = time.perf_counter_ns()
    py_adv, py_ret = py_gae(rewards, values, cu_seqlens, 1.0, 1.0)
    t2 = time.perf_counter_ns()
    adv, ret = cu_gae.cu_gae_func(rewards, values, cu_seqlens, 1.0, 1.0)
    t3 = time.perf_counter_ns()
    assert torch.allclose(adv, py_adv), (adv - py_adv).abs().max()
    assert torch.allclose(ret, py_ret), (ret - py_ret).abs().max()

    print(f"max_seqlen={max_seqlen},bs={bs}, flash acceleration", (t2 - t1) / (t3 - t2))
