from typing import Tuple
import functools
import itertools
import timeit

import matplotlib.pyplot as plt
import pytest
import torch

from pygae import *
import cugae

gamma = 0.9
lam = 0.5
repeat = 10

torch.random.manual_seed(0)


def cuda_synced(cuda_fn):
    def wrapped_cuda_fn():
        res = cuda_fn()
        torch.cuda.synchronize()
        return res

    return wrapped_cuda_fn


@pytest.mark.parametrize("max_seqlen", [32, 128, 512])
@pytest.mark.parametrize("bs", [2, 4])
def benchmark_gae1d_nolp_misalign(max_seqlen: int, bs: int):
    seqlens = torch.randint(1, max_seqlen, (bs,), dtype=torch.int32, device="cuda")
    rewards = torch.randn(seqlens.sum(), dtype=torch.float32, device="cuda")
    values = torch.randn(seqlens.sum() + bs, dtype=torch.float32, device="cuda")
    bootstrap = torch.ones(bs, dtype=torch.bool, device="cuda")
    cu_seqlens = torch.nn.functional.pad(seqlens.cumsum(0), (1, 0)).int()

    cuda_fn = functools.partial(
        cugae.cugae1d_nolp_misalign_func, rewards, values, cu_seqlens, bootstrap, gamma, lam
    )
    py_fn = functools.partial(pygae1d_nolp_misalign, rewards, values, cu_seqlens, bootstrap, gamma, lam)

    torch.cuda.synchronize()
    cuda_t = timeit.timeit(cuda_synced(cuda_fn), number=repeat)

    torch.cuda.synchronize()
    py_t = timeit.timeit(cuda_synced(py_fn), number=repeat)

    print(f"max_seqlen={max_seqlen},bs={bs}, CUDA acceleration ratio", py_t / cuda_t)
    return py_t / cuda_t


@pytest.mark.parametrize("seqlen", [32, 128, 512, 1024])
@pytest.mark.parametrize("bs", [8, 16, 32, 100])
def test_gae2d_olp(bs: int, seqlen: int):
    rewards = torch.randn(bs, seqlen).cuda()
    values = torch.randn(bs, seqlen + 1).cuda()
    dones = torch.randint(0, 2, (bs, seqlen + 1)).bool().cuda()
    truncates = dones.logical_and(torch.randint(0, 2, (bs, seqlen + 1)).bool().cuda())

    py_fn = functools.partial(pygae2d_olp, rewards, values, dones, truncates, gamma, lam)
    cuda_fn = functools.partial(cugae.cugae2d_olp_func, rewards, values, dones, truncates, gamma, lam)

    torch.cuda.synchronize()
    cuda_t = timeit.timeit(cuda_synced(cuda_fn), number=repeat)

    torch.cuda.synchronize()
    py_t = timeit.timeit(cuda_synced(py_fn), number=repeat)

    print(f"seqlen={seqlen},bs={bs}, CUDA acceleration ratio", py_t / cuda_t)
    return py_t / cuda_t


@pytest.mark.parametrize("seqlen", [32, 128, 512, 1024])
@pytest.mark.parametrize("bs", [8, 16, 32, 100])
def test_gae2d_nolp(bs: int, seqlen: int):
    torch.random.manual_seed(0)
    rewards = torch.randn(bs, seqlen).cuda()
    values = torch.randn(bs, seqlen + 1).cuda()
    on_reset_ = torch.randint(0, 2, (bs, seqlen + 2)).bool().cuda()
    on_reset = on_reset_[:, :-1].contiguous()
    truncates = on_reset_[:, 1:].logical_and(torch.randint(0, 2, (bs, seqlen + 1)).bool().cuda()).contiguous()

    py_fn = functools.partial(pygae2d_nolp, rewards, values, on_reset, truncates, gamma, lam)
    cuda_fn = functools.partial(cugae.cugae2d_nolp_func, rewards, values, on_reset, truncates, gamma, lam)

    torch.cuda.synchronize()
    cuda_t = timeit.timeit(cuda_synced(cuda_fn), number=repeat)

    torch.cuda.synchronize()
    py_t = timeit.timeit(cuda_synced(py_fn), number=repeat)

    print(f"seqlen={seqlen},bs={bs}, CUDA acceleration ratio", py_t / cuda_t)
    return py_t / cuda_t


def get_barplot():
    gae1d_res = {}
    for max_seqlen, bs in itertools.product([32, 128, 512], [2, 4]):
        gae1d_res[(max_seqlen, bs)] = benchmark_gae1d_nolp_misalign(max_seqlen, bs)
    gae2d_olp_res = {}
    for seqlen, bs in itertools.product([32, 128, 512, 1024], [8, 16, 32, 100]):
        gae2d_olp_res[(seqlen, bs)] = test_gae2d_olp(bs, seqlen)
    gae2d_nolp_res = {}
    for seqlen, bs in itertools.product([32, 128, 512, 1024], [8, 16, 32, 100]):
        gae2d_nolp_res[(seqlen, bs)] = test_gae2d_nolp(bs, seqlen)

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle("CUDA Acceleration Ratios for Different Configurations", fontsize=20)

    for ax, res, title in zip(
        axes, [gae1d_res, gae2d_olp_res, gae2d_nolp_res], ["GAE 1D", "GAE 2D OVERLAPPED", "GAE 2D NON-OVERLAPPED"]
    ):
        bars = ax.bar(
            [f"seqlen={seqlen},bs={bs}" for seqlen, bs in res.keys()], list(res.values())
        )

        # Adding labels and title
        ax.set_ylabel("CUDA Acceleration Ratio", fontsize=16)
        ax.set_xlabel("Configuration", fontsize=16)
        ax.set_title(title, fontsize=16)
        # ax.set_yscale("log")

        # Adding values on top of the bars
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha="center", va="bottom")

        # Increase y-axis label size
        ax.tick_params(axis="y", labelsize=10)

        # Rotate x-axis labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    # Adjust layout for individual subplots
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the plot with a higher DPI for better quality
    plt.savefig("benchmark.png", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    get_barplot()
