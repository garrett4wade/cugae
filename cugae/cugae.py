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
    """Compute GAE over a batch of packed sequences with different lengths.

    This function assumes that rewards and values are packed into an 1D tensor.
    Values are longer than rewards by the number of sequences in rewards because of bootstrapping.
    cu_seqlens marks the bounary of sequences in rewards.

    The final step of each sequence is *NOT* overlapped with the first step of the next sequence,
    and rewards/values do not have the same length, so this function is suffixed with
    "nolp" (non-overlap) and "misalign".

    Args:
        rewards (torch.FloatTensor): Shape [total_seqlen], rewards across sequences.
        values (torch.FloatTensor): Shape [total_seqlen + batch_size], values across sequences.
            Values are bootstrapped, so it's longer than rewards.
        cu_seqlens (torch.IntTensor): Marker of sequence boundaries in rewards,
            e.g., [0, s1, s1+s2, ..., total_seqlen]. It should starts with 0 and ends with total_seqlen.
        truncate (torch.BoolTensor): Whether each sequence is truncated because of exceeding max length.
            If truncate, the next value of the last step will be bootstraped, otherwise 0.
        gamma (float): Discount factor.
        lam (float): GAE discount factor.

    Returns:
        Tuple[torch.FloatTensor, torch.FloatTensor]: Advantages and returns (value targets).
            Both have the same shape as rewards.
    """
    assert len(rewards.shape) == len(values.shape) == len(cu_seqlens.shape) == 1
    assert cu_seqlens[0] == 0 and cu_seqlens[-1] == rewards.shape[0]
    return gae_cuda.gae_1d_nolp_misalign(rewards, values, cu_seqlens, truncate, gamma, lam)


def cugae2d_olp_func(
    rewards: torch.FloatTensor,
    values: torch.FloatTensor,
    dones: torch.BoolTensor,
    truncates: torch.BoolTensor,
    gamma: float,
    lam: float,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """Compute GAE over batched sequences with variable lengths, assuming overlapped sequences.

    This function assumes that rewards and values are batched as 2D tensors.
    The first dimension is batch_size and the second dimension is the number of collected timesteps.
    Each batch slot may contain multiple sequences, and sequences may have different lengths.
    The length of each sequence is marked by dones.

    `dones` marks the termination of each sequence, no matter it's truncated or not.
    `truncates` marks truncation and its nonzero indices must be the subset of `dones`.
    If truncate, abandon GAE computation of the last step (because we don't have the bootstrapped
    value in this case) and start from the second last step.

    The final step of each sequence *is overlapped* by the first step of the next sequence,
    i.e., auto-reset, which has widely used in libraries such as gym. In other words, the
    steps where `dones` is True are actually the first steps of sequences. Therefore,
    this function is suffixed with "olp" (overlap).

    Args:
        rewards (torch.FloatTensor): Shape [batch_size, seqlen].
        values (torch.FloatTensor): Shape [batch_size, seqlen + 1], with one more bootstrap step.
        dones (torch.BoolTensor): Shape [batch_size, seqlen + 1].
        truncates (torch.BoolTensor): Shape [batch_size, seqlen + 1].
        gamma (float): Discount factor.
        lam (float): GAE discount factor.

    Returns:
        Tuple[torch.FloatTensor, torch.FloatTensor]: Advantages and returns (value targets).
            Both have the same shape as rewards.
    """
    truncates_indices = truncates.nonzero()
    assert torch.all(dones[truncates_indices[:, 0], truncates_indices[:, 1]])
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
    truncates: torch.BoolTensor,
    gamma: float,
    lam: float,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """Compute GAE over batched sequences with variable lengths, assuming non-overlapped sequences.

    This function assumes that rewards and values are batched as 2D tensors.
    The first dimension is batch_size and the second dimension is the number of collected timesteps.
    Each batch slot may contain multiple sequences, and sequences may have different lengths.
    The length of each sequence is marked by `on_reset`.

    `on_reset` marks the beginning of each sequence. `truncates` marks truncation.
    If truncate, values will be bootstrapped from the `done` step.

    The final step of each sequence is *NOT* overlapped by the first step of the next sequence.
    Each sequence will be complete. The last step should only have observations but no rewards.
    This is used in SRL. Therefore, this function is suffixed with "nolp" (non-overlap).

    Args:
        rewards (torch.FloatTensor): Shape [batch_size, seqlen].
        values (torch.FloatTensor): Shape [batch_size, seqlen + 1], with one more bootstrap step.
        dones (torch.BoolTensor): Shape [batch_size, seqlen + 1].
        truncates (torch.BoolTensor): Shape [batch_size, seqlen + 1].
        gamma (float): Discount factor.
        lam (float): GAE discount factor.

    Returns:
        Tuple[torch.FloatTensor, torch.FloatTensor]: Advantages and returns (value targets).
            Both have the same shape as rewards.
    """
    dones = on_reset[:, 1:]
    truncates_indices = truncates[:, :-1].nonzero()
    assert torch.all(dones[truncates_indices[:, 0], truncates_indices[:, 1]])
    on_reset_indices = on_reset.nonzero()
    num_resets = on_reset.float().sum(1)
    max_num_resets = int(num_resets.max())
    cu_num_resets = torch.nn.functional.pad(num_resets.cumsum(0), (1, 0), value=0).int()
    truncates = torch.cat([torch.zeros_like(truncates[:, 0:1]), truncates[:, :-1]], dim=1)
    bootstrap = truncates[on_reset_indices[:, 0], on_reset_indices[:, 1]]
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
