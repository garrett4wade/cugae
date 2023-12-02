from typing import Tuple
import torch


def pygae2d_olp(
    rewards: torch.FloatTensor,
    values: torch.FloatTensor,
    dones: torch.BoolTensor,
    truncates: torch.BoolTensor,
    gamma: float,
    lam: float,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    episode_length = int(rewards.shape[1])
    masks = 1 - dones.float()
    truncate_mask = 1 - truncates.float()
    delta = rewards + gamma * values[:, 1:] * masks[:, 1:] - values[:, :-1]
    adv = torch.zeros_like(rewards)
    gae = torch.zeros_like(rewards[:, 0])
    m = gamma * lam * masks[:, 1:]
    step = episode_length - 1
    while step >= 0:
        # if env is terminated compulsively, then abandon the finnal step
        # i.e. advantage of final step is 0, values target of final step is predicted values
        gae = (delta[:, step] + m[:, step] * gae) * truncate_mask[:, step + 1]
        adv[:, step] = gae
        step -= 1
    return adv, adv + values[:, :-1]


def pygae1d_nolp_misalign(
    rewards: torch.FloatTensor,
    values: torch.FloatTensor,
    cu_seqlens_: torch.IntTensor,
    bootstrap: torch.FloatTensor,
    gamma: float,
    lam: float,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    cu_seqlens = cu_seqlens_.clone()
    cu_seqlens[1:] += torch.ones_like(cu_seqlens_[1:]).cumsum(0)

    bs = cu_seqlens_.shape[0] - 1
    assert values.shape[0] == rewards.shape[0] + bs
    advantages_reversed = []
    returns_reversed = []
    for i in reversed(range(bs)):
        v_offset = cu_seqlens[i]
        r_offset, r_end = cu_seqlens_[i], cu_seqlens_[i + 1]
        assert cu_seqlens[i + 1] - v_offset - 1 == r_end - r_offset
        lastgaelam = 0
        for t in reversed(range(r_end - r_offset)):
            nextvalues = values[v_offset + t + 1]
            if t == r_end - r_offset - 1:
                nextvalues *= bootstrap[i]
            delta = rewards[r_offset + t] + gamma * nextvalues - values[v_offset + t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
            returns_reversed.append(lastgaelam + values[v_offset + t])

    advantages = torch.stack(advantages_reversed[::-1])
    returns = torch.stack(returns_reversed[::-1])
    return advantages, returns


def pygae2d_nolp(
    rewards: torch.FloatTensor,
    values: torch.FloatTensor,
    on_reset: torch.BoolTensor,
    truncates: torch.BoolTensor,
    gamma: float,
    lam: float,
) -> torch.FloatTensor:
    on_reset = on_reset.float()
    truncates = truncates.float()
    episode_length = int(rewards.shape[1])
    delta = rewards + gamma * values[:, 1:] * (1 - on_reset[:, 1:]) - values[:, :-1]

    gae = torch.zeros_like(rewards[:, 0])
    adv = torch.zeros_like(rewards)

    # 1. If the next step is a new episode, GAE doesn't propagate back
    # 2. If the next step is a truncated final step, the backpropagated GAE is -V(t),
    #    which is not correct. We ignore it such that the current GAE is r(t-1)+ɣV(t)-V(t-1)
    # 3. If the next step is a done final step, the backpropagated GAE is zero.
    m = gamma * lam * (1 - on_reset[:, 1:]) * (1 - truncates[:, 1:])

    step = episode_length - 1
    while step >= 0:
        gae = delta[:, step] + m[:, step] * gae
        adv[:, step] = gae
        step -= 1

    return adv, adv + values[:, :-1]
