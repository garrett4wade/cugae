from typing import Tuple
import torch


def pygae2d(
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
        # i.e. advantage of final step is 0, value target of final step is predicted value
        gae = (delta[:, step] + m[:, step] * gae) * truncate_mask[:, step + 1]
        adv[:, step] = gae
        step -= 1
    return adv, adv + values[:, :-1]
