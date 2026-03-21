import torch


@torch.jit.script
def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


def grad_norm(params):
    total = 0.
    for p in params:
        if p.grad is not None:
            total += torch.sum(p.grad ** 2)
    return torch.sqrt(total)
