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


# ---------------------------------------------------------------------------
# Quaternion utilities -- MuJoCo [w, x, y, z] convention
# ---------------------------------------------------------------------------

@torch.jit.script
def normalize(x, eps: float = 1e-9):
    """Normalize vectors along the last dimension."""
    return x / x.norm(p=2, dim=-1).clamp(min=eps).unsqueeze(-1)


@torch.jit.script
def quat_mul(a, b):
    """Quaternion multiplication (Hamilton product). Convention: [w, x, y, z]."""
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


@torch.jit.script
def quat_conjugate(a):
    """Quaternion conjugate. [w, x, y, z] -> [w, -x, -y, -z]."""
    return torch.cat([a[..., :1], -a[..., 1:]], dim=-1)


@torch.jit.script
def quat_rotate(q, v):
    """Rotate vector v by quaternion q. Convention: [w, x, y, z]."""
    q_vec = q[..., 1:4]
    q_w = q[..., 0:1]
    t = 2.0 * torch.cross(q_vec, v, dim=-1)
    return v + q_w * t + torch.cross(q_vec, t, dim=-1)


@torch.jit.script
def quat_rotate_inverse(q, v):
    """Rotate vector v by the inverse of quaternion q. Convention: [w, x, y, z]."""
    q_vec = q[..., 1:4]
    q_w = q[..., 0:1]
    t = 2.0 * torch.cross(q_vec, v, dim=-1)
    return v - q_w * t + torch.cross(q_vec, t, dim=-1)


@torch.jit.script
def quat_from_angle_axis(angle, axis):
    """Create quaternion from angle (scalar per batch) and axis (3-vec per batch).

    Returns [w, x, y, z].
    """
    theta = (angle / 2).unsqueeze(-1)
    ax = normalize(axis)
    xyz = ax * theta.sin()
    w = theta.cos()
    return normalize(torch.cat([w, xyz], dim=-1))
