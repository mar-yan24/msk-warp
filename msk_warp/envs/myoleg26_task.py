"""Versioned, simulator-independent flat-ground MyoLeg26 task definition.

The reward measures velocity tracking and command effort. It is neither a
metabolic-energy model nor an imitation objective. Quantities use SI units.
"""

from dataclasses import asdict, dataclass
import math

import torch


def excitation_from_action(actions: torch.Tensor) -> torch.Tensor:
    """Signed policy command to bounded muscle excitation; passive action is -1."""
    return 0.5 * (actions.clamp(-1.0, 1.0) + 1.0)


@dataclass(frozen=True)
class MyoLegTaskContract:
    version: str = "myoleg26-walk-v1"
    target_speed: float = 1.0
    velocity_variance: float = 0.25
    effort_weight: float = 0.01
    termination_height: float = 0.55
    termination_upright: float = 0.5

    def __post_init__(self):
        for name in ("target_speed", "velocity_variance", "effort_weight",
                     "termination_height", "termination_upright"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.velocity_variance <= 0 or self.effort_weight < 0:
            raise ValueError("velocity_variance must be positive and effort_weight nonnegative")
        if self.termination_height <= 0 or not 0 <= self.termination_upright <= 1:
            raise ValueError("invalid fall thresholds")

    def as_dict(self):
        return asdict(self)

    def reward_terms(self, obs, excitation, control_dt):
        """Return per-step terms; velocity features are world x/y/z at obs[5:8].

        A perfect forward gait earns nearly one unit per second. Stationary,
        upright, passive posture earns exp(-4) per second at the default target.
        Both tracking and effort are integrated over the actual control period.
        """
        if not math.isfinite(control_dt) or control_dt <= 0:
            raise ValueError("control_dt must be finite and positive")
        nu = excitation.shape[-1]
        error = (obs[:, 5] - self.target_speed).square() + obs[:, 6].square()
        tracking = torch.exp(-error / self.velocity_variance)
        upright = obs[:, -(nu + 2)].clamp(0.0, 1.0)
        heading = obs[:, -(nu + 1)].clamp(0.0, 1.0)
        effort = excitation.square().mean(dim=-1)
        locomotion = tracking * upright * heading
        return {
            "velocity_tracking": tracking,
            "upright": upright,
            "heading": heading,
            "excitation_effort": effort,
            "locomotion_reward": control_dt * locomotion,
            "effort_cost": control_dt * self.effort_weight * effort,
        }

    def reward(self, obs, excitation, control_dt):
        terms = self.reward_terms(obs, excitation, control_dt)
        return terms["locomotion_reward"] - terms["effort_cost"]

    def failures(self, obs, num_actions):
        """Tensor-only fall indicators; contacts are checked by the environment."""
        finite = torch.isfinite(obs).all(dim=-1)
        return {
            "nonfinite": ~finite,
            "low_pelvis": obs[:, 0] < self.termination_height,
            "low_upright": obs[:, -(num_actions + 2)] < self.termination_upright,
        }
