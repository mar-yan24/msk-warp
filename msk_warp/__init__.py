"""msk-warp: Differentiable RL with SHAC and MuJoCo Warp."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent


def get_asset_path(filename: str) -> str:
    """Return absolute path to a file in msk_warp/assets/."""
    return str(PACKAGE_ROOT / "assets" / filename)


def resolve_model_path(model_path: str) -> str:
    """Resolve a model_path from a config YAML.

    Handles three cases:
    - Absolute path: returned unchanged
    - 'assets/foo.xml': resolved to PACKAGE_ROOT/assets/foo.xml
    - Bare 'foo.xml': resolved to PACKAGE_ROOT/assets/foo.xml
    """
    p = Path(model_path)
    if p.is_absolute():
        return str(p)
    if p.parts[0] == "assets":
        return str(PACKAGE_ROOT / p)
    return str(PACKAGE_ROOT / "assets" / p)


# Lazy re-exports: deferred until accessed so that importing msk_warp
# for path helpers doesn't pull in torch/warp/mujoco.
def __getattr__(name):
    if name == "WarpSimStep":
        from msk_warp.bridge import WarpSimStep
        return WarpSimStep
    if name == "ENV_MAP":
        from msk_warp.envs import ENV_MAP
        return ENV_MAP
    if name == "ACTOR_MAP":
        from msk_warp.networks import ACTOR_MAP
        return ACTOR_MAP
    if name == "CRITIC_MAP":
        from msk_warp.networks import CRITIC_MAP
        return CRITIC_MAP
    raise AttributeError(f"module 'msk_warp' has no attribute {name!r}")
