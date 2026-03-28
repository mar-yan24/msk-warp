"""Visualize training progression as a grid of snapshots (DiffRL paper style).

Discovers policy checkpoints saved during training, runs one episode per
checkpoint, and arranges representative frames in a labeled grid showing
how the policy improves over training.

Usage:
    python scripts/visualize_progression.py --logdir logs/cartpole --output progression.png
    python scripts/visualize_progression.py --logdir logs/ant --output ant_prog.png --cols 2
    python scripts/visualize_progression.py --logdir logs/ant --max-checkpoints 8 --composite 5
"""

import argparse
import math
import os
import re
import sys

import numpy as np
import torch
import mujoco
import yaml
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

# Import shared components from visualize.py
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)
from visualize import load_policy, ADAPTER_MAP

from msk_warp import resolve_model_path


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def discover_checkpoints(logdir):
    """Find policy checkpoints and their iteration numbers.

    Returns list of (iteration, path) sorted by iteration.
    Recognises:
      - init_policy.pt          -> iteration 0
      - *_policy_iter{N}_*.pt   -> iteration N
    """
    checkpoints = []
    for f in os.listdir(logdir):
        if not f.endswith('.pt'):
            continue
        path = os.path.join(logdir, f)
        if f == 'init_policy.pt':
            checkpoints.append((0, path))
            continue
        m = re.match(r'.*_policy_iter(\d+)_reward.*\.pt', f)
        if m:
            checkpoints.append((int(m.group(1)), path))
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def subsample_checkpoints(checkpoints, max_n):
    """Evenly subsample checkpoints, always keeping first and last."""
    if len(checkpoints) <= max_n:
        return checkpoints
    indices = np.round(np.linspace(0, len(checkpoints) - 1, max_n)).astype(int)
    return [checkpoints[i] for i in np.unique(indices)]


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

CAMERA_DEFAULTS = {
    'CartPoleSwingUp': dict(distance=4.0, elevation=-15.0, lookat=[0, 0, 0.5]),
    'Ant':             dict(distance=6.0, elevation=-20.0, lookat=[0, 0, 0.5]),
}


def make_camera(env_name, lookat_override=None):
    """Return an MjvCamera configured for the given environment."""
    defaults = CAMERA_DEFAULTS.get(env_name, dict(
        distance=5.0, elevation=-20.0, lookat=[0, 0, 0.5]))
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = defaults['distance']
    cam.elevation = defaults['elevation']
    cam.lookat[:] = lookat_override or defaults['lookat']
    return cam


# ---------------------------------------------------------------------------
# Episode rendering
# ---------------------------------------------------------------------------

def _run_step(mjm, mjd, adapter, actor, obs_rms):
    """One control step: obs -> policy -> physics -> (reward, done)."""
    obs = adapter.compute_obs(mjd.qpos.copy(), mjd.qvel.copy())
    obs_norm = obs_rms.normalize(obs) if obs_rms is not None else obs

    with torch.no_grad():
        action = actor(obs_norm, deterministic=True)
        action = torch.tanh(action)

    action_np = action.squeeze(0).cpu().numpy()
    adapter.update_actions(action_np)

    mjd.ctrl[:len(action_np)] = action_np * adapter.action_strength
    for _ in range(adapter.substeps):
        mujoco.mj_step(mjm, mjd)

    obs_after = adapter.compute_obs(mjd.qpos.copy(), mjd.qvel.copy())
    done = adapter.check_done(obs_after)
    return done


def render_single_frame(mjm, mjd, adapter, actor, obs_rms, renderer,
                        env_name, frame_frac=0.5):
    """Run an episode and return a single representative frame (RGB array).

    The frame is captured at *frame_frac* of the way through the episode.
    For Ant, the camera tracks the torso so the agent stays centred.
    """
    adapter.reset_state(mjm, mjd)
    target_step = int(adapter.episode_length * frame_frac)
    frame = None

    for t in range(adapter.episode_length):
        done = _run_step(mjm, mjd, adapter, actor, obs_rms)

        if t == target_step or (done and frame is None):
            cam = make_camera(env_name)
            # Track the agent for locomotion envs
            if env_name == 'Ant':
                cam.lookat[:] = mjd.qpos[0:3].copy()
                cam.lookat[2] = max(cam.lookat[2], 0.3)
            renderer.update_scene(mjd, camera=cam)
            frame = renderer.render().copy()

        if done:
            break

    # Fallback: capture final frame if we never hit the target
    if frame is None:
        cam = make_camera(env_name)
        if env_name == 'Ant':
            cam.lookat[:] = mjd.qpos[0:3].copy()
            cam.lookat[2] = max(cam.lookat[2], 0.3)
        renderer.update_scene(mjd, camera=cam)
        frame = renderer.render().copy()

    return frame


def render_composite_frame(mjm, mjd, adapter, actor, obs_rms, renderer,
                           env_name, num_overlays=5):
    """Run an episode and return a multi-frame composite (motion trail).

    Captures *num_overlays* evenly spaced frames and blends them, giving a
    stroboscopic / motion-trail effect similar to the DiffRL paper figures.
    Earlier frames are rendered at lower opacity; the final capture is full
    strength.
    """
    adapter.reset_state(mjm, mjd)
    total = adapter.episode_length
    capture_steps = set(np.linspace(0, total - 1, num_overlays, dtype=int))

    raw_frames = []
    for t in range(total):
        done = _run_step(mjm, mjd, adapter, actor, obs_rms)

        if t in capture_steps:
            cam = make_camera(env_name)
            if env_name == 'Ant':
                cam.lookat[:] = mjd.qpos[0:3].copy()
                cam.lookat[2] = max(cam.lookat[2], 0.3)
            renderer.update_scene(mjd, camera=cam)
            raw_frames.append(renderer.render().copy())

        if done:
            # Grab final frame if we haven't captured enough
            if len(raw_frames) == 0 or t not in capture_steps:
                cam = make_camera(env_name)
                if env_name == 'Ant':
                    cam.lookat[:] = mjd.qpos[0:3].copy()
                    cam.lookat[2] = max(cam.lookat[2], 0.3)
                renderer.update_scene(mjd, camera=cam)
                raw_frames.append(renderer.render().copy())
            break

    if len(raw_frames) == 1:
        return raw_frames[0]

    # Blend: last frame is the base; earlier frames are layered with
    # increasing opacity so the most recent pose is sharpest.
    composite = raw_frames[-1].astype(np.float32)
    n = len(raw_frames) - 1
    for i, frame in enumerate(raw_frames[:-1]):
        alpha = 0.12 + 0.18 * (i / max(n - 1, 1))     # 0.12 .. 0.30
        composite = composite * (1.0 - alpha) + frame.astype(np.float32) * alpha

    return np.clip(composite, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Grid assembly
# ---------------------------------------------------------------------------

def _try_load_font(size):
    """Try common system fonts, fall back to PIL default."""
    candidates = [
        'arial.ttf', 'Arial.ttf',
        'DejaVuSans.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        'C:/Windows/Fonts/arial.ttf',
    ]
    for name in candidates:
        try:
            return PIL.ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return PIL.ImageFont.load_default()


def build_grid(frames, labels, sublabels, cols=3, padding=20, label_gap=60):
    """Arrange rendered frames in a labelled grid image.

    Each cell shows the frame with a primary label (e.g. "Iteration 100")
    and an optional sublabel (e.g. "(2 minutes of training)") centred below.
    """
    n = len(frames)
    rows = math.ceil(n / cols)
    fh, fw = frames[0].shape[:2]

    cell_w = fw + padding
    cell_h = fh + label_gap + padding
    grid_w = cols * cell_w + padding
    grid_h = rows * cell_h + padding

    bg_color = (230, 230, 232)
    grid = PIL.Image.new('RGB', (grid_w, grid_h), color=bg_color)
    draw = PIL.ImageDraw.Draw(grid)

    font = _try_load_font(16)
    font_sm = _try_load_font(13)

    for idx in range(n):
        r, c = divmod(idx, cols)
        x = padding + c * cell_w
        y = padding + r * cell_h

        grid.paste(PIL.Image.fromarray(frames[idx]), (x, y))

        # Primary label
        lx = x + fw // 2
        ly = y + fh + 6
        bbox = draw.textbbox((0, 0), labels[idx], font=font)
        tw = bbox[2] - bbox[0]
        draw.text((lx - tw // 2, ly), labels[idx], fill=(30, 30, 30), font=font)

        # Sublabel
        if sublabels[idx]:
            sy = ly + 22
            bbox = draw.textbbox((0, 0), sublabels[idx], font=font_sm)
            tw = bbox[2] - bbox[0]
            draw.text((lx - tw // 2, sy), sublabels[idx],
                      fill=(110, 110, 110), font=font_sm)

    return grid


# ---------------------------------------------------------------------------
# Time formatting
# ---------------------------------------------------------------------------

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f} seconds"
    if seconds < 3600:
        mins = seconds / 60
        return f"{mins:.1f} minutes" if mins >= 2 else f"{mins:.1f} minute"
    hours = seconds / 3600
    return f"{hours:.1f} hours"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate a training progression grid (DiffRL paper style)')
    parser.add_argument('--logdir', required=True,
                        help='Log directory with checkpoints and cfg.yaml')
    parser.add_argument('--output', default='progression.png',
                        help='Output image path (default: progression.png)')
    parser.add_argument('--max-checkpoints', type=int, default=6,
                        help='Maximum number of checkpoints to show (default: 6)')
    parser.add_argument('--cols', type=int, default=3,
                        help='Grid columns (default: 3)')
    parser.add_argument('--width', type=int, default=640,
                        help='Frame width in pixels (default: 640)')
    parser.add_argument('--height', type=int, default=480,
                        help='Frame height in pixels (default: 480)')
    parser.add_argument('--composite', type=int, default=0, metavar='N',
                        help='Overlay N frames per cell for motion trails '
                             '(0 = single frame, default: 0)')
    parser.add_argument('--frame-frac', type=float, default=0.5,
                        help='Where in the episode to capture the single '
                             'frame, 0.0-1.0 (default: 0.5)')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible episodes')
    args = parser.parse_args()

    # ---- Load config ----
    cfg_path = os.path.join(args.logdir, 'cfg.yaml')
    if not os.path.exists(cfg_path):
        print(f"Error: {cfg_path} not found")
        sys.exit(1)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    env_cfg = cfg['params']['env']
    env_name = env_cfg['name']

    # ---- Discover & subsample checkpoints ----
    checkpoints = discover_checkpoints(args.logdir)
    if not checkpoints:
        print(f"No checkpoints found in {args.logdir}")
        sys.exit(1)

    checkpoints = subsample_checkpoints(checkpoints, args.max_checkpoints)
    print(f"Env: {env_name}  |  {len(checkpoints)} checkpoints")
    for it, path in checkpoints:
        print(f"  iter {it:>5d}  {os.path.basename(path)}")

    # ---- MuJoCo setup ----
    model_path = resolve_model_path(env_cfg['model_path'])
    xml_string = open(model_path).read()
    # Strip Warp-only flags for native MuJoCo rendering
    xml_vis = xml_string.replace('jacobian="sparse"', '')
    xml_vis = xml_vis.replace(
        '<flag contact="disable" constraint="disable" eulerdamp="disable"/>', '')
    mjm = mujoco.MjModel.from_xml_string(xml_vis)
    mjd = mujoco.MjData(mjm)
    renderer = mujoco.Renderer(mjm, width=args.width, height=args.height)

    # ---- Adapter ----
    if env_name not in ADAPTER_MAP:
        print(f"No adapter for '{env_name}'. Available: {list(ADAPTER_MAP)}")
        sys.exit(1)
    adapter = ADAPTER_MAP[env_name](env_cfg, args.device)

    # ---- Reference timestamp for training time estimates ----
    init_pt = os.path.join(args.logdir, 'init_policy.pt')
    t0 = os.path.getmtime(init_pt) if os.path.exists(init_pt) else None

    # ---- Render each checkpoint ----
    frames, labels, sublabels = [], [], []

    for iter_num, ckpt_path in checkpoints:
        print(f"  Rendering iteration {iter_num} ...", end=' ', flush=True)

        # Deterministic seeding so every checkpoint sees the same init state
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        actor, obs_rms = load_policy(ckpt_path, device=args.device)
        actor.eval()

        if args.composite > 0:
            frame = render_composite_frame(
                mjm, mjd, adapter, actor, obs_rms, renderer,
                env_name, num_overlays=args.composite)
        else:
            frame = render_single_frame(
                mjm, mjd, adapter, actor, obs_rms, renderer,
                env_name, frame_frac=args.frame_frac)

        frames.append(frame)
        labels.append(f"Iteration {iter_num}")

        if t0 is not None:
            elapsed = max(os.path.getmtime(ckpt_path) - t0, 0)
            sublabels.append(f"({format_time(elapsed)} of training)")
        else:
            sublabels.append('')

        print('done')

    # ---- Assemble grid ----
    grid = build_grid(frames, labels, sublabels, cols=args.cols)
    grid.save(args.output)
    print(f"\nSaved {args.output}  ({grid.width}x{grid.height})")


if __name__ == '__main__':
    main()
