# Repository Guidelines

## Project Structure & Module Organization
`msk_warp/` contains the package code. Core areas are `bridge.py` for the Warp-to-PyTorch autograd bridge, `envs/` for environments, `algorithms/` for SHAC, `networks/` for actor/critic models, `utils/` for shared helpers, and `assets/` plus `configs/` for MJCF/XML and YAML defaults. Use `scripts/` for runnable entry points such as training and visualization. Keep regression and gradient checks in `tests/`.

Active environment scope is:
- `CartPoleSwingUp`
- `Ant`
- `MyoLeg26Walk`

`MyoLegWalk` scaffolding and one-off experiments are archived under local-only `archive/`.

Docs policy:
- Canonical, tracked docs live in `docs/` (`README.md`, `project_scope.md`, `ant_research_playbook.md`, `testing_matrix.md`).
- Session notes and historical writeups live in local-only `archive/docs/`.

Training outputs belong in `logs/` or `outputs/`, not in tracked source folders.

## Build, Test, and Development Commands
Install the package in editable mode with dev dependencies:

```bash
pip install -e ".[dev]"
```

Run training with a concrete config:

```bash
python scripts/train.py --cfg configs/cartpole_shac.yaml --logdir logs/cartpole
python scripts/train.py --cfg configs/ant_shac.yaml --logdir logs/ant --device cuda:0
```

Visualize a saved policy:

```bash
python scripts/visualize.py --cfg logs/cartpole/cfg.yaml --policy logs/cartpole/best_policy.pt --episodes 5
```

Run the full test suite or a focused gradient test:

```bash
pytest tests/
pytest tests/test_ant_gradient.py::test_forward_vel_gradient_tape -v
```

## Coding Style & Naming Conventions
Target Python 3.11+ and follow existing style: 4-space indentation, module-level docstrings, and straightforward PEP 8 naming. Use `snake_case` for files, functions, variables, and YAML keys; use `CamelCase` for classes. Keep environment and config names aligned, for example `ant_shac.yaml` with code in `msk_warp/envs/ant.py`. No formatter or linter is configured in `pyproject.toml`, so match surrounding style closely and keep imports tidy.

## Testing Guidelines
Pytest is the test runner, with discovery rooted at `tests/`. Name files `test_*.py` and keep new tests close to the subsystem they validate. Prefer deterministic, targeted checks over long training runs. For physics or gradient changes, add or update an assertion in the existing gradient-focused tests before changing tolerances.

## Commit & Pull Request Guidelines
Recent commits use short, imperative summaries such as `actor init fix` and `unit tests`. Keep commit messages concise and specific to one change. PRs should describe the behavior change, list the configs or tests exercised, and call out any dependency or environment assumptions. Include screenshots only when a visualization or rendered output changed.

## Configuration Notes
This repo depends on a custom editable `mujoco_warp` build in addition to `torch`, `warp-lang`, and `mujoco`. Do not hardcode machine-specific paths in committed code. Keep large logs, checkpoints, and generated frames out of git.
