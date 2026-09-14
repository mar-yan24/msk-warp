# msk-warp

Research code for learning musculoskeletal locomotion policies with simulation gradients.
PyTorch owns the actor, critic, observations and rewards; a custom MuJoCo Warp backend supplies
state and control derivatives. SHAC is the current policy-learning implementation, with PPO
available as a comparison. The target is MyoLeg26; accelerated training on that target is unproven.

## Current status (2026-09-13)

| Environment | What is established |
| --- | --- |
| CartPole | Gradient and training regression control; convergence depends on seed/configuration. |
| Motor hopper | Working SHAC locomotion control, measured across three seeds. |
| Muscle hopper | Activation gradients are threaded, but reliable forward locomotion has not been demonstrated. Tendon state-gradient defects remain. |
| Ant | Historical PPO locomotion control; SHAC standing/fine-tuning failures and a free-root contact-gradient defect remain. |
| MyoLeg26 | Pinned official beta model: `nq=47`, `nv=46`, 26 muscles. Five-seed PPO walking baseline failed its bounded behavior gate. Strict visited-state forward checks block derivative qualification; autodiff training remains guarded. |

Trajectory optimization and CPU return-map shooting investigate candidate gaits. They are separate
from policy learning. Finding an orbit does not establish robust policy control or training speed;
failing to find one in a bounded search does not establish physical impossibility.

## Backend and setup

The tested backend is the local PR #1423 stack at `C:/Projects/mujoco_warp_pr1423`, branch
`mark/pr1423-fixes`, commit `8f5636ce6b8d399c9452c04694eff22237219b8a`, based on PR head `eeac6f2`.
Its APIs and muscle derivatives are experimental; installing upstream `mujoco-warp` is not an
equivalent setup. The backend checkout and its local fixes are not vendored in this repository.

`pyproject.toml` specifies Python >=3.11, MuJoCo 3.10.0, Warp 1.14.0, the editable backend path,
and PyTorch's `cu130` index. The working machine uses Python 3.12 and an NVIDIA CUDA GPU.
Provision the pinned checkout first; on another machine, update the local source path before syncing.
Run from the repository root in PowerShell:

```powershell
uv sync --extra dev
.venv/Scripts/python.exe -m pytest tests/unit -q
.venv/Scripts/python.exe scripts/train.py --cfg configs/cartpole_shac.yaml --logdir logs/cartpole --seed 0
.venv/Scripts/python.exe scripts/train.py --cfg configs/hopper_motor_shac.yaml --logdir logs/hopper_motor --seed 0
```

Configs resolve relative to `msk_warp/` or the current directory. Training also accepts `--device`,
`--max-epochs` and `--init-policy`. **`--init-policy` is a warm start, not a training resume:** it
loads saved networks/normalizers and resets optimizers; iteration and training state are not restored.

## Gradient contract

`WarpSimStep` takes `(ctrl, qpos, qvel, act)` and returns `(qpos, qvel, act)`. Activation must remain
in the graph across physics substeps and policy steps. Empty activation tensors serve motor models.
The explicit `backward_mode` setting selects:

| Mode | Role and limitations |
| --- | --- |
| `tape_per_substep` | Chains one taped VJP per physics substep; used by the current CartPole/hopper configs. |
| `tape` | Tapes all substeps; tested on the pinned stack. Old fork failures do not describe this implementation. |
| `fd` | Finite-difference diagnostic/control implementation. Selected local checks pass; an existing hopper training comparison failed. It is not a validated training fallback. |

All modes have targeted coverage in `tests/gpu/test_bridge_gradients.py`. Known failures are kept
as explicit expected failures: free-joint contact `qpos` gradients and multi-tendon chain `qpos`
gradients. A passing local derivative check does not validate every state, horizon or objective.
The bridge also sanitizes nonfinite gradients and clamps components; diagnostics must detect this.
Replay currently omits solver `qacc_warmstart`; its numerical impact still needs measurement.

The H16 trajectory-objective audit (ten random parameter sets, ten directions per block) passes
native forward-output checks on muscle hopper but fails all 80 strict block/term derivative gates.
A motor seed-0 check passes; a ten-state motor run stops on one forward mismatch before backward.
These findings justify investigating derivative validity, not claims that muscle gaits cannot exist.

Environment construction checks the backend's model/data gradient contract. Contact experiments
use Newton with dense Jacobians and supported geometry pairs. Do not bypass those checks to label
an unsupported model differentiable.

## MyoLeg26 reference and task

The reference is MyoHub's `myolegs26` at
[`eb327acbae0fad12279495040607f5235d962328`](https://github.com/MyoHub/myo_sim/tree/eb327acbae0fad12279495040607f5235d962328),
built with its canonical `myo_sim.load_spec` API. Upstream labels this reduced model **beta**;
the preceding 80-muscle MyoLeg section of its documentation describes a different model.
The selected model has a passive torso without arms, a quaternion free root, 28 equality constraints
and total mass 78.090468479 kg. The old asset remains at `assets/myoleg/myoLeg26_BASELINE.xml`.
Its arms, root coordinates, EDL/FDL gains, offsets and old keyframes are not carried into the new task.

`assets/myoleg26/reference.xml` preserves the official assembly and demo scene.
`assets/myoleg26/flat_boxes.xml` defines a separate task: 14 massless boxes measured from the
original collision surfaces, articulated calcaneus/toe bodies, one ground plane and no self-contact.
Body masses/inertias are unchanged; tests also check joint, tendon and actuator invariants.
The boxes are a conservative approximation, not validated anatomical contact surfaces.
The task uses Euler/Newton/dense, 2 ms physics steps and disabled solver warmstart.
A nominal reset aligns forward with world +X and sets 1 mm foot clearance; it is not an equilibrium.

The `myoleg26-walk-v1` task targets **1.0 m/s without imitation**. Its 145-value observation includes
world pelvis pose/velocities, joint coordinates/velocities, all 26 muscle activations and previous
commands. Pure Torch kinematics are checked against native MuJoCo, including root quaternion and
velocity frame conventions. A signed action maps to excitation `u=(clamp(a,-1,1)+1)/2`;
passive is `a=-1`, while neutral policy output `a=0` gives 50% excitation.
Reward integrates forward/lateral velocity tracking, upright/heading factors and a
`0.01*mean(u**2)` effort penalty over the control period. This is command effort, not metabolic energy.
Episodes distinguish failures from time limits; policies receive reset observations, and timeouts
bootstrap from preserved final observations. Old MyoLeg checkpoints and returns require rebaselining.

The default official environment refuses gradient training because the pinned backend's free-root
and tendon derivatives have unresolved failures. `allow_unvalidated_gradients=True` is a diagnostic
override, not a training recipe. Primitive contact support alone does not validate those gradients.
The legacy mesh asset still has its separate MULTICCD/margin import and mesh-derivative blockers.

To reproduce assets from a clean checkout at the exact source pin, use a **new** output directory:

```powershell
.venv/Scripts/python.exe scripts/build_myoleg26_assets.py --source logs/upstream_myo_sim_20260912 --out logs/myoleg26_rebuild
.venv/Scripts/python.exe scripts/check_myoleg26_task.py --out logs/myoleg26_task_check.json
```

The checked-in manifest records source, builder and asset hashes plus all task overrides.
The diagnostic compares full native/Warp state at 1/4/16 physics steps and measures passive,
neutral and random-action first episodes. Ten sampled seeds pass the bounded forward gate;
this does not establish long-horizon engine equivalence or learned gait.
`configs/experiments/myoleg26_ppo.yaml` is a forward-policy feasibility baseline.
`configs/myoleg26_shac.yaml` uses the same task but remains blocked pending derivative validation.

Validation includes 297 CPU unit tests and 58 GPU tests (three known backend expected failures).
Sixteen passive, neutral and random-action episodes per condition all fail before four seconds.
The PPO update/checkpoint test and a two-epoch run of the configured 64-actor baseline complete;
neither is evidence of a learned gait or accelerated training.

### Frozen five-seed baseline and derivative gates

`configs/experiments/myoleg26_baseline_v1.json` freezes the task, source/model hashes,
backend/packages and five PPO seeds at 1,048,576 training control transitions each.
All five completed: 5,242,880 training controls / 20,971,520 physics steps, plus
77,712 evaluation controls / 310,848 physics steps. Inclusive batch wall time was
3,905.04 s (65.08 min) on the RTX 4060 Laptop; training loops used 3,710.44 s and
evaluation 168.92 s. Kernel caches were populated; brief CPU review/test activity
occurred while the GPU was reserved for this batch. This is application wall time.

No scheduled checkpoint met the four-second walking gate; selected-checkpoint
confirmation had 0/160 survivors across five seeds. Mean episode duration was
0.708 s and mean forward velocity -0.381 m/s. Selected epochs were 0/0/0/0/96.
The preregistered ranking ignores subthreshold survival duration when all survival
fractions are zero; its pre-failure speed error can favor an earlier fall. Preserve
this result, but address that selection limitation in a separately frozen protocol.
These results reject this bounded recipe as a successful walking baseline, not
PPO or the model as unlearnable. They cannot establish time-to-gait acceleration.

The selector captured 50 episode states from all five selected policies:
27 double-support, 11 left-support and 12 right-support. Forty states therefore
come from initially random actors. Shared reset/noise IDs create clustered data;
there are 15 distinct reset IDs in this sample, not 50 IID initializations.
The local checker compares actions, free-root/internal position tangents,
velocities and activations against native float64 finite differences, with
forward, epsilon-window, sanitizer and coverage prerequisites.

In this run 65/100 forward cases exceeded the preregistered derivative-check
tolerance (20/50 at one control step, 45/50 at four). This tolerance uses
absolute 1e-4 for joint velocities, stricter than the prior task smoke's 1e-3;
small constrained-joint velocity differences dominate, not a demonstrated
catastrophic simulation failure. The population FD/AD stage stopped before
measurement, and complete-policy qualification remained blocked. A separate
one-state GPU plumbing check found clipped adjoints; it is not population evidence.
Three fresh native/direct-Warp/bridge replays of the worst normalized case also
show the mismatch in direct Warp; the first direct/bridge trajectories are
bitwise identical. Direct Warp varies across repetitions. The numerical cause
remains unisolated, so the result cannot be assigned to bridge plumbing alone.
The previous-action observation also has an unresolved detached path in SHAC.
Do not promote the beta asset or primitive collision support to validated gradients.

Reproduce the bounded run in a new directory:

```powershell
.venv/Scripts/python.exe scripts/run_myoleg26_baseline.py --frozen-manifest msk_warp/configs/experiments/myoleg26_baseline_v1.json --outdir logs/myoleg26_baseline_replay
```

`scripts/select_myoleg26_states.py` verifies the freeze and selected-checkpoint
provenance before constructing an archive. `scripts/check_myoleg26_derivatives.py`
consumes it; `scripts/check_myoleg26_policy_gradient.py` fails closed on missing
local evidence or fewer than 30 distinct reset IDs. A positive policy report
would qualify sampled parameter directions of the complete stochastic objective,
not prove its full high-dimensional gradient. `scripts/summarize_myoleg26_baseline.py`
retains all five seeds and separate costs. Local artifacts are in
`logs/myoleg26_ppo_baseline_v1*`, `logs/myoleg26_visited_v1*`,
`logs/myoleg26_derivatives_v1.json` and `logs/myoleg26_policy_gradient_v1.json`.

## Verification and diagnostics

```powershell
.venv/Scripts/python.exe -m pytest tests/unit -q
.venv/Scripts/python.exe -m pytest tests/gpu -q
.venv/Scripts/python.exe -m pytest tests/slow --run-slow -q
.venv/Scripts/python.exe scripts/policy_gradient_cosine.py --cfg configs/hopper_motor_shac.yaml --out logs/policy_gradient.json
.venv/Scripts/python.exe scripts/check_trajopt_gradients.py --cfg configs/hopper_motor_shac.yaml --scales docs/research/phase4-capability/gait_motor_seed2.json --cycle 16 --samples 10 --directions 10 --auglag --out logs/trajopt_motor_gradient.json
```

GPU tests skip without CUDA; slow tests require `--run-slow` and can take substantial training time.
The trajectory diagnostic performs no optimization: it checks control/initial-state directional
derivatives against native float64 MuJoCo, with forward parity, epsilon checks and sanitizer reporting.
Repeat with `configs/hopper_muscle_shac.yaml` and a separate output file for the muscle comparison.
Its findings apply to the recorded sampled states. The scales argument above requires a local
research artifact; a fresh clone does not contain it. The policy cosine script checks a deterministic
initialized-policy objective, not stochastic gradients throughout training.

`docs/` is entirely ignored at the owner's request, including the local validity ledger and research
protocols. `logs/` and generated outputs are also ignored. Code, tests and commit messages provide the
versioned record; preserve local research artifacts separately. Commit verified units independently.

## References

- [SHAC / Xu et al., ICLR 2022](https://arxiv.org/abs/2204.07137) and [DiffRL implementation](https://github.com/NVlabs/DiffRL).
- [Suh et al., ICML 2022: Do Differentiable Simulators Give Better Policy Gradients?](https://proceedings.mlr.press/v162/suh22b.html)
- [Adaptive Horizon Actor-Critic / Georgiev et al., 2024](https://arxiv.org/abs/2405.17784)
- [MuJoCo Warp upstream](https://github.com/google-deepmind/mujoco_warp), [PR #1423](https://github.com/google-deepmind/mujoco_warp/pull/1423), and [differentiability roadmap](https://github.com/google-deepmind/mujoco_warp/issues/500).

Code license: Apache 2.0. Vendored MyoSim model attribution and applicable source notices are
preserved in `msk_warp/assets/myoleg26/SOURCE_NOTICES.txt` and `LICENSE.upstream`;
upstream identifies the reduced-leg model lineage as CC-BY 3.0.
