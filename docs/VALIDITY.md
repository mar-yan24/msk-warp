# Validity ledger

A **running, append-only** record of everything known to be wrong, unverified, or
load-bearing-but-fragile in this project: backend autodiff correctness, muscle-model fidelity,
instrument defects, protocol claims that outrun their evidence, and operational constraints.

This file exists because nothing else in the repo does this job. `POSTMORTEM.md` is a closed
retrospective frozen at 2026-05-14. `research/phase1-bakeoff/findings.md` is a closed two-defect
decision record. `HANDOFF.md` is session-scoped and **restates** rather than accumulates, so a defect
found in one session and not repeated in the next handoff simply disappears.

**It is the one file under `docs/` that is tracked in git** (`.gitignore:40-41`: `docs/*` plus
`!docs/VALIDITY.md`; the directory pattern had to become `docs/*` because git cannot re-include a
file whose parent directory is excluded). It therefore has history, blame and review, which the rest
of the research record deliberately does not.

## Conventions

- **Append-only.** Never rewrite or delete an entry. To change one, add a dated status line beneath
  it. A retracted claim is more useful than a missing one.
- **IDs are stable and never reused**: `BE` backend, `MF` muscle fidelity, `IN` instruments,
  `TR` training, `CL` claims and protocol, `OP` operational.
- **Status**: `open` (known wrong, unfixed) | `characterised` (understood and pinned by a test, not
  fixed) | `fixed` | `retracted` (the entry itself was wrong) | `by-design` (a real limit, not a bug).
- Every entry carries **evidence** as `file:line` or a measurement with its number. An entry with no
  evidence does not belong here.
- **Closes** states what measurement or change would let the entry be marked `fixed`.

---

## BE - Backend autodiff (mujoco_warp PR #1423, worktree `mark/pr1423-fixes` @ `8f5636c`)

### BE-01 `open` (2026-09-07) Free-joint root gradient through contact is too large (pr1423 defect 2)

- Evidence: `research/phase1-bakeoff/findings.md:27`; AD/FD ratios up to 15.1 on the quaternion z
  component. Pinned by xfail `tests/gpu/test_bridge_gradients.py::test_ant_qpos_matches_fd`. The only
  open backend defect carrying no fix marker in `findings.md`.
- Impact: the ant. Does **not** affect either hopper - `hopper_*.xml` has slide/hinge roots, no free
  joint.
- Closes: ant `H=1` qpos relative error <= 5e-2 against PR #1535 as oracle.

### BE-02 `characterised` (2026-09-08) Tendon `dL/dqpos` is wrong when two tendons share a kinematic chain (pr1423 defect 3)

- Evidence: `research/phase3-hopper/results.md:120-202`. Muscle hopper cosine **0.926** against
  float64 FD on smooth dynamics (min 0.865 over 8 seeds) where the same skeleton with torque motors
  gives 1.00000. Error **scales with tendon force** - the 25-line `TWO_TENDON_CHAIN_XML` reproduction
  gives 0.981 at gear 800, 0.585 at 3000, **0.116 at 12000**, with individual seeds reaching cosine
  **-0.98** (gradient points the wrong way). Localised to the tendon Jacobian's own derivative,
  `(d ten_J / dq)^T F`, the same object defect 1 lived in.
- Impact: **live for every muscle-hopper run.** `hopper_muscle.xml` has three antagonist pairs on one
  chain. `ctrl`, `act` and `qvel` are exact; only the BPTT state path is biased. G3.5 bounded the
  aggregate *policy* gradient at cosine 0.99, so SHAC training is usable.
- Closes: `test_two_tendon_chain_qpos_matches_fd` flips from xfail to pass.

### BE-03 `open` (2026-09-07) `nv > 60` dense-Jacobian cap, and MyoLeg26 sits exactly on it

- Evidence: `research/phase1-bakeoff/findings.md:33-36`. pr1423 verified exact at `nv = 60`
  (`chain60_eq`, `chain60_eq_soft`, qpos/qvel/ctrl within 1e-2 at H=1); above that the
  blocked-Cholesky path is untested. `myoLeg26_BASELINE.xml` compiles to `nq = nv = 60`.
- Closes: a bake-off case above `nv = 60` against the float64 CPU reference.

### BE-04 `open` (2026-09-08) Neither defect has been reported upstream

- Evidence: `research/phase1-bakeoff/findings.md:42` says both "are worth reporting to the PR author;
  the harness reproduces them in minutes". No evidence anywhere in the tree that this happened.
- Impact: the project carries a private fix branch for defect 1 and two permanent xfails.

### BE-05 `by-design` (2026-09-11) Model parameters are not differentiable through the bridge

- Evidence: verified this session - after `backend.put_model`, every one of `actuator_gainprm`,
  `actuator_biasprm`, `actuator_dynprm`, `body_mass`, `geom_size`, `site_pos`, `dof_damping`,
  `actuator_gear`, `tendon_stiffness` has `requires_grad = False`. `backend.py:83-85` is a bare
  `mjw.put_model` with no grad request; `WarpSimStep.forward` takes only `(ctrl, qpos, qvel, act)`
  (`bridge.py:127`) and `_taped_backward` extracts `.grad` from exactly those four handles
  (`bridge.py:178, 193-198`). `backend.zero_grad` would not zero a model-array grad either
  (`backend.py:106-112`), so one would accumulate silently across epochs.
- Impact: **HANDOFF ledger items 8 (geometry / moment arms) and 10 (tendon compliance) cannot be
  pursued by gradient optimisation.** They are discrete sweeps only. Corroborated by
  `scripts/trajopt_hopper.py:289-290`, which optimises exactly two leaves and nothing model-side.
- Closes: `requires_grad = True` on the target array before the tape is recorded, that array added as
  a differentiable input to `WarpSimStep.forward`, and its grad captured in `_taped_backward` - none
  of which exists, and whether mjw's adjoint kernels propagate into `Model` is untested.

### BE-06 `open` (2026-09-11) `GRAD_CLAMP` silently converts `inf` to 0 and clamps per element at 1e4

- Evidence: `bridge.py:37-38, 107-116`. The only witness is a process-global `sanitized_nan_count`.
- Impact: a saturated-reward run cannot distinguish "no gradient" from "sanitised gradient" without
  reading that counter. Relevant to TR-05, where median gradient norms were 0.03-0.05.

### BE-07 `open` (2026-09-11) The trajectory-optimisation objective gradient has never been validated against finite differences on the muscle model

- Evidence: G3.5 bounds the **SHAC policy gradient** at cosine 0.99 despite BE-02
  (`research/phase3-hopper/results.md:195-197`). That is a different object from
  `scripts/trajopt_hopper.py`'s per-world open-loop `dJ/du`, whose objective contains a residual term
  depending explicitly on `qpos` at both ends of the cycle - precisely the path BE-02 corrupts. No
  measurement of it exists.
- Impact: the Phase 4 muscle **search** descended a possibly-biased gradient while its motor control
  descended an exact one, so "muscle fails at `T_c` 24/27/32 while motor passes" is confounded.
  Phase 4's existence claim itself is forward-verified and unaffected.
- Closes: AD-vs-FD cosine on `dJ/du` at >= 10 random muscle-hopper configurations, motor as control.

### BE-08 `fixed` (2026-09-11) A defect-free analysis route exists: float64 CPU MuJoCo reproduces the Warp muscle rollout exactly

- Evidence: measured this session. Replaying `hopper_muscle_T16_gait.npz`'s own 16 controls in float64
  CPU MuJoCo (4 substeps, `solver=Newton`, `jacobian=dense`) from the recorded phase-0 state
  reproduces the Warp-recorded closing state to max `|dqpos| =` **8.4e-07**, `|dqvel| =` **3.8e-06**,
  `|dact| =` **4.4e-08**, with per-phase agreement at the same order for all 16 steps. A second
  measurement put the cycled open-loop *divergence rate* of the motor gait at 3.74 cycles (CPU
  float64) against 3.70 (Warp float32) - 1% agreement on an exponentially growing quantity.
- Impact: any analysis expressible as forward rollouts plus finite differences can be done in float64
  CPU MuJoCo with **no dependence on BE-01, BE-02 or BE-06 at all**, at 0.4 ms per 16-step cycle.
  This is the basis of the Phase 5 return-map instrument. It does **not** mean the adjoint is fixed -
  anything that *trains* still carries BE-02.

---

## MF - Muscle model fidelity

### MF-01 `open` (2026-09-11) Neither the hopper nor myoLeg26 has any tendon compliance, so the muscle must supply all elastic energy return

- Evidence: verified by compiling all four assets. `tendon_stiffness` is identically zero in
  `hopper_muscle.xml`, `hopper_muscle_fast.xml`, `hopper_muscle_weak.xml` **and**
  `myoLeg26_BASELINE.xml`; `tendon_damping` likewise. myoLeg26 declares `springlength` on exactly four
  tendons (`edl_r/l_tendon` 0.368875, `fdl_r/l_tendon` 0.378773) but **no tendon declares
  `stiffness`**, so all four are inert.
- Impact: real hopping is dominated by tendon elastic energy return. A rigid-tendon model forces the
  muscle to do all the work at a bandwidth MF-10 measures as insufficient. This is a **substantive
  criticism of the myoLeg26 muscle class**, not a workaround the project declined.
- Related: MF-09 (MuJoCo cannot express series elasticity for a tendon-bound muscle at all).

### MF-02 `open` (2026-09-11) myoLeg26 uses `tausmooth = 0`, the discontinuous-time-constant gradient hazard the hopper deliberately patched

- Evidence: `myoLeg26_BASELINE.xml`, `dynprm = [0.01, 0.04, 0, 0, ...]` on all 26 actuators.
  `hopper_muscle.xml:11-14` sets `tausmooth = 0.02` with a header explaining that at 0 the time
  constant switches discontinuously on `sign(ctrl - act)` and `d(act_dot)/d(ctrl)` loses the tau term
  exactly where the gradient is needed.
- Closes: set `tausmooth` on myoLeg26 before any differentiable run, and record the fidelity cost.

### MF-03 `retracted` (2026-09-11) myoLeg26's solver is **not** a problem - a plausible-sounding claim that measurement refuted

- The claim, drafted and then checked: "myoLeg26 declares no `solver` attribute, so it defaults to CG,
  which `backend.check_model` rejects."
- **Refuted the same day.** `mujoco.MjModel.from_xml_path("myoleg/myoLeg26_BASELINE.xml")` compiles to
  `mjSOL_NEWTON`. MuJoCo's documented default solver *is* Newton, not CG. No override is needed.
- Recorded rather than deleted because it is an easy and confident-sounding wrong assumption about a
  model nobody has run yet, and because it is the cheapest possible illustration of the project's own
  rule: the entry cost one line of Python to falsify. The real myoLeg26 blockers are MF-02
  (`tausmooth = 0`) and MF-04 (contact geometry).

### MF-04 `open` (2026-09-11) myoLeg26 has **no** differentiable contact geometry at all, not merely mesh feet

- Evidence: measured by compiling the asset. Its 33 geoms are **31 mesh, 1 hfield, 1 plane**, and
  **every one of the 33 is collidable** (`geom_contype` or `geom_conaffinity` nonzero). There is not a
  single capsule or sphere in the model. The gradient contract admits plane/sphere/capsule only,
  because mesh, box and hfield return zero contact gradients upstream without warning
  (`backend.unsupported_pairs`, `backend.py:114-117`), and `assert_grad_contract` fails loudly at env
  construction (`base_env.py:70-73`).
- Impact: this is **stronger than the "all-mesh feet" framing in CLAUDE.md's layout note.** It is not
  a foot problem that a proxy on two geoms would fix - the whole collision model would have to be
  re-authored, and the hfield is a second, separate blocker. In May 2026 the contact-geometry grad
  fields were never enabled at all, so every contact gradient in that era was silently inert
  (`POSTMORTEM.md`); this entry is the model-side counterpart of that.
- Closes: capsule/sphere contact proxies covering every collidable geom that can touch the floor, the
  hfield replaced by a plane, and `assert_grad_contract` passing on the result.

### MF-05 `characterised` (2026-09-10) myoLeg26's muscle class derives peak force from `scale`/`acc0`, which is wrong on any other body

- Evidence: on a 15.82 kg hopper it gives 10-20x the motor variant's matched torque.
  `scripts/calibrate_hopper_muscle.py --check` verifies an explicit `gainprm[2]`.
- Note (2026-09-11): myoLeg26 itself already sets `scale = 1.0` on all 26 actuators with explicit
  per-muscle peak forces from 509.4 to 9500.9 N, so the hazard is in *porting* its class, not in the
  asset.

### MF-06 `open` (2026-09-11) The only published SHAC muscle result does not use a Hill muscle, so the comparison that motivated this project is not like for like

- Evidence: read from source. DiffRL's SNU humanoid (152 MTUs, its hardest task) computes
  `muscle_activation = actions * muscle_strengths` (`C:\Projects\DiffRL\envs\snu_humanoid.py:271`) and
  then `f = n * muscle_activation` (`dflex/dflex/sim.py:1237`, whose own comment reads "todo: add
  passive elastic and viscosity terms"). That is **constant tension along a routed path**: no
  activation state, no ODE, zero lag, no force-length curve, no force-velocity curve, no passive
  elasticity.
- Impact: **both mechanisms Phase 3 measured as this hopper's blockers are absent from it** -
  co-activation stiffness needs the force-length slope, and deactivation bandwidth needs the
  activation ODE. "SHAC trains muscle humanoids" is not evidence that SHAC trains Hill muscles.

### MF-07 `by-design` (2026-09-09) `hopper_muscle_fast.xml` is deliberately unphysiological

- Evidence: its own header. Differs from the base asset in exactly one number, `dynprm[1]`
  0.04 -> 0.01 (tau_deact 40 ms -> 10 ms), raising measured release rate from ~24.5 to ~96 N.m per
  control step. Useful precisely because it is a one-variable lever; must be labelled
  fidelity-relaxed in every result.

### MF-08 `open` (2026-09-11) This model does not clamp `act`, so activation outside [0,1] yields a spurious unit Floquet multiplier

- Evidence: `actuator_actlimited = False` and `actrange = [0, 0]` on `hopper_muscle.xml`.
  `mju_muscleDynamics` clamps internally, so `act > 1` with `ctrl = 1` gives `act_dot = 0` and a
  marginal mode of exactly 1.
- Impact: any stability analysis must count activation excursions outside [0,1] and flag them. The
  `T_c` 16 candidate stays within [0.037, 0.989], so it does not bite there.

### MF-09 `by-design` (2026-09-11) MuJoCo cannot express a series-elastic muscle-tendon unit

- Evidence: a muscle actuator bound to a tendon uses `ten_length` as the MTU length, so the tendon is
  rigid by construction; `<spatial stiffness>` adds a spring in **parallel**, not in series.
- Impact: any compliant-tendon experiment in this engine tests *elastic energy return in the leg*,
  **not** *series elasticity in the MTU*, and must be described that way. A genuine series element
  would need an extra DOF in the chain, changing `nq` and breaking comparability with every existing
  number.

### MF-10 `open` (2026-09-11) The deactivation constant is comparable to the entire stance phase

- Evidence: measured on the `T_c` 16 candidate. Stance is **5 of 16 control steps** (`ncon > 0` at
  phases 13, 14, 15, 0, 1) with peak ground reaction **866 N = 5.6 body weights** (weight 155.2 N),
  against a measured one-step activation-gap closure of 0.21-0.56, i.e. a release constant of
  **1.2-4.3 control steps**. Phase 3 measured the same mechanism as deactivation being 20-25x slower
  than a motor at matched peak torque.
- Impact: a correction requiring a muscle to *release* inside stance cannot be delivered at any gain.
  This is the quantitative form of the bandwidth argument, and it is the reason changing the task to
  walking (~36-step stance) is a live option.

---

## IN - Instruments and tooling

### IN-01 `open` (2026-09-11) `inspect_trajopt_candidate.py` reports multi-cycle state range as within-cycle amplitude

- Evidence: `scripts/inspect_trajopt_candidate.py:70-77` takes `height.max() - height.min()` over the
  whole `cycles * cycle` window, so on a diverging open-loop replay the figure is dominated by drift.
  Reported **0.508 m** for the `T_c` 16 muscle candidate (`inspect_muscle_T16.json`,
  `summary.height_range = 0.5077`) and 0.524 m for the motor candidate; the true **within-cycle**
  range, re-measured directly from the reference asset on 2026-09-11, is **0.0714 m** (min -0.0032,
  max +0.0682) against the real motor gait's per-cycle 0.1857 m.
- Impact: the "verdict: hopping" reasoning in `phase4-capability/results.md:271, 383, 390` rests
  partly on "the torso oscillates half a metre". It oscillates 38% of a real gait's amplitude.
- Closes: compute the range within a single cycle; see CL-04 for the documents still carrying it.
- **Note 2026-09-11.** Two within-cycle numbers are now on record and they differ by 2.5% because
  of the sample set, not the physics: **0.071374 m** over the reference table's 16 pre-step phases,
  and **0.073186 m** from `ReturnMap.roll`, which additionally samples the closing state after step
  16. Either is the right order; both are ~7x below the retracted 0.508 m. Quote which one you mean.

### IN-02 `open` (2026-09-11) `inspect_trajopt_candidate.py` has a second, previously unrecorded off-by-one in advance-per-cycle

- Evidence: `:73`, `advances = [x[(c+1)*cycle - 1] - x[c*cycle - 1 if c else 0] for c in ...]`.
  Because `trace()` appends the **post**-step state, `x[0]` is already one step in, so cycle 0 spans
  `cycle - 1` steps while every later cycle spans `cycle`. The first entry is systematically short.
- Impact: feeds `advance_drift = std/|mean|` (`:80`) and therefore the `DRIFTING` / `HOPPING` verdict.

### IN-03 `open` (2026-09-11) `inspect_trajopt_candidate.py` is blind to activation

- Evidence: `trace()` at `:53-65` records `cat([qpos[0], qvel[0]])` - 12 columns. `act` is never
  recorded, so the tool that classifies muscle candidates cannot see the muscle state.

### IN-04 `open` (2026-09-11) `inspect_trajopt_candidate.py` rounds states, which destroys a fixed point

- Evidence: `:149` writes states through `.round(5)`. A periodic-orbit state 1e-5 away from the true
  one can sit on a contact-event surface, where a Jacobian finite difference blows up as `1/eps`
  (measured: `||J||_F` rising from 4.7e3 at eps 1e-5 to 4.3e5 at 1e-7, with 11 of 11 columns changing
  the contact sequence).
- Impact: any orbit state must be serialised at full float64 precision.

### IN-05 `open` (2026-09-11) `trajopt_hopper.verify()` computes no residual and never inspects `act`

- Evidence: `scripts/trajopt_hopper.py:156-176` returns only `(alive, velocity, survived_fraction)`.
- Impact: amendment 1 justified excluding activation from the residual on the grounds that "the
  10-cycle open-loop verification catches exactly that". **That enforcement mechanism was never
  implemented**, and amendment 2 then cut verification to 3 cycles. See CL-01.

### IN-06 `open` (2026-09-11) `trajopt_hopper.py:423` hardcodes the residual bar instead of using `--residual-bar`

- Evidence: `restarts_above_gate_residual = int((residual > 0.4684).sum())` ignores
  `args.residual_bar`.

### IN-07 `characterised` (2026-09-11) `--verify-only` inflates the iteration counter

- Evidence: `trajopt_hopper.py:438` saves `done = already_done + args.iters` regardless of whether the
  loop ran, so verify-only invocations must pass `--iters 0`. `logs/phase4/muscle_T16.json` records
  `iters: 0` while its npz holds `iterations_done = 320`.

### IN-08 `characterised` (2026-09-11) `--skip-verify` writes no JSON at all

- Evidence: `trajopt_hopper.py:350-354` saves the checkpoint and returns.

### IN-09 `open` (2026-09-10) `find_period`'s 1.2 tolerance reports the harmonic in a third of stochastic worlds

- Evidence: `msk_warp/utils/gait.py:74-94`; `phase4-capability/results.md:61-65` records that in
  **3 of 8** stochastic-init worlds the residual at `T = 55` was low enough that the fundamental
  `T = 27` fell outside the window. A tolerance near 1.5 would recover all 8. Not changed, because
  changing it after the fact is a post-hoc choice and no gate depends on `T_ref`.

### IN-10 `open` (2026-09-11) `extract_gait_cycle.py --scales-from` has no dimension guard

- Evidence: `:144-147` substitutes external scales with no width check; an 11-wide scales file against
  a 17-wide muscle trace fails as a broadcast error inside `periodicity_residual`.

### IN-11 `characterised` (2026-09-10) SHAC's own evaluator does not report velocity, which is half of every behavioural gate

- Evidence: `shac.py:465-495` reports return, length and fall rate only. Always use
  `scripts/evaluate_hopper.py`.

### IN-12 `characterised` (2026-09-10) Checkpoint filenames record the *training* reward under the stochastic policy

- Impact: deterministic evaluation can differ enormously. Never read behaviour off a filename.

### IN-13 `open` (2026-09-11) Actor gradient norms survive only as 2-decimal stdout text

- Evidence: `shac.py:675-690` prints `{:.2f}`, which is 25-50% relative resolution in the 0.01-0.20
  band where the tracking run lived. Full-precision scalars go to tensorboard (`shac.py:640-643`, tag
  `grad_norm/before_clip`), but the venv has `tensorboardX` (writer) and **no** reader - no
  `tensorboard`, `tbparse`, or `torch.utils.tensorboard`.
- Note: the critic's `value iter` prints use `end='\r'`, so the `iter N:` summary lands mid-line and
  line-oriented grep will not isolate it; use `grep -oE "grad norm before clip [-0-9.]+"`.

### IN-14 `by-design` (2026-09-11) `scipy` is not installed and not in `pyproject.toml`

- Impact: numerical solvers (Newton, Levenberg-Marquardt, Riccati, eigen) must be numpy-only. The venv
  pins are fixed by ADR-0001 and are not worth a resolver risk.

### IN-15 `fixed` (2026-09-11) Projecting a state onto `jnt_range` moves it *off* a valid orbit, because MuJoCo's joint limits are soft

- Evidence: found by running the instrument's own controls, which all three failed. MuJoCo enforces
  joint limits through `solimp`/`solref` exactly like contact, so a physically valid state sits
  slightly outside `jnt_range`: the muscle hopper's standing orbit settles with `leg` at
  **+3.29e-04** against an upper limit of **0**. A hard clamp is therefore not a projection onto the
  feasible set but a move off the orbit. Measured from the settled standing point: the raw Newton
  step reaches residual **1.9e-12**, the *projected* step **4.3e-04**, and the solver then stalls
  forever at a point it had already found.
- Impact while live: the negative control stalled at 2.2e-07 instead of converging, and **both**
  positive controls stalled -- the motor gait at 9.36e-02 with its thigh limit flagged active. Had
  this not been caught by a control with a known answer, it would have produced a **false negative
  on the central muscle question**, and the false negative would have looked clean.
- Fixed: `shoot` does not project by default (`msk_warp/analysis/stability.py`). The constraint that
  actually binds is enforced where it belongs -- a trial step whose rollout terminates, by falling or
  by the engine's instability flags, is rejected outright. `Bounds` is now a reporting device;
  `active()` still matters for interpreting an orbit that rests on a limit, whose Jacobian there is
  one-sided. Pinned by `test_soft_joint_limits_must_not_be_projected`.
- Second-order lesson, also pinned: the first version of that regression test passed while the bug
  was still reachable, because `shoot` silently overrode an explicit `project=True` when `bounds`
  was `None`. A flag that cannot be exercised cannot be regression-tested.

---

## TR - Training and algorithm

### TR-01 `open` (2026-09-11) Staged training discards the optimiser state at every stage boundary

- Evidence: `scripts/train.py:58` calls `algo.load(init_policy, reset_optimizers=True)` with the flag
  **hardcoded**; no caller anywhere sets it to `False`. `SHAC.save` (`shac.py:727-733`) stores
  `[actor, critic, target_critic, obs_rms, ret_rms]` and no optimiser state; `SHAC.load` calls
  `_build_optimizers()`.
- Impact: with `betas = (0.7, 0.95)` the second moment needs ~20 steps of bias correction, so on
  60-90-epoch stages roughly 20-30% of every stage runs on a cold optimiser. **The 500-epoch G4.3
  result was six staged processes**, so this is a confound on that number.
  `scripts/trajopt_hopper.py` already does the right thing (`save_params` / `restore_optimiser`), so
  this is an internal inconsistency rather than a missing idea.
- Closes: append an optimiser dict to the checkpoint list (never restructure it - `ckpt[0]` and
  `ckpt[3]` are indexed positionally by `evaluate_hopper.py`, `extract_gait_cycle.py` and
  `diag_policy_engine_gap.py`) and flip the default.

### TR-02 `open` (2026-09-11) `init_policy.pt` is overwritten with a random network on every resumed stage

- Evidence: `shac.py:196` calls `self.save('init_policy')` inside `__init__`, i.e. **before**
  `train.py:58` loads the checkpoint. `logs/phase4/track_seed0/init_policy.pt` is timestamped at the
  start of the *last* stage.
- Impact: `scripts/visualize_progression.py:52, 355` treats that file as iteration 0 and is misled.

### TR-03 `open` (2026-09-11) `best_policy.pt` selection resets every stage

- Evidence: `best_policy_loss = np.inf` (`shac.py:235`) and `best_eval_return = -np.inf`
  (`shac.py:183`) are constructor state.
- Impact: "best" means best-within-the-last-stage. Visible in the artefacts as `iter_count` restarting
  and `eval iter 50` appearing four times in one stdout capture.

### TR-04 `open` (2026-09-11) `task_weight: 0.0` silently disables the fall penalty, and the failure mode it would have opposed is exactly what happened

- Evidence: `msk_warp/envs/hopper.py:456-460` short-circuits the whole of
  `HopperBaseEnv._compute_reward` when `task_weight == 0`. That removes the progress term, the height
  term, the angle term, the action penalty, **and the `-200 * height_diff^2` branch below the height
  threshold**. `configs/hopper_muscle_track.yaml:37` sets `task_weight: 0.0`.
- Impact: G4.3's measured failure is *monotonic height loss* (+0.034 at step 1 to -0.830 at step 48),
  and no reward term opposed it. This is a second, independent cause alongside TR-05, and it was not
  in the record before 2026-09-11.

### TR-05 `open` (2026-09-11) The tracking reward's *scale* accounts for roughly a quarter of the gradient deficit blamed entirely on its *shape*

- Evidence: the tracking run earns ~0.19 per step (return 7.1 over length 38.3) against the working
  motor run's ~4.54 (4544 over 1000) - a **24x** gap. `rew_scale` is unset and defaults to 1.0
  (`shac.py:122`). The observed actor-gradient-norm deficit is ~100x (median 0.03-0.05 against
  1.8-3.0), so it decomposes as roughly 24x scale times 4x shape.
- The shape half is real and separate: `r = exp(-k e)` gives `dr/dtheta = -k r de/dtheta`, so the
  per-step gradient weight **is** the reward, for every `k > 0`. Measured collapse across one 38-step
  episode: r = 0.809 (step 1), 0.125 (16), 0.029 (32), 0.0002 (48).
- Closes: derive `rew_scale` from the measured per-step ratio **and** raise `grad_norm` so it still
  never binds (see TR-08), then report the norms as the mechanism check.

### TR-06 `characterised` (2026-09-11) The tracking reward has exactly zero direct gradient with respect to activation

- Evidence: `HopperMuscleTrackEnv.shape_dims = 11` (`hopper.py:394`) excludes `act` from the tracked
  shape vector. The only `ctrl`-to-reward path is through the dynamics within the substep window.
- Note: deliberate, and defensible - `protocol.md:194-198` argues a phase-indexed activation target
  would demand a jump at the wrap beyond the deactivation bandwidth. Recorded because it bounds what
  the tracking reward can teach.

### TR-07 `by-design` (2026-09-11) `lr_schedule` supports only `linear`; every other value is a silent no-op

- Evidence: `shac.py:581-590`. The `else` branch sets the *logged* `lr` and leaves both optimisers at
  their constructed rate. So `constant` works, but so would any typo.

### TR-08 `characterised` (2026-09-11) `grad_norm: 1.0` never binds on the tracking run

- Evidence: max observed norm **0.20** over 320 epochs, with before == after clipping on every single
  epoch. So clipping is not the limiter - but it **would** become one if TR-05's 24x scale fix landed
  without raising the threshold.

---

## CL - Claims and protocol

### CL-01 `open` (2026-09-11) The periodicity-residual gate cannot answer the existence question at any activation scale, so the owed "re-run G4.2 with activation in the residual" cannot be discharged as written

- Evidence: measured on `hopper_muscle_T16_gait.npz`, with `R_mech` reproducing the recorded 0.31744
  exactly as an arithmetic check. The same candidate scores, against the 0.4684 gate: `R_mech`
  (11 comp, stage-0 motor scales) **0.3174 pass**; `R_full` with **unit** activation scales
  **0.4216 pass**; `R_full` with the **standing trace's** activation scales **12.1, fail by 26x**.
- No principled scale exists: there is no muscle gait to calibrate one from, and the motor gait's
  `act` array is zero-width, so the protocol's single-ruler requirement is unsatisfiable. The obvious
  choice (unit scales, since `act` is dimensionless on [0,1]) makes the muscle gate **easier**,
  because the `sqrt(D)` denominator grows from `sqrt(11)` to `sqrt(17)` faster than the activation
  error accumulates - the opposite of what amendment 1 feared.
- Compounded by IN-05: amendment 1's stated fallback enforcement was never implemented.
- Response: replace the residual gate with a fixed-point test (Phase 5). The residual stays what
  amendment 1 said it was, a shaping term.

### CL-02 `open` (2026-09-11) Activation was never the blocker - `trajopt_hopper.py` is over-parameterised by six physically redundant degrees of freedom

- Evidence: MuJoCo's `dyntype=muscle` computes `act_dot` from `(ctrl, act)` alone, with no dependence
  on length or velocity, so under a periodic `ctrl` the activation subsystem is autonomous and
  exponentially stable. Confirmed numerically: from a **different mechanical state and a different
  initial activation**, the end-of-cycle activation converges to
  `act* = [0.941422, 0.168312, 0.063191, 0.985747, 0.037268, 0.977356]` with **exactly 0.0e+00**
  difference, reaching machine precision in three cycles (deltas 1.3e-06, 8.4e-12, 4.2e-17, 0).
  Independently, the activation-to-mechanics block of the cycle Jacobian is bitwise zero.
- The candidate's chosen phase-0 activation is `[0.776, 0.595, 0.921, 0.100, 0.406, 0.765]`, up to
  **0.886** away from `act*` - and the celebrated per-muscle "activation closing errors"
  `[0.166, 0.427, 0.858, 0.886, 0.369, 0.212]` **are exactly `|act_0 - act*|`**.
- Impact: given a periodic control the periodic activation is *determined*, not free, yet
  `trajopt_hopper.py:124, 290` hands the optimiser `z[:, 11:17]` as six free initial-activation
  parameters with no gradient penalty for being wrong. It used them to buy one good-looking mechanical
  cycle out of a transient. **This applies to every muscle cell in the Phase 4 horizon sweep.** The
  fix is a removal of parameters, not the addition of a gate.

### CL-03 `open` (2026-09-11) No periodic orbit was found in the `T_c` 16 candidate's basin - but the negative is not yet clean

- Evidence: with `act` pinned to `act*`, the candidate's one-cycle mechanical residual is **0.410**
  (worse than the reported 0.3174, as expected once the crutch is removed). Damped Newton on
  `P(x) = x` stalls at 0.390; Levenberg-Marquardt reaches **0.076**, 5.4x better, then stalls with
  lambda climbing to 4e3. Two independent 20-start runs found 0 of 20 converged, floor 7.6e-2, and 0
  of 20 on `P^2` (32 steps).
- The instrument is sound: `cond(J - I) = 330`; the Jacobian is identical to 4 decimals across
  finite-difference steps from 1e-5 to 3e-3 with 0 of 11 columns perturbing the contact sequence; the
  linear model holds to cosine **1.00000** and relative error 0.000-0.012 over 12 random directions;
  and two structurally independent Jacobian routes (composed per-substep `mjd_transitionFD` versus
  full-cycle central differences) agree to **4.4e-9** relative Frobenius.
- **The caveat that keeps this open**: at an LM-stalled iterate, `||J||_F` scales as `1/eps` over six
  decades with 11 of 11 columns changing the contact sequence - the signature of a point *on* a
  contact-event surface, where the merit function has a kink. So LM may be converging to the kink
  rather than to a zero, and "did not converge" is **not** by itself evidence that no orbit exists.
  Every terminal iterate needs the eps-sweep slope test (slope <= -0.5 means discontinuous) before a
  negative is reported.
- Also open: the searched radius was sigma <= 0.60 from one candidate. A negative claim must be
  scoped by start count and radius.
- **Refinement 2026-09-11, and it strengthens the negative.** The kink caveat was *measured at this
  terminal iterate and does not apply*: the eps sweep there reports slope **+0.0000** over three
  clean decades, verdict `stable`, zero contact-sequence changes at every step at or below 1e-4.
  So Levenberg-Marquardt stalled at a genuine smooth local minimum of `||F||`, not on a
  contact-event surface. (The large-eps end of the grid *does* straddle an event -- `||J||_F` of
  `[16.5, 75.6, 215.3, 623.4, 7.92, 7.92, 7.92, 7.92]` across eps 1e-2 down to 1e-7, with 3 to 10
  columns changing the contact sequence in the first four -- which is why the slope must be fitted
  on the clean sub-window only. Fitting all eight gives +0.24 and hides a perfectly smooth
  small-eps regime.) The remaining caveat is scope alone: one start. `Outcome.ON_EVENT_BOUNDARY`
  and the sweep now ship in `msk_warp/analysis/stability.py`, so any future negative carries this
  check automatically.
- **New 2026-09-11:** a shooting step reached a state where MuJoCo warned "Nan, Inf or huge value
  in QACC" while `qpos` stayed finite, and the rollout was accepted, reporting an advance of
  **-918389 m**. A finite-state check is not sufficient; `ReturnMap.roll` now reads the engine's
  own `mjWARN_BADQACC`/`BADQVEL`/`BADQPOS` counters and terminates. Found by a test.
- **Positive control now passes, 2026-09-11, so the muscle negative is interpretable.** With the
  soft-limit defect IN-15 fixed, 24 perturbed starts (sigma 0.25) on the **motor** hopper at
  `T_c` 27 gave **5 converged fixed points**, the best at residual **1.57e-14**, advance
  **+1.7708 m/cycle** = **+3.935 m/s**, within-cycle torso range **0.1788 m**, eps verdict `stable`
  over 4 decades. That is the trained policy's own gait (3.83 m/s, 0.1857 m per cycle) recovered as
  an *exact* limit cycle -- the strongest validation the instrument has. The identical protocol on
  the **muscle** candidate gave **0 of 24**, best residual 7.59e-02. Still 24 starts at one radius:
  the multistart sweep is what scopes the claim.

### CL-04 `open` (2026-09-11) `phase4-capability/results.md` still quotes the retracted height figures

- Evidence: `results.md:271` (0.524 m, motor) and `:383, :390` (0.508 m, muscle, "the torso oscillates
  half a metre"). `HANDOFF.md:202-207` corrects them to a within-cycle 0.0714 m, re-measured
  independently on 2026-09-11, but `results.md` was never updated. See IN-01 for the cause.
- **`fixed` 2026-09-11.** `results.md` now carries a dated retraction note at `:279` and marks all
  four affected cells, at `:271`, `:277`, `:393` and `:400`. The tool bug itself (IN-01) and its
  companion off-by-one (IN-02) remain open.

### CL-05 `open` (2026-09-11) `docs/README.md:3` claims the research record is tracked in git; it is not

- Evidence: `git ls-files docs/` returned nothing before this file was added. The claim was true on
  `backup-pre-solo-author-2026-09-07` (commit `799d7f7`) and was lost in the 2026-09-07 history
  rewrite.

### CL-06 `open` (2026-09-11) `docs/README.md:10` points at a file that does not exist

- Evidence: references `docs/decisions/backend.md` ("gradient contract of the chosen backend, written
  in Phase 2"). `docs/decisions/` holds only ADR-0001 and ADR-0002.

### CL-07 `open` (2026-09-11) `docs/README.md:21`'s no-generated-artefacts rule is violated by 548 KB of them

- Evidence: `phase4-capability/` is 548 KB across 26 files, 38% of all of `docs/`, largest being
  `gait_motor_seed2_stoch_trace.npz` at 180 KB. `phase3-hopper/` has 39 files of the same kind.

### CL-08 `open` (2026-09-11) `logs/phase4/track_smoke`, cited as the cost basis for amendment 3, has been deleted

- Evidence: `phase4-capability/protocol.md:166` cites it; not on disk.

### CL-09 `open` (2026-09-11) The "cycles to fall from the spectral radius" arithmetic does not reproduce the one case where the answer is known

- Evidence: the trained motor gait's cycled open-loop replay falls at **3.74 cycles** (CPU float64;
  3.70 in Warp). Its measured spectral radius at the recorded start is **9.157** with four unstable
  multipliers, which predicts 1.77 cycles at an escape radius of 1.0 and 2.08 at 2.0 - short by a
  factor of ~1.8. Three checkable reasons: the recorded start is not a fixed point (`|F| = 0.18`), so
  the radius is evaluated off-orbit; the asymptotic rate understates the finite-time rate (the
  per-cycle Jacobian product's geometric mean falls from 38.2 to 6.05 by cycle 4); and the escape
  radius is not 1.0 (the muscle candidate survives a scaled residual of 1.35).
- Impact: **this arithmetic must be reported, never gated.** A coincidental agreement was observed for
  the muscle candidate (3.8 predicted against 3-4 measured) and must not be read as calibration. The
  calibratable quantity is finite-time growth inside the linear regime.

### CL-10 `open` (2026-09-11) The `T_c` 16 reference's feedforward sits on the control box boundary in 45% of cells, so a feedback correction has nowhere to go

- Evidence: of the 96 (phase, muscle) cells, `1 - ctrl < 0.05` in **22**, `ctrl < 0.05` in **21**, and
  only **53** have >= 0.05 headroom both ways (19 have >= 0.2). Worst: `ankle_plantar` mean upward
  headroom 0.169, `ankle_dorsi` mean downward headroom 0.163 - the pair that does the hopping work.
  Separately, `|ctrl - act| < 0.02` in 13 of 96 cells, inside the `tausmooth` smoothing window where a
  correction flips the activation/deactivation time-constant branch (large-signal gains differ by up
  to 4x even though the derivative is symmetric to 2e-4).
- Impact: the trajopt `tanh` parameterisation drove the control into the corners because nothing
  penalised it. `trajopt_hopper.py` already insets the *joint* parameterisation by
  `JOINT_MARGIN = 0.05` for the same class of reason; the control box needs the same treatment before
  any feedback experiment is fair.

### CL-11 `open` (2026-09-11) "It is open-loop unstable" was recorded as a limitation of the muscle result; the working motor gait is four times worse

- Evidence: measured spectral radius of the cycle Jacobian, same instrument, same protocol.
  **Muscle `T_c` 16 candidate: rho 2.167, two unstable multipliers** out of eleven. **Trained motor
  gait at its own period 27: rho 9.157, four unstable multipliers** -- and that gait evaluates at
  3.83 m/s with a 0% fall rate over 16 episodes. The converged motor fixed point found by shooting
  reads rho 9.008.
- Impact: `phase4-capability/results.md:448-454` lists "it is open-loop unstable" as one of four
  things bounding the muscle existence claim. It is not a muscle property. **Strong open-loop
  instability is what a hopping gait is**, which is why every one of them -- motor included -- needs
  feedback, and why Phase 4's own amendment 2 had to cut the verification from 10 cycles to 3. If
  anything the muscle candidate is the *more* benign orbit of the two, and a two-dimensional
  unstable subspace against 96 control parameters is a great deal of authority.
- Closes: restate the caveat in the phase record as "hopping orbits are open-loop unstable, this one
  included and less so than its control", with both radii cited.

### CL-12 `open` (2026-09-11) Phase 4's existence claim is retracted: none of its eleven candidates is a forward-travelling periodic orbit, for either actuation

- Evidence: `research/phase5-orbits/results.md`. Each of the eleven `logs/phase4/*_params.npz`
  candidates was tested for a fixed point of its own `T_c`-step return map -- activation pinned to
  `act*`, 96 starts per cell at sigma 0.05 to 0.60 of component scale, 1056 shoots, each with a
  two-route Jacobian and an eps certification. Control reconstruction verified against the exported
  reference to `max |dctrl| = 4.3e-08`.
- **Nine of eleven cells contain no fixed point**, best residuals 2.18e-02 to 9.94e-02. That
  includes **five of five motor cells**, which Phase 4 reported as passing at all four horizons and
  relied on as its positive control. So the gate's failure is **actuation-independent**: it could
  not distinguish a periodic orbit from a well-shaped transient, for muscle or motor.
- Two cells do yield genuine orbits, and neither travels forward: `motor_T16` world 185 at residual
  5.98e-15 hops **backwards** at -0.246 m/s (airborne 6 of 16 steps, `rho` 3.948), and
  `muscle_T32` world 154 at residual 1.42e-13 is a **bob**, airborne 1 of 32 steps, 0.1335 m of
  vertical excursion at +0.005 m/s (`rho` 6.984).
- The positive control that does pass is the **trained** motor policy's own control at its measured
  period 27: 5 of 24 starts converge, best residual 1.57e-14, +1.7708 m/cycle = **+3.935 m/s**,
  airborne 22 of 27, within-cycle height range 0.1788 m. That is the real gait recovered as an exact
  limit cycle at its own speed and amplitude, which is what licenses reading the negatives.
- **What is licensed**: "these candidates are not periodic orbits". **Not** "no periodic muscle orbit
  exists" -- the return map takes a control as given, so the sweep cannot search over controls. Five
  of the nine negatives have a terminal point reading `no_window`, so for those "a real local
  minimum" is not established either.
- Also retracted by implication: the framing that the muscle model "cannot close a cycle". It can --
  the `T_c` 32 bob is a real orbit of the unmodified physiological model. What has not been found is
  a muscle orbit with **both** a flight phase and forward travel.
- Closes: (i) remove the six redundant initial-activation parameters from `trajopt_hopper.py` per
  CL-02 and re-run `T_c` 16; (ii) joint `(x*, u)` shooting on CPU, which at 0.4 ms per cycle can
  afford thousands of restarts and is the only experiment that can answer the existence question.

### CL-13 `open` (2026-09-11) Velocity reach along the periodic manifold tracks **flight fraction**, not actuator type

- Evidence: project the velocity gradient onto the null space of the periodicity Jacobian, in a
  physical metric (state in units of its own scale, control in units of its own range), and read off
  the achievable velocity change per unit step along the manifold. Measured on four orbits spanning
  both actuators, same instrument, same metric:

  | orbit | flight | `rho` | retained | dv per unit step |
  |---|---|---|---|---|
  | motor trained gait `T_c` 27 | 22/27 | 9.008 | 0.178 | **7.80e-01** |
  | motor backwards hop `T_c` 16 | 6/16 | 3.948 | 0.288 | **6.11e-01** |
  | muscle bob `T_c` 32 | 1/32 | 6.984 | 0.011 | **2.52e-02** |
  | muscle standing `T_c` 16 | 0/16 | 0.736 | 0.000 | **7.97e-10** |

  The ordering of `dv per unit step` is **identical** to the ordering of flight fraction, and it
  spans both actuators. It does not track `rho`.
- Mechanism, and it is elementary once seen: in flight the body is ballistic and horizontal velocity
  is a free constant of the motion; in stance the foot is anchored by contact, so changing velocity
  means working against the constraint. **An orbit with no flight phase has no lever, whatever
  drives it** -- the standing orbit's 7.97e-10 is that statement at the limit.
- Impact: it explains, and retires, a failed experiment. Continuing in velocity from the muscle bob
  needs ~40 physical units of motion along the manifold to gain 1 m/s, against ~1.3 from the motor
  gait. That continuation was doomed by the **choice of starting orbit**, not by the muscle. Any
  future search for a fast muscle gait must start from, or first create, an orbit with a flight
  phase.
- Closes: the same measurement on a muscle orbit that *does* have flight, once one is found. If its
  `dv per unit step` lands near the motor values, muscle actuation is not the velocity constraint at
  all.

### IN-16 `characterised` (2026-09-11) Joint `(x, u)` shooting does not converge from the `T_c` 16 candidate, and four candidate explanations are ruled out

- The setup: unknowns `(x, u)`, 11 equations in 107 unknowns for single shooting, solved by
  minimum-norm damped Gauss-Newton in the dual form `delta = -A^T (A A^T + lambda I)^-1 F`.
  `msk_warp/analysis/gaitsearch.py`.
- **Not conditioning.** The scaled periodicity Jacobian has singular values 8.20 down to 0.111,
  condition **74**.
- **Not the control box.** Relaxed from [0, 1] to [-3, 4] with **zero** cells saturated, the solve
  stalls at exactly the same 1.9431e-02 it reaches inside the physiological box. Identical to four
  decimals across four box widths, so every run finds the same local minimum.
- **Not contact non-smoothness.** Per-segment linear-model error for a random step of 1e-2 in the
  physical metric is **0.0005 overall and below 0.002 in every segment**, including the two segments
  that contain a contact transition (2 and 14). The residual is smooth where the solver works.
- **Not single-vs-multiple shooting**, though it helps. Splitting the cycle raises the usable step
  from `k = 0.02` to `k = 0.10` and the gain per step from 1.0% to 6.2% (segments 1, 2, 4, 8, 16),
  and the Jacobian gets **9x cheaper** (0.382 s to 0.043 s) because it becomes 82% sparse. 301
  iterations at `S = 16` moved the closing error 8.24e-02 to 1.78e-02 and then stalled.
- **What is left: distance.** With the smallest singular value around 0.04 and a closing gap of 1.36
  RMS scale-units, the Newton step is of order 30 physical units while the linear model is exact
  only to about 1e-2. The solution, if it exists, is roughly three orders of magnitude further away
  than the trust region, so a local method seeded at the candidate cannot reach it.
- A closure homotopy (`set_homotopy`: demand a fraction `alpha` of the gap) was built to cross that
  distance and does not: adaptive stepping reached `alpha = 0.002` and needed 120 iterations there.
- Verified, so the negative is about the problem rather than the code: the hand-assembled sparse
  Jacobian agrees with a brute-force dense difference of the same residual to **4.3e-13**
  (`test_sparse_jacobian_matches_dense_finite_differences`), and four-segment chaining reproduces a
  full sixteen-step roll to 2.7e-15 with activation closing exactly.
- Closes: a method that is not seeded locally -- direct collocation with a proper NLP, or multistart
  over random controls rather than continuation from one candidate. CL-13 also says any such search
  should be seeded on an orbit that already has a flight phase.

### CL-14 `open` (2026-09-11) The Phase 4 ruler measured the noise floor of a policy rollout, not the tolerance for periodicity, and a soft penalty cannot fix it

- `R_ref = 0.2342` was extracted by `scripts/extract_gait_cycle.py` from a **trained policy's
  trajectory** -- stochastic initialisation, 400 control steps, not a limit cycle. A genuine periodic
  orbit has `R = 0` exactly. So the gate `R <= 2 * R_ref = 0.4684` calibrated *how noisy a policy
  rollout is*, and admits trajectories nowhere near periodic. That is the root cause of CL-12: nine
  of eleven Phase 4 candidates cleared the gate and none of them was an orbit.
- **Fixing the activation defect (CL-02) does not fix this.** Re-running `T_c` 16 muscle with the
  initial activation pinned and a warm-up cycle, 165 iterations at 256 worlds, gives best
  `v +1.37 m/s` at `R 0.244` and 1 of 256 verified at `+0.914 m/s, R 0.3468`. Put through the
  return map: **0 of 96 starts converge**, best residual 6.36e-02, terminal point certified
  `stable`. Still not an orbit. (The reconstruction agrees with the optimiser -- 0.338 against
  0.347 -- so the pipeline is sound; it is the criterion that is not.)
- **The penalty trades, it does not enforce, and that is actuation-independent.** Same cycle length,
  same iteration budget, same seed, only `lambda` changed:

  | model | `lambda` 4 | `lambda` 40 |
  |---|---|---|
  | muscle | `v +1.4`, `R 0.25` | `v +0.01`, **`R 0.036`** |
  | motor | `v +2.9`, `R 0.55` | `v -0.08`, **`R 0.058`** |

  Raising `lambda` tenfold buys a 7x tighter cycle and costs essentially all the speed, in **both**
  models. There is no setting that delivers `R -> 0` with `v > 0.5`; `J = v - lambda R` simply picks
  a point on a trade-off curve.
- Impact: **periodicity has to be a constraint, not a penalty term.** That is what the return-map
  shooting in `msk_warp/analysis` does, and it is why the two instruments disagree systematically.
  The trajectory optimiser remains the right tool for a *global* search over controls -- it explores
  256 restarts at once, which no local method can -- but its output is a starting point for shooting,
  never a periodic-orbit claim on its own.
- Closes: a formulation with `P(x) - x = 0` enforced (augmented Lagrangian on the existing GPU
  driver, or a proper NLP), with the return map as the acceptance test either way.
- **`fixed` 2026-09-12.** Implemented as `--auglag` in `scripts/trajopt_hopper.py`: the objective
  stays `v`, the 11-component closing error is the constraint, multipliers and per-world penalty
  weights are carried in the checkpoint. The constraint is the **vector** `e`, not the scalar
  `R = ||e||`, because `||e||` has an infinite-derivative kink at `e = 0` -- exactly where
  convergence has to happen -- while `||e||^2` is smooth there. It works: see CL-15.

### CL-15 `open` (2026-09-12) At `T_c` 16 every closed orbit is static, for **both** actuators -- the cycle length excludes forward travel, not the muscle

- Evidence. With periodicity **enforced** rather than penalised, the muscle run at `T_c` 16 drove the
  closing residual from 0.244 to **0.0269**, twelve times tighter than any Phase 4 candidate. Two of
  its tightest worlds were then put through the return map: **48 of 48 starts converged**, residuals
  down to **1.7e-15**. These are genuine periodic orbits to machine precision -- the first the
  trajectory optimiser has ever produced.
- **Every one of them is static**: advance +0.0001 m/cycle, **flight 0/16**, within-cycle height
  range 0.0036 to 0.0114 m, `rho` about 2.23 with one unstable multiplier.
- The population-level picture is a Pareto frontier, not a failure:

  | closing residual | worlds | fastest among them |
  |---|---|---|
  | `R < 0.30` | 55 | +0.916 m/s |
  | `R < 0.20` | 36 | +0.640 m/s |
  | `R < 0.10` | 8 | +0.287 m/s |
  | `R -> 0` | -- | **-> 0** |

- **The motor control, run identically, behaves the same way.** Its tightest worlds reach `R` 0.055
  at `v -0.17`, and worlds under `R < 0.1` top out near +0.3 m/s before vanishing as the constraint
  tightens. So the closure-versus-speed conflict at this cycle length is **actuation-independent**.
- Reading, and it reframes the phase: `T_c` 16 is the wrong cycle length for a travelling gait, for
  muscle *and* motor. Phase 4 chose it because it was the only horizon its broken residual gate
  passed for the muscle (CL-12, CL-14), and the choice has been inherited ever since. The motor's
  one **known** fast orbit -- +3.935 m/s, airborne 22 of 27, recovered to residual 1.57e-14 -- lives
  at `T_c` **27**, and CL-13 explains why a short cycle cannot travel: too little flight, and in
  stance the foot is anchored.
- Closes: sweep `T_c` with the augmented Lagrangian and the return map as acceptance, for both
  models. The right question is no longer "does a muscle orbit exist at `T_c` 16" -- it does, and it
  stands still -- but **"at what cycle length does a moving muscle orbit appear, and how does its
  speed compare with the motor's at the same `T_c`"**.

### IN-17 `fixed` (2026-09-12) Ranking worlds by the objective is meaningless under an augmented Lagrangian

- Evidence: each world carries its own multipliers, and `max_e (-y.e - rho/2 |e|^2) = |y|^2 / 2 rho`,
  so `argmax(J)` selects the world with the **largest multipliers** rather than the best gait.
  Observed directly: a run reported `J +80.8, R 1.436` while the tightest world in the same
  population sat at `R 0.058` and the fastest world under `R < 0.1` ran at +0.287 m/s. Read off the
  progress line alone, that looks like divergence; it was the ranking.
- Fixed: `summarise` ranks by constraint violation in `--auglag` mode and reports a Pareto row --
  how many worlds sit under each residual bar and how fast the quickest of them is. `orbit_shoot`
  also caps its per-orbit printing, since converged starts usually land on one orbit (48 of 48 in
  the cell above, all reporting identical advance, flight and `rho`).

---

## OP - Operational

### OP-01 `open` (2026-09-10) This machine cannot run background GPU jobs

- Evidence: ~24.8 GB committed of a 31.3 GB limit with zero Python running, 2.7-3.7 GB free RAM. The
  harness kills background tasks under memory pressure; **five runs died mid-flight**, one after it
  had converged and cleared its gate.
- Mitigation: foreground only, sized under ~10 minutes per call, chained with `--resume`.

### OP-02 `characterised` (2026-09-10) 512 worlds is 2.7x cheaper per world-step than 256

- Evidence: 284 against 767 microseconds per 1000 world-steps - the same super-linear batching win the
  ant speed recipe found. Blocked in practice by OP-01.

### OP-03 `characterised` (2026-09-10) Piping a run through `grep` block-buffers its output

- Mitigation: `grep --line-buffered`, or read the output file.

### OP-04 `by-design` (2026-09-11) The Phase 5 return-map analysis is float64 CPU and never touches the GPU, which removes OP-01 from its critical path

- Evidence: measured 0.40 ms per 16-step cycle (~2850 cycles/s); a full-cycle Jacobian is 22-35
  rollouts ~ 9-14 ms; a 60-iteration Levenberg-Marquardt shoot is 0.55 s. A 1000-start multistart is
  about 9 minutes on one core, roughly 5000x cheaper than the 256-world GPU trajectory optimisation it
  replaces, and it parallelises trivially.
- Forward warning: on MyoLeg26 (`nv` 60, `na` 26) the same routes cost ~200x200 Jacobians and several
  hundred rollouts per cycle, so this is cheap for the hopper only.
