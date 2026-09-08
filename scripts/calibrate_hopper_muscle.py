"""Calibrate hopper_muscle.xml peak isometric forces against hopper_motor.xml torque authority.

myoLeg26's muscle class derives peak force from ``gainprm[3]`` (scale) and the actuator's
``acc0``, which is calibrated to myoLeg26's segment inertias. On the 15.8 kg hopper that yields
1700-4300 N.m of joint torque, 10-20x what hopper_motor.xml's ``gear=200`` motors deliver, so the
two models would not be the same control problem.

This script sweeps each muscle's joint through its full range at full activation, finds the pose
where that muscle produces the most joint torque, and reports the explicit ``gainprm[2]`` peak
force that puts the maximum at the motor variant's 200 N.m. Peak force enters the muscle gain
linearly, so one measurement at a reference force determines it. Run with ``--check`` to verify
the values currently written into the asset.

    .venv/Scripts/python.exe scripts/calibrate_hopper_muscle.py
    .venv/Scripts/python.exe scripts/calibrate_hopper_muscle.py --check
"""

import argparse

import mujoco
import numpy as np

from msk_warp import resolve_model_path

TARGET_TORQUE = 200.0   # N.m, matches hopper_motor.xml gear=200 at ctrl=+-1
REFERENCE_FORCE = 1000.0
GRID = 41


def _torques(model, qpos, act):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.act[:] = act
    mujoco.mj_forward(model, data)
    return np.array(data.qfrc_actuator, dtype=float)


def _driven_dof(model):
    """DOF each muscle produces the most torque on, at the neutral pose."""
    base = _torques(model, model.qpos0, np.zeros(model.nu))
    eye = np.eye(model.nu)
    return [int(np.abs(_torques(model, model.qpos0, eye[i]) - base).argmax()) for i in range(model.nu)]


def sweep(model):
    """Per muscle: (name, joint name, max active torque, pose, max passive torque)."""
    dofs = _driven_dof(model)
    eye = np.eye(model.nu)
    out = []
    for i in range(model.nu):
        dof = dofs[i]
        jid = int(np.where(model.dof_jntid == dof)[0][0])
        adr = model.jnt_qposadr[jid]
        lo, hi = model.jnt_range[jid]
        best = best_q = passive = 0.0
        for q in np.linspace(lo, hi, GRID):
            qpos = model.qpos0.copy()
            qpos[adr] = q
            rest = _torques(model, qpos, np.zeros(model.nu))
            active = abs(float(_torques(model, qpos, eye[i])[dof] - rest[dof]))
            passive = max(passive, abs(float(rest[dof])))
            if active > best:
                best, best_q = active, q
        out.append((
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i),
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid),
            best, best_q, passive,
        ))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify the asset's forces instead of solving for them")
    ap.add_argument("--tol", type=float, default=0.02, help="relative tolerance for --check")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(resolve_model_path("assets/hopper_muscle.xml"))
    if not args.check:
        model.actuator_gainprm[:, 2] = REFERENCE_FORCE
        model.actuator_biasprm[:, 2] = REFERENCE_FORCE

    rows = sweep(model)
    worst = 0.0
    for name, joint, torque, pose, passive in rows:
        if args.check:
            err = abs(torque - TARGET_TORQUE) / TARGET_TORQUE
            worst = max(worst, err)
            print(f"{name:14s} {joint:12s} peak torque {torque:7.2f} N.m at q={pose:+.3f}  "
                  f"passive <= {passive:5.2f} N.m  rel err {err:.3%}")
        else:
            force = TARGET_TORQUE / torque * REFERENCE_FORCE
            print(f"{name:14s} {joint:12s} peak torque {torque:6.2f} N.m at F={REFERENCE_FORCE:.0f} N, "
                  f"q={pose:+.3f}  ->  force = {force:7.0f} N")

    if args.check:
        print(f"\nworst relative error {worst:.3%} (tolerance {args.tol:.1%})")
        if worst > args.tol:
            raise SystemExit(f"calibration drift: {worst:.3%} > {args.tol:.1%}")
        print("hopper_muscle.xml peak forces match hopper_motor.xml torque authority")


if __name__ == "__main__":
    main()
