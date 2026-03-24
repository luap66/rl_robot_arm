#!/usr/bin/env python3
"""
Förderband-Test: 8 Würfel in verschiedenen Orientierungen fallen auf das Band.
Misst X-Transport und Y-Drift.  Nutzt den Slide-Joint-Ansatz (MuJoCo issue #547).
"""

import numpy as np
import mujoco
import mujoco.viewer
import time
from pathlib import Path

SCENE = Path(__file__).parent / "test_belt_scene.xml"

# Bandparameter (World-Koordinaten; conveyor_base bei 0.7 -0.6 0.0)
BELT_X_MIN = 0.7 - 1.0   # -0.3
BELT_X_MAX = 0.7 + 1.0   #  1.7
BELT_Y_MIN = -0.6 - 0.25
BELT_Y_MAX = -0.6 + 0.25
BELT_SURFACE_Z = 0.185    # Oberseite des Bandgeoms

BELT_SPEED = 0.25         # m/s  (ctrl = speed, da gainprm == damping)

CUBE_NAMES = [
    ("cube0", "aufrecht"),
    ("cube1", "45° um Z"),
    ("cube2", "45° um X"),
    ("cube3", "30° um Y"),
    ("cube4", "90° um X  (Kante)"),
    ("cube5", "60° um Z"),
    ("cube6", "diag. 45° (X+Z)"),
    ("cube7", "180° um Z (Kopf)"),
]


def on_belt(pos):
    return (BELT_X_MIN <= pos[0] <= BELT_X_MAX and
            BELT_Y_MIN <= pos[1] <= BELT_Y_MAX and
            pos[2] < BELT_SURFACE_Z + 0.12)


def _ids(model):
    belt_drive = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "conv_belt_drive")
    belt_slide = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,    "conv_belt_slide")
    return belt_drive, belt_slide


def belt_step(model, data, belt_drive_id, belt_slide_qposadr, belt_speed):
    if belt_drive_id >= 0:
        data.ctrl[belt_drive_id] = belt_speed
    mujoco.mj_step(model, data)
    # Infinite-belt trick: keep belt surface visually in place every step
    if belt_slide_qposadr >= 0:
        data.qpos[belt_slide_qposadr] = 0.0


def run_gui():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data  = mujoco.MjData(model)

    belt_drive_id, belt_slide_id = _ids(model)
    belt_slide_qposadr = int(model.jnt_qposadr[belt_slide_id]) if belt_slide_id >= 0 else -1
    cube_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name, _ in CUBE_NAMES}

    print(f"conv_belt_drive actuator: {belt_drive_id}")
    print(f"Band läuft mit {BELT_SPEED} m/s\n")

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    start_pos = {name: data.xpos[cid].copy() for name, cid in cube_ids.items()}

    landed    = {name: False for name, _ in CUBE_NAMES}
    land_x    = {}

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.5, -0.6, 0.3]
        viewer.cam.distance  = 4.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth   = 160

        step = 0
        wall_t0 = time.perf_counter()
        while viewer.is_running():
            target_sim_time = time.perf_counter() - wall_t0
            substeps = 0
            while data.time < target_sim_time and substeps < 30:
                belt_step(model, data, belt_drive_id, belt_slide_qposadr, BELT_SPEED)
                step += 1
                substeps += 1
                for name, _ in CUBE_NAMES:
                    cid = cube_ids[name]
                    pos = data.xpos[cid]
                    if not landed[name] and on_belt(pos):
                        landed[name] = True
                        land_x[name] = pos[0]
                if step % 400 == 0:
                    print(f"t={data.time:.1f}s", end="  |  ")
                    for name, label in CUBE_NAMES:
                        cid = cube_ids[name]
                        pos = data.xpos[cid]
                        if on_belt(pos):
                            dx = pos[0] - start_pos[name][0]
                            dy = pos[1] - start_pos[name][1]
                            print(f"{label}: ΔX={dx:+.3f} ΔY={dy:+.4f}", end="  ")
                    print()
            viewer.sync()
            if substeps == 0:
                time.sleep(0.0005)

    print("\n" + "=" * 65)
    print(f"{'Orientierung':<22} {'Aufprall-X':>10} {'End-X':>8} "
          f"{'ΔX':>8} {'ΔY (Drift)':>12}")
    print("-" * 65)
    for name, label in CUBE_NAMES:
        cid = cube_ids[name]
        pos = data.xpos[cid]
        lx  = land_x.get(name, float("nan"))
        dx  = pos[0] - start_pos[name][0]
        dy  = pos[1] - start_pos[name][1]
        print(f"{label:<22} {lx:>10.3f} {pos[0]:>8.3f} {dx:>+8.3f} {dy:>+12.4f}")


def run_headless():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data  = mujoco.MjData(model)

    belt_drive_id, belt_slide_id = _ids(model)
    belt_slide_qposadr = int(model.jnt_qposadr[belt_slide_id]) if belt_slide_id >= 0 else -1
    cube_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name, _ in CUBE_NAMES}

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    start_pos = {name: data.xpos[cid].copy() for name, cid in cube_ids.items()}

    steps = int(15.0 / model.opt.timestep)
    for _ in range(steps):
        belt_step(model, data, belt_drive_id, belt_slide_qposadr, BELT_SPEED)

    print(f"\nBandgeschwindigkeit: {BELT_SPEED} m/s  |  Simulationszeit: 15 s")
    print("=" * 65)
    print(f"{'Orientierung':<22} {'End-X':>8} {'ΔX':>8} {'ΔY (Drift)':>12}  {'Auf Band?':>9}")
    print("-" * 65)
    for name, label in CUBE_NAMES:
        cid  = cube_ids[name]
        pos  = data.xpos[cid]
        dx   = pos[0] - start_pos[name][0]
        dy   = pos[1] - start_pos[name][1]
        belt = "✓" if on_belt(pos) else "am Bandende"
        print(f"{label:<22} {pos[0]:>8.3f} {dx:>+8.3f} {dy:>+12.4f}  {belt:>11}")


if __name__ == "__main__":
    import sys
    if "--headless" in sys.argv or "-n" in sys.argv:
        run_headless()
    else:
        run_gui()
