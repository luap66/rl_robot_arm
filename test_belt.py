#!/usr/bin/env python3
"""
Förderband-Test: 8 Würfel in verschiedenen Orientierungen fallen nacheinander
auf das Band und werden transportiert. Misst X-Bewegung und Y-Drift.
"""

import numpy as np
import mujoco
import mujoco.viewer
import time
from pathlib import Path

SCENE = Path(__file__).parent / "test_belt_scene.xml"

# Bandparameter (World-Koordinaten)
BELT_X_MIN = 0.7 - 1.0   # -0.3
BELT_X_MAX = 0.7 + 1.0   #  1.7
BELT_Y_MIN = -0.6 - 0.25
BELT_Y_MAX = -0.6 + 0.25
BELT_SURFACE_Z = 0.185    # Oberseite des Bandgeoms

BELT_SPEED = 0.25         # m/s

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


def _belt_geom_id(model):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "conv_belt_surface")


_BELT_MU    = 1.5
_BELT_GAIN  = 40.0   # proportional → kein Jitter, kein Deadband nötig
_LAT_GAIN   = 100.0
_YAW_GAIN   = 0.5


def _apply_belt_forces(model, data, belt_geom_id: int, belt_speed_x: float) -> None:
    """Apply belt friction forces via qfrc_applied (belt surface is condim=1)."""
    # Belt axes in world frame (X = transport, Y = lateral).
    belt_x = np.array([1.0, 0.0, 0.0])
    belt_y = np.array([0.0, 1.0, 0.0])

    processed: set = set()
    for i in range(data.ncon):
        con = data.contact[i]
        if con.geom1 == belt_geom_id:
            other_geom = con.geom2
        elif con.geom2 == belt_geom_id:
            other_geom = con.geom1
        else:
            continue
        body_id = model.geom_bodyid[other_geom]
        if body_id == 0 or body_id in processed:
            continue
        processed.add(body_id)

        # cvel linear part is world-space linear velocity of the body frame.
        v_world = data.cvel[body_id][3:6]
        v_x     = float(np.dot(v_world, belt_x))
        v_y     = float(np.dot(v_world, belt_y))
        omega_z = float(data.cvel[body_id][2])
        mass    = float(model.body_mass[body_id])
        F_cap   = _BELT_MU * mass * 9.81

        F_x = float(np.clip(_BELT_GAIN * (belt_speed_x - v_x), -F_cap, F_cap))
        F_y = float(np.clip(-_LAT_GAIN * v_y,                  -F_cap, F_cap))
        T_z = float(np.clip(-_YAW_GAIN * omega_z,              -0.2,   0.2))

        mujoco.mj_applyFT(
            model, data,
            np.array([F_x, F_y, 0.0]),
            np.array([0.0, 0.0, T_z]),
            data.xpos[body_id].copy(), body_id,
            data.qfrc_applied,
        )


def belt_step(model, data, belt_geom_id: int, belt_speed: float):
    data.qfrc_applied[:] = 0.0
    if belt_speed != 0.0:
        _apply_belt_forces(model, data, belt_geom_id, belt_speed)
    mujoco.mj_step(model, data)


def run_gui():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data  = mujoco.MjData(model)

    geom_id = _belt_geom_id(model)
    cube_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name, _ in CUBE_NAMES
    }

    print(f"conv_belt_surface geom: {geom_id}")
    print(f"Band läuft mit {BELT_SPEED} m/s\n")

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    start_pos = {name: data.xpos[cid].copy() for name, cid in cube_ids.items()}

    landed    = {name: False for name, _ in CUBE_NAMES}
    land_x    = {}
    land_time = {}

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.5, -0.6, 0.3]
        viewer.cam.distance  = 4.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth   = 160

        step = 0
        while viewer.is_running():
            belt_step(model, data, geom_id, BELT_SPEED)
            step += 1
            t = data.time

            for name, label in CUBE_NAMES:
                cid = cube_ids[name]
                pos = data.xpos[cid]
                if not landed[name] and on_belt(pos):
                    landed[name] = True
                    land_x[name] = pos[0]
                    land_time[name] = t

            if step % 400 == 0:
                print(f"t={t:.1f}s", end="  |  ")
                for name, label in CUBE_NAMES:
                    cid = cube_ids[name]
                    pos = data.xpos[cid]
                    if on_belt(pos):
                        dx = pos[0] - start_pos[name][0]
                        dy = pos[1] - start_pos[name][1]
                        print(f"{label}: ΔX={dx:+.3f} ΔY={dy:+.4f}", end="  ")
                print()

            viewer.sync()
            time.sleep(model.opt.timestep)

        print("\n" + "=" * 60)
        print(f"{'Orientierung':<22} {'Aufprall-X':>10} {'End-X':>8} "
              f"{'ΔX (Transport)':>15} {'ΔY (Drift)':>12}")
        print("-" * 60)
        for name, label in CUBE_NAMES:
            cid = cube_ids[name]
            pos = data.xpos[cid]
            lx  = land_x.get(name, float("nan"))
            dx  = pos[0] - start_pos[name][0]
            dy  = pos[1] - start_pos[name][1]
            print(f"{label:<22} {lx:>10.3f} {pos[0]:>8.3f} {dx:>15.3f} {dy:>12.4f}")


def run_headless():
    """Kein Fenster – nur Ausgabe der Messwerte."""
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data  = mujoco.MjData(model)

    geom_id  = _belt_geom_id(model)
    cube_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name, _ in CUBE_NAMES}

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    start_pos = {name: data.xpos[cid].copy() for name, cid in cube_ids.items()}

    steps = int(15.0 / model.opt.timestep)
    for _ in range(steps):
        belt_step(model, data, geom_id, BELT_SPEED)

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
