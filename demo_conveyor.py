#!/usr/bin/env python3
"""
Demo: Franka Panda with Conveyor Belt
Demonstrates the robot reaching towards the conveyor belt
"""

import numpy as np
import time
import mujoco
import sys
import threading
import tkinter as tk
import argparse
from env_conveyor import PandaConveyorEnv
from gym_env import PandaConveyorGym, ConveyorTaskConfig


def demo_reach_conveyor():
    """Demo: Robot reaches towards conveyor belt"""
    print("=" * 60)
    print("DEMO: Franka Panda with Conveyor Belt")
    print("=" * 60)
    
    # Initialize gym environment to use training reward/state logic
    config = ConveyorTaskConfig(
        action_mode="velocity",
        control_conveyor=False,
        conveyor_speed=0.25,
        randomize_cube=False,
        max_steps=500,
    )
    env = PandaConveyorGym(gui=True, dt=0.01, config=config)
    
    print("\n1. Reset robot to home position...")
    obs, _ = env.reset()
    print(f"   ✓ Home position: {obs}")
    print(f"   ✓ Joint angles (rad): {obs}")
    
    # Simulation parameters
    num_steps = 500
    conveyor_speeds = [0.25]
    
    print(f"\n2. Running {num_steps} simulation steps...")
    print("   Demonstrating different conveyor speeds\n")
    
    # IDs for pickup cube and hand (for scripted pickup/drop)
    cube_joint_id = mujoco.mj_name2id(env.env.model, mujoco.mjtObj.mjOBJ_JOINT, "pickup_cube_joint")
    hand_body_id = mujoco.mj_name2id(env.env.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    cube_body_id = mujoco.mj_name2id(env.env.model, mujoco.mjtObj.mjOBJ_BODY, "pickup_cube")
    qpos_adr = env.env.model.jnt_qposadr[cube_joint_id] if cube_joint_id >= 0 else -1
    qvel_adr = env.env.model.jnt_dofadr[cube_joint_id] if cube_joint_id >= 0 else -1

    pickup_time = 2.0
    carry_time = 2.0
    drop_pos = np.array([0.7, -0.6, 0.25])

    cumulative_reward = 0.0
    terminated = False
    truncated = False
    for speed_idx, speed in enumerate(conveyor_speeds):
        print(f"   Conveyor speed {speed_idx+1}/3: {speed:.1f}")
        
        for step in range(num_steps // len(conveyor_speeds)):
            sim_time = env.env.data.time
            # Simple trajectory: move joints in a coordinated way
            t = (speed_idx * num_steps // len(conveyor_speeds) + step) / num_steps
            
            # Create a reaching motion
            action = np.array([
                0.3 * np.sin(2*np.pi*t),           # joint1: rotate base
                -0.2 * np.cos(2*np.pi*t),          # joint2: lift
                0.1 * np.sin(3*np.pi*t),           # joint3: rotate
                -np.pi/4 - 0.3 * np.sin(2*np.pi*t), # joint4: wrist
                0.2 * np.cos(2*np.pi*t),           # joint5: rotate
                0.1 * np.sin(2*np.pi*t),           # joint6: rotate
                0.2 * np.cos(2*np.pi*t),           # joint7: rotate
            ])
            
            # Gym action space includes release command as the 8th entry.
            gym_action = np.concatenate([action, [0.0]]).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(gym_action)
            cumulative_reward += float(reward)

            # Visual marker at the cube position used in state + reward
            if env.env.viewer is not None and cube_body_id >= 0:
                cube_pos = env.env.data.xpos[cube_body_id].copy()
                env.env.viewer.add_marker(
                    pos=cube_pos,
                    size=np.array([0.012, 0.012, 0.012]),
                    rgba=np.array([1.0, 0.2, 0.2, 0.9]),
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                )

            # Scripted pickup/drop: attach cube to hand, then release above belt
            if cube_joint_id >= 0 and hand_body_id >= 0 and qpos_adr >= 0:
                if pickup_time <= sim_time < pickup_time + carry_time:
                    hand_pos = env.env.data.xpos[hand_body_id].copy()
                    hand_mat = env.env.data.xmat[hand_body_id].reshape(3, 3)
                    quat = np.zeros(4)
                    mujoco.mju_mat2Quat(quat, hand_mat)
                    cube_pos = hand_pos + np.array([0.0, 0.0, 0.06])
                    env.env.data.qpos[qpos_adr:qpos_adr+3] = cube_pos
                    env.env.data.qpos[qpos_adr+3:qpos_adr+7] = quat
                    if qvel_adr >= 0:
                        env.env.data.qvel[qvel_adr:qvel_adr+6] = 0.0
                elif sim_time >= pickup_time + carry_time:
                    env.env.data.qpos[qpos_adr:qpos_adr+3] = drop_pos
                    env.env.data.qpos[qpos_adr+3:qpos_adr+7] = np.array([1.0, 0.0, 0.0, 0.0])
                    if qvel_adr >= 0:
                        env.env.data.qvel[qvel_adr:qvel_adr+6] = 0.0
            time.sleep(env.env.dt)
            
            ee_pos = env.env.get_end_effector_pose()
            joint_summary = f"J1={obs[0]}, J4={obs[3]}, J7={obs[6]}"
            ee_summary = f"EE pos: {ee_pos}"
            reward_parts = info.get("reward_parts", {})
            print(
                f"      Step {step:3d}: reward={reward:+.3f} cum={cumulative_reward:+.3f} | "
                f"{joint_summary} | {ee_summary} | parts={reward_parts}",
                flush=True,
            )

            if terminated or truncated:
                reason = "success_on_belt_end" if terminated else "max_steps"
                print(
                    f"      Episode beendet bei Step {step} ({reason}), cum_reward={cumulative_reward:+.3f}",
                    flush=True,
                )
                break
        if terminated or truncated:
            break
    
    print("\n3. Return to home position...")
    for step in range(100):
        # Generate home position action (simple PID-like)
        home_q = np.array([0.0, 0.0, 0.0, -np.pi/4, 0.0, np.pi/2, np.pi/4])
        action = 0.5 * (home_q - obs)  # Simple P-control
        home_action = np.concatenate([action, [0.0]]).astype(np.float32)
        obs, _, _, _, _ = env.step(home_action)
    
    print("   ✓ Returned to home position")
    
    print("\n" + "=" * 60)
    print("✓ Demo completed!")
    print("=" * 60)
    if env.env.gui and env.env.viewer is not None:
        print("\nViewer läuft weiter mit Bewegung. Fenster schließen oder Strg+C zum Beenden.")
        try:
            t = 0.0
            while env.env.viewer.is_running():
                # Keep a gentle motion so the scene stays alive
                t += env.env.dt
                action = np.array([
                    0.2 * np.sin(2*np.pi*t),
                    -0.15 * np.cos(2*np.pi*t),
                    0.08 * np.sin(3*np.pi*t),
                    -np.pi/4 - 0.2 * np.sin(2*np.pi*t),
                    0.15 * np.cos(2*np.pi*t),
                    0.08 * np.sin(2*np.pi*t),
                    0.15 * np.cos(2*np.pi*t),
                ])
                keep_alive_action = np.concatenate([action, [0.0]]).astype(np.float32)
                env.step(keep_alive_action)
                if env.env.viewer is not None and cube_body_id >= 0:
                    cube_pos = env.env.data.xpos[cube_body_id].copy()
                    env.env.viewer.add_marker(
                        pos=cube_pos,
                        size=np.array([0.012, 0.012, 0.012]),
                        rgba=np.array([1.0, 0.2, 0.2, 0.9]),
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    )
                time.sleep(1.0 / 60.0)
        except KeyboardInterrupt:
            pass

    env.close()


def demo_gui_control():
    """Manual control with a mouse GUI (sliders), classic behavior."""
    print("=" * 60)
    print("MANUAL CONTROL - Franka Panda + Conveyor (Mouse GUI)")
    print("=" * 60)

    env = PandaConveyorEnv(gui=True)
    env.reset()

    # Shared state for GUI -> simulation
    lock = threading.Lock()
    action_vals = [float(env.data.qpos[jid]) for jid in env.joint_ids]
    conveyor_val = 0.25
    running = True

    def set_action(idx, val):
        nonlocal action_vals
        with lock:
            action_vals[idx] = float(val)

    def set_conveyor(val):
        nonlocal conveyor_val
        with lock:
            conveyor_val = float(val)

    def zero_all():
        nonlocal action_vals, conveyor_val
        with lock:
            action_vals = [float(env.data.qpos[jid]) for jid in env.joint_ids]
            conveyor_val = 0.0
        for i, s in enumerate(joint_sliders):
            s.set(action_vals[i])
        conveyor_slider.set(0.0)

    def release_cube():
        if hasattr(env, "set_magnetic_grasp"):
            env.set_magnetic_grasp(False)

    def sim_loop():
        nonlocal running
        try:
            while running and env.viewer is not None and env.viewer.is_running():
                with lock:
                    action = np.array(action_vals, dtype=np.float64)
                    speed = float(conveyor_val)
                env.step(action, conveyor_speed=speed, action_mode="position")
                time.sleep(env.dt)
        finally:
            running = False

    # Build GUI
    root = tk.Tk()
    root.title("Panda Control - Sliders")

    frame = tk.Frame(root, padx=10, pady=10)
    frame.pack()

    joint_sliders = []
    for i in range(7):
        if i < len(env.joint_ids):
            jid = env.joint_ids[i]
            jrange = env.model.jnt_range[jid].copy()
            if jrange[0] == jrange[1]:
                jrange = np.array([-2.9, 2.9], dtype=float)
        else:
            jrange = np.array([-2.9, 2.9], dtype=float)
        tk.Label(frame, text=f"Joint {i+1}").grid(row=i, column=0, sticky="w")
        slider = tk.Scale(
            frame,
            from_=float(jrange[0]),
            to=float(jrange[1]),
            resolution=0.01,
            orient=tk.HORIZONTAL,
            length=300,
            command=lambda v, idx=i: set_action(idx, v),
        )
        slider.set(action_vals[i])
        slider.grid(row=i, column=1, padx=5, pady=2)
        joint_sliders.append(slider)

    tk.Label(frame, text="Conveyor Speed").grid(row=7, column=0, sticky="w")
    conveyor_slider = tk.Scale(
        frame,
        from_=0.0,
        to=20.0,
        resolution=0.01,
        orient=tk.HORIZONTAL,
        length=300,
        command=set_conveyor,
    )
    conveyor_slider.set(conveyor_val)
    conveyor_slider.grid(row=7, column=1, padx=5, pady=6)

    btn_frame = tk.Frame(frame)
    btn_frame.grid(row=8, column=0, columnspan=2, pady=10)
    tk.Button(btn_frame, text="Zero All", command=zero_all).pack(side=tk.LEFT, padx=5)
    tk.Button(btn_frame, text="Release Cube", command=release_cube).pack(side=tk.LEFT, padx=5)
    tk.Button(btn_frame, text="Quit", command=root.destroy).pack(side=tk.LEFT, padx=5)

    sim_thread = threading.Thread(target=sim_loop, daemon=True)
    sim_thread.start()

    def on_close():
        nonlocal running
        running = False
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

    env.close()


def demo_gui_control_training():
    """Manual control mapped to training action space and reward logic."""
    print("=" * 60)
    print("MANUAL CONTROL - PandaConveyorGym (Training Settings)")
    print("=" * 60)

    config = ConveyorTaskConfig(
        action_mode="velocity",
        control_conveyor=False,
        conveyor_speed=0.25,
        randomize_cube=False,
        max_steps=500,
        render_every_n_episodes=0,
        gui_sleep_s=0.02,
    )
    env = PandaConveyorGym(gui=True, dt=0.01, config=config)
    _, _ = env.reset()
    print(f"obs_dim={env.observation_space.shape[0]}, act_dim={env.action_space.shape[0]}")

    lock = threading.Lock()
    act_dim = int(env.action_space.shape[0])
    action_vals = [0.0 for _ in range(act_dim)]
    running = True
    step_count = 0
    ep_reward = 0.0

    def set_action(idx, val):
        nonlocal action_vals
        with lock:
            action_vals[idx] = float(val)

    def zero_all():
        nonlocal action_vals
        with lock:
            action_vals = [0.0 for _ in range(act_dim)]
        for i, s in enumerate(joint_sliders):
            s.set(action_vals[i])

    def reset_episode():
        nonlocal step_count, ep_reward
        env.reset()
        step_count = 0
        ep_reward = 0.0

    def sim_loop():
        nonlocal running, step_count, ep_reward
        try:
            while running and env.env.viewer is not None and env.env.viewer.is_running():
                with lock:
                    action = np.array(action_vals, dtype=np.float32)
                _, reward, terminated, truncated, info = env.step(action)
                step_count += 1
                ep_reward += float(reward)
                print(
                    f"[manual-train] step={step_count} reward={reward:+.3f} ep_reward={ep_reward:+.3f} "
                    f"parts={info.get('reward_parts', {})}",
                    flush=True,
                )
                if terminated or truncated:
                    reason = "success_on_belt_end" if terminated else "max_steps"
                    print(
                        f"[manual-train] done reason={reason} step={step_count} ep_reward={ep_reward:+.3f}",
                        flush=True,
                    )
                    env.reset()
                    step_count = 0
                    ep_reward = 0.0
                time.sleep(env.env.dt)
        finally:
            running = False

    root = tk.Tk()
    root.title("PandaConveyorGym Control - Training Action Space")

    frame = tk.Frame(root, padx=10, pady=10)
    frame.pack()

    joint_sliders = []
    labels = [f"Action {i}" for i in range(act_dim)]
    for i in range(min(7, act_dim)):
        labels[i] = f"JointVel {i+1}"
    if act_dim >= 8:
        labels[7] = "Release Cmd"
    if act_dim >= 9:
        labels[8] = "Conveyor Cmd"
    for i in range(act_dim):
        tk.Label(frame, text=labels[i]).grid(row=i, column=0, sticky="w")
        slider = tk.Scale(
            frame,
            from_=-1.0,
            to=1.0,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            length=300,
            command=lambda v, idx=i: set_action(idx, v),
        )
        slider.set(0.0)
        slider.grid(row=i, column=1, padx=5, pady=2)
        joint_sliders.append(slider)

    btn_frame = tk.Frame(frame)
    btn_frame.grid(row=act_dim, column=0, columnspan=2, pady=10)
    tk.Button(btn_frame, text="Zero All", command=zero_all).pack(side=tk.LEFT, padx=5)
    tk.Button(btn_frame, text="Reset Episode", command=reset_episode).pack(side=tk.LEFT, padx=5)
    tk.Button(btn_frame, text="Quit", command=root.destroy).pack(side=tk.LEFT, padx=5)

    sim_thread = threading.Thread(target=sim_loop, daemon=True)
    sim_thread.start()

    def on_close():
        nonlocal running
        running = False
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

    env.close()


def demo_ik_pick():
    """Inverse kinematics demo: pick the cube and place it on the belt."""
    print("=" * 60)
    print("IK PICK DEMO - Franka Panda + Conveyor")
    print("=" * 60)

    env = PandaConveyorEnv(gui=True)
    env.reset()

    # IDs
    hand_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    cube_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "pickup_cube")
    weld_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_weld")
    if hand_body_id < 0 or cube_body_id < 0:
        print("❌ Hand oder Cube nicht gefunden. Prüfe die Szene.")
        env.close()
        return
    if weld_id >= 0:
        env.data.eq_active[weld_id] = 0

    # IK gains
    kp = 2.5
    max_vel = 0.6
    grasp_dist = 0.04
    posture_gain = 0.05

    # Targets
    lift_offset = np.array([0.0, 0.0, 0.12])
    belt_drop = np.array([0.7, -0.6, 0.25])

    phase = "approach"
    phase_start = env.data.time
    qpos_target = np.array([env.data.qpos[jid] for jid in env.joint_ids], dtype=np.float64)
    home_q = np.array([0.0, 0.0, 0.0, -np.pi/4, 0.0, np.pi/2, np.pi/4], dtype=np.float64)
    is_grasped = False

    try:
        while env.viewer is not None and env.viewer.is_running():
            # Current positions
            cube_pos = env.data.xpos[cube_body_id].copy()
            hand_pos = env.data.xpos[hand_body_id].copy()

            if phase == "approach":
                target = cube_pos + np.array([0.0, 0.0, 0.08])
                if np.linalg.norm(hand_pos - target) < grasp_dist:
                    phase = "grasp"
                    phase_start = env.data.time
            elif phase == "grasp":
                target = cube_pos + np.array([0.0, 0.0, 0.02])
                # Activate weld when close enough
                if not is_grasped and weld_id >= 0 and np.linalg.norm(hand_pos - cube_pos) < 0.03:
                    env.data.eq_active[weld_id] = 1
                    is_grasped = True
                if env.data.time - phase_start > 0.8:
                    phase = "lift"
                    phase_start = env.data.time
            elif phase == "lift":
                target = cube_pos + lift_offset
                if env.data.time - phase_start > 1.0:
                    phase = "move"
                    phase_start = env.data.time
            elif phase == "move":
                target = belt_drop
                if np.linalg.norm(hand_pos - target) < 0.05:
                    phase = "release"
                    phase_start = env.data.time
            elif phase == "release":
                target = belt_drop
                if is_grasped and weld_id >= 0:
                    env.data.eq_active[weld_id] = 0
                    is_grasped = False
                if env.data.time - phase_start > 0.8:
                    phase = "done"
            else:
                target = belt_drop

            # Jacobian for hand position
            jacp = np.zeros((3, env.model.nv))
            mujoco.mj_jacBody(env.model, env.data, jacp, None, hand_body_id)

            # Position error
            err = target - hand_pos
            dq = jacp.T @ (kp * err)
            # Posture regularization to keep arm stable
            dq[:7] += posture_gain * (home_q - qpos_target)
            dq = np.clip(dq, -max_vel, max_vel)

            # Map to 7 joints
            action = np.zeros(7, dtype=np.float64)
            for i, joint_id in enumerate(env.joint_ids):
                dof_adr = env.model.jnt_dofadr[joint_id]
                action[i] = dq[dof_adr]

            # Gripper command
            if phase in ("grasp", "lift", "move"):
                gripper = 1.0
            else:
                gripper = -1.0

            # Integrate to position targets for stability
            qpos_target = qpos_target + action * env.dt
            env.step(np.concatenate([qpos_target, [gripper]]), conveyor_speed=0.25, action_mode="position")
            time.sleep(env.dt)

    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conveyor demos and manual control")
    parser.add_argument("--manual-train", action="store_true", help="Manual control in PandaConveyorGym action space")
    parser.add_argument("--manual", action="store_true", help="Manual control with base PandaConveyorEnv")
    parser.add_argument("--ik", action="store_true", help="Run IK pick-and-place demo")
    args = parser.parse_args()

    if args.manual_train:
        demo_gui_control_training()
    elif args.manual:
        demo_gui_control()
    elif args.ik:
        demo_ik_pick()
    else:
        demo_reach_conveyor()
