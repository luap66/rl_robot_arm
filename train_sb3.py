#!/usr/bin/env python3
"""
Minimal Stable-Baselines3 training script for PandaConveyorGym.
"""

import argparse
from datetime import datetime
from pathlib import Path

from gym_env import PandaConveyorGym, ConveyorTaskConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--control-conveyor", action="store_true")
    parser.add_argument("--render-every", type=int, default=0)
    parser.add_argument("--gui-sleep", type=float, default=0.02)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--run-name", type=str, default="")
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("stable-baselines3 is required to run train_sb3.py") from exc
    try:
        import tensorboard  # noqa: F401
        has_tensorboard = True
    except Exception:
        has_tensorboard = False

    run_name = args.run_name.strip() or f"ppo_panda_{datetime.now():%Y%m%d_%H%M%S}"
    log_root = Path(args.log_dir)
    run_dir = log_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = ConveyorTaskConfig(
        action_mode="velocity",
        control_conveyor=args.control_conveyor,
        conveyor_speed=5.0,
        randomize_cube=True,
        max_steps=500,
        render_every_n_episodes=args.render_every,
        gui_sleep_s=args.gui_sleep,
    )

    env = PandaConveyorGym(gui=args.gui, dt=0.01, config=config)
    env = Monitor(
        env,
        filename=str(run_dir / "monitor.csv"),
        info_keywords=("is_success", "done_reason"),
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=1024,
        batch_size=256,
        learning_rate=3e-4,
        ent_coef=args.ent_coef,
        device="cpu",
        tensorboard_log=str(log_root) if has_tensorboard else None,
    )
    if has_tensorboard:
        model.learn(total_timesteps=args.timesteps, tb_log_name=run_name)
    else:
        print("TensorBoard package not installed; continuing without TensorBoard logging.")
        model.learn(total_timesteps=args.timesteps)
    model.save(str(run_dir / "ppo_panda_conveyor"))
    print(f"Run directory: {run_dir}")
    if has_tensorboard:
        print(f"TensorBoard: tensorboard --logdir {log_root}")
    env.close()


if __name__ == "__main__":
    main()
