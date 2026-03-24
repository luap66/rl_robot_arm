#!/usr/bin/env python3
"""
Minimal Stable-Baselines3 training script for PandaConveyorGym.
"""

import argparse
from datetime import datetime
from pathlib import Path

from gym_env import PandaConveyorGym, ConveyorTaskConfig


def linear_schedule(initial_value: float, final_value: float):
    def _schedule(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * progress_remaining

    return _schedule


class EntCoefScheduleCallback:
    def __init__(self, initial_value: float, final_value: float):
        from stable_baselines3.common.callbacks import BaseCallback

        self.initial_value = float(initial_value)
        self.final_value = float(final_value)

        class _CallbackImpl(BaseCallback):
            def __init__(self, outer):
                super().__init__()
                self.outer = outer

            def _scheduled_value(self) -> float:
                progress_remaining = float(self.model._current_progress_remaining)
                return self.outer.final_value + (
                    self.outer.initial_value - self.outer.final_value
                ) * progress_remaining

            def _on_training_start(self) -> None:
                self.model.ent_coef = self._scheduled_value()

            def _on_rollout_end(self) -> None:
                value = self._scheduled_value()
                self.model.ent_coef = value
                self.logger.record("train/ent_coef", value)

            def _on_step(self) -> bool:
                return True

        self.callback = _CallbackImpl(self)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--control-conveyor", action="store_true")
    parser.add_argument("--render-every", type=int, default=0)
    parser.add_argument("--gui-sleep", type=float, default=0.02)
    parser.add_argument("--ent-coef", type=float, default=0.03)
    parser.add_argument("--ent-coef-final", type=float, default=0.005)
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
        conveyor_speed=3.0,
        randomize_cube=False,
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
    ent_coef_callback = EntCoefScheduleCallback(args.ent_coef, args.ent_coef_final).callback
    if has_tensorboard:
        model.learn(total_timesteps=args.timesteps, tb_log_name=run_name, callback=ent_coef_callback)
    else:
        print("TensorBoard package not installed; continuing without TensorBoard logging.")
        model.learn(total_timesteps=args.timesteps, callback=ent_coef_callback)
    model.save(str(run_dir / "ppo_panda_conveyor"))
    print(f"Run directory: {run_dir}")
    if has_tensorboard:
        print(f"TensorBoard: tensorboard --logdir {log_root}")
    env.close()


if __name__ == "__main__":
    main()
