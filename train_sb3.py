#!/usr/bin/env python3
"""
Minimal Stable-Baselines3 training script for PandaConveyorGym.
"""

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

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

            @staticmethod
            def _mean_ep_info(ep_info_buffer, key: str):
                values = []
                for item in ep_info_buffer:
                    if key not in item:
                        continue
                    value = item[key]
                    if isinstance(value, bool):
                        values.append(1.0 if value else 0.0)
                    else:
                        try:
                            values.append(float(value))
                        except (TypeError, ValueError):
                            continue
                if not values:
                    return None
                return sum(values) / len(values)

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

                ep_info_buffer = getattr(self.model, "ep_info_buffer", None)
                if ep_info_buffer:
                    reached_grasped_rate = self._mean_ep_info(ep_info_buffer, "reached_grasped")
                    reached_on_belt_rate = self._mean_ep_info(ep_info_buffer, "reached_on_belt")
                    grasped_ratio_mean = self._mean_ep_info(ep_info_buffer, "grasped_ratio")
                    on_belt_ratio_mean = self._mean_ep_info(ep_info_buffer, "on_belt_ratio")

                    if reached_grasped_rate is not None:
                        self.logger.record("rollout/reached_grasped_rate", reached_grasped_rate)
                    if reached_on_belt_rate is not None:
                        self.logger.record("rollout/reached_on_belt_rate", reached_on_belt_rate)
                    if grasped_ratio_mean is not None:
                        self.logger.record("rollout/grasped_ratio_mean", grasped_ratio_mean)
                    if on_belt_ratio_mean is not None:
                        self.logger.record("rollout/on_belt_ratio_mean", on_belt_ratio_mean)

            def _on_step(self) -> bool:
                return True

        self.callback = _CallbackImpl(self)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--control-conveyor", action="store_true")
    parser.add_argument("--render-every", type=int, default=0)
    parser.add_argument("--gui-sleep", type=float, default=0.02)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--ent-coef-final", type=float, default=0.003)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--run-name", type=str, default="")
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("stable-baselines3 is required to run train_sb3.py") from exc
    try:
        import tensorboard  # noqa: F401
        has_tensorboard = True
    except Exception:
        has_tensorboard = False

    run_name = args.run_name.strip() or f"ppo_panda_{datetime.now():%Y%m%d_%H%M%S}"
    if args.num_envs < 1:
        raise ValueError("--num-envs must be >= 1")

    log_root = Path(args.log_dir)
    run_dir = log_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = ConveyorTaskConfig(
        action_mode="velocity",
        control_conveyor=args.control_conveyor,
        conveyor_speed=3.0,
        randomize_cube=True,
        max_steps=500,
        render_every_n_episodes=args.render_every,
        gui_sleep_s=args.gui_sleep,
    )

    info_keywords = (
        "is_success", "done_reason",
        "reached_grasped", "reached_on_belt",
        "grasped_ratio", "on_belt_ratio",
    )

    if args.num_envs == 1:
        env = PandaConveyorGym(gui=args.gui, dt=0.01, config=config)
        env = Monitor(env, filename=str(run_dir / "monitor.csv"), info_keywords=info_keywords)
    else:
        def make_env(rank: int) -> Callable[[], PandaConveyorGym]:
            def _init() -> PandaConveyorGym:
                return PandaConveyorGym(gui=False, dt=0.01, config=config)
            return _init

        env_fns = [make_env(i) for i in range(args.num_envs)]
        env = SubprocVecEnv(env_fns)
        env = VecMonitor(env, filename=str(run_dir / "vec_monitor.csv"), info_keywords=info_keywords)

    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    print(f"Using {args.num_envs} parallel env(s)")

    # GUI render env: runs in main thread so MuJoCo viewer works.
    # The viewer is opened for one episode, then closed again so it doesn't
    # sit idle and waste resources between renders.
    render_callback = None
    if args.gui and args.num_envs > 1:
        from stable_baselines3.common.callbacks import BaseCallback
        render_every = max(1, args.render_every) if args.render_every else 5

        # render_every_n_episodes=0 disables the internal GUI-toggle mechanism
        # in PandaConveyorGym.reset() that would otherwise close our viewer.
        # gui_sleep_s=0: sleep is handled in the callback loop below.
        render_config = ConveyorTaskConfig(**{
            **config.__dict__,
            "gui_sleep_s": 0.0,
            "render_every_n_episodes": 0,
        })

        class _RenderCallback(BaseCallback):
            def __init__(self):
                super().__init__()
                self._rollout = 0

            def _on_rollout_end(self) -> None:
                self._rollout += 1
                if self._rollout % render_every != 0:
                    return
                env = PandaConveyorGym(gui=True, dt=0.01, config=render_config)
                vec_norm = self.model.get_vec_normalize_env()
                try:
                    obs, _ = env.reset()
                    done = False
                    while not done:
                        obs_input = vec_norm.normalize_obs(obs[np.newaxis])[0] if vec_norm is not None else obs
                        action, _ = self.model.predict(obs_input, deterministic=False)
                        obs, _, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated
                        time.sleep(args.gui_sleep)
                finally:
                    env.close()

            def _on_step(self) -> bool:
                return True

        render_callback = _RenderCallback()
        print(f"GUI viewer will open at rollout {render_every} (every {render_every} rollout(s) thereafter)")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=512,
        batch_size=256,
        n_epochs=5,
        clip_range=0.15,
        vf_coef=0.5,
        max_grad_norm=0.5,
        learning_rate=linear_schedule(3e-4, 3e-5),
        ent_coef=args.ent_coef,
        use_sde=True,
        sde_sample_freq=4,
        device="cpu",
        tensorboard_log=str(log_root) if has_tensorboard else None,
    )
    from stable_baselines3.common.callbacks import CallbackList
    ent_coef_callback = EntCoefScheduleCallback(args.ent_coef, args.ent_coef_final).callback
    callbacks = [ent_coef_callback]
    if render_callback is not None:
        callbacks.append(render_callback)
    callback = CallbackList(callbacks)

    if has_tensorboard:
        model.learn(total_timesteps=args.timesteps, tb_log_name=run_name, callback=callback)
    else:
        print("TensorBoard package not installed; continuing without TensorBoard logging.")
        model.learn(total_timesteps=args.timesteps, callback=callback)
    model.save(str(run_dir / "ppo_panda_conveyor"))
    env.save(str(run_dir / "vecnormalize.pkl"))
    print(f"Run directory: {run_dir}")
    if has_tensorboard:
        print(f"TensorBoard: tensorboard --logdir {log_root}")
    env.close()


if __name__ == "__main__":
    main()
