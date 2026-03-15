#!/usr/bin/env python3
"""
Evaluate a trained SB3 model in the PandaConveyorGym.
"""

import argparse
import time

from gym_env import PandaConveyorGym, ConveyorTaskConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ppo_panda_conveyor")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--gui-sleep", type=float, default=0.03)
    parser.add_argument("--control-conveyor", action="store_true")
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("stable-baselines3 is required to run eval_sb3.py") from exc

    config = ConveyorTaskConfig(
        action_mode="velocity",
        control_conveyor=args.control_conveyor,
        conveyor_speed=0.25,
        randomize_cube=False,
        max_steps=500,
        render_every_n_episodes=1,
        gui_sleep_s=args.gui_sleep,
    )

    env = PandaConveyorGym(gui=True, dt=0.01, config=config)
    model = PPO.load(args.model, device="cpu")

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
        print(f"[eval] episode {ep} reward={ep_reward:.3f}")
        time.sleep(0.5)

    env.close()


if __name__ == "__main__":
    main()
