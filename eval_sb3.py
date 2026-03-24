#!/usr/bin/env python3
"""
Evaluate a trained SB3 model in the PandaConveyorGym.
"""

import argparse
from collections import Counter
from pathlib import Path
import time


def load_model(algo, model_path, env):
    if algo == "ppo":
        from stable_baselines3 import PPO

        return PPO.load(model_path, env=env, device="cpu")

    if algo == "a2c":
        from stable_baselines3 import A2C

        return A2C.load(model_path, env=env, device="cpu")

    raise ValueError(f"Unsupported algorithm: {algo}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, choices=("ppo", "a2c"), default="ppo")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--gui-sleep", type=float, default=0.03)
    parser.add_argument("--control-conveyor", action="store_true")
    parser.add_argument("--conveyor-speed", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--vecnormalize", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", dest="deterministic", action="store_true")
    parser.add_argument("--stochastic", dest="deterministic", action="store_false")
    parser.add_argument("--randomize-cube", dest="randomize_cube", action="store_true")
    parser.add_argument("--fixed-cube", dest="randomize_cube", action="store_false")
    parser.set_defaults(deterministic=True, randomize_cube=True)
    args = parser.parse_args()

    from gym_env import PandaConveyorGym, ConveyorTaskConfig

    try:
        import stable_baselines3  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("stable-baselines3 is required to run eval_sb3.py") from exc

    try:
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("stable-baselines3 is required to run eval_sb3.py") from exc

    config = ConveyorTaskConfig(
        action_mode="velocity",
        control_conveyor=args.control_conveyor,
        conveyor_speed=args.conveyor_speed,
        randomize_cube=args.randomize_cube,
        max_steps=args.max_steps,
        render_every_n_episodes=1,
        gui_sleep_s=args.gui_sleep,
    )

    model_path = args.model or f"{args.algo}_panda_conveyor"
    model_dir = Path(model_path).resolve().parent

    env = DummyVecEnv([lambda: PandaConveyorGym(gui=args.gui, dt=0.01, config=config)])
    env.seed(args.seed)

    vecnorm_path = Path(args.vecnormalize).resolve() if args.vecnormalize else model_dir / "vecnormalize.pkl"
    if vecnorm_path.exists():
        env = VecNormalize.load(str(vecnorm_path), env)
        env.training = False
        env.norm_reward = False
        print(f"[eval] using VecNormalize stats from {vecnorm_path}")
    else:
        print("[eval] vecnormalize stats not found; evaluating without normalization wrapper")

    model = load_model(args.algo, model_path, env)

    success_count = 0
    done_reasons = Counter()
    release_sources = Counter()
    milestone_grasp_count = 0
    milestone_on_belt_count = 0
    milestone_end_count = 0
    rewards = []

    for ep in range(1, args.episodes + 1):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        last_info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, dones, infos = env.step(action)
            done = bool(dones[0])
            ep_reward += float(reward[0])
            last_info = infos[0] if infos else {}

        rewards.append(ep_reward)
        is_success = bool(last_info.get("is_success", False))
        success_count += int(is_success)
        done_reason = last_info.get("done_reason", "unknown")
        release_source = str(last_info.get("release_source", "none")).lower()
        milestone_grasp_count += int(bool(last_info.get("milestone_grasped", False)))
        milestone_on_belt_count += int(bool(last_info.get("milestone_on_belt", False)))
        milestone_end_count += int(bool(last_info.get("milestone_end", False)))
        done_reasons[str(done_reason)] += 1
        release_sources[release_source] += 1
        print(
            f"[eval] episode {ep:03d} reward={ep_reward:+.3f} "
            f"success={is_success} done_reason={done_reason} release_source={release_source}"
        )
        time.sleep(0.5)

    success_rate = success_count / max(1, args.episodes)
    mean_reward = sum(rewards) / max(1, len(rewards))
    print(f"[summary] deterministic={args.deterministic} episodes={args.episodes}")
    print(f"[summary] mean_reward={mean_reward:+.3f}")
    print(f"[summary] success_rate={success_rate:.3f} ({success_count}/{args.episodes})")
    print(f"[summary] done_reason_counts={dict(done_reasons)}")
    print(f"[summary] release_source_counts={dict(release_sources)}")
    print(
        "[summary] milestone_rates="
        f"grasp:{milestone_grasp_count/args.episodes:.3f} "
        f"on_belt:{milestone_on_belt_count/args.episodes:.3f} "
        f"end:{milestone_end_count/args.episodes:.3f}"
    )

    env.close()


if __name__ == "__main__":
    main()
