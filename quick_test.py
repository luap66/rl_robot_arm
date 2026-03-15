#!/usr/bin/env python3
"""
Quick test to verify reward shaping works and agent can grasp.
"""
import numpy as np
from gym_env import PandaConveyorGym, ConveyorTaskConfig


def test_reward_signal():
    """Test that positive rewards are being received."""
    config = ConveyorTaskConfig(
        randomize_cube=False,
        max_steps=200,
    )
    
    env = PandaConveyorGym(gui=False, dt=0.01, config=config)
    
    print("Testing reward signal for 5 episodes...")
    print("=" * 60)
    
    episode_rewards = []
    
    for ep in range(5):
        obs, _ = env.reset()
        ep_reward = 0.0
        grasped = False
        
        for step in range(config.max_steps):
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            
            # Check for grasp
            if not grasped and info.get("reward_parts", {}).get("grasp", 0) > 0:
                grasped = True
                print(f"  Episode {ep+1}: GRASP at step {step+1}! Cumulative reward: {ep_reward:.2f}")
            
            if terminated or truncated:
                break
        
        episode_rewards.append(ep_reward)
        if not grasped:
            print(f"  Episode {ep+1}: No grasp. Cumulative reward: {ep_reward:.2f}")
    
    print("=" * 60)
    print(f"Mean episode reward: {np.mean(episode_rewards):.2f}")
    print(f"Reward range: [{min(episode_rewards):.2f}, {max(episode_rewards):.2f}]")
    
    if np.mean(episode_rewards) > -10:
        print("✓ Reward signal looks reasonable! Ready to train.")
    else:
        print("✗ Reward signal still too negative. Check config.")
    
    env.close()


if __name__ == "__main__":
    test_reward_signal()
