#!/usr/bin/env python3
"""
Visualize training progress from logs.
"""
import re
import matplotlib.pyplot as plt
from collections import defaultdict


def parse_training_log(filepath):
    """Parse PPO training logs to extract metrics."""
    metrics = defaultdict(list)
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find all episode reward mean values
    ep_rew_pattern = r'\|    ep_rew_mean\s+\|\s+([-\d.]+)'
    ep_rew_matches = re.findall(ep_rew_pattern, content)
    
    # Find all timesteps
    timestep_pattern = r'\|    total_timesteps\s+\|\s+(\d+)'
    timestep_matches = re.findall(timestep_pattern, content)
    
    for ts, reward in zip(timestep_matches, ep_rew_matches):
        metrics['timesteps'].append(int(ts))
        metrics['ep_rew_mean'].append(float(reward))
    
    return metrics


def plot_training(metrics):
    """Plot training curves."""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 1, 1)
    plt.plot(metrics['timesteps'], metrics['ep_rew_mean'], marker='o', linewidth=2)
    plt.xlabel('Total Timesteps', fontsize=12)
    plt.ylabel('Mean Episode Reward', fontsize=12)
    plt.title('PPO Training Progress', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add target line
    plt.axhline(y=0, color='r', linestyle='--', label='Zero reward')
    plt.axhline(y=1, color='g', linestyle='--', label='Grasp reward')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=150)
    print("Saved plot to training_progress.png")
    
    # Print summary
    if metrics['ep_rew_mean']:
        print(f"\nTraining Summary:")
        print(f"  Starting reward: {metrics['ep_rew_mean'][0]:.3f}")
        print(f"  Latest reward: {metrics['ep_rew_mean'][-1]:.3f}")
        print(f"  Best reward: {max(metrics['ep_rew_mean']):.3f}")
        print(f"  Timesteps: {metrics['timesteps'][-1]}")


if __name__ == "__main__":
    import sys
    logfile = sys.argv[1] if len(sys.argv) > 1 else "training.log"
    metrics = parse_training_log(logfile)
    
    if metrics['ep_rew_mean']:
        plot_training(metrics)
    else:
        print("No training metrics found in log")
