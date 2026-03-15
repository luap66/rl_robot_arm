"""
Demo script for Franka Emika Panda robot arm environment using MuJoCo
"""

import numpy as np
from env import PandaRobotEnv
import time


def demo_basic_movement():
    """Demonstrate basic movement of the Panda robot."""
    print("Starting Panda Robot Environment Demo (MuJoCo)...")
    
    # Create environment with GUI
    env = PandaRobotEnv(gui=True, dt=0.001)
    
    try:
        # Reset environment
        print("\n1. Resetting environment to home position...")
        obs = env.reset()
        print(f"   Initial joint positions: {obs}")
        time.sleep(1)
        
        # Get initial end effector pose
        pos, orn = env.get_end_effector_pose()
        print(f"   End effector position: {pos}")
        print(f"   End effector orientation (quaternion): {orn}")
        time.sleep(1)
        
        # Demonstrate joint movement
        print("\n2. Executing random joint velocity commands...")
        for step in range(500):
            # Generate random action
            action = np.random.uniform(-0.5, 0.5, size=env.num_dofs)
            
            # Execute action
            obs, reward, done, info = env.step(action)
            
            # Print status every 50 steps
            if step % 50 == 0:
                pos, orn = env.get_end_effector_pose()
                print(f"   Step {step}: EE Position: {pos}, Reward: {reward:.4f}")
        
        # Return to home position
        print("\n3. Returning to home position...")
        home_position = np.array([0, 0, 0, -np.pi/4, 0, np.pi/2, np.pi/4])
        
        for _ in range(200):
            # Use position control to move to home
            current_pos = env._get_observation()
            action = (home_position - current_pos) * 2  # Simple proportional control
            action = np.clip(action, -1, 1)
            obs, reward, done, info = env.step(action)
        
        final_pos, final_orn = env.get_end_effector_pose()
        print(f"   Final end effector position: {final_pos}")
        print(f"   Final joint positions: {obs}")
        
        print("\nDemo completed successfully!")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("Environment closed.")


if __name__ == "__main__":
    demo_basic_movement()
