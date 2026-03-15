"""
Franka Emika Panda Robot Arm Environment with MuJoCo
This module provides a simulation environment for the Franka Emika Panda robot arm
using the official MuJoCo Menagerie model from Google DeepMind.
"""

import numpy as np
import mujoco as mj
import mujoco.viewer as mjviewer
from pathlib import Path
from typing import Tuple, Dict, Any, Optional


class PandaRobotEnv:
    """
    Franka Emika Panda robot arm environment for reinforcement learning.
    
    This environment simulates the Franka Emika Panda 7-DoF robot arm in MuJoCo
    using the official model from Google DeepMind's mujoco_menagerie.
    """
    
    def __init__(self, gui: bool = True, dt: float = 0.001, model_path: Optional[str] = None):
        """
        Initialize the Panda robot environment.
        
        Args:
            gui: Whether to display MuJoCo viewer
            dt: Time step for simulation
            model_path: Path to the Panda XML model file. If None, uses mujoco_menagerie path.
        """
        self.gui = gui
        self.dt = dt
        self.viewer = None
        self.model = None
        self.data = None
        self.model_path = model_path
        
        # Robot configuration
        self.num_dofs = 7
        self.action_space_size = self.num_dofs
        
        # Joint constraints (from Franka specification)
        self.max_velocity = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
        self.max_force = np.array([87, 87, 87, 87, 12, 120, 120])
        
        # Joint indices for the arm
        self.joint_names = [f"joint{i}" for i in range(1, 8)]
        
        self._setup()
    
    def _setup(self):
        """Initialize the MuJoCo environment and load the official Panda robot model."""
        # Find the model path
        if self.model_path is None:
            # Try to find mujoco_menagerie in parent directory
            menagerie_path = Path(__file__).parent.parent / "mujoco_menagerie" / "franka_emika_panda" / "panda.xml"
            if menagerie_path.exists():
                self.model_path = str(menagerie_path)
            else:
                raise FileNotFoundError(
                    "Could not find mujoco_menagerie. Please provide model_path or ensure "
                    "mujoco_menagerie is cloned to /home/paulw/projects/mujoco_menagerie"
                )
        
        # Load the official Panda model
        self.model = mj.MjModel.from_xml_path(self.model_path)
        self.data = mj.MjData(self.model)
        
        # Set timestep
        self.model.opt.timestep = self.dt
        
        # Get joint indices for the arm (first 7 revolute joints)
        self.joint_ids = []
        for i in range(1, 8):  # Joints 1-7 are the arm
            joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, f"joint{i}")
            if joint_id >= 0:
                self.joint_ids.append(joint_id)
        
        # Get actuator indices for the arm (first 7 actuators)
        self.actuator_ids = []
        for i in range(1, 8):  # Actuators 1-7 are for the arm joints
            actuator_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}")
            if actuator_id >= 0:
                self.actuator_ids.append(actuator_id)
        
        # Initialize viewer if GUI is enabled
        if self.gui:
            self.viewer = mjviewer.launch_passive(self.model, self.data)
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.
        
        Returns:
            Initial observation (joint positions)
        """
        # Reset to home position (neutral position)
        home_position = np.array([0, 0, 0, -np.pi/4, 0, np.pi/2, np.pi/4])
        
        # Set joint positions using the actual joint IDs
        for i, joint_id in enumerate(self.joint_ids):
            if i < len(home_position):
                self.data.qpos[joint_id] = home_position[i]
                self.data.qvel[joint_id] = 0.0
        
        # Forward kinematics
        mj.mj_forward(self.model, self.data)
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step of the environment with the given action.
        
        Args:
            action: Joint velocity commands for the robot (size: num_dofs)
            
        Returns:
            observation: Current joint positions
            reward: Reward for this step
            done: Whether episode is finished
            info: Additional information
        """
        # Clip action to valid range
        action = np.clip(action, -1, 1)
        action = action * self.max_velocity
        
        # Apply control to the arm actuators
        for i, actuator_id in enumerate(self.actuator_ids):
            if i < len(action):
                self.data.ctrl[actuator_id] = action[i]
        
        # Step simulation
        mj.mj_step(self.model, self.data)
        
        # Render if viewer is active
        if self.viewer and self.gui:
            self.viewer.sync()
        
        # Get observation
        observation = self._get_observation()
        
        # Calculate reward (simple: penalize joint movement)
        reward = -np.sum(np.abs(action)) * self.dt
        
        # Check if episode is done
        done = False
        
        return observation, reward, done, {}
    
    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation from the environment.
        
        Returns:
            Array of joint positions for the first 7 arm joints
        """
        # Get joint positions from the first 7 arm joints
        positions = np.array([self.data.qpos[joint_id] for joint_id in self.joint_ids])
        
        return positions
    
    def get_end_effector_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the current end effector position and orientation.
        
        Returns:
            position: 3D position of end effector (center of hand)
            orientation: Quaternion orientation [x, y, z, w]
        """
        # Get the hand body position and orientation
        body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "hand")
        
        if body_id >= 0:
            # Position is the center of the hand body
            position = self.data.xpos[body_id].copy()
            # Orientation from the rotation matrix
            xmat = self.data.xmat[body_id].reshape(3, 3)
            orientation = self._matrix_to_quaternion(xmat)
        else:
            position = np.array([0.0, 0.0, 0.0])
            orientation = np.array([0.0, 0.0, 0.0, 1.0])
        
        return position, orientation
    
    @staticmethod
    def _matrix_to_quaternion(mat: np.ndarray) -> np.ndarray:
        """
        Convert a 3x3 rotation matrix to a quaternion [x, y, z, w].
        
        Args:
            mat: 3x3 rotation matrix
            
        Returns:
            Quaternion [x, y, z, w]
        """
        trace = mat[0, 0] + mat[1, 1] + mat[2, 2]
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (mat[2, 1] - mat[1, 2]) * s
            y = (mat[0, 2] - mat[2, 0]) * s
            z = (mat[1, 0] - mat[0, 1]) * s
        elif mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
            s = 2.0 * np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2])
            w = (mat[2, 1] - mat[1, 2]) / s
            x = 0.25 * s
            y = (mat[0, 1] + mat[1, 0]) / s
            z = (mat[0, 2] + mat[2, 0]) / s
        elif mat[1, 1] > mat[2, 2]:
            s = 2.0 * np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2])
            w = (mat[0, 2] - mat[2, 0]) / s
            x = (mat[0, 1] + mat[1, 0]) / s
            y = 0.25 * s
            z = (mat[1, 2] + mat[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1])
            w = (mat[1, 0] - mat[0, 1]) / s
            x = (mat[0, 2] + mat[2, 0]) / s
            y = (mat[1, 2] + mat[2, 1]) / s
            z = 0.25 * s
        
        return np.array([x, y, z, w])
    
    def set_joint_angles(self, angles: np.ndarray):
        """
        Set robot joint angles directly.
        
        Args:
            angles: Target joint angles (size: num_dofs)
        """
        for i, joint_id in enumerate(self.joint_ids):
            if i < len(angles):
                self.data.qpos[joint_id] = angles[i]
        
        # Forward kinematics
        mj.mj_forward(self.model, self.data)
    
    def render(self):
        """Render the environment (if GUI is enabled)."""
        if self.viewer and self.gui:
            self.viewer.sync()
    
    def close(self):
        """Close the environment and disconnect from MuJoCo viewer."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
    
    def __del__(self):
        """Cleanup when environment is deleted."""
        self.close()

if __name__ == "__main__":
    # Example usage
    env = PandaRobotEnv(gui=True)
    
    try:
        obs = env.reset()
        print(f"Initial observation: {obs}")
        
        # Run for a few steps
        for i in range(500):
            action = np.random.uniform(-1, 1, size=env.num_dofs)
            obs, reward, done, info = env.step(action)
            
            if i % 50 == 0:
                # Get end effector pose
                pos, orn = env.get_end_effector_pose()
                print(f"Step {i} - End effector position: {pos}")
        
    finally:
        env.close()
