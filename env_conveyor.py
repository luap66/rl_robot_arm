"""
Franka Panda with Conveyor Belt Environment
"""

import numpy as np
import mujoco as mj
import mujoco.viewer as mjviewer
from pathlib import Path
from typing import Tuple, Dict, Any, Optional


def _apply_belt_forces(model, data, belt_geom_id: int, belt_speed_x: float) -> None:
    """Apply belt friction forces via qfrc_applied to bodies touching the belt.

    The belt surface uses condim=1 (normal support only, no tangential friction
    from the contact model).  This function provides the full friction behaviour
    programmatically so nothing fights the contact solver.

    Physics per contact body:
      X: velocity-error force drives body toward belt_speed_x, capped at mu*m*g.
      Y: velocity-proportional force damps lateral drift, same cap.
      Z-rot: damping torque prevents spin.
    """
    _BELT_MU   = 1.5    # combined sliding friction coefficient
    _BELT_GAIN = 40.0   # proportional → steady-state, no oscillation
    _LAT_GAIN  = 100.0  # lateral damping gain [N/(m/s)]
    _YAW_GAIN  = 0.5    # yaw damping gain [N·m/(rad/s)]

    processed: set = set()
    for i in range(data.ncon):
        con = data.contact[i]
        if con.geom1 == belt_geom_id:
            other_geom = con.geom2
        elif con.geom2 == belt_geom_id:
            other_geom = con.geom1
        else:
            continue
        body_id = model.geom_bodyid[other_geom]
        if body_id == 0 or body_id in processed:
            continue
        processed.add(body_id)

        # World-space linear velocity of the body
        xmat   = data.xmat[body_id].reshape(3, 3)
        v_body = xmat @ data.cvel[body_id][3:6]
        v_x    = float(v_body[0])
        v_y    = float(v_body[1])
        omega_z = float(data.cvel[body_id][2])   # angular vel about world-Z

        # Normal force estimate: mass * g (conservative lower bound)
        mass  = float(np.sum(model.body_mass[body_id]))
        F_cap = _BELT_MU * mass * 9.81

        F_x = float(np.clip(_BELT_GAIN * (belt_speed_x - v_x), -F_cap, F_cap))
        F_y = float(np.clip(-_LAT_GAIN * v_y,                  -F_cap, F_cap))
        T_z = float(np.clip(-_YAW_GAIN * omega_z,              -0.2,   0.2))

        force  = np.array([F_x, F_y, 0.0])
        torque = np.array([0.0, 0.0, T_z])
        mj.mj_applyFT(
            model, data, force, torque,
            data.xpos[body_id].copy(), body_id,
            data.qfrc_applied,
        )


class PandaConveyorEnv:
    """
    Franka Emika Panda robot arm environment with conveyor belt.
    """
    
    def __init__(self, gui: bool = True, dt: float = 0.01):
        """
        Initialize the Panda + Conveyor environment.
        
        Args:
            gui: Whether to display MuJoCo viewer
            dt: Time step for simulation
        """
        self.gui = gui
        self.dt = dt
        self.viewer = None
        self.model = None
        self.data = None
        
        # Robot configuration
        self.num_dofs = 7
        self.action_space_size = self.num_dofs
        
        # Joint constraints
        self.max_velocity = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
        
        self.joint_ids = []
        self.actuator_ids = []
        self.conveyor_geom_id = -1
        self.conveyor_speed_scale = 0.25
        self.grasp_weld_id = -1
        self.cube_body_id = -1
        self.hand_body_id = -1
        self.cube_joint_id = -1
        self.cube_qpos_adr = -1
        self.cube_qvel_adr = -1
        self.grasp_site_id = -1
        self.hand_subtree_mask = None
        self.magnetic_grasped = False
        self.magnetic_release_until = 0.0
        self.magnetic_release_cooldown = 1.0
        
        self._setup()
    
    def _setup(self):
        """Initialize MuJoCo environment with Panda and conveyor belt."""
        # Load scene with conveyor belt
        scene_file = Path(__file__).parent / "conveyor_scene.xml"
        
        if not scene_file.exists():
            raise FileNotFoundError(f"Scene file not found: {scene_file}")
        
        self.model = mj.MjModel.from_xml_path(str(scene_file))
        self.data = mj.MjData(self.model)
        
        # Set timestep
        self.model.opt.timestep = self.dt
        
        # Get joint IDs for the arm
        for i in range(1, 8):
            joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, f"joint{i}")
            if joint_id >= 0:
                self.joint_ids.append(joint_id)
        
        # Get actuator IDs for arm (1-7)
        for i in range(1, 8):
            actuator_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}")
            if actuator_id >= 0:
                self.actuator_ids.append(actuator_id)
        # Gripper actuator removed
        self.gripper_actuator_id = -1
        
        print(f"✓ Loaded {len(self.joint_ids)} arm joints and {len(self.actuator_ids)} actuators")

        # Conveyor belt surface geom (for contact-based transport)
        # Belt surface geom (from assets conveyor model, prefixed)
        self.conveyor_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "conv_belt_surface")
        # Optional grasp weld (for IK demo)
        self.grasp_weld_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_EQUALITY, "grasp_weld")
        # Bodies for magnetic grasp
        self.cube_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "pickup_cube")
        self.hand_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "hand")
        self.grasp_site_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, "grasp_site")
        self.cube_joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "pickup_cube_joint")
        if self.cube_joint_id >= 0:
            self.cube_qpos_adr = int(self.model.jnt_qposadr[self.cube_joint_id])
            self.cube_qvel_adr = int(self.model.jnt_dofadr[self.cube_joint_id])
        # Precompute hand subtree mask for contact checks
        if self.hand_body_id >= 0:
            self.hand_subtree_mask = np.zeros(self.model.nbody, dtype=bool)
            for b in range(self.model.nbody):
                cur = b
                while cur > 0:
                    if cur == self.hand_body_id:
                        self.hand_subtree_mask[b] = True
                        break
                    cur = self.model.body_parentid[cur]
            self.hand_subtree_mask[self.hand_body_id] = True

        
        # Initialize viewer if GUI is enabled
        if self.gui:
            self.viewer = mjviewer.launch_passive(self.model, self.data)
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        home_position = np.array([0, 0, 0, -np.pi/4, 0, np.pi/2, np.pi/4])
        
        for i, joint_id in enumerate(self.joint_ids):
            if i < len(home_position):
                self.data.qpos[joint_id] = home_position[i]
                self.data.qvel[joint_id] = 0.0

        # Ensure grasp weld is disabled on reset
        if self.grasp_weld_id >= 0:
            self.data.eq_active[self.grasp_weld_id] = 0
        self.magnetic_grasped = False
        self.magnetic_release_until = 0.0

        mj.mj_forward(self.model, self.data)
        return self._get_observation()
    
    def step(
        self,
        action: np.ndarray,
        conveyor_speed: float = 0.0,
        action_mode: str = "velocity",
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step.
        
        Args:
            action: Joint command (size: num_dofs)
            conveyor_speed: Conveyor belt speed (-1 to 1, normalized)
            action_mode: "velocity" or "position"
            
        Returns:
            observation, reward, done, info
        """
        gripper_cmd = None

        if action_mode == "velocity":
            # Clip actions
            action = np.clip(action, -1, 1)
            arm_action = action[: self.num_dofs]
            arm_action = arm_action * self.max_velocity
            if len(action) > self.num_dofs:
                action = np.concatenate([arm_action, action[self.num_dofs:]])
            else:
                action = arm_action
        elif action_mode == "position":
            # Clip to actuator ctrlrange if available
            if self.model.actuator_ctrlrange is not None and self.model.nu > 0:
                ctrl_min = self.model.actuator_ctrlrange[:, 0]
                ctrl_max = self.model.actuator_ctrlrange[:, 1]
                action = np.clip(action, ctrl_min[: len(action)], ctrl_max[: len(action)])
        else:
            raise ValueError(f"Unknown action_mode: {action_mode}")

        # Apply arm control (first 7 entries)
        for i, actuator_id in enumerate(self.actuator_ids):
            if i < len(action):
                self.data.ctrl[actuator_id] = action[i]
        # Gripper control removed
        
        # Spin rollers for visual fidelity (radius 0.05 m → ω = v / r).
        roller_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, "conv_roller_vel")
        if roller_id >= 0:
            self.data.ctrl[roller_id] = conveyor_speed * self.conveyor_speed_scale / 0.05
        
        # Apply belt friction forces before stepping.
        # Belt surface has condim=1 (normal support only), so these forces
        # are the sole source of tangential friction – no fighting the solver.
        self.data.qfrc_applied[:] = 0.0
        if conveyor_speed != 0.0 and self.conveyor_geom_id >= 0:
            _apply_belt_forces(
                self.model, self.data,
                self.conveyor_geom_id,
                conveyor_speed * self.conveyor_speed_scale,
            )
        mj.mj_step(self.model, self.data)

        # Auto-grasp with latch: once grasped, hold until released externally.
        # Check contacts after stepping so new contacts are detected immediately.
        if (
            not self.magnetic_grasped
            and self.data.time >= self.magnetic_release_until
            and self._hand_touches_cube()
        ):
            self.magnetic_grasped = True

        # If magnetically grasped, keep cube attached to grasp site
        if self.magnetic_grasped and self.cube_qpos_adr >= 0:
            if self.grasp_site_id >= 0:
                hand_pos = self.data.site_xpos[self.grasp_site_id].copy()
                hand_mat = self.data.site_xmat[self.grasp_site_id].copy()
            else:
                hand_pos = self.data.xpos[self.hand_body_id].copy()
                hand_mat = self.data.xmat[self.hand_body_id].copy()
            quat = np.zeros(4)
            mj.mju_mat2Quat(quat, hand_mat)
            self.data.qpos[self.cube_qpos_adr:self.cube_qpos_adr+3] = hand_pos
            self.data.qpos[self.cube_qpos_adr+3:self.cube_qpos_adr+7] = quat
            if self.cube_qvel_adr >= 0:
                self.data.qvel[self.cube_qvel_adr:self.cube_qvel_adr+6] = 0.0
            mj.mj_forward(self.model, self.data)
        
        # Render
        if self.viewer and self.gui:
            self.viewer.sync()
        
        obs = self._get_observation()
        reward = -np.sum(np.abs(action)) * self.dt
        done = False
        
        return obs, reward, done, {}
    
    def _get_observation(self) -> np.ndarray:
        """Get observation from environment."""
        positions = np.array([self.data.qpos[joint_id] for joint_id in self.joint_ids])
        return positions
    
    def get_end_effector_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get end effector position and orientation."""
        body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "hand")
        
        if body_id >= 0:
            position = self.data.xpos[body_id].copy()
            xmat = self.data.xmat[body_id].reshape(3, 3)
            orientation = self._matrix_to_quaternion(xmat)
        else:
            position = np.array([0.0, 0.0, 0.0])
            orientation = np.array([0.0, 0.0, 0.0, 1.0])
        
        return position, orientation
    
    @staticmethod
    def _matrix_to_quaternion(mat: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to quaternion."""
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
    
    def close(self):
        """Close environment."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def set_magnetic_grasp(self, active: bool, use_cooldown: bool = True):
        """Enable or disable magnetic grasp latch."""
        self.magnetic_grasped = bool(active)
        if not active and use_cooldown:
            # Prevent immediate re-grasp on contact
            self.magnetic_release_until = self.data.time + self.magnetic_release_cooldown
        elif not active:
            self.magnetic_release_until = self.data.time

    def _hand_touches_cube(self) -> bool:
        if self.cube_body_id < 0 or self.hand_body_id < 0 or self.hand_subtree_mask is None:
            return False
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            b1 = self.model.geom_bodyid[con.geom1]
            b2 = self.model.geom_bodyid[con.geom2]
            if (self.hand_subtree_mask[b1] and b2 == self.cube_body_id) or (
                self.hand_subtree_mask[b2] and b1 == self.cube_body_id
            ):
                return True
        return False
    
    def __del__(self):
        """Cleanup."""
        self.close()


if __name__ == "__main__":
    print("Testing Panda + Conveyor Environment...")
    
    env = PandaConveyorEnv(gui=False)
    
    try:
        obs = env.reset()
        print(f"✓ Reset: {obs}")
        
        # Run a few steps
        for step in range(100):
            # Random arm action
            action = np.random.uniform(-0.3, 0.3, size=7)
            # Conveyor speed
            conveyor_speed = 0.5  # Move conveyor
            
            obs, reward, done, info = env.step(action, conveyor_speed)
            
            if step % 20 == 0:
                try:
                    pos, orn = env.get_end_effector_pose()
                    print(f"Step {step}: EE pos {pos}")
                except:
                    print(f"Step {step}: joints {obs}")
        
        print("✓ Test completed!")
        
    finally:
        env.close()
