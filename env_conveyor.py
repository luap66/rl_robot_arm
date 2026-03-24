"""
Franka Panda with Conveyor Belt Environment
"""

import numpy as np
import mujoco as mj
import mujoco.viewer as mjviewer
from pathlib import Path
from typing import Tuple, Dict, Any, Optional


class PandaConveyorEnv:
    """
    Franka Emika Panda robot arm environment with conveyor belt.

    Belt transport uses the official MuJoCo slide-joint approach (issue #547):
    the belt surface is a body with a slide joint driven by a general actuator.
    gainprm = damping → terminal velocity = ctrl (m/s).  No custom force code.
    """

    def __init__(self, gui: bool = True, dt: float = 0.01):
        self.gui = gui
        self.dt = dt
        self.viewer = None
        self.model = None
        self.data = None

        self.num_dofs = 7
        self.action_space_size = self.num_dofs
        self.max_velocity = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])

        self.joint_ids = []
        self.actuator_ids = []
        self.conveyor_geom_id = -1
        self.belt_drive_id = -1    # general actuator that drives belt slide
        self.belt_slide_qposadr = -1  # for periodic qpos reset
        self.belt_slide_qdofadr = -1
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
        scene_file = Path(__file__).parent / "conveyor_scene.xml"
        if not scene_file.exists():
            raise FileNotFoundError(f"Scene file not found: {scene_file}")

        self.model = mj.MjModel.from_xml_path(str(scene_file))
        self.data = mj.MjData(self.model)
        self.model.opt.timestep = self.dt

        for i in range(1, 8):
            jid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, f"joint{i}")
            if jid >= 0:
                self.joint_ids.append(jid)

        for i in range(1, 8):
            aid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}")
            if aid >= 0:
                self.actuator_ids.append(aid)
        self.gripper_actuator_id = -1

        print(f"✓ Loaded {len(self.joint_ids)} arm joints and {len(self.actuator_ids)} actuators")

        self.conveyor_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "conv_belt_surface")
        self.grasp_weld_id    = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_EQUALITY, "grasp_weld")
        self.cube_body_id     = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "pickup_cube")
        self.hand_body_id     = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "hand")
        self.grasp_site_id    = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, "grasp_site")
        self.cube_joint_id    = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "pickup_cube_joint")
        if self.cube_joint_id >= 0:
            self.cube_qpos_adr = int(self.model.jnt_qposadr[self.cube_joint_id])
            self.cube_qvel_adr = int(self.model.jnt_dofadr[self.cube_joint_id])

        # Belt slide joint for periodic position reset
        belt_slide_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "conv_belt_slide")
        if belt_slide_id >= 0:
            self.belt_slide_qposadr = int(self.model.jnt_qposadr[belt_slide_id])
            self.belt_slide_qdofadr = int(self.model.jnt_dofadr[belt_slide_id])

        # Belt drive actuator (conv_ prefix from attach)
        self.belt_drive_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, "conv_belt_drive")

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

        if self.gui:
            self.viewer = mjviewer.launch_passive(self.model, self.data)

    def reset(self) -> np.ndarray:
        home_position = np.array([0, 0, 0, -np.pi/4, 0, np.pi/2, np.pi/4])
        for i, joint_id in enumerate(self.joint_ids):
            if i < len(home_position):
                self.data.qpos[joint_id] = home_position[i]
                self.data.qvel[joint_id] = 0.0

        # Reset belt slide position so it doesn't drift across episodes
        if self.belt_slide_qposadr >= 0:
            self.data.qpos[self.belt_slide_qposadr] = 0.0
            self.data.qvel[self.belt_slide_qdofadr] = 0.0

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
        if action_mode == "velocity":
            action = np.clip(action, -1, 1)
            arm_action = action[: self.num_dofs] * self.max_velocity
            action = np.concatenate([arm_action, action[self.num_dofs:]]) if len(action) > self.num_dofs else arm_action
        elif action_mode == "position":
            if self.model.actuator_ctrlrange is not None and self.model.nu > 0:
                ctrl_min = self.model.actuator_ctrlrange[:, 0]
                ctrl_max = self.model.actuator_ctrlrange[:, 1]
                action = np.clip(action, ctrl_min[: len(action)], ctrl_max[: len(action)])
        else:
            raise ValueError(f"Unknown action_mode: {action_mode}")

        for i, actuator_id in enumerate(self.actuator_ids):
            if i < len(action):
                self.data.ctrl[actuator_id] = action[i]

        # Drive belt: ctrl = target speed in m/s (gainprm == damping in XML)
        if self.belt_drive_id >= 0:
            self.data.ctrl[self.belt_drive_id] = conveyor_speed * self.conveyor_speed_scale

        mj.mj_step(self.model, self.data)

        # Infinite-belt trick: reset slide qpos to 0 every step so the belt
        # surface stays visually in place.  qvel is preserved, so friction
        # (which depends on velocity, not position) is unaffected.
        if self.belt_slide_qposadr >= 0:
            self.data.qpos[self.belt_slide_qposadr] = 0.0

        # Auto-grasp latch
        if (
            not self.magnetic_grasped
            and self.data.time >= self.magnetic_release_until
            and self._hand_touches_cube()
        ):
            self.magnetic_grasped = True

        # Keep cube attached to grasp site while magnetically grasped
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

        if self.viewer and self.gui:
            self.viewer.sync()

        obs = self._get_observation()
        reward = -np.sum(np.abs(action)) * self.dt
        return obs, reward, False, {}

    def _get_observation(self) -> np.ndarray:
        return np.array([self.data.qpos[j] for j in self.joint_ids])

    def get_end_effector_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "hand")
        if body_id >= 0:
            position = self.data.xpos[body_id].copy()
            xmat = self.data.xmat[body_id].reshape(3, 3)
            orientation = self._matrix_to_quaternion(xmat)
        else:
            position = np.zeros(3)
            orientation = np.array([0.0, 0.0, 0.0, 1.0])
        return position, orientation

    @staticmethod
    def _matrix_to_quaternion(mat: np.ndarray) -> np.ndarray:
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
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def set_magnetic_grasp(self, active: bool, use_cooldown: bool = True):
        self.magnetic_grasped = bool(active)
        if not active and use_cooldown:
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
        self.close()


if __name__ == "__main__":
    print("Testing Panda + Conveyor Environment...")
    env = PandaConveyorEnv(gui=False)
    try:
        obs = env.reset()
        print(f"✓ Reset: {obs}")
        for step in range(100):
            action = np.random.uniform(-0.3, 0.3, size=7)
            obs, reward, done, info = env.step(action, conveyor_speed=0.5)
            if step % 20 == 0:
                try:
                    pos, _ = env.get_end_effector_pose()
                    print(f"Step {step}: EE pos {pos}")
                except Exception:
                    print(f"Step {step}: joints {obs}")
        print("✓ Test completed!")
    finally:
        env.close()
