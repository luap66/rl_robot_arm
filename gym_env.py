"""
Gymnasium wrapper for PandaConveyorEnv.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np

try:
    import gymnasium as gym
except Exception as exc:  # pragma: no cover - optional dependency
    raise ImportError("gymnasium is required for gym_env.py") from exc

import mujoco as mj

from env_conveyor import PandaConveyorEnv


@dataclass
class ConveyorTaskConfig:
    action_mode: str = "velocity"  # "velocity" or "position"
    control_conveyor: bool = False
    conveyor_speed: float = 1.0
    randomize_cube: bool = False
    cube_xy_range: float = 0.15
    cube_z: float = 0.05
    cube_fixed_pos: tuple[float, float, float] = (0.35, 0.2, 0.05)
    max_steps: int = 500
    grasp_dist: float = 0.04
    grasp_reward: float = 0.5
    release_height: float = 0.18
    release_over_belt: bool = False
    release_belt_margin: float = 0.04
    release_xy_margin: float = 0.02
    auto_release_inner_xy_margin: float = 0.005
    agent_controls_release: bool = True
    render_every_n_episodes: int = 0
    gui_sleep_s: float = 0.02
    belt_collision_penalty: float = 0.3  # Reduziert
    belt_end_reward: float = 2.0  # Reduziert (zu spät für gutes Lernensignal)
    belt_end_margin: float = 0.03
    max_reach_dist: float = 0.6
    auto_max_reach: bool = True
    reach_margin: float = 0.05
    out_of_reach_penalty: float = 2.0
    min_steps_before_terminate: int = 100
    drop_below_belt_margin: float = 0.03
    step_penalty: float = 0.000
    hold_over_belt_penalty: float = 0.02  # Reduziert
    release_success_reward: float = 1.0  # Reduziert
    terminate_on_success: bool = True
    to_belt_scale: float = 3.0
    hand_to_belt_scale: float = 0.5
    on_belt_reward: float = 5.0
    on_belt_z_below_margin: float = 0.04
    on_belt_z_above_margin: float = 0.08
    release_cmd_threshold: float = 0.5
    release_cmd_reward: float = 0.0  # Entfernt (verursacht Verwirrung)
    release_cmd_penalty: float = 0.0  # Entfernt
    hand_below_belt_penalty: float = 0.1  # Reduziert
    hand_below_belt_margin: float = 0.02
    simple_goal_reward: bool = False
    goal_xy_only: bool = True
    cube_velocity_penalty: float = 0.1
    auto_release_when_over_belt: bool = True
    ground_collision_penalty: float = 0.5
    action_scale: float = 0.7
    milestone_grasp_reward: float = 2.0
    milestone_on_belt_reward: float = 1.0  # Reduziert
    milestone_end_reward: float = 2.0  # Reduziert
    grasp_sustain_reward: float = 0.0  # Entfernt (verursacht kontinuierliche Streuung)
    push_without_grasp_penalty: float = 0.02  # Reduziert
    lift_up_reward_scale: float = 6.0
    lift_down_penalty_scale: float = 2.0
    lift_min_height_margin: float = 0.01
    lift_low_height_penalty: float = 0.02
    belt_progress_reward_scale: float = 20.0


class PandaConveyorGym(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, gui: bool = False, dt: float = 0.01, config: ConveyorTaskConfig | None = None):
        super().__init__()
        self.config = config or ConveyorTaskConfig()
        self.base_gui = gui
        self.env = PandaConveyorEnv(gui=gui, dt=dt)
        self.step_count = 0
        self.episode_count = 0
        self.episode_reward = 0.0
        self.prev_is_grasped = False
        self.prev_cube_z = 0.0
        self.prev_cube_x = 0.0
        self.in_track_on_belt = False
        self.milestone_grasped = False
        self.milestone_on_belt = False
        self.milestone_end = False
        self._refresh_ids()

        # Observation: joint pos (7) + joint vel (7)
        # + cube rel hand (3) + belt rel hand (3)
        # + cube vel (3) + cube quat (4) + grasped (1)
        obs_dim = 28
        high = np.ones(obs_dim, dtype=np.float32) * np.inf
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        # Action: 7 joints (+ release if enabled + optional conveyor)
        act_dim = 7 + (1 if self.config.agent_controls_release else 0) + (1 if self.config.control_conveyor else 0)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(act_dim,), dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        qpos = np.array([self.env.data.qpos[j] for j in self.env.joint_ids], dtype=np.float32)
        qvel = np.array([self.env.data.qvel[j] for j in self.env.joint_ids], dtype=np.float32)
        hand_pos = (
            self.env.data.xpos[self.hand_body_id].astype(np.float32)
            if self.hand_body_id >= 0
            else np.zeros(3, dtype=np.float32)
        )
        cube_pos = (
            self.env.data.xpos[self.cube_body_id].astype(np.float32)
            if self.cube_body_id >= 0
            else np.zeros(3, dtype=np.float32)
        )
        cube_vel = (
            self.env.data.cvel[self.cube_body_id][3:6].astype(np.float32)
            if self.cube_body_id >= 0
            else np.zeros(3, dtype=np.float32)
        )
        cube_quat = (
            self.env.data.xquat[self.cube_body_id].astype(np.float32)
            if self.cube_body_id >= 0
            else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        )
        belt_pos = (
            self.env.data.geom_xpos[self.belt_geom_id].astype(np.float32)
            if self.belt_geom_id >= 0
            else np.zeros(3, dtype=np.float32)
        )
        grasped = np.array([1.0 if self.is_grasped else 0.0], dtype=np.float32)
        cube_rel_hand = cube_pos - hand_pos
        belt_rel_hand = belt_pos - hand_pos
        return np.concatenate([qpos, qvel, cube_rel_hand, belt_rel_hand, cube_vel, cube_quat, grasped], axis=0)

    def _cube_on_belt(self) -> bool:
        if self.cube_body_id < 0 or self.belt_geom_id < 0:
            return False
        cube_pos = self.env.data.xpos[self.cube_body_id]
        belt_pos = self.env.data.geom_xpos[self.belt_geom_id]
        belt_size = self.env.model.geom_size[self.belt_geom_id]
        cube_half = self._cube_half_extents()
        # Robust XY check: accept overlap of cube footprint with belt surface.
        x_ok = abs(cube_pos[0] - belt_pos[0]) <= belt_size[0] + cube_half[0]
        y_ok = abs(cube_pos[1] - belt_pos[1]) <= belt_size[1] + cube_half[1]
        # Robust Z check using cube bottom relative to belt top.
        belt_top_z = float(belt_pos[2] + belt_size[2])
        cube_bottom_z = float(cube_pos[2] - cube_half[2])
        z_ok = (
            cube_bottom_z >= belt_top_z - self.config.on_belt_z_below_margin
            and cube_bottom_z <= belt_top_z + self.config.on_belt_z_above_margin
        )
        return bool(x_ok and y_ok and z_ok)

    def _cube_half_extents(self) -> np.ndarray:
        if self.cube_geom_id < 0:
            return np.zeros(3, dtype=np.float64)
        size = np.asarray(self.env.model.geom_size[self.cube_geom_id], dtype=np.float64)
        return size[:3]

    def _cube_over_belt_xy(self, cube_pos: np.ndarray, belt_pos: np.ndarray, belt_size: np.ndarray) -> bool:
        cube_half = self._cube_half_extents()
        # Require cube footprint to lie within belt surface (with optional inner safety margin).
        x_limit = float(belt_size[0] - cube_half[0] - self.config.auto_release_inner_xy_margin)
        y_limit = float(belt_size[1] - cube_half[1] - self.config.auto_release_inner_xy_margin)
        if x_limit <= 0.0 or y_limit <= 0.0:
            return False
        return bool(abs(cube_pos[0] - belt_pos[0]) <= x_limit and abs(cube_pos[1] - belt_pos[1]) <= y_limit)

    def _compute_subtree_mask(self, root_body_id: int) -> np.ndarray:
        mask = np.zeros(self.env.model.nbody, dtype=bool)
        if root_body_id < 0:
            return mask
        for b in range(self.env.model.nbody):
            cur = b
            while cur > 0:
                if cur == root_body_id:
                    mask[b] = True
                    break
                cur = self.env.model.body_parentid[cur]
        mask[root_body_id] = True
        return mask

    def _compute_max_reach(self) -> float:
        if self.hand_body_id < 0 or self.robot_root_id < 0:
            return self.config.max_reach_dist
        # Sum link offsets along the chain hand -> root as a rough max reach
        total = 0.0
        cur = self.hand_body_id
        while cur > 0 and cur != self.robot_root_id:
            total += float(np.linalg.norm(self.env.model.body_pos[cur]))
            cur = self.env.model.body_parentid[cur]
        return total

    def _robot_hits_belt(self) -> bool:
        if self.belt_geom_id < 0 or self.robot_subtree_mask is None:
            return False
        for i in range(self.env.data.ncon):
            con = self.env.data.contact[i]
            if con.geom1 == self.belt_geom_id or con.geom2 == self.belt_geom_id:
                other = con.geom2 if con.geom1 == self.belt_geom_id else con.geom1
                other_body = self.env.model.geom_bodyid[other]
                if self.robot_subtree_mask[other_body]:
                    return True
        return False

    def _robot_hits_ground(self) -> bool:
        ground_geom_id = mj.mj_name2id(self.env.model, mj.mjtObj.mjOBJ_GEOM, "ground")
        if ground_geom_id < 0 or self.robot_subtree_mask is None:
            return False
        for i in range(self.env.data.ncon):
            con = self.env.data.contact[i]
            if con.geom1 == ground_geom_id or con.geom2 == ground_geom_id:
                other = con.geom2 if con.geom1 == ground_geom_id else con.geom1
                other_body = self.env.model.geom_bodyid[other]
                if self.robot_subtree_mask[other_body]:
                    return True
        return False

    def _cube_at_belt_end(self) -> bool:
        if self.cube_body_id < 0 or self.belt_geom_id < 0:
            return False
        cube_pos = self.env.data.xpos[self.cube_body_id]
        belt_pos = self.env.data.geom_xpos[self.belt_geom_id]
        belt_size = self.env.model.geom_size[self.belt_geom_id]
        # Assume belt runs along +X direction; end is +X edge
        end_x = belt_pos[0] + belt_size[0]
        x_ok = cube_pos[0] >= end_x - self.config.belt_end_margin
        y_ok = abs(cube_pos[1] - belt_pos[1]) <= belt_size[1]
        z_ok = cube_pos[2] >= belt_pos[2] - 0.02
        return bool(x_ok and y_ok and z_ok)

    def _goal_pos(self) -> np.ndarray:
        if self.belt_geom_id < 0:
            return np.zeros(3, dtype=np.float64)
        belt_pos = self.env.data.geom_xpos[self.belt_geom_id]
        belt_size = self.env.model.geom_size[self.belt_geom_id]
        return np.array([belt_pos[0] + belt_size[0], belt_pos[1], belt_pos[2]], dtype=np.float64)

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self.episode_count += 1
        self.step_count = 0
        self.episode_reward = 0.0

        # Render only every N episodes (if configured)
        if self.config.render_every_n_episodes and self.base_gui:
            show_gui = (self.episode_count % self.config.render_every_n_episodes) == 0
            if show_gui != self.env.gui:
                self.env.close()
                self.env = PandaConveyorEnv(gui=show_gui, dt=self.env.dt)
                self._refresh_ids()

        obs = self.env.reset()
        # Reset magnetic grasp latch
        if hasattr(self.env, "set_magnetic_grasp"):
            self.env.set_magnetic_grasp(False, use_cooldown=False)
        self.is_grasped = False
        self.in_track_on_belt = False
        self.prev_is_grasped = self.is_grasped
        if self.cube_body_id >= 0:
            self.prev_cube_z = float(self.env.data.xpos[self.cube_body_id][2])
            self.prev_cube_x = float(self.env.data.xpos[self.cube_body_id][0])
        else:
            self.prev_cube_z = 0.0
            self.prev_cube_x = 0.0
        self.milestone_grasped = False
        self.milestone_on_belt = False
        self.milestone_end = False

        if self.cube_body_id >= 0:
            # Place cube on table near robot (fixed or randomized)
            if self.config.randomize_cube:
                dx = self.np_random.uniform(-self.config.cube_xy_range, self.config.cube_xy_range)
                dy = self.np_random.uniform(-self.config.cube_xy_range, self.config.cube_xy_range)
                cube_pos = np.array([0.35 + dx, 0.15 + dy, self.config.cube_z], dtype=np.float64)
            else:
                cube_pos = np.array(self.config.cube_fixed_pos, dtype=np.float64)
            cube_jid = mj.mj_name2id(self.env.model, mj.mjtObj.mjOBJ_JOINT, "pickup_cube_joint")
            if cube_jid >= 0:
                qpos_adr = self.env.model.jnt_qposadr[cube_jid]
                self.env.data.qpos[qpos_adr:qpos_adr+3] = cube_pos
                self.env.data.qpos[qpos_adr+3:qpos_adr+7] = np.array([1.0, 0.0, 0.0, 0.0])
                self.env.data.qvel[self.env.model.jnt_dofadr[cube_jid]:self.env.model.jnt_dofadr[cube_jid]+6] = 0.0
                mj.mj_forward(self.env.model, self.env.data)

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self.step_count += 1
        action = np.asarray(action, dtype=np.float32)

        idx = 7
        release_cmd = 0.0
        if self.config.agent_controls_release:
            if len(action) > idx:
                release_cmd = float(action[idx])
            idx += 1
        if self.config.control_conveyor:
            conveyor_speed = float(np.clip(action[idx], -1.0, 1.0))
        else:
            conveyor_speed = self.config.conveyor_speed
        arm_action = action[:7]

        # Map [-1,1] to position ctrlrange if using position mode
        if self.config.action_mode == "position":
            ctrl = np.zeros(7, dtype=np.float64)
            ctrl[:7] = arm_action[:7] * self.config.action_scale
            self.env.step(ctrl, conveyor_speed=conveyor_speed, action_mode="position")
        else:
            self.env.step(arm_action * self.config.action_scale, conveyor_speed=conveyor_speed, action_mode="velocity")

        obs = self._get_obs()
        # Sync grasp state from physics env if available
        if hasattr(self.env, "magnetic_grasped"):
            self.is_grasped = bool(self.env.magnetic_grasped)

        # Reward: reach cube, then move it toward the belt
        if self.cube_body_id >= 0 and self.hand_body_id >= 0:
            cube_pos = self.env.data.xpos[self.cube_body_id]
            hand_pos = self.env.data.xpos[self.hand_body_id]
            dist = np.linalg.norm(hand_pos - cube_pos)
        else:
            dist = 0.0
        cube_on_belt_now = self._cube_on_belt()

        # Enter TRACK_ON_BELT once cube was released and is resting on the belt.
        if not self.in_track_on_belt and self.prev_is_grasped and not self.is_grasped and cube_on_belt_now:
            self.in_track_on_belt = True

        # In TRACK_ON_BELT grasping is forbidden: force release if needed.
        if self.in_track_on_belt and self.is_grasped:
            if hasattr(self.env, "set_magnetic_grasp"):
                self.env.set_magnetic_grasp(False)
            self.is_grasped = False

        # Base step penalty always applies
        reward = 0.0 if self.in_track_on_belt else -self.config.step_penalty
        reward_parts = {
            "reach": 0.0,
            "goal_dist": 0.0,
            "step": 0.0 if self.in_track_on_belt else -self.config.step_penalty,
            "grasp": 0.0,
            "lift": 0.0,
            "to_belt": 0.0,
            "on_belt": 0.0,
            "end": 0.0,
            "belt_progress": 0.0,
            "collide": 0.0,
            "hand_below_belt": 0.0,
            "belt_collision": 0.0,
            "ground_collision": 0.0,
            "push_without_grasp": 0.0,
            "drop_off_belt": 0.0,
            "hold_over_belt": 0.0,
            "release_success": 0.0,
            "release_cmd": 0.0,
            "milestone": 0.0,
        }
        if self.in_track_on_belt:
            if self.cube_body_id >= 0:
                cube_pos = self.env.data.xpos[self.cube_body_id]
                progress_x = float(cube_pos[0] - self.prev_cube_x)
                progress_reward = self.config.belt_progress_reward_scale * max(progress_x, 0.0)
                reward += progress_reward
                reward_parts["belt_progress"] += progress_reward
            if self._cube_at_belt_end():
                reward += self.config.belt_end_reward
                reward_parts["end"] += self.config.belt_end_reward
        elif self.config.simple_goal_reward:
            goal = self._goal_pos()
            if self.config.goal_xy_only:
                goal_dist = float(np.linalg.norm((cube_pos - goal)[:2]))
            else:
                goal_dist = float(np.linalg.norm(cube_pos - goal))
            reward -= goal_dist
            reward_parts["goal_dist"] = -goal_dist
        else:
            # Stop reach penalty once cube is already on belt.
            if not self.is_grasped and not cube_on_belt_now:
                reward -= dist
                reward_parts["reach"] -= dist
        # Pseudo-grasp: weld when close, release above belt
        # Magnetic grasp logic (latched in env_conveyor)
        if not self.in_track_on_belt and self.cube_body_id >= 0 and self.hand_body_id >= 0:
            if not self.is_grasped and dist < self.config.grasp_dist:
                if hasattr(self.env, "set_magnetic_grasp"):
                    self.env.set_magnetic_grasp(True)
                self.is_grasped = True
                if not self.config.simple_goal_reward:
                    reward += self.config.grasp_reward
                    reward_parts["grasp"] += self.config.grasp_reward
            elif self.is_grasped:
                cube_pos = self.env.data.xpos[self.cube_body_id]
                release = False
                release_reason = None
                if self.config.agent_controls_release:
                    # Auto-release when cube footprint is over belt surface in XY.
                    if self.belt_geom_id >= 0:
                        belt_pos = self.env.data.geom_xpos[self.belt_geom_id]
                        belt_size = self.env.model.geom_size[self.belt_geom_id]
                        over_belt_xy = self._cube_over_belt_xy(cube_pos, belt_pos, belt_size)
                    else:
                        over_belt_xy = False
                    if over_belt_xy and self.config.auto_release_when_over_belt:
                        release = True
                        release_reason = "auto_over_belt"
                    elif over_belt_xy and release_cmd > self.config.release_cmd_threshold:
                        release = True
                        release_reason = "agent"
                else:
                    if cube_pos[2] > self.config.release_height:
                        release = True
                        release_reason = "height"
                    elif self.config.release_over_belt and self.belt_geom_id >= 0:
                        belt_pos = self.env.data.geom_xpos[self.belt_geom_id]
                        belt_size = self.env.model.geom_size[self.belt_geom_id]
                        over_belt_xy = (
                            abs(cube_pos[0] - belt_pos[0]) <= belt_size[0]
                            and abs(cube_pos[1] - belt_pos[1]) <= belt_size[1]
                        )
                        if over_belt_xy and cube_pos[2] <= belt_pos[2] + self.config.release_belt_margin:
                            release = True
                            release_reason = "over_belt"
                if release:
                    if hasattr(self.env, "set_magnetic_grasp"):
                        self.env.set_magnetic_grasp(False)
                    self.is_grasped = False
                    if self._cube_on_belt():
                        self.in_track_on_belt = True
                    if self.env.gui:
                        print(f"[grasp] release triggered ({release_reason})")
        if not self.in_track_on_belt and not self.config.simple_goal_reward:
            # Explicitly reward lifting motion while grasping.
            if self.is_grasped and self.cube_body_id >= 0:
                cube_pos = self.env.data.xpos[self.cube_body_id]
                dz = float(cube_pos[2] - self.prev_cube_z)
                lift_term = (
                    self.config.lift_up_reward_scale * max(dz, 0.0)
                    - self.config.lift_down_penalty_scale * max(-dz, 0.0)
                )
                reward += lift_term
                reward_parts["lift"] += lift_term
                min_lift_z = self.config.cube_z + self.config.lift_min_height_margin
                if cube_pos[2] < min_lift_z:
                    reward -= self.config.lift_low_height_penalty
                    reward_parts["lift"] -= self.config.lift_low_height_penalty

            # Encourage moving cube toward belt center (only when grasped)
            if self.is_grasped and self.belt_geom_id >= 0 and self.cube_body_id >= 0:
                belt_pos = self.env.data.geom_xpos[self.belt_geom_id]
                cube_pos = self.env.data.xpos[self.cube_body_id]
                belt_xy_dist = np.linalg.norm((cube_pos - belt_pos)[:2])
                shaped = self.config.to_belt_scale * (1.0 - np.tanh(3.0 * belt_xy_dist))
                reward += shaped
                reward_parts["to_belt"] += shaped

            # Legacy per-step on_belt reward removed to avoid reward farming.
        if not self.in_track_on_belt and self._cube_at_belt_end():
            reward += self.config.belt_end_reward
            reward_parts["end"] += self.config.belt_end_reward

        if not self.in_track_on_belt and not self.config.simple_goal_reward:
            # Reward explicit release command when over belt
            if self.config.agent_controls_release and self.is_grasped and self.belt_geom_id >= 0 and self.cube_body_id >= 0:
                belt_pos = self.env.data.geom_xpos[self.belt_geom_id]
                belt_size = self.env.model.geom_size[self.belt_geom_id]
                cube_pos = self.env.data.xpos[self.cube_body_id]
                over_belt_xy = (
                    abs(cube_pos[0] - belt_pos[0]) <= belt_size[0]
                    and abs(cube_pos[1] - belt_pos[1]) <= belt_size[1]
                )
                if over_belt_xy and release_cmd > self.config.release_cmd_threshold:
                    reward += self.config.release_cmd_reward
                    reward_parts["release_cmd"] += self.config.release_cmd_reward

        if not self.in_track_on_belt and not self.config.simple_goal_reward:
            # Penalize dropping the cube off the belt after a grasp
            if self.prev_is_grasped and not self.is_grasped and not self._cube_on_belt():
                if self.belt_geom_id >= 0 and self.cube_body_id >= 0:
                    belt_z = self.env.data.geom_xpos[self.belt_geom_id][2]
                    cube_z = self.env.data.xpos[self.cube_body_id][2]
                    if cube_z < belt_z - self.config.drop_below_belt_margin:
                        reward -= 1.0
                        reward_parts["drop_off_belt"] -= 1.0

            # Penalize holding the cube over the belt without releasing
            if self.is_grasped and self.belt_geom_id >= 0 and self.cube_body_id >= 0:
                belt_pos = self.env.data.geom_xpos[self.belt_geom_id]
                belt_size = self.env.model.geom_size[self.belt_geom_id]
                cube_pos = self.env.data.xpos[self.cube_body_id]
                over_belt_xy = (
                    abs(cube_pos[0] - belt_pos[0]) <= belt_size[0]
                    and abs(cube_pos[1] - belt_pos[1]) <= belt_size[1]
                )
                if over_belt_xy and cube_pos[2] > belt_pos[2] + 0.02:
                    reward -= self.config.hold_over_belt_penalty
                    reward_parts["hold_over_belt"] -= self.config.hold_over_belt_penalty

        # Milestone rewards (one-time per episode)
        if not self.in_track_on_belt and self.is_grasped and not self.milestone_grasped:
            reward += self.config.milestone_grasp_reward
            reward_parts["milestone"] += self.config.milestone_grasp_reward
            self.milestone_grasped = True
        # Removed grasp_sustain_reward - causes signal noise
        if not self.in_track_on_belt and self._cube_on_belt() and not self.milestone_on_belt:
            reward += self.config.milestone_on_belt_reward
            reward_parts["milestone"] += self.config.milestone_on_belt_reward
            self.milestone_on_belt = True
        if not self.in_track_on_belt and self._cube_at_belt_end() and not self.milestone_end:
            reward += self.config.milestone_end_reward
            reward_parts["milestone"] += self.config.milestone_end_reward
            self.milestone_end = True

        # Penalize hand going below the belt surface
        if not self.in_track_on_belt and self.belt_geom_id >= 0 and self.hand_body_id >= 0:
            belt_z = self.env.data.geom_xpos[self.belt_geom_id][2]
            hand_z = self.env.data.xpos[self.hand_body_id][2]
            if hand_z < belt_z - self.config.hand_below_belt_margin:
                reward -= self.config.hand_below_belt_penalty
                reward_parts["collide"] -= self.config.hand_below_belt_penalty
                reward_parts["hand_below_belt"] -= self.config.hand_below_belt_penalty

        if not self.in_track_on_belt and not self.config.simple_goal_reward:
            # Bonus for releasing successfully onto the belt
            if self.prev_is_grasped and not self.is_grasped and self._cube_on_belt():
                reward += self.config.release_success_reward
                reward_parts["release_success"] += self.config.release_success_reward

        if not self.in_track_on_belt and self._robot_hits_belt():
            reward -= self.config.belt_collision_penalty
            reward_parts["collide"] -= self.config.belt_collision_penalty
            reward_parts["belt_collision"] -= self.config.belt_collision_penalty

        if not self.in_track_on_belt and self._robot_hits_ground():
            reward -= self.config.ground_collision_penalty
            reward_parts["collide"] -= self.config.ground_collision_penalty
            reward_parts["ground_collision"] -= self.config.ground_collision_penalty

        # Penalize pushing the cube without grasping
        if not self.in_track_on_belt and not self.is_grasped and self.cube_body_id >= 0 and self.hand_body_id >= 0:
            pushing = False
            for i in range(self.env.data.ncon):
                con = self.env.data.contact[i]
                b1 = self.env.model.geom_bodyid[con.geom1]
                b2 = self.env.model.geom_bodyid[con.geom2]
                if (b1 == self.hand_body_id and b2 == self.cube_body_id) or (
                    b2 == self.hand_body_id and b1 == self.cube_body_id
                ):
                    pushing = True
                    break
            if pushing:
                reward -= self.config.push_without_grasp_penalty
                reward_parts["collide"] -= self.config.push_without_grasp_penalty
                reward_parts["push_without_grasp"] -= self.config.push_without_grasp_penalty

        # Slow down GUI playback during rendering episodes
        if self.env.gui and self.config.gui_sleep_s > 0:
            time.sleep(self.config.gui_sleep_s)

        terminated = False
        # Out-of-reach termination disabled

        # Terminate on success if configured
        done_reason = None
        if self.config.terminate_on_success and self._cube_at_belt_end() and not self.is_grasped:
            terminated = True
            done_reason = "success_on_belt_end"
        truncated = self.step_count >= self.config.max_steps
        if truncated:
            done_reason = "max_steps"
        step_reward = float(reward)
        self.episode_reward += step_reward
        is_success = bool(terminated and done_reason == "success_on_belt_end")
        info = {
            "dist": float(dist),
            "reward_parts": reward_parts,
            "episode_reward": float(self.episode_reward),
            "is_success": is_success,
            "done_reason": done_reason,
        }
        self.prev_is_grasped = self.is_grasped
        if self.cube_body_id >= 0:
            self.prev_cube_z = float(self.env.data.xpos[self.cube_body_id][2])
            self.prev_cube_x = float(self.env.data.xpos[self.cube_body_id][0])

        # When GUI is on (e.g. render episodes during training), print reward breakdown every step.
        if self.env.gui:
            parts = ", ".join([f"{k}={v:+.3f}" for k, v in reward_parts.items()])
            print(f"[reward] total={reward:+.3f} ({parts})", flush=True)
            if (terminated or truncated) and done_reason is not None:
                print(f"[done] reason={done_reason}, episode_reward={self.episode_reward:+.3f}", flush=True)
        return obs, step_reward, terminated, truncated, info

    def render(self):
        # Viewer is handled inside PandaConveyorEnv when gui=True
        return None

    def close(self):
        self.env.close()

    def _refresh_ids(self):
        # IDs
        self.hand_body_id = mj.mj_name2id(self.env.model, mj.mjtObj.mjOBJ_BODY, "hand")
        self.cube_body_id = mj.mj_name2id(self.env.model, mj.mjtObj.mjOBJ_BODY, "pickup_cube")
        self.belt_body_id = mj.mj_name2id(self.env.model, mj.mjtObj.mjOBJ_BODY, "conv_belt")
        self.belt_geom_id = mj.mj_name2id(self.env.model, mj.mjtObj.mjOBJ_GEOM, "conv_belt_surface")
        self.weld_id = mj.mj_name2id(self.env.model, mj.mjtObj.mjOBJ_EQUALITY, "grasp_weld")
        self.cube_geom_id = -1
        if self.cube_body_id >= 0:
            geom_count = int(self.env.model.body_geomnum[self.cube_body_id])
            if geom_count > 0:
                self.cube_geom_id = int(self.env.model.body_geomadr[self.cube_body_id])
        self.is_grasped = False
        self.robot_root_id = mj.mj_name2id(self.env.model, mj.mjtObj.mjOBJ_BODY, "link0")
        self.robot_subtree_mask = self._compute_subtree_mask(self.robot_root_id)
        self.computed_max_reach = self._compute_max_reach()
