"""Apple Silicon CPU experiment; not the upstream mjlab training recipe.

Uses the unchanged upstream MJCF and BAM CPU loader (Apache-2.0).
Keeps the 61D input layout; rewards, noise, delay and PPO implementation differ.
No resulting policy should be deployed to hardware without separate validation.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "upstream"
UPSTREAM_SHA = "bae39f5900b318f7e87fc485c30a31213cfa14eb"
JOINTS = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)
_helper = None


def upstream_helper():
    global _helper
    if _helper is None:
        path = UPSTREAM / "scripts/infer_policy.py"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; run bash scripts/setup.sh")
        spec = importlib.util.spec_from_file_location("microduck_cpu_helpers", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        _helper = module
    return _helper


class MicroDuckEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, task="stand", actuator="bam", render_mode=None,
                 episode_seconds=10.0, command=None, policy_contract="local", scene_path=None):
        super().__init__()
        if task not in ("stand", "walk") or actuator not in ("bam", "xml"):
            raise ValueError("task=stand|walk, actuator=bam|xml")
        self.task, self.actuator, self.render_mode = task, actuator, render_mode
        if policy_contract not in ("local", "official"):
            raise ValueError("policy_contract must be local or official")
        self.policy_contract = policy_contract
        self.fixed_command = None if command is None else np.asarray(command, dtype=np.float32)
        if self.fixed_command is not None and self.fixed_command.shape != (3,):
            raise ValueError("command must contain vx, vy, wz")
        self.dt = 0.02
        self.frame_skip = 4
        self.max_steps = int(episode_seconds / self.dt)
        if self.max_steps < 1:
            raise ValueError("episode_seconds must be at least 0.02")
        helper = upstream_helper()
        self.scene_path = Path(scene_path).resolve() if scene_path is not None else (
            UPSTREAM / "src/mjlab_microduck/robot/microduck/scene_walk.xml")
        scene = str(self.scene_path)
        self.bam = None
        if actuator == "bam":
            motor = helper.load_bam_model(200.0, 7.4, None)
            self.model, self.data, self.bam, _ = helper.load_mujoco_with_bam(
                scene, motor, 0.005, vin_drop_gain=0.1, vin_min=6.0)
        else:
            self.model = mujoco.MjModel.from_xml_path(scene)
            self.model.opt.timestep = 0.005
            self.data = mujoco.MjData(self.model)
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.model.opt.iterations = 30
        self.body_id = self.model.body("trunk_base").id
        self.joint_ids = np.array([self.model.joint(n).id for n in JOINTS])
        self.qadr = self.model.jnt_qposadr[self.joint_ids]
        self.vadr = self.model.jnt_dofadr[self.joint_ids]
        self.act_ids = np.array([self.model.actuator(n).id for n in JOINTS])
        assert self.model.nu == len(JOINTS) == 14
        self.home = helper.DEFAULT_POSE.astype(np.float64).copy()
        self.limits = self.model.jnt_range[self.joint_ids].copy()
        self.foot_geom_ids = [self.model.geom(n).id for n in
                              ("left_foot_collision", "right_foot_collision")]
        self.floor_id = self.model.geom("floor").id
        self.support_geom_ids = {self.floor_id} | {
            i for i in range(self.model.ngeom)
            if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("terrain_")
        }
        for g in self.foot_geom_ids:
            self.model.geom_friction[g, 0] = 1.0
            self.model.geom_condim[g] = 3
            # Custom terrain has equal priority so its measured contact settings participate.
            self.model.geom_priority[g] = 1 if scene_path is None else 0
        self.gyro_slice = self._sensor_slice("imu_ang_vel")
        self.vel_slice = self._sensor_slice("imu_lin_vel")
        self.action_space = spaces.Box(-1.0, 1.0, (14,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (61,), dtype=np.float32)
        self.last_action = np.zeros(14, np.float32)
        self.command = np.zeros(3, np.float32)
        self.steps = 0
        self.viewer = self.renderer = None
        self.camera = mujoco.MjvCamera()
        self.camera.distance = 0.65
        self.camera.azimuth = 145
        self.camera.elevation = -15
        self.reset()

    def _sensor_slice(self, name):
        sensor = self.model.sensor(name).id
        start = self.model.sensor_adr[sensor]
        return slice(start, start + self.model.sensor_dim[sensor])

    def _rotation(self):
        return self.data.xmat[self.body_id].reshape(3, 3)

    def _observation(self):
        gravity = self._rotation().T @ np.array([0.0, 0.0, -1.0])
        return np.concatenate((
            self.data.sensordata[self.gyro_slice], gravity,
            self.data.qpos[self.qadr] - self.home,
            self.data.qvel[self.vadr], self.last_action, self.command,
            np.zeros(10),  # head_pose(4), body_pose(6), fixed zero in this experiment
        )).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("STAND").id)
        noise = float((options or {}).get("joint_noise", 0.01))
        self.data.qpos[self.qadr] = np.clip(
            self.home + self.np_random.uniform(-noise, noise, 14),
            self.limits[:, 0] + 0.001, self.limits[:, 1] - 0.001)
        self.data.qvel[:] = 0
        self.last_action[:] = 0
        self.steps = 0
        if self.fixed_command is not None:
            self.command[:] = self.fixed_command
        elif self.task == "stand" or self.np_random.random() < 0.25:
            self.command[:] = 0
        else:
            self.command[:] = self.np_random.uniform([-0.2, -0.1, -0.5], [0.3, 0.1, 0.5])
        mujoco.mj_forward(self.model, self.data)
        if self.bam is not None:
            self.bam.reset(self.data.qpos)
            self.bam.q_target[self.act_ids] = self.home
        else:
            self.data.ctrl[self.act_ids] = self.home
        return self._observation(), {"command": self.command.copy()}

    def _contacts(self):
        found = np.zeros(2, dtype=bool)
        for contact in self.data.contact:
            pair = (contact.geom1, contact.geom2)
            if any(g in self.support_geom_ids for g in pair):
                for i, g in enumerate(self.foot_geom_ids):
                    found[i] |= g in pair
        return found

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (14,) or not np.isfinite(action).all():
            raise ValueError("action must be a finite 14D vector")
        if self.policy_contract == "local":
            action = np.clip(action, -1, 1)
        target = self._action_to_target(action)
        previous = self.last_action.copy()
        self._advance_physics(target)
        # Refresh kinematics/sensors at the integrated state for consistent observations.
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        self.last_action[:] = action
        finite = bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())
        height = float(self.data.xpos[self.body_id, 2])
        upright = float(self._rotation()[2, 2]) if finite else -1.0
        velocity = self.data.sensordata[self.vel_slice].copy()
        gyro = self.data.sensordata[self.gyro_slice].copy()
        vel_error = float(np.sum((velocity[:2] - self.command[:2]) ** 2))
        yaw_error = float((gyro[2] - self.command[2]) ** 2)
        pose_error = float(np.mean((self.data.qpos[self.qadr] - self.home) ** 2))
        terms = {
            "alive": 1.0,
            "upright": 2.0 * max(upright, 0.0) ** 2,
            "velocity": 2.0 * np.exp(-vel_error / 0.04),
            "yaw": 1.0 * np.exp(-yaw_error / 0.25),
            "pose": 0.5 * np.exp(-pose_error / (0.08 if self.task == "walk" else 0.02)),
            "action_rate": -0.05 * float(np.sum((action - previous) ** 2)),
            "joint_speed": -0.0002 * float(np.sum(self.data.qvel[self.vadr] ** 2)),
        }
        failed = not finite or height < 0.045 or upright < np.cos(np.deg2rad(70))
        reward = float(sum(terms.values()) * self.dt - (1.0 if failed else 0.0))
        obs = self._observation()
        if not np.isfinite(obs).all() or not np.isfinite(reward):
            failed, reward = True, -1.0
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        info = {"height": height, "upright": upright, "velocity": velocity,
                "velocity_error_sq": vel_error, "yaw_error_sq": yaw_error,
                "foot_contacts": self._contacts(), "reward_terms": terms,
                "failure": bool(failed)}
        if self.render_mode == "human":
            self.render()
        return obs, reward, bool(failed), self.steps >= self.max_steps, info

    def _action_to_target(self, action):
        # Bounded legacy experimental actions, retained for old checkpoint replay.
        if self.policy_contract == "official":
            target = self.home + action
        else:
            target = np.clip(self.home + 0.35 * action,
                             self.limits[:, 0] + 0.01, self.limits[:, 1] - 0.01)
        return target

    def _advance_physics(self, target):
        if self.bam is not None:
            self.bam.q_target[self.act_ids] = target
        else:
            self.data.ctrl[self.act_ids] = target
        for _ in range(self.frame_skip):
            if self.bam is not None:
                self.bam.update()
            mujoco.mj_step(self.model, self.data)

    def render(self):
        self.camera.lookat[:] = self.data.xpos[self.body_id]
        if self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.renderer.update_scene(self.data, camera=self.camera)
            return self.renderer.render()
        if self.render_mode == "human":
            if self.viewer is None:
                if getattr(self, "viewer_key_callback", None) is not None:
                    from driving_viewer import DrivingViewer
                    self.viewer = DrivingViewer(self.model, self.data, self.viewer_key_callback,
                                                self.viewer_command_provider, self.viewer_stop_callback)
                else:
                    from mujoco import viewer as mj_viewer
                    self.viewer = mj_viewer.launch_passive(self.model, self.data)
                self.viewer.cam.distance = self.camera.distance
                self.viewer.cam.azimuth = self.camera.azimuth
                self.viewer.cam.elevation = self.camera.elevation
            if self.viewer.is_running():
                self.viewer.cam.lookat[:] = self.camera.lookat
                self.viewer.sync()
        return None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
