"""Native MuJoCo training using a pinned robotd stand/walk runtime contract.

Only sensor and physics uncertainty vary across stages. Action interpretation,
raw-action feedback, target filtering and firmware gain never change with stage.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import mujoco
import numpy as np
from gymnasium import spaces

from microduck_env import MicroDuckEnv, JOINTS, upstream_helper
from robot_contract import DEFAULT_CONTRACT, build_observation, load_contract, targets_from_action


class RuntimeMicroDuckEnv(MicroDuckEnv):
    def __init__(self, task="stand", actuator="bam", render_mode=None,
                 episode_seconds=10.0, command=None, contract_path=str(DEFAULT_CONTRACT),
                 dr_profile="curriculum", training_progress=0,
                 nominal_steps=200_000, sensor_steps=600_000, policy_contract=None,
                 head_command=None, scene_path=None):
        if actuator != "bam":
            raise ValueError("runtime recipe requires BAM; XML is legacy-only")
        if dr_profile not in ("nominal", "sensors", "dynamics", "curriculum"):
            raise ValueError("unknown domain randomization profile")
        if nominal_steps < 0 or sensor_steps <= nominal_steps:
            raise ValueError("curriculum needs 0 <= nominal_steps < sensor_steps")
        self.contract_path = str(Path(contract_path).resolve())
        self.contract = load_contract(self.contract_path)
        if self.contract["joint_names"] != list(JOINTS):
            raise ValueError("MJCF servo order does not match runtime")
        self.dr_profile = dr_profile
        self.training_progress = training_progress
        self.fixed_head_command = None if head_command is None else np.asarray(head_command, dtype=np.float32)
        if self.fixed_head_command is not None and self.fixed_head_command.shape != (4,):
            raise ValueError("head_command must have four entries")
        self.nominal_steps, self.sensor_steps = nominal_steps, sensor_steps
        self._runtime_ready = False
        super().__init__(task=task, actuator=actuator, render_mode=render_mode,
                         episode_seconds=episode_seconds, command=command,
                         policy_contract="official", scene_path=scene_path)
        if not np.allclose(self.home, self.contract["home"], atol=1e-7):
            raise ValueError("asset HOME does not match runtime HOME")
        # Finite action bounds are an internal policy output clamp. They are
        # exported inside the graph; robotd receives that exact raw output.
        self.action_space = spaces.Box(-2.0, 2.0, (14,), dtype=np.float32)
        self.slot = task
        self.nominal_mass = self.model.body_mass.copy()
        self.nominal_inertia = self.model.body_inertia.copy()
        self.nominal_ipos = self.model.body_ipos.copy()
        self.nominal_armature = self.model.dof_armature.copy()
        self.nominal_damping = self.model.dof_damping.copy()
        self.nominal_dof_friction = self.model.dof_frictionloss.copy()
        self.nominal_friction = self.model.geom_friction.copy()
        self.nominal_force = self.model.actuator_forcerange.copy()
        self.head_command = np.zeros(4, dtype=np.float32)
        self.body_command = np.zeros(6, dtype=np.float32)
        self._runtime_ready = True
        self.reset()

    def set_training_progress(self, transitions):
        """Stage is sampled at episode reset, so parameters stay fixed per episode."""
        self.training_progress = int(transitions)

    def selected_profile(self):
        if self.dr_profile != "curriculum":
            return self.dr_profile
        if self.training_progress < self.nominal_steps:
            return "nominal"
        if self.training_progress < self.sensor_steps:
            return "sensors"
        return "dynamics"

    def _restore_and_randomize(self):
        # Always restore compile-time baselines: never multiply already-randomized values.
        self.model.body_mass[:] = self.nominal_mass
        self.model.body_inertia[:] = self.nominal_inertia
        self.model.body_ipos[:] = self.nominal_ipos
        self.model.dof_armature[:] = self.nominal_armature
        self.model.dof_damping[:] = self.nominal_damping
        self.model.dof_frictionloss[:] = self.nominal_dof_friction
        self.model.geom_friction[:] = self.nominal_friction
        self.model.actuator_forcerange[:] = self.nominal_force
        self.active_profile = self.selected_profile()
        a = self.contract["actuator"]
        self.bam.model.actuator.kp = self.contract["slots"][self.slot]["kp_fw"]
        self.bam.model.actuator.vin = a["voltage"]
        self.bam.vin_drop_gain = a["voltage_sag_gain"]
        self.bam.vin_min = a["voltage_min"]
        self.friction_scale = 1.0
        self.actuation_lag_substeps = 0
        self.encoder_bias = np.zeros(14)
        self.imu_rotation = np.eye(3)
        self.sensor_lag_steps = self.joint_velocity_lag_steps = 0
        self.noise_strength = 0.0
        if self.active_profile in ("sensors", "dynamics"):
            self.noise_strength = 0.5 if self.active_profile == "sensors" else 1.0
            s = self.noise_strength
            self.encoder_bias = self.np_random.uniform(-0.015*s, 0.015*s, 14)
            axis = self.np_random.normal(size=3)
            axis /= max(np.linalg.norm(axis), 1e-12)
            angle = self.np_random.uniform(-np.deg2rad(6*s), np.deg2rad(6*s))
            skew = np.array([[0,-axis[2],axis[1]], [axis[2],0,-axis[0]], [-axis[1],axis[0],0]])
            self.imu_rotation = np.eye(3) + np.sin(angle)*skew + (1-np.cos(angle))*(skew @ skew)
            self.sensor_lag_steps = int(self.np_random.integers(0, 2))
            self.joint_velocity_lag_steps = 1
        if self.active_profile == "dynamics":
            self.bam.model.actuator.vin = float(self.np_random.uniform(6.5, 8.2))
            self.bam.vin_drop_gain = float(self.np_random.uniform(0, 0.2))
            self.friction_scale = float(self.np_random.uniform(0.9, 1.1))
            self.actuation_lag_substeps = int(self.np_random.integers(3, 7))  # 15..30 ms
            mass_scale = float(self.np_random.uniform(0.95, 1.05))
            self.model.body_mass[self.body_id] *= mass_scale
            self.model.body_inertia[self.body_id] *= mass_scale
            self.model.body_ipos[self.body_id] += self.np_random.uniform(-0.003, 0.003, 3)
            self.model.dof_armature[self.vadr] *= self.np_random.uniform(0.9, 1.1, 14)
            self.model.geom_friction[self.foot_geom_ids, 0] = self.np_random.uniform(0.7, 1.3)
        # The motor torque bounds match both nominal and randomized voltage.
        limit = self.bam.model.actuator.vin * self.bam.model.kt.value / self.bam.model.R.value
        self.model.actuator_forcerange[self.act_ids] = [-limit, limit]
        mujoco.mj_setConst(self.model, self.data)

    def reset(self, *, seed=None, options=None):
        if not self._runtime_ready:
            return super().reset(seed=seed, options=options)
        # Initialize the Gym RNG before sampling episode-level physical uncertainty.
        if seed is not None:
            super(MicroDuckEnv, self).reset(seed=seed)
        self._restore_and_randomize()
        self.previous_targets = None
        self._sensor_stamp = None
        self._sensor_history = deque(maxlen=2)
        self._observation_cache = None
        # The three unbound body channels and all body controls remain zero here.
        # Train only small head deltas: beyond these ranges is outside this policy's task.
        self.head_command[:] = self.np_random.uniform([-0.05,-0.05,-0.07,-0.015], [0.05,0.05,0.07,0.015])
        if self.np_random.random() < 0.2:
            self.head_command[:] = 0
        if self.fixed_head_command is not None:
            self.head_command[:] = self.fixed_head_command
        self.body_command[:] = 0
        self._target_queue = deque([self.home.copy()] * 7, maxlen=7)
        obs, info = super().reset(seed=None, options=options)
        self.bam.model.reset()
        self.bam.last_ts = self.data.time
        info.update(dr_profile=self.active_profile, runtime_slot=self.slot)
        return obs, info

    def _observation(self):
        if not self._runtime_ready:
            return super()._observation()
        # Repeated get_observation at one instant must not resample noise or history.
        stamp = (self.steps, float(self.data.time))
        if self._sensor_stamp == stamp:
            return self._observation_cache.copy()
        sensor = (self.data.sensordata[self.gyro_slice].copy(),
                  self._rotation().T @ np.array([0.0,0.0,-1.0]),
                  self.data.qvel[self.vadr].copy())
        if not self._sensor_history:
            self._sensor_history.extend([sensor, sensor])
        else:
            self._sensor_history.append(sensor)
        gyro, gravity, _ = self._sensor_history[-1-self.sensor_lag_steps]
        velocity = self._sensor_history[-1-self.joint_velocity_lag_steps][2]
        strength = self.noise_strength
        gyro = self.imu_rotation @ gyro + self.np_random.uniform(-0.03*strength, 0.03*strength, 3)
        gravity = self.imu_rotation @ gravity + self.np_random.uniform(-0.01*strength, 0.01*strength, 3)
        position = (self.data.qpos[self.qadr] - self.home + self.encoder_bias
                    + self.np_random.uniform(-0.001*strength, 0.001*strength, 14))
        velocity = velocity + self.np_random.uniform(-0.25*strength, 0.25*strength, 14)
        try:
            obs = build_observation(gyro, gravity, position, velocity, self.last_action,
                                    self.command, self.head_command, self.body_command)
        except ValueError:
            # Base step detects physical non-finites and terminates the episode.
            obs = np.full(61, np.nan, dtype=np.float32)
        self._sensor_stamp, self._observation_cache = stamp, obs
        return obs.copy()

    def _action_to_target(self, action):
        # This action is the exact raw output emitted by the exported policy graph.
        if np.any(np.abs(action) > 2.000001):
            raise ValueError("policy exceeded its exported internal ±2 action bound")
        target, self.previous_targets = targets_from_action(
            action, self.previous_targets, self.contract, self.slot)
        self.commanded_target = target.copy()
        return target

    def _advance_physics(self, target):
        for _ in range(self.frame_skip):
            self._target_queue.append(target.copy())
            delayed = self._target_queue[-1-self.actuation_lag_substeps]
            self.bam.q_target[self.act_ids] = delayed
            self.bam.update()
            # BAM rewrites friction every substep; scale that fresh budget, not prior values.
            self.model.dof_frictionloss[self.vadr] *= self.friction_scale
            mujoco.mj_step(self.model, self.data)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if not terminated:
            head_error = self.data.qpos[self.qadr[5:9]] - self.home[5:9] - self.head_command
            head_reward = 0.5 * float(np.mean(np.exp(-(head_error / 0.15)**2)))
            # Measure actual joint limits without changing the runtime target mapping.
            joint_position = self.data.qpos[self.qadr]
            center = self.limits.mean(axis=1)
            half_width = np.diff(self.limits, axis=1).ravel() / 2
            margin_violation = np.maximum(np.abs(joint_position-center) - 0.9*half_width, 0)
            limit_cost = -2.0 * float(np.sum(margin_violation**2))
            reward += (head_reward + limit_cost) * self.dt
            info["reward_terms"].update(head_tracking=head_reward, joint_limit=limit_cost)
        info.update(dr_profile=self.active_profile, runtime_slot=self.slot,
                    actuation_lag_substeps=self.actuation_lag_substeps)
        return obs, float(reward), terminated, truncated, info
