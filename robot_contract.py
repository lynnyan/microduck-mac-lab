"""Versioned steady-state runtime contract, independent of PPO implementation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

DEFAULT_CONTRACT = Path(__file__).resolve().parent / "contracts/runtime_default_v1.json"


def contract_hash(contract):
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_contract(path=DEFAULT_CONTRACT):
    contract = json.loads(Path(path).read_text())
    if (contract["observation_dim"], contract["action_dim"], contract["control_hz"]) != (61, 14, 50):
        raise ValueError("unsupported contract shape or frequency")
    if len(contract["home"]) != 14 or len(set(contract["joint_names"])) != 14:
        raise ValueError("invalid joint contract")
    if not np.isfinite(contract["home"]).all():
        raise ValueError("non-finite HOME")
    if contract.get("schema_version") != 1 or contract["physics_dt"] != 0.005:
        raise ValueError("unsupported schema or physics timestep")
    if contract["actuator"]["motor"] != "xl330" or contract["actuator"]["model"] != "m6":
        raise ValueError("unsupported actuator model")
    if contract["actuator"]["max_current"] is not None:
        raise ValueError("this profile does not implement firmware current limits")
    if contract["body_zero_indices"] != [55,56,60] or contract["mouth_hardware_index"] != 9:
        raise ValueError("unsupported command or mouth layout")
    if (contract["target_filter"]["first_tick"] != "unfiltered" or
            contract["target_filter"]["history"] != "filtered_targets_before_safety_clamp"):
        raise ValueError("unsupported target-filter semantics")
    if any(not 0 <= contract["target_filter"][k] <= 1 for k in ("head_alpha","legs_alpha")):
        raise ValueError("invalid low-pass coefficient")
    if contract["policy_to_hardware"] != list(range(9)) + list(range(10, 15)):
        raise ValueError("mouth exclusion mapping differs")
    return contract


def build_observation(gyro, gravity, positions, velocities, last_action, twist, head, body):
    """Mirror Observation::build, including hardware's unbound body x/y/yaw."""
    parts = [gyro, gravity, positions, velocities, last_action, twist, head, body]
    widths = [3, 3, 14, 14, 14, 3, 4, 6]
    if any(np.asarray(p).shape != (n,) for p, n in zip(parts, widths)):
        raise ValueError("observation block shape mismatch")
    out = np.concatenate(parts).astype(np.float32)
    out[[55, 56, 60]] = 0
    if not np.isfinite(out).all():
        raise ValueError("non-finite runtime observation")
    return out


def targets_from_action(action, previous, contract, slot):
    """Mirror robotd steady-state scale/filter, then Safety actuator travel clamp.

    previous is pre-safety filter history, not the physical or delayed target.
    The raw action is returned unchanged to the next observation elsewhere.
    """
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (14,) or not np.isfinite(action).all():
        raise ValueError("14 finite raw policy actions required")
    target = np.asarray(contract["home"]) + contract["slots"][slot]["action_scale"] * action
    if previous is not None:
        alpha = np.full(14, contract["target_filter"]["legs_alpha"])
        alpha[5:9] = contract["target_filter"]["head_alpha"]
        target = alpha * target + (1 - alpha) * previous
    history = target.copy()
    return np.clip(target, *contract["actuator_target_bounds"]), history
