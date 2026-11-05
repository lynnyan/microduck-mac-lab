"""Export a runtime-contract SB3 run with observation normalization inside ONNX.

Produces a local bundle. Does not upload, install or command a physical robot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from stable_baselines3 import PPO

from microduck_env import JOINTS
from robot_contract import contract_hash, load_contract
from runtime_env import RuntimeMicroDuckEnv


class RuntimeActor(torch.nn.Module):
    def __init__(self, model, normalizer):
        super().__init__()
        policy = model.policy
        if policy.squash_output or not normalizer.norm_obs or isinstance(normalizer.obs_rms, dict):
            raise ValueError("exporter requires normalized flat MLP observations and unsquashed PPO")
        self.features = policy.features_extractor
        self.actor = policy.mlp_extractor.policy_net
        self.action_net = policy.action_net
        self.register_buffer("mean", torch.tensor(normalizer.obs_rms.mean, dtype=torch.float32))
        self.register_buffer("variance", torch.tensor(normalizer.obs_rms.var, dtype=torch.float32))
        self.register_buffer("low", torch.tensor(model.action_space.low, dtype=torch.float32))
        self.register_buffer("high", torch.tensor(model.action_space.high, dtype=torch.float32))
        self.epsilon = normalizer.epsilon
        self.clip_obs = normalizer.clip_obs

    def forward(self, obs):
        normalized = torch.clamp((obs - self.mean) / torch.sqrt(self.variance + self.epsilon),
                                 -self.clip_obs, self.clip_obs)
        raw = self.action_net(self.actor(self.features(normalized)))
        # SB3 predict() clips continuous actions to action_space. It is part of
        # the exported policy, before robotd scale/filter and last_action feedback.
        return torch.maximum(torch.minimum(raw, self.high), self.low)


def export(run: Path, destination: Path):
    config = json.loads((run / "config.json").read_text())
    if config.get("recipe") != "runtime_contract_v1":
        raise ValueError("legacy experimental models cannot be exported as runtime-contract policies")
    contract = load_contract(run / "contract.json")
    if config["contract_sha256"] != contract_hash(contract):
        raise ValueError("contract hash mismatch")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("bundle directory must be empty; refusing to overwrite")
    destination.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    model = PPO.load(run / "model.zip", device="cpu")
    with (run / "vecnormalize.pkl").open("rb") as f:
        normalizer = pickle.load(f)  # only load trusted local training artifacts
    normalizer.training = False
    normalizer.norm_reward = False
    if model.observation_space.shape != (61,) or model.action_space.shape != (14,):
        raise ValueError("policy shape mismatch")
    if not np.allclose(model.action_space.high, 2) or not np.allclose(model.action_space.low, -2):
        raise ValueError("unexpected internal action clamp")
    model.policy.set_training_mode(False)
    wrapper = RuntimeActor(model, normalizer).eval()
    onnx_file = destination / "policy.onnx"
    with torch.no_grad():
        torch.onnx.export(wrapper, torch.zeros(1,61), str(onnx_file),
                          input_names=["obs"], output_names=["actions"],
                          opset_version=17, dynamo=False, external_data=False)
    graph = onnx.load(str(onnx_file))
    metadata = {
        "joint_names": ",".join(JOINTS),
        "default_joint_pos": ",".join(map(str, contract["home"])),
        "observation_names": "base_ang_vel,projected_gravity,joint_pos,joint_vel,actions,command,head_command,body_command",
        "action_scale": str(contract["slots"][config["task"]]["action_scale"]),
        "microduck.contract_json": json.dumps(contract, sort_keys=True),
        "microduck.contract_sha256": contract_hash(contract),
        "microduck.slot": config["task"],
        "microduck.recipe": "runtime_contract_v1",
        "microduck.normalization": "included; feed raw 61D obs; do not normalize twice",
        "microduck.internal_action_bound": "2.0",
    }
    onnx.helper.set_model_props(graph, metadata)
    onnx.checker.check_model(graph)
    onnx.save(graph, str(onnx_file))
    session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    if [(i.name, i.shape) for i in session.get_inputs()] != [("obs", [1,61])]:
        raise ValueError("runtime ONNX input contract failed")
    if [(i.name, i.shape) for i in session.get_outputs()] != [("actions", [1,14])]:
        raise ValueError("runtime ONNX output contract failed")

    def reference(obs):
        return model.predict(normalizer.normalize_obs(obs.reshape(1,-1)), deterministic=True)[0][0]

    # Collect real observation distributions under every uncertainty profile,
    # plus off-distribution inputs to exercise normalization clipping.
    samples = []
    for profile in ("nominal", "sensors", "dynamics"):
        env = RuntimeMicroDuckEnv(task=config["task"], contract_path=str(run / "contract.json"),
                                  dr_profile=profile, episode_seconds=5)
        try:
            obs, _ = env.reset(seed=8191)
            for _ in range(100):
                samples.append(obs.copy())
                obs, _, failed, timeout, _ = env.step(reference(obs))
                if failed or timeout:
                    obs, _ = env.reset()
        finally:
            env.close()
    rng = np.random.default_rng(4093)
    samples.extend(rng.normal(0, 10, size=(64,61)).astype(np.float32))
    samples = np.asarray(samples, dtype=np.float32)
    max_error = 0.0
    for obs in samples:
        result = session.run(["actions"], {"obs": obs.reshape(1,-1)})[0][0]
        if not np.isfinite(result).all():
            raise ValueError("non-finite exported action")
        max_error = max(max_error, float(np.max(np.abs(result-reference(obs)))))
    if max_error > 1e-4:
        raise ValueError(f"ONNX numerical mismatch {max_error}")
    np.savez_compressed(destination / "replay_inputs.npz", obs=samples)
    digest = hashlib.sha256(onnx_file.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 2, "model_api": 1, "obs_len": 61, "action_len": 14,
        "robot": {"model": "microduck", "hw_rev": 1, "servos": "xl330", "control_hz": 50},
        "name": f"mac-runtime-{config['task']}", "kind": "perpetual", "slot": config["task"],
        "entry_pose": "standing",
        "description": "Native CPU PPO trained with a pinned robotd control contract; hardware validation pending",
        "command": {"twist": contract["supported_commands"][config["task"]],
                    "head": "deltas: ±0.05, ±0.05, ±0.07, ±0.015 rad", "body": "zeros only"},
        "training": {"task_id": config["task"], "recipe": config["recipe"],
                     "run": str(run.resolve()), "runtime_commit": contract["runtime_commit"]},
        "eval": {"onnx_max_abs_error": max_error, "observations": len(samples),
                 "onnx_sha256": digest, "hardware_tested": False},
    }
    requirements = {"runtime_commit": contract["runtime_commit"], "slot": config["task"],
                    "tuning": contract["slots"][config["task"]],
                    "robotd_tuning": {
                        "action_scale": contract["slots"]["walk"]["action_scale"],
                        "standing_action_scale": contract["slots"]["stand"]["action_scale"],
                        "gain": contract["slots"]["walk"]["kp_fw"],
                        "standing_gain_ratio": contract["slots"]["stand"]["kp_fw"] / contract["slots"]["walk"]["kp_fw"],
                        "head_lowpass": contract["target_filter"]["head_alpha"],
                        "legs_lowpass": contract["target_filter"]["legs_alpha"],
                    },
                    "target_filter": contract["target_filter"],
                    "control_hz": 50, "contract_sha256": contract_hash(contract),
                    "normalization_inside_onnx": True, "external_action_clamp": False,
                    "body_command": "zero", "hardware_calibration_verified": False,
                    "note": "Gait manifest action_scale is not applied as runtime tuning automatically. Verify robotd parameters separately."}
    report = {"status": "passed", "samples": len(samples), "max_abs_error": max_error,
              "threshold": 1e-4, "onnx_sha256": digest, "contract_sha256": contract_hash(contract),
              "scope": "ONNX arithmetic and source-pinned steady-state interface; not hardware validation"}
    for filename, value in (("manifest.json", manifest), ("contract.json", contract),
                            ("deployment_requirements.json", requirements), ("validation.json", report)):
        (destination / filename).write_text(json.dumps(value, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.run, args.output)
