"""View or evaluate a local SB3 bundle, upstream ONNX, or zero-action baseline."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from microduck_env import MicroDuckEnv, JOINTS
from robot_contract import contract_hash, load_contract
from runtime_env import RuntimeMicroDuckEnv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--run", type=Path, help="local training output directory")
    group.add_argument("--onnx", type=Path, help="upstream 61D/14D ONNX with normalizer baked in")
    group.add_argument("--zero", action="store_true", help="hold HOME without a trained policy")
    p.add_argument("--task", choices=["stand", "walk"], default="stand")
    p.add_argument("--scene", type=Path, help="custom scene XML (overrides the run's scene)")
    p.add_argument("--actuator", choices=["bam", "xml"], default="bam")
    p.add_argument("--command", nargs=3, type=float, default=[0.0, 0.0, 0.0], metavar=("VX", "VY", "WZ"))
    p.add_argument("--seconds", type=float, default=10.0, help="simulated seconds per episode")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--keyboard", action="store_true", help="persistent keyboard velocity commands in the viewer")
    p.add_argument("--video", type=Path)
    p.add_argument("--report", type=Path)
    p.add_argument("--dr-profile", choices=["nominal", "sensors", "dynamics"], default="nominal")
    p.add_argument("--head-command", nargs=4, type=float, default=[0,0,0,0])
    args = p.parse_args()
    if args.seconds <= 0 or args.episodes < 1:
        p.error("positive seconds and episodes required")
    if args.keyboard:
        if args.headless or args.video or args.zero:
            p.error("--keyboard requires an interactive viewer and a walking policy")
        if args.run is None and args.onnx is None:
            args.onnx = Path(__file__).resolve().parent / "models/alpha_walking.onnx"
        args.task = "walk"
    torch.set_num_threads(1)
    mode = "rgb_array" if args.video else None if args.headless else "human"
    task, actuator = args.task, args.actuator
    runtime = False
    contract_path = None
    if args.run:
        config = json.loads((args.run / "config.json").read_text())
        if args.scene is None and config.get("scene"):
            args.scene = Path(config["scene"])
        task, actuator = config["task"], config["actuator"]
        runtime = config.get("recipe") == "runtime_contract_v1"
        if runtime:
            contract_path = args.run / "contract.json"
            if contract_hash(load_contract(contract_path)) != config["contract_sha256"]:
                p.error("training run contract hash mismatch")
    if args.onnx:
        import onnxruntime as ort
        session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
        metadata = session.get_modelmeta().custom_metadata_map
        if metadata.get("microduck.recipe") == "runtime_contract_v1":
            runtime = True
            task = metadata["microduck.slot"]
            actuator = "bam"
            contract_path = args.onnx.parent / "contract.json"
            if contract_hash(load_contract(contract_path)) != metadata["microduck.contract_sha256"]:
                p.error("ONNX bundle contract hash mismatch")
    if args.keyboard and task != "walk":
        p.error("--keyboard requires a walking policy; the selected run is a standing policy")
    kwargs = dict(task=task, actuator=actuator, render_mode=mode,
                  episode_seconds=args.seconds, command=args.command, scene_path=args.scene)
    if runtime:
        kwargs.update(contract_path=str(contract_path), dr_profile=args.dr_profile,
                      head_command=args.head_command)
        env = RuntimeMicroDuckEnv(**kwargs)
    else:
        env = MicroDuckEnv(**kwargs, policy_contract="official" if args.onnx else "local")
    normalizer = None
    if args.run:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        # Used only for normalization. The physics loop below steps the scalar env.
        normalizer = VecNormalize.load(str(args.run / "vecnormalize.pkl"), DummyVecEnv([lambda: env]))
        normalizer.training = False
        normalizer.norm_reward = False
        model = PPO.load(args.run / "model.zip", device="cpu")

        def policy(obs):
            norm = normalizer.normalize_obs(obs.reshape(1, -1))
            return model.predict(norm, deterministic=True)[0][0]
    elif args.onnx:
        inp, out = session.get_inputs()[0], session.get_outputs()[0]
        if inp.shape != [1, 61] or out.shape != [1, 14]:
            p.error(f"expected [1,61] -> [1,14], got {inp.shape} -> {out.shape}")
        metadata = session.get_modelmeta().custom_metadata_map
        if metadata.get("joint_names", "").split(",") != list(JOINTS):
            p.error("ONNX joint order does not match this MicroDuck model")
        expected_scale = env.contract["slots"][task]["action_scale"] if runtime else 1.0
        if float(metadata.get("action_scale", "nan")) != expected_scale:
            p.error("ONNX action scale does not match its runtime contract")
        defaults = np.fromstring(metadata.get("default_joint_pos", ""), sep=",")
        if defaults.shape != (14,) or not np.allclose(defaults, env.home, atol=0.002):
            p.error("ONNX HOME pose does not match this MicroDuck model")

        def policy(obs):
            return session.run(None, {inp.name: obs.reshape(1, -1)})[0][0]
    else:
        def policy(obs):
            return np.zeros(14, dtype=np.float32)

    keyboard = None
    if args.keyboard:
        from keyboard_control import KeyboardCommands
        keyboard = KeyboardCommands()
        env.viewer_key_callback = keyboard.on_key
        env.viewer_command_provider = keyboard.snapshot
        env.viewer_stop_callback = keyboard.stop
        env.camera.distance = 1.6
        env.camera.elevation = -35
        print("Click the simulation window. W/S: forward/back; A/D: turn; Q/E: sidestep; SPACE: stop.", flush=True)
        print("Commands persist after key release. SPACE clears all axes. New episodes start stopped.", flush=True)

    writer = None
    records = []
    closed = False
    try:
        if args.video:
            import imageio.v2 as imageio
            args.video.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(str(args.video), fps=50, codec="libx264")
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            if keyboard:
                keyboard.stop()
            total, squared_errors, yaw_errors = 0.0, [], []
            start_x = float(env.data.xpos[env.body_id, 0])
            done = truncated = False
            while not (done or truncated):
                started = time.monotonic()
                if keyboard:
                    env.command[:] = keyboard.snapshot()
                    # The previous observation must contain the latest user command before inference.
                    obs[48:51] = env.command
                obs, reward, done, truncated, info = env.step(policy(obs))
                total += reward
                squared_errors.append(info["velocity_error_sq"])
                yaw_errors.append(info["yaw_error_sq"])
                if writer:
                    writer.append_data(env.render())
                if mode == "human":
                    if env.viewer is not None and not env.viewer.is_running():
                        closed = True
                        break
                    time.sleep(max(0.0, env.dt - (time.monotonic() - started)))
            if closed:
                break
            record = {"seed": args.seed + episode, "seconds": env.steps * env.dt,
                      "survived": bool(truncated and not done), "failed": done,
                      "return": total, "velocity_rmse": float(np.sqrt(np.mean(squared_errors))),
                      "yaw_rmse": float(np.sqrt(np.mean(yaw_errors))),
                      "world_x_displacement": float(env.data.xpos[env.body_id, 0] - start_x)}
            records.append(record)
            print(json.dumps(record))
    finally:
        if writer:
            writer.close()
        if normalizer:
            normalizer.close()
        else:
            env.close()
    result = {"policy": str(args.run or args.onnx or "HOME zero-action baseline"),
              "contract": "runtime_contract_v1" if runtime else "official" if args.onnx else "local", "actuator": actuator,
              "dr_profile": args.dr_profile if runtime else "legacy_nominal",
              "head_command": args.head_command if runtime else [0,0,0,0],
              "command": args.command, "episode_seconds": args.seconds,
              "scene": str(env.scene_path),
              "keyboard": args.keyboard,
              "completed_episodes": len(records), "window_closed_early": closed,
              "survival_rate": float(np.mean([r["survived"] for r in records])) if records else None,
              "episodes": records,
              "scope": ("source-pinned steady-state runtime contract with requested uncertainty profile; not hardware validation"
                        if runtime else "native CPU rehearsal, simplified nominal physics; not official training benchmark")}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
