"""Train CPU PPO using the pinned robotd contract (or explicit legacy recipe)."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from microduck_env import MicroDuckEnv, ROOT, UPSTREAM_SHA
from robot_contract import DEFAULT_CONTRACT, contract_hash, load_contract
from runtime_env import RuntimeMicroDuckEnv


class SaveBundle(BaseCallback):
    def __init__(self, directory, every):
        super().__init__()
        self.directory, self.every, self.last_save = directory, every, 0

    def _on_step(self):
        if self.num_timesteps - self.last_save >= self.every:
            self.model.save(self.directory / "checkpoint.zip")
            self.training_env.save(self.directory / "checkpoint_vecnormalize.pkl")
            self.last_save = self.num_timesteps
        return True

    def _on_rollout_start(self):
        # Change uncertainty only; never change control mapping across curriculum stages.
        if self.model.runtime_recipe:
            self.training_env.env_method("set_training_progress", self.model.num_timesteps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=["runtime", "legacy"], default="runtime")
    parser.add_argument("--scene", type=Path, help="custom scene XML; the task reward is unchanged")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--dr-profile", choices=["nominal", "sensors", "dynamics", "curriculum"], default="curriculum")
    parser.add_argument("--nominal-steps", type=int, default=200_000)
    parser.add_argument("--sensor-steps", type=int, default=600_000)
    parser.add_argument("--task", choices=["stand", "walk"], default="stand")
    parser.add_argument("--actuator", choices=["bam", "xml"], default="bam")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=20_000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path, help="existing run directory; also restores normalization")
    args = parser.parse_args()
    runtime = args.recipe == "runtime"
    contract = load_contract(args.contract) if runtime else None
    if runtime and args.actuator != "bam":
        parser.error("runtime recipe requires --actuator bam")
    if args.nominal_steps < 0 or args.sensor_steps <= args.nominal_steps:
        parser.error("0 <= nominal_steps < sensor_steps required")
    if min(args.timesteps, args.envs, args.n_steps, args.batch_size, args.threads, args.save_every) < 1:
        parser.error("numeric budgets must be positive")
    if args.envs * args.n_steps < 2 or (args.envs * args.n_steps) % args.batch_size:
        parser.error("envs*n_steps must be >=2 and divisible by batch_size")
    if args.resume:
        previous = json.loads((args.resume / "config.json").read_text())
        if (previous["task"], previous["actuator"]) != (args.task, args.actuator):
            parser.error("resume requires the same task and actuator as the original run")
        was_runtime = previous.get("recipe") == "runtime_contract_v1"
        if was_runtime != runtime:
            parser.error("cannot resume legacy weights into the runtime recipe; train from scratch")
        if runtime and previous.get("contract_sha256") != contract_hash(contract):
            parser.error("runtime contract changed; resume refused")
    torch.set_num_threads(args.threads)
    if args.scene is None and args.resume and previous.get("scene"):
        args.scene = Path(previous["scene"])
    if args.scene is not None:
        args.scene = args.scene.resolve()
        if not args.scene.is_file():
            parser.error("scene XML does not exist; generate it first")
    directory = args.output or ROOT / "outputs" / f"{args.recipe}_{args.task}_{time.strftime('%Y%m%d_%H%M%S')}"
    directory.mkdir(parents=True, exist_ok=True)
    # Refuse to overwrite a previous run accidentally.
    if (directory / "config.json").exists():
        parser.error("output already contains a run; choose a new output directory")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(upstream_commit=UPSTREAM_SHA, platform=platform.platform(),
                  python=sys.version, torch=torch.__version__, device="cpu",
                  action_scale=contract["slots"][args.task]["action_scale"] if runtime else 0.35,
                  recipe="runtime_contract_v1" if runtime else "native_cpu_experimental_v1")
    env_kwargs = {"task": args.task, "actuator": args.actuator, "scene_path": args.scene}
    if runtime:
        (directory / "contract.json").write_text(json.dumps(contract, indent=2))
        config.update(contract_sha256=contract_hash(contract),
                      runtime_commit=contract["runtime_commit"])
        env_kwargs.update(contract_path=str((directory / "contract.json").resolve()),
                          dr_profile=args.dr_profile, nominal_steps=args.nominal_steps,
                          sensor_steps=args.sensor_steps)
    (directory / "config.json").write_text(json.dumps(config, indent=2))
    vec = make_vec_env(RuntimeMicroDuckEnv if runtime else MicroDuckEnv, n_envs=args.envs, seed=args.seed,
                       env_kwargs=env_kwargs,
                       vec_env_cls=SubprocVecEnv if args.envs > 1 else DummyVecEnv,
                       monitor_dir=str(directory / "monitor"))
    if args.resume:
        vec = VecNormalize.load(str(args.resume / "vecnormalize.pkl"), vec)
        vec.training = True
        vec.norm_reward = True
        model = PPO.load(args.resume / "model.zip", env=vec, device="cpu", seed=args.seed)
        if model.n_steps != args.n_steps or model.batch_size != args.batch_size:
            vec.close()
            parser.error("resume requires the original n_steps and batch_size")
    else:
        vec = VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10.0)
        model = PPO("MlpPolicy", vec, device="cpu", seed=args.seed,
                    learning_rate=3e-4, n_steps=args.n_steps, batch_size=args.batch_size,
                    n_epochs=5, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                    ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, target_kl=0.03,
                    policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]},
                                   "activation_fn": torch.nn.ELU, "log_std_init": -2.0 if runtime else -1.0},
                    verbose=1)
    model.runtime_recipe = runtime
    if runtime:
        vec.env_method("set_training_progress", model.num_timesteps)
    model.set_logger(configure(str(directory), ["stdout", "csv", "tensorboard"]))
    interrupted = False
    status = "failed"
    try:
        model.learn(total_timesteps=args.timesteps, reset_num_timesteps=not bool(args.resume),
                    callback=SaveBundle(directory, args.save_every))
        status = "complete"
    except KeyboardInterrupt:
        interrupted = True
        status = "interrupted"
        print("Interrupted; saving current policy and normalization before closing.")
    finally:
        model.save(directory / "model.zip")
        vec.save(directory / "vecnormalize.pkl")
        (directory / "training_status.json").write_text(json.dumps({
            "timesteps": model.num_timesteps, "interrupted": interrupted, "status": status}, indent=2))
        vec.close()
    print(f"Saved local CPU experiment: {directory}")


if __name__ == "__main__":
    main()
