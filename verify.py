"""Native physics, Gym API, reset determinism and HOME stability measurement."""
import json
import platform
from importlib.metadata import version

import numpy as np
import torch
from stable_baselines3.common.env_checker import check_env

from microduck_env import MicroDuckEnv, ROOT, UPSTREAM_SHA


def main():
    env = MicroDuckEnv()
    try:
        check_env(env, warn=True)
        a, _ = env.reset(seed=123)
        env.step(np.zeros(14, np.float32))
        b, _ = env.reset(seed=123)
        np.testing.assert_allclose(a, b, atol=1e-7)
        env.reset(seed=42, options={"joint_noise": 0.0})
        for step in range(150):
            obs, reward, terminated, _, info = env.step(np.zeros(14, np.float32))
            assert np.isfinite(obs).all() and np.isfinite(reward)
            if terminated:
                break
        home = {"simulated_seconds": (step + 1) * env.dt,
                "failed": terminated, "height": info["height"], "upright": info["upright"],
                "note": "HOME is a reference pose, not a balancing controller"}
        # A falling untrained robot is expected. Verify reset restores a valid state.
        restored, _ = env.reset(seed=42)
        assert np.isfinite(restored).all() and env.steps == 0
        report = {"platform": platform.platform(), "upstream_commit": UPSTREAM_SHA,
                  "versions": {n: version(n) for n in
                               ("mujoco", "torch", "stable-baselines3", "gymnasium", "better-actuator-models")},
                  "cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available(),
                  "training_device": "cpu", "gym_api_check": "passed",
                  "reset_determinism": "passed", "observation_dim": 61, "action_dim": 14,
                  "home_baseline": home, "finite": True}
    finally:
        env.close()
    directory = ROOT / "outputs"
    directory.mkdir(exist_ok=True)
    (directory / "verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
