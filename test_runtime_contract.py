"""Regression checks for the robotd contract and physical uncertainty pipeline."""
import json
import unittest
import tempfile
from pathlib import Path

import numpy as np
from stable_baselines3.common.env_checker import check_env

from microduck_env import ROOT
from robot_contract import build_observation, load_contract, targets_from_action
from runtime_env import RuntimeMicroDuckEnv


class ContractTests(unittest.TestCase):
    def test_runtime_golden_targets(self):
        c = load_contract()
        home = np.array(c["home"])
        action = np.zeros(14)
        action[[0,5]] = [1.0, 0.4]
        first, history = targets_from_action(action, None, c, "stand")
        np.testing.assert_allclose(first[[0,5]], home[[0,5]] + [1,0.4])
        second, _ = targets_from_action(np.zeros(14), history, c, "stand")
        np.testing.assert_allclose(second[[0,5]], home[[0,5]] + [0.3,0.2])
        walk, _ = targets_from_action(action, None, c, "walk")
        np.testing.assert_allclose(walk[[0,5]], home[[0,5]] + [0.9,0.36])
        extreme, history = targets_from_action(np.full(14,10), None, c, "stand")
        np.testing.assert_allclose(extreme, np.pi)
        self.assertGreater(history[0], np.pi)  # filter history precedes Safety clamp

    def test_observation_layout_and_unbound_commands(self):
        obs = build_observation(np.arange(3), np.array([0,0,-1]), np.arange(14)+10,
                                np.arange(14)+30, np.arange(14)+50,
                                np.array([0.2,-0.1,0.3]), np.arange(4)+70, np.arange(6)+80)
        self.assertEqual(obs.shape, (61,))
        np.testing.assert_array_equal(obs[34:48], np.arange(14)+50)
        np.testing.assert_array_equal(obs[51:55], [70,71,72,73])
        np.testing.assert_array_equal(obs[55:61], [0,0,82,83,84,0])
        c = load_contract()
        hardware = np.arange(15)
        np.testing.assert_array_equal(hardware[c["policy_to_hardware"]],
                                      [0,1,2,3,4,5,6,7,8,10,11,12,13,14])

    def test_refuse_nonfinite_action(self):
        with self.assertRaises(ValueError):
            targets_from_action(np.full(14,np.nan), None, load_contract(), "stand")

    def test_both_slots_and_raw_feedback(self):
        for task, gain in (("stand",160),("walk",200)):
            env = RuntimeMicroDuckEnv(task=task, dr_profile="nominal")
            try:
                env.reset(seed=1)
                action = np.linspace(-0.15,0.15,14,dtype=np.float32)
                obs, _, _, _, _ = env.step(action)
                np.testing.assert_array_equal(obs[34:48], action)
                self.assertEqual(env.bam.model.actuator.kp,gain)
                np.testing.assert_array_equal(env._observation(),obs)
            finally:
                env.close()

    def test_profiles_reset_determinism_and_no_accumulation(self):
        for profile in ("nominal","sensors","dynamics"):
            env = RuntimeMicroDuckEnv(dr_profile=profile)
            try:
                first, _ = env.reset(seed=44)
                mass, ipos = env.model.body_mass.copy(), env.model.body_ipos.copy()
                voltage = env.bam.model.actuator.vin
                rollout = [env.step(np.zeros(14,np.float32))[0] for _ in range(5)]
                again, _ = env.reset(seed=44)
                np.testing.assert_array_equal(first,again)
                np.testing.assert_array_equal(env.model.body_mass,mass)
                np.testing.assert_array_equal(env.model.body_ipos,ipos)
                self.assertEqual(env.bam.model.actuator.vin,voltage)
                for expected in rollout:
                    actual = env.step(np.zeros(14,np.float32))[0]
                    np.testing.assert_allclose(actual,expected,atol=1e-6)
                self.assertTrue(0.95 <= mass[env.body_id]/env.nominal_mass[env.body_id] <= 1.05)
            finally:
                env.close()

    def test_curriculum_does_not_change_control_contract(self):
        env = RuntimeMicroDuckEnv(dr_profile="curriculum",nominal_steps=100,sensor_steps=200)
        try:
            for steps, profile in ((0,"nominal"),(100,"sensors"),(200,"dynamics")):
                env.set_training_progress(steps)
                env.reset(seed=5)
                self.assertEqual(env.active_profile,profile)
                target = env._action_to_target(np.full(14,0.2))
                np.testing.assert_allclose(target,env.home+0.2)
                self.assertEqual(env.bam.model.actuator.kp,160)
        finally:
            env.close()

    def test_actuation_delay_is_in_physics_substeps(self):
        env = RuntimeMicroDuckEnv(dr_profile="nominal")
        try:
            env.reset(seed=2)
            env.actuation_lag_substeps=3
            samples=[]
            original=env.bam.update
            def record():
                samples.append(env.bam.q_target.copy())
                original()
            env.bam.update=record
            target=env.home+0.1
            env._advance_physics(target)
            for sample in samples[:3]:
                np.testing.assert_allclose(sample,env.home)
            np.testing.assert_allclose(samples[3],target)
        finally:
            env.close()

    def test_gym_api_and_timeout(self):
        env = RuntimeMicroDuckEnv(dr_profile="nominal",episode_seconds=0.06)
        try:
            check_env(env,warn=True)
            env.reset(seed=17)
            for _ in range(3):
                _, _, terminated, truncated, _ = env.step(np.zeros(14,np.float32))
            self.assertFalse(terminated)
            self.assertTrue(truncated)
        finally:
            env.close()

    def test_exporter_refuses_legacy_run(self):
        from export_runtime import export
        with tempfile.TemporaryDirectory() as temporary:
            run=Path(temporary)
            (run / "config.json").write_text(json.dumps({"recipe":"native_cpu_experimental_v1"}))
            with self.assertRaisesRegex(ValueError,"legacy"):
                export(run,run / "forbidden_export")

    def test_resume_uses_model_counter_on_first_rollout(self):
        from types import SimpleNamespace
        from train import SaveBundle
        events=[]
        vector=SimpleNamespace(env_method=lambda name,value: events.append((name,value)))
        callback=SaveBundle(ROOT / "outputs",20_000)
        callback.model=SimpleNamespace(runtime_recipe=True,num_timesteps=700_000,get_env=lambda:vector)
        # BaseCallback's own counter is still zero on the first resumed rollout.
        callback.num_timesteps=0
        callback._on_rollout_start()
        self.assertEqual(events,[("set_training_progress",700_000)])


if __name__ == "__main__":
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={"tests":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),
            "passed":result.wasSuccessful(),"runtime_commit":load_contract()["runtime_commit"]}
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs/runtime_contract_tests.json").write_text(json.dumps(report,indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
