"""Scene24 task loading and Gym integration with the original Q-Planning runners."""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
from . import episode_policy_seed


_CAMERAS = {"image": "agentview", "wrist_image": "robot0_eye_in_hand"}


def register_bowls(assets):
    from libero.libero.envs.base_object import OBJECTS_DICT, register_object
    from robosuite.models.objects import MujocoXMLObject

    class ColoredBowl(MujocoXMLObject):
        def __init__(self, name):
            xml = assets / "extension_objects" / "akita_bowl" / (self.kind + ".xml")
            super().__init__(str(xml), name=name, joints=[dict(type="free", damping="0.0005")],
                             obj_type="all", duplicate_collision_geoms=False)
            self.category_name = self.kind
            self.rotation = (0, 0)
            self.rotation_axis = "x"
            self.object_properties = {"vis_site_names": {}}

    class AkitaGreenBowl(ColoredBowl):
        kind = "akita_green_bowl"

    class AkitaBlueBowl(ColoredBowl):
        kind = "akita_blue_bowl"

    for bowl in (AkitaGreenBowl, AkitaBlueBowl):
        if bowl.kind not in OBJECTS_DICT:
            register_object(bowl)


class Scene24Env(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, root, width, height, max_steps, seed_pair):
        import torch
        from libero.libero.envs import OffScreenRenderEnv

        root = Path(root)
        snapshot = root / "task_snapshot"
        manifest = json.loads((snapshot / "task.json").read_text())
        self.task = self.task_description = manifest["task"]["goal_prompt"]
        self.states = torch.load(snapshot / manifest["files"]["init"], map_location="cpu",
                                 weights_only=True).numpy()
        self.seed_pair = seed_pair
        self._max_episode_steps = max_steps
        self._camera_width = width
        self._camera_height = height
        self.action_space = gym.spaces.Box(-1.0, 1.0, (7,), dtype=np.float32)
        rgb = gym.spaces.Box(0, 255, (height, width, 3), dtype=np.uint8)
        self.observation_space = gym.spaces.Dict({
            "pixels": gym.spaces.Dict({"image": rgb, "wrist_image": rgb}),
            "agent_pos": gym.spaces.Box(-np.inf, np.inf, (8,), dtype=np.float32),
        })
        register_bowls(root / "assets")
        self.sim = OffScreenRenderEnv(
            bddl_file_name=str(snapshot / manifest["files"]["bddl"]),
            scene_xml=str((root / "assets/scenes/libero_kitchen_tabletop_base_style.xml").resolve()),
            use_camera_obs=False, camera_heights=height, camera_widths=width,
            camera_names=tuple(_CAMERAS.values()),
            control_freq=20, horizon=max_steps,
        )
        self._last_obs = None

    def _observation(self, raw):
        from robosuite.utils.transform_utils import quat2axisangle

        self.camera_raw = {
            name: self.sim.sim.render(width=self._camera_width, height=self._camera_height, camera_name=camera)
            for name, camera in _CAMERAS.items()
        }
        state = np.concatenate((raw["robot0_eef_pos"], quat2axisangle(raw["robot0_eef_quat"]),
                                raw["robot0_gripper_qpos"])).astype(np.float32)
        self._last_obs = {
            "pixels": {name: np.ascontiguousarray(image[::-1]) for name, image in self.camera_raw.items()},
            "agent_pos": state,
        }
        return self._last_obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options and "init_index" in options:
            scene_seed, init = int(seed), int(options["init_index"])
        else:
            scene_seed, init = self.seed_pair(0 if seed is None else seed)
        if not 0 <= init < len(self.states):
            raise ValueError(f"initial state {init} outside [0, {len(self.states)})")
        self.sim.seed(scene_seed)
        self.sim.reset()
        raw = self.sim.set_init_state(self.states[init])
        episode_policy_seed.set(scene_seed)
        self.elapsed = 0
        return self._observation(raw), {"seed": scene_seed, "init_index": init, "is_success": False}

    def step(self, action):
        raw, _, done, _ = self.sim.step(np.asarray(action))
        # Refresh derived body and camera poses after the final physics substep.
        self.sim.sim.forward()
        self.elapsed += 1
        success = bool(done)
        return (self._observation(raw), float(success), success,
                self.elapsed >= self._max_episode_steps and not success, {"is_success": success})

    def render(self):
        return self._last_obs["pixels"]["image"]

    def close(self):
        self.sim.close()


@dataclass(frozen=True)
class Scene24Config:
    type: str
    episode_length: int
    observation_height: int
    observation_width: int


def episode_seed_pair(cfg, token):
    protocol = cfg.bc.extra["episode_protocol"]
    n_init = protocol["num_initial_states"]
    first_train_token = protocol["upstream_train_seed_offset"] + cfg.seed
    if token >= first_train_token:
        iteration, within = divmod(token - first_train_token, cfg.self_improve.train_episodes_per_task)
        round_number = protocol["first_collection_round"] + iteration
        seed = protocol["train_seed_base"] + protocol["train_round_stride"] * round_number + within
        return seed, within % n_init
    seed, init = divmod(token - cfg.eval.start_seed, n_init)
    return protocol["eval_seed_start"] + seed, init


class Scene24Suite:
    @staticmethod
    def build_config(cfg, task):
        return Scene24Config(task, cfg.benchmark.episode_length, cfg.benchmark.observation_height,
                             cfg.benchmark.observation_width)

    @staticmethod
    @contextmanager
    def open(cfg, env_cfg):
        def make():
            return Scene24Env(os.environ["SCENE24_RESOURCES"], env_cfg.observation_width,
                              env_cfg.observation_height, env_cfg.episode_length,
                              lambda token: episode_seed_pair(cfg, token))

        env = gym.vector.SyncVectorEnv([make], autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
        try:
            yield {0: env}
        finally:
            env.close()
