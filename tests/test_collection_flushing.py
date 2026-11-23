"""Exercise the upstream rollout, LeRobot writer and OnlineQDataset together on CPU."""

import hashlib
import json
import random
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pytest.importorskip("lerobot.policies.q_function.online_dataset")
from qplanning import self_improve
from qplanning.config.schema import RunConfig
from qplanning.data import online_buffer as buffer


CAMERAS = ("observation.images.main", "observation.images.wrist")


def identity(value):
    return value


class SmallEnv:
    def __init__(self):
        self.envs = [SimpleNamespace(task_description="pick")]
        self.seeds = []

    def call(self, method):
        return [5 if method == "_max_episode_steps" else self.envs[0].task_description]

    def reset(self, seed):
        self.seed = int(seed[0])
        self.seeds.append(self.seed)
        self.rng = np.random.default_rng(self.seed)
        self.t = 0
        return self.observation(), {}

    def observation(self):
        return {"pixels": {c: self.rng.integers(0, 256, (1, 12, 16, 3), dtype=np.uint8) for c in ("main", "wrist")}}

    def step(self, action):
        self.t += 1
        done = self.t == 3 + self.seed % 3
        success = bool(done and self.seed % 2)
        return (self.observation(), np.array([float(success)]), np.array([done]), np.array([False]),
                {"final_info": {"is_success": np.array([success])}})


class SmallPolicy:
    bc = SimpleNamespace(policy_config=None, preprocessor=identity, postprocessor=identity)

    def reset(self):
        pass

    def select_action(self, observation):
        return torch.rand(1, 7) + random.random() + np.random.random() + observation[CAMERAS[0]].mean()


def content_digest(value):
    if isinstance(value, torch.Tensor):
        value = value.contiguous().numpy().tobytes()
    else:
        value = json.dumps(value).encode()
    return hashlib.sha256(value).hexdigest()


def collect_trial(root, flush_every, module=self_improve):
    random.seed(92)
    np.random.seed(92)
    torch.manual_seed(92)
    cfg = RunConfig()
    cfg.paths.output_dir = str(root)
    cfg.qfunction.camera_keys = list(CAMERAS)
    cfg.self_improve.flush_every = flush_every
    cfg.self_improve.heldout_episodes_per_task = 1
    cfg.self_improve.train_episodes_per_task = 4
    env = SmallEnv()
    receipts = []
    original_episode = module.run_episode

    def record(*args):
        ep = original_episode(*args)
        receipts.append({"seed": int(args[-1]), **{k: content_digest(v) for k, v in ep.items()}})
        return ep

    @contextmanager
    def open_env(*args):
        yield None, {0: env}

    with (patch("qplanning.eval.prepare"), patch("qplanning.eval.build_policy", return_value=SmallPolicy()),
          patch("qplanning.envs.build_env_config"), patch.object(module, "resolve_tasks", return_value=["pick"]),
          patch.object(module, "open_task_envs", open_env), patch.object(module, "task_label", return_value="pick"),
          patch("lerobot.envs.factory.make_env_pre_post_processors", return_value=(identity, identity)),
          patch.object(module, "run_episode", record)):
        assert module.run(cfg, stage="collect") == 0

    expected = [cfg.seed + module.TRAIN_SEED_STRIDE + j for j in range(4)]
    assert env.seeds == [cfg.self_improve.heldout_seed_base] + expected
    dirs = buffer.iteration_dirs(root)
    assert buffer.validate_shards(root, 0) == (dirs, [])
    if flush_every:
        seeds = [s for d in dirs for s in json.loads((d.parent / "manifest.json").read_text())["seeds"]]
        assert seeds == expected and len(dirs) == (len(expected) + flush_every - 1) // flush_every
    dataset = buffer.OnlineQDataset(root, h=2, camera_keys=CAMERAS)
    lengths = [int(ep["length"]) for ds in dataset._sub_datasets for ep in ds.meta.episodes]
    assert lengths == [3 + seed % 3 for seed in expected]
    assert len(dataset) == sum(lengths)
    frames = [{k: content_digest(v) for k, v in row.items()} for row in dataset]
    assert all(set((*CAMERAS, "action", "q_reward_chunk_first")) <= row.keys() for row in frames)
    rewards = [float(dataset[i]["q_reward_chunk_first"][0]) for i in range(len(dataset))]
    assert sum(rewards) == sum(seed % 2 for seed in expected)
    return receipts, lengths, frames


@pytest.mark.parametrize("flush_every", [2, 3])
def test_split_collection_keeps_every_episode_byte_exact(tmp_path, flush_every):
    reference = collect_trial(tmp_path / "single", 0)
    assert collect_trial(tmp_path / "split", flush_every) == reference


def test_replayed_parts_are_reused_and_incomplete_job_is_rejected(tmp_path):
    reference = collect_trial(tmp_path, 2)
    parts = buffer.iteration_dirs(tmp_path)
    files = [p for part in parts for p in part.rglob("*.parquet")]
    times = [p.stat().st_mtime_ns for p in files]
    assert collect_trial(tmp_path, 2) == reference
    assert [p.stat().st_mtime_ns for p in files] == times
    (tmp_path / "iter_000" / "collection.json").unlink()
    assert buffer.validate_shards(tmp_path, 0) == ([], parts)


@pytest.mark.parametrize("broken", ["unreadable", "missing_part", "all_parts_missing"])
def test_finetune_rejects_bad_shards_with_a_readable_job(tmp_path, broken):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from qplanning import self_improve
    from qplanning.config.schema import RunConfig

    healthy = tmp_path / "iter_001" / "online_episodes"
    (healthy / "meta").mkdir(parents=True)
    (healthy / "meta" / "info.json").write_text(json.dumps({"total_episodes": 1}))
    (healthy / "data").mkdir()
    pq.write_table(pa.table({"action": [0]}), healthy / "data" / "chunk.parquet")
    if broken == "unreadable":
        bad = tmp_path / "iter_000" / "online_episodes"
        (bad / "meta").mkdir(parents=True)
        (bad / "meta" / "info.json").write_text(json.dumps({"total_episodes": 1}))
        (bad / "data").mkdir()
        (bad / "data" / "chunk.parquet").write_bytes(b"truncated")
    else:
        job = tmp_path / "iter_000"
        job.mkdir()
        names = ["iter_000_part0000", "iter_000_part0001"]
        (job / "collection.json").write_text(json.dumps({"parts": names}))
        if broken == "missing_part":
            (tmp_path / names[0] / "online_episodes").mkdir(parents=True)
    # The current iteration is readable. The cumulative reader would also open the older job.
    cfg = RunConfig()
    cfg.self_improve.min_shards_to_finetune = 1
    with pytest.raises(RuntimeError, match="unreadable or incomplete.*iter_000"):
        self_improve.finetune(cfg, None, tmp_path, tmp_path / "checkpoint", "", iteration=1)
    assert not (tmp_path / "checkpoint").exists()


@pytest.mark.parametrize("damaged", ["frame_rows", "episode_rows", "part_manifest"])
def test_published_part_with_incomplete_contents_is_rejected(tmp_path, damaged):
    import pyarrow.parquet as pq

    collect_trial(tmp_path, flush_every=2)
    part, intact = buffer.iteration_dirs(tmp_path)
    if damaged == "part_manifest":
        (part.parent / "manifest.json").unlink()
    else:
        folder = part / ("data" if damaged == "frame_rows" else "meta/episodes")
        data_file = next(folder.rglob("*.parquet"))
        table = pq.read_table(data_file)
        # The file still has a valid footer, so opening each Parquet alone cannot detect this.
        pq.write_table(table.slice(0, table.num_rows - 1), data_file)
    assert buffer.validate_shards(tmp_path, 0) == ([intact], [part])
    with pytest.raises(RuntimeError, match=part.parent.name):
        self_improve.finetune(RunConfig(), None, tmp_path, tmp_path / "ckpt", source_checkpoint="unused", iteration=0)
