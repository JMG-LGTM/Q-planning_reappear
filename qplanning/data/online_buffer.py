"""The growing replay buffer of deployment rollouts.

Rollouts are written as LeRobot datasets, one directory per iteration (and per shard), and
read back lazily. The buffer is cumulative: iteration N trains on everything collected up to
N, not just the newest batch, which is what lets the Q-function keep the earlier failure
modes in view instead of forgetting them.

Both successful and failed episodes are kept. That asymmetry is the point of the method -- a
behaviour-cloning policy could only learn from the successes, while a value function learns
from a failure too, because a failed episode is simply one whose return is zero.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import logging
import random
import re
from tempfile import TemporaryDirectory
from collections.abc import Iterable, Sequence
from pathlib import Path

from lerobot.policies.q_function.online_dataset import (
    OnlineQDataset,
    save_episodes_lerobot,
)

log = logging.getLogger("qplanning.data")

__all__ = ["OnlineQDataset", "save_episodes_lerobot", "save_iteration", "validate_shards",
           "iteration_dirs", "write_manifest"]

#: Leaf directory name every collection job writes into. ``OnlineQDataset`` globs for it, so
#: a shard that names its directory anything else is silently invisible to training.
EPISODES_DIRNAME = "online_episodes"


def iteration_dir(output_dir: Path, iteration: int, shard: int | None = None) -> Path:
    """Where one collection job writes, following the layout the buffer globs for."""
    name = f"iter_{iteration:03d}" if shard is None else f"iter_{iteration:03d}_shard{shard:02d}"
    return Path(output_dir) / name


def iteration_dirs(output_dir: Path) -> list[Path]:
    """Every episode directory currently in the buffer, oldest first."""
    return sorted(Path(output_dir).glob(f"iter_*/{EPISODES_DIRNAME}"))


def write_manifest(episodes_dir: Path, **fields: object) -> Path:
    """Record how a shard was produced, next to the data itself.

    Without this, a buffer is a pile of frames with no record of which policy or which
    Q-function generated them, which makes a surprising result impossible to attribute
    after the fact.
    """
    path = Path(episodes_dir).parent / "manifest.json"
    path.write_text(json.dumps(fields, indent=2, default=str))
    return path


def save_iteration(
    episode_dicts: Sequence[dict],
    episodes_dir: Path,
    *,
    camera_keys: Sequence[str],
    image_shape: tuple[int, int, int] | None,
    action_dim: int,
    manifest: dict | None = None,
) -> Path:
    """Write one job's episodes into the buffer."""
    episodes_dir = Path(episodes_dir)
    kwargs: dict = {"action_dim": action_dim, "camera_keys": tuple(camera_keys)}
    if image_shape is not None:
        kwargs["image_shape"] = image_shape
    save_episodes_lerobot(list(episode_dicts), episodes_dir, **kwargs)
    if manifest is not None:
        write_manifest(episodes_dir, **manifest)
    return episodes_dir


def collection_name(episodes_dir: Path) -> str:
    """The logical collection job owning a legacy dataset or a numbered part."""
    return re.sub(r"_part[0-9]+$", "", episodes_dir.parent.name)


def incomplete_parts(dirs: Sequence[Path]) -> list[Path]:
    """A part is trainable only once the job has published its complete ordered index."""
    groups: dict[Path, set[str]] = {}
    for path in dirs:
        job = path.parent.parent / collection_name(path)
        if job != path.parent:
            groups.setdefault(job, set()).add(path.parent.name)
    bad = []
    for job, names in groups.items():
        try:
            expected = set(json.loads((job / "collection.json").read_text())["parts"])
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            expected = set()
        if expected != names or (job / EPISODES_DIRNAME).exists():
            bad.extend(job.parent / name / EPISODES_DIRNAME for name in sorted(names))
    return bad


class EpisodeParts:
    """Flush complete LeRobot datasets, retaining only the current group of episodes."""

    def __init__(self, job: Path, cfg, iteration: int, image_shape):
        self.job = job
        self.limit = cfg.self_improve.flush_every
        self.cameras = tuple(cfg.qfunction.camera_keys)
        self.shape = image_shape
        self.pending = []
        self.seeds = []
        self.parts = []
        self.identity = {
            "iteration": iteration, "shard": cfg.eval.shard, "q_checkpoint": str(cfg.qfunction.checkpoint),
            "planner": cfg.planner.type, "collection": job.name,
            "config_sha256": hashlib.sha256(json.dumps(asdict(cfg), sort_keys=True, default=str).encode()).hexdigest(),
        }
        if (job / EPISODES_DIRNAME).exists():
            raise FileExistsError(f"an unpartitioned collection already exists in {job}")
        job.mkdir(parents=True, exist_ok=True)

    def append(self, episode, seed):
        self.pending.append(episode)
        self.seeds.append(seed)
        if len(self.pending) == self.limit:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        import numpy as np
        import torch

        name = f"{self.job.name}_part{len(self.parts):06d}"
        target = self.job.parent / name
        digest = hashlib.sha256()
        for episode in self.pending:
            for key, value in sorted(episode.items()):
                digest.update(key.encode() + b"\0")
                if isinstance(value, torch.Tensor):
                    array = value.detach().cpu().contiguous().numpy()
                    digest.update(str((array.shape, array.dtype.str)).encode() + b"\0")
                    digest.update(memoryview(array).cast("B"))
                else:
                    digest.update(json.dumps(value).encode() + b"\0")
        manifest = dict(self.identity, part=len(self.parts), n_saved=len(self.pending),
                        seeds=list(self.seeds), content_sha256=digest.hexdigest())
        if not target.exists():
            # The CPU-only dataset writer must not change the rollout's random streams.
            py_rng, np_rng = random.getstate(), np.random.get_state()
            with TemporaryDirectory(prefix=".collect-", dir=self.job.parent) as tmp:
                part = Path(tmp) / name
                try:
                    with torch.random.fork_rng(devices=[]):
                        save_iteration(self.pending, part / EPISODES_DIRNAME, camera_keys=self.cameras,
                                       image_shape=self.shape, action_dim=int(self.pending[0]["action"].shape[1]),
                                       manifest=manifest)
                finally:
                    random.setstate(py_rng)
                    np.random.set_state(np_rng)
                try:
                    part.rename(target)
                except OSError:
                    if not target.is_dir():
                        raise
        # On retry, replay from the original seeds and reuse only byte-identical batches.
        if json.loads((target / "manifest.json").read_text()) != manifest:
            raise FileExistsError(f"existing collection part has different episodes or settings: {target}")
        if (problem := _shard_problem(target / EPISODES_DIRNAME)) is not None:
            raise ValueError(f"unreadable collection part {target}: {problem}")
        self.parts.append(name)
        self.pending.clear()
        self.seeds.clear()

    def finish(self, n_saved, n_heldout, failed):
        self.flush()
        actual = {p.parent.name for p in self.job.parent.glob(self.job.name + "_part*/" + EPISODES_DIRNAME)}
        if actual != set(self.parts):
            raise ValueError(f"existing parts do not match this collection: {self.job}")
        index = dict(self.identity, parts=self.parts, n_saved=n_saved, n_heldout=n_heldout, failed_tasks=failed)
        with TemporaryDirectory(prefix=".index-", dir=self.job) as tmp:
            path = Path(tmp) / "collection.json"
            path.write_text(json.dumps(index, indent=2))
            path.replace(self.job / "collection.json")


def validate_shards(output_dir: Path, iteration: int | None = None) -> tuple[list[Path], list[Path]]:
    """Separate readable episode directories from damaged ones.

    A collection job killed mid-write leaves a truncated parquet file behind. Training then
    fails partway through an epoch with an error that points at the reader rather than at
    the shard, and the whole iteration is lost. Checking up front costs a directory listing
    and turns that into a named, skippable shard.

    Returns:
        ``(good, quarantined)`` directories.
    """
    candidates = iteration_dirs(output_dir)
    incomplete = incomplete_parts(candidates)
    if iteration is not None:
        prefix = f"iter_{iteration:03d}"
        candidates = [p for p in candidates if p.parent.name.startswith(prefix)]

    good: list[Path] = []
    quarantined: list[Path] = list(incomplete)
    for episodes_dir in candidates:
        if episodes_dir in incomplete:
            continue
        problem = _shard_problem(episodes_dir)
        if problem is None:
            good.append(episodes_dir)
        else:
            quarantined.append(episodes_dir)
            log.warning("quarantining %s: %s", episodes_dir.parent.name, problem)
    return good, quarantined


def _shard_problem(episodes_dir: Path) -> str | None:
    """Why this directory cannot be trained on, or None if it is fine."""
    info = episodes_dir / "meta" / "info.json"
    if not info.exists():
        return "no meta/info.json (the job probably died before finalising)"
    try:
        total = json.loads(info.read_text()).get("total_episodes", 0)
    except json.JSONDecodeError:
        return "meta/info.json is not valid JSON"
    if not total:
        return "contains no episodes"

    parquets = list((episodes_dir / "data").rglob("*.parquet"))
    if not parquets:
        return "no parquet files"
    try:
        import pyarrow.parquet as pq

        for path in parquets:
            pq.read_metadata(path)          # reads the footer; a torn file fails here
    except ImportError:
        return None                          # cannot check without pyarrow; assume fine
    except Exception as exc:  # noqa: BLE001
        return f"unreadable parquet ({type(exc).__name__})"
    return None
