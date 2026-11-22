"""Convert manifest trajectories to a local LeRobot v3 dataset, optionally including failures."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


# The released dataset fixes the serialized feature order. Upstream aggregate_stats
# iterates a set, so otherwise the same numbers produce different JSON bytes per process.
STATS_COLUMN_ORDER = (
    "index", "task_index", "observation.images.wrist_image", "episode_index",
    "observation.state", "next.done", "timestamp", "frame_index", "action",
    "observation.images.image", "next.reward",
)


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify(path, expected):
    actual = digest(path)
    if actual != expected:
        raise ValueError(f"{path.name} has SHA256 {actual}, expected {expected}")


def features(height, width):
    image = {"dtype": "image", "shape": (height, width, 3), "names": ["height", "width", "channels"]}
    return {
        "observation.images.image": image.copy(), "observation.images.wrist_image": image.copy(),
        "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
        "action": {"dtype": "float32", "shape": (7,), "names": None},
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
    }


def frame(image, wrist, state, action, done, task):
    return {"observation.images.image": np.ascontiguousarray(image),
            "observation.images.wrist_image": np.ascontiguousarray(wrist),
            "observation.state": np.asarray(state, dtype=np.float32),
            "action": np.asarray(action, dtype=np.float32),
            "next.done": np.array([done], dtype=bool),
            "next.reward": np.array([float(done)], dtype=np.float32), "task": task}


def failed_rollout(row, run):
    # Historical demonstration manifests also carry derived training labels. The
    # platform's recorded outcome is the field shared with deployment rollouts.
    outcomes = [value for value in (row.get("platform_success"), run.get("success"))
                if isinstance(value, bool)]
    if len(set(outcomes)) > 1:
        raise ValueError(f"inconsistent recorded outcome in {row['run_id']}")
    return bool(outcomes) and not outcomes[0]


def convert(manifest_path, root, repo_id, *, include_failures=False):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import write_json

    manifest = json.loads(manifest_path.read_text())
    package = manifest.get("training_package")
    if package is not None:
        verify(Path(package["local"]) / "manifest.json", package["manifest_file_sha256"])
    records = {row["run_id"]: row for row in manifest["members"] + manifest.get("takeover_ancestors", [])}
    run_dirs = {rid: (Path(package["local"]) / "runs" / rid if package is not None else Path(row["platform_dir"]))
                for rid, row in records.items()}
    for rid, row in records.items():
        episode = run_dirs[rid] / "episodes" / "episode_000"
        verify(episode / "trajectory.npz", row["trajectory_sha256"])
        verify(episode / "trajectory_observations.npz", row["observations_sha256"])

    if root.exists():
        raise FileExistsError(f"choose a fresh output directory: {root}")
    trajectories = []
    for row in manifest["members"]:
        rid = row["run_id"]
        run = json.loads((run_dirs[rid] / "run.json").read_text())
        failed = failed_rollout(row, run)
        if failed and not include_failures:
            raise ValueError(f"unsuccessful rollout {rid}; enable --include-failures to convert it")
        if failed and run.get("error"):
            raise ValueError(f"rollout {rid} records a runtime error: {run['error']}")
        trajectories.append((row, run, failed))

    dataset = LeRobotDataset.create(repo_id, fps=20, root=root, features=features(256, 256), use_videos=False)
    provenance = []
    for row, run, failed in trajectories:
        rid = row["run_id"]
        limit = None
        if failed:
            lengths = (run.get("action_count"), row.get("recorded_actions"))
            limit = min([950, *(int(n) for n in lengths if n is not None)])
        ancestry = set()
        ancestor = rid
        while ancestor:
            if ancestor in ancestry:
                raise ValueError(f"cycle in ancestry for {rid}")
            ancestry.add(ancestor)
            ancestor = records[ancestor].get("parent_run_id")
        intervals = []
        index = 0
        done = False
        for seg in run["trajectory_segments"]:
            origin, start, stop = seg["origin_run_id"], seg["start"], seg["end"]
            if start != index or stop <= start or origin not in ancestry:
                raise ValueError(f"invalid source interval for {rid}: {seg}")
            files = run_dirs[origin] / "episodes" / "episode_000"
            with np.load(files / "trajectory.npz", allow_pickle=False) as traj, np.load(
                files / "trajectory_observations.npz", allow_pickle=False
            ) as imgs:
                actions, terminal = traj["env_action"], traj["done"]
                state = np.concatenate((traj["eef_position"], traj["eef_axis_angle"], traj["gripper_qpos"]), axis=1)
                front, wrist = imgs["agentview_image"], imgs["wrist_image"]
                if limit is not None:
                    stop = min(stop, limit, len(actions))
                if stop <= start or stop > len(actions) or len(terminal) != len(actions) or not (
                    len(front) == len(wrist) == len(state) == len(actions) + 1
                ):
                    raise ValueError(f"frame/action boundary in {origin}")
                for at in range(start, stop):
                    done = not failed and bool(terminal[at])
                    dataset.add_frame(frame(front[at, ::-1], wrist[at, ::-1], state[at], actions[at],
                                            done, run["task"]))
                    index = at + 1
                    if done:
                        break
            intervals.append({"id": origin, "start": start, "end_exclusive": index})
            if done or (limit is not None and (index >= limit or index < seg["end"])):
                break
        if index == 0:
            raise ValueError(f"no recorded actions in {rid}")
        if not failed and not done:
            raise ValueError(f"no successful transition in {rid}")
        dataset.save_episode()
        terminal_source = "recorded_failure_without_terminal_reward" if failed else "first_raw_done"
        provenance.append({"run_id": rid, "frames": index, "intervals": intervals, "terminal": terminal_source})
    for row in manifest.get("official_demonstrations", []):
        base = Path(row["local_dir"])
        (source,) = (base.parents[1] / "source" / "data").glob(f"*/episode_{row['episode_index']:06d}.parquet")
        verify(source, row["source_sha256"])
        for name, key in (("numeric.npz", "numeric_sha256"), ("images.npz", "images_sha256"),
                          ("bc_chunks_h8.npz", "bc_chunks_h8_sha256")):
            verify(base / name, row[key])
        meta = json.loads((base / "metadata.json").read_text())
        with np.load(base / "numeric.npz", allow_pickle=False) as num, np.load(base / "images.npz", allow_pickle=False) as imgs:
            state, action = num["state"], num["actions"]
            front, wrist = imgs["image"], imgs["wrist_image"]
            length = row["recorded_action_count"]
            if not all(len(x) == length for x in (state, action, front, wrist)):
                raise ValueError(f"official frame/action count differs for {row['run_id']}")
            for i in range(length):
                dataset.add_frame(frame(front[i], wrist[i], state[i], action[i], i == length - 1, meta["prompt"]))
        dataset.save_episode()
        provenance.append({"run_id": row["run_id"], "frames": length, "source_sha256": row["source_sha256"],
                           "terminal": "last action of published successful demonstration"})
    dataset.finalize()
    stats_file = root / "meta" / "stats.json"
    stats = json.loads(stats_file.read_text())
    ordered = {key: stats[key] for key in STATS_COLUMN_ORDER if key in stats}
    ordered.update((key, stats[key]) for key in sorted(stats.keys() - ordered.keys()))
    write_json(ordered, stats_file)
    return {"manifest_sha256": digest(manifest_path), "episodes": provenance,
            "files": {p.relative_to(root).as_posix(): digest(p) for p in sorted(root.rglob("*")) if p.is_file()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("root", type=Path)
    parser.add_argument("provenance", type=Path)
    parser.add_argument("--repo-id", default="demos_bc_v1")
    parser.add_argument("--include-failures", action="store_true",
                        help="Include recorded failures up to 950 actions, with zero reward and done false throughout.")
    args = parser.parse_args()
    report = convert(args.manifest, args.root, args.repo_id, include_failures=args.include_failures)
    args.provenance.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
