"""Convert successful manifest trajectories into a lossless, local LeRobot v3 dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


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


def convert(manifest_path, root, repo_id):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    manifest = json.loads(manifest_path.read_text())
    package = Path(manifest["training_package"]["local"])
    verify(package / "manifest.json", manifest["training_package"]["manifest_file_sha256"])
    records = {row["run_id"]: row for row in manifest["members"] + manifest["takeover_ancestors"]}
    for rid, row in records.items():
        episode = package / "runs" / rid / "episodes" / "episode_000"
        verify(episode / "trajectory.npz", row["trajectory_sha256"])
        verify(episode / "trajectory_observations.npz", row["observations_sha256"])

    if root.exists():
        raise FileExistsError(f"choose a fresh output directory: {root}")
    dataset = LeRobotDataset.create(repo_id, fps=20, root=root, features=features(256, 256), use_videos=False)
    provenance = []
    for row in manifest["members"]:
        rid = row["run_id"]
        run = json.loads((package / "runs" / rid / "run.json").read_text())
        ancestry = set()
        ancestor = rid
        while ancestor:
            if ancestor in ancestry:
                raise ValueError(f"cycle in ancestry for {rid}")
            ancestry.add(ancestor)
            ancestor = records[ancestor]["parent_run_id"]
        intervals = []
        index = 0
        done = False
        for seg in run["trajectory_segments"]:
            origin, start, stop = seg["origin_run_id"], seg["start"], seg["end"]
            if start != index or stop <= start or origin not in ancestry:
                raise ValueError(f"invalid source interval for {rid}: {seg}")
            files = package / "runs" / origin / "episodes" / "episode_000"
            with np.load(files / "trajectory.npz", allow_pickle=False) as traj, np.load(
                files / "trajectory_observations.npz", allow_pickle=False
            ) as imgs:
                actions, terminal = traj["env_action"], traj["done"]
                state = np.concatenate((traj["eef_position"], traj["eef_axis_angle"], traj["gripper_qpos"]), axis=1)
                front, wrist = imgs["agentview_image"], imgs["wrist_image"]
                if stop > len(actions) or not (len(front) == len(wrist) == len(state) == len(actions) + 1):
                    raise ValueError(f"frame/action boundary in {origin}")
                for at in range(start, stop):
                    done = bool(terminal[at])
                    dataset.add_frame(frame(front[at, ::-1], wrist[at, ::-1], state[at], actions[at],
                                            done, run["task"]))
                    index = at + 1
                    if done:
                        break
            intervals.append({"id": origin, "start": start, "end_exclusive": index})
            if done:
                break
        if not done:
            raise ValueError(f"no successful transition in {rid}")
        dataset.save_episode()
        provenance.append({"run_id": rid, "frames": index, "intervals": intervals, "terminal": "first_raw_done"})
    for row in manifest["official_demonstrations"]:
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
    return {"manifest_sha256": digest(manifest_path), "episodes": provenance,
            "files": {p.relative_to(root).as_posix(): digest(p) for p in sorted(root.rglob("*")) if p.is_file()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("root", type=Path)
    parser.add_argument("provenance", type=Path)
    parser.add_argument("--repo-id", default="demos_bc_v1")
    args = parser.parse_args()
    report = convert(args.manifest, args.root, args.repo_id)
    args.provenance.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
