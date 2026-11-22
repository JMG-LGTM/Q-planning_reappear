"""Exercise deployment failures through the converter and the native Parquet writer."""

import json

import pytest

np = pytest.importorskip("numpy")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("lerobot.datasets.lerobot_dataset")
from scene24_qplanning import convert_demos as converter


def record(tmp, length, recorded_end, success, terminal_at):
    source = tmp / "rollout"
    episode = source / "episodes" / "episode_000"
    episode.mkdir(parents=True)
    idx = np.arange(length + 1, dtype=np.float32)
    sim = episode / "trajectory.npz"
    terminal = np.zeros(length, dtype=bool)
    if terminal_at is not None:
        terminal[terminal_at] = True
    action = np.broadcast_to(idx[:-1, None], (length, 7)).copy()
    np.savez(sim, env_action=action, done=terminal, eef_position=np.tile(idx[:, None], (1, 3)),
             eef_axis_angle=np.zeros((length + 1, 3)), gripper_qpos=np.zeros((length + 1, 2)))
    cameras = episode / "trajectory_observations.npz"
    images = np.broadcast_to(idx.astype(np.uint8)[:, None, None, None], (length + 1, 12, 16, 3))
    np.savez_compressed(cameras, agentview_image=images, wrist_image=255-images)
    run = {"action_count": recorded_end, "success": success, "task": "pick", "trajectory_segments": [
        {"origin_run_id": "one", "start": 0, "end": length}]}
    (source / "run.json").write_text(json.dumps(run))
    manifest = tmp / "manifest.json"
    manifest.write_text(json.dumps({"members": [{"run_id": "one", "parent_run_id": None,
        "platform_dir": str(source), "platform_success": success, "label_success": False,
        "trajectory_sha256": converter.digest(sim), "observations_sha256": converter.digest(cameras)}]}))
    return manifest


def tiny_images(monkeypatch):
    original = converter.features
    monkeypatch.setattr(converter, "features", lambda _h, _w: original(12, 16))


def read_fields(root):
    names = ["next.reward", "next.done", "action"]
    files = sorted((root / "data").rglob("*.parquet"))
    assert len(files) == 1
    return pq.ParquetFile(files[0]).read(columns=names).to_pydict()


@pytest.mark.parametrize("length,recorded_end,expected", [(956, 956, 950), (17, 13, 13), (5, 5, 5)])
def test_recorded_failure_has_no_reward_or_done(tmp_path, monkeypatch, length, recorded_end, expected):
    tiny_images(monkeypatch)
    manifest = record(tmp_path, length, recorded_end, False, terminal_at=2)
    report = converter.convert(manifest, tmp_path / "out", "example", include_failures=True)
    data = read_fields(tmp_path / "out")
    assert len(data["action"]) == report["episodes"][0]["frames"] == expected
    assert not np.asarray(data["next.done"]).any()
    assert np.array_equal(np.asarray(data["next.reward"]).ravel(), np.zeros(expected))
    assert np.array_equal(np.asarray(data["action"])[:, 0], np.arange(expected))
    assert report["episodes"][0]["intervals"][-1]["end_exclusive"] == expected


def test_default_rejects_a_failed_episode(tmp_path, monkeypatch):
    tiny_images(monkeypatch)
    manifest = record(tmp_path, 4, 4, False, terminal_at=None)
    with pytest.raises(ValueError, match="include-failures"):
        converter.convert(manifest, tmp_path / "default", "example")
    assert not (tmp_path / "default").exists()


def test_success_has_the_same_first_terminal_in_both_modes(tmp_path, monkeypatch):
    tiny_images(monkeypatch)
    manifest = record(tmp_path, 7, 7, True, terminal_at=3)
    for mode in (False, True):
        target = tmp_path / str(mode)
        report = converter.convert(manifest, target, "example", include_failures=mode)
        assert report["episodes"][0]["frames"] == 4
        data = read_fields(target)
        assert np.asarray(data["next.done"]).ravel().tolist() == [False, False, False, True]
        assert np.asarray(data["next.reward"]).ravel().tolist() == [0.0, 0.0, 0.0, 1.0]
    assert read_fields(tmp_path / "False") == read_fields(tmp_path / "True")
    stats = (tmp_path / "False" / "meta" / "stats.json").read_bytes()
    assert list(json.loads(stats)) == list(converter.STATS_COLUMN_ORDER)
    assert stats == (tmp_path / "True" / "meta" / "stats.json").read_bytes()
