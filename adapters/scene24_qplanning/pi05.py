"""Frozen OpenPI sampler. The worker isolates OpenPI's pinned Transformers version."""
from __future__ import annotations

import atexit
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
from . import episode_policy_seed


def write_packet(stream, **arrays):
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    data = buf.getbuffer()
    stream.write(struct.pack(">Q", len(data)))
    stream.write(data)
    stream.flush()


def read_packet(stream):
    header = stream.read(8)
    if not header:
        return None
    size, = struct.unpack(">Q", header)
    payload = stream.read(size)
    if len(payload) != size:
        raise EOFError("OpenPI worker closed during a packet")
    with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
        return dict(arrays)


def stats_path(checkpoint):
    identity = json.loads((checkpoint / "pi05_identity.json").read_text())
    (relative,) = [name for name in identity["files"] if name.endswith("/norm_stats.json")]
    return checkpoint / relative


class OpenPIAdapter:
    """All action values returned to the planner are in OpenPI's quantile-normalized space."""

    @classmethod
    def build(cls, cfg, env_cfg=None, **kwargs):
        from qplanning.policies.base import BCBundle

        sampler = cls(cfg)
        return BCBundle(sampler, lambda obs: obs, sampler.unnormalize_actions, cfg,
                        int(cfg.extra.get("n_action_steps", sampler.chunk_size)))

    def __init__(self, cfg):
        import torch

        self.checkpoint = Path(cfg.checkpoint).resolve()
        metadata = json.loads((self.checkpoint / "config.json").read_text())
        self.chunk_size = metadata["action_horizon"]
        self.latent_dim = metadata["action_dim"]
        stats = json.loads(stats_path(self.checkpoint).read_text())["norm_stats"]["actions"]
        self.lo = torch.tensor(stats["q01"], dtype=torch.float32)
        self.hi = torch.tensor(stats["q99"], dtype=torch.float32)
        self.action_dim = len(stats["q01"])
        self.steps = cfg.baseline_denoise_steps
        self.python = os.environ.get("OPENPI_PYTHON", sys.executable)
        self.worker = None
        self.episode_generator = None
        self.to(cfg.device)
        atexit.register(self.close)

    def reset(self):
        self.episode_generator = None

    def parameters(self):
        return iter((self.lo, self.hi))

    def to(self, device):
        import torch

        device = torch.device(device)
        if self.worker is not None and device != self.device:
            self.close()
            self.reset()
        self.device = device
        self.lo, self.hi = self.lo.to(device), self.hi.to(device)
        return self

    def unnormalize_actions(self, actions):
        return (actions + 1.0) / 2.0 * (self.hi.to(actions) - self.lo.to(actions) + 1e-6) + self.lo.to(actions)

    def close(self):
        if self.worker is not None:
            self.worker.stdin.close()
            self.worker.wait()
            self.worker.stdout.close()
            self.worker = None

    def sample_chunks(self, batch, n_samples, *, denoise_steps=None, generator=None):
        import torch

        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        if self.worker is None:
            numerical_settings = {
                "deterministic": torch.are_deterministic_algorithms_enabled(),
                "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
            }
            self.worker = subprocess.Popen(
                [self.python, "-m", "scene24_qplanning.pi05_worker", str(self.checkpoint), str(self.device),
                 json.dumps(numerical_settings)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            )
        if generator is None and episode_policy_seed.get() is not None:
            if self.episode_generator is None:
                self.episode_generator = torch.Generator(device=self.device).manual_seed(episode_policy_seed.get())
            generator = self.episode_generator
        noise = torch.randn((n_samples, self.chunk_size, self.latent_dim), device=self.device,
                            dtype=torch.float32, generator=generator)
        def image(key):
            value = batch["observation.images." + key]
            if value.shape[0] != 1:
                raise ValueError("OpenPI expects one observation")
            value = value[0].permute(1, 2, 0)
            if value.is_floating_point():
                value = (value * 255.0).round().clamp(0, 255)
            return value.to(device="cpu", dtype=torch.uint8).numpy()

        task = batch["task"]
        write_packet(self.worker.stdin, image=image("image"), wrist=image("wrist_image"),
                     state=batch["observation.state"][0].detach().cpu().numpy(),
                     prompt=np.asarray(task if isinstance(task, str) else task[0]),
                     noise=noise.cpu().numpy(), steps=np.asarray(denoise_steps or self.steps))
        result = read_packet(self.worker.stdout)
        if result is None:
            raise RuntimeError(f"OpenPI worker exited with status {self.worker.wait()}")
        return torch.from_numpy(result["actions"]).to(self.device)
