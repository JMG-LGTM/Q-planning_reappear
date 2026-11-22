"""Use the checkpoint's own OpenPI model and transforms in its dedicated interpreter."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .pi05 import read_packet, stats_path, write_packet


def verify_checkpoint(root):
    identity = json.loads((root / "pi05_identity.json").read_text())
    path = root / "model.safetensors"
    expected = identity["files"][path.name]
    with path.open("rb") as handle:
        os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    for _ in range(2):
        reader = subprocess.Popen(["dd", "if=" + str(path), "iflag=direct", "bs=4M", "status=none"],
                                  stdout=subprocess.PIPE)
        with reader.stdout as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if reader.wait():
            raise OSError("O_DIRECT checkpoint read failed")
        if actual == expected:
            break
    else:
        raise ValueError(f"checkpoint O_DIRECT SHA256 {actual}, expected {expected}")
    stats = stats_path(root)
    if hashlib.sha256(stats.read_bytes()).hexdigest() != identity["files"][stats.relative_to(root).as_posix()]:
        raise ValueError("normalization statistics SHA256 mismatch")
    return identity


def sample(model, obs, noise, steps):
    import torch
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

    if noise.shape[0] == 1:
        return model.sample_actions(noise.device, obs, noise=noise, num_steps=steps)
    images, masks, tokens, token_masks, state = model._preprocess_observation(obs, train=False)
    emb, pad, attention = model.embed_prefix(images, masks, tokens, token_masks)
    mask4d = model._prepare_attention_masks_4d(make_att_2d_masks(pad, attention))
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
    _, cache = model.paligemma_with_expert.forward(attention_mask=mask4d,
            position_ids=torch.cumsum(pad, dim=1) - 1, past_key_values=None,
            inputs_embeds=[emb, None], use_cache=True)
    n = noise.shape[0]
    cache.batch_repeat_interleave(n)
    state, pad = state.expand(n, -1), pad.expand(n, -1)
    dt = torch.tensor(-1.0 / steps, device=noise.device, dtype=torch.float32)
    time = torch.tensor(1.0, device=noise.device, dtype=torch.float32)
    x = noise
    while time >= -dt / 2:
        velocity = model.denoise_step(state, pad, cache, x, time.expand(n))
        x = x + dt * velocity
        time = time + dt
    return x


def main(root, device, settings):
    import jax
    import numpy as np
    import torch
    import safetensors.torch
    from openpi import transforms
    from openpi.models import model as model_api, pi0_config, tokenizer
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from openpi.policies.libero_policy import LiberoInputs
    from openpi.shared import normalize

    torch.backends.cuda.matmul.allow_tf32 = settings["matmul_tf32"]
    torch.backends.cudnn.allow_tf32 = settings["cudnn_tf32"]
    torch.backends.cudnn.benchmark = settings["cudnn_benchmark"]
    torch.use_deterministic_algorithms(settings["deterministic"], warn_only=settings["warn_only"])
    identity = verify_checkpoint(root)
    architecture = json.loads((root / "config.json").read_text())
    cfg = pi0_config.Pi0Config(pi05=True, action_dim=architecture["action_dim"],
            action_horizon=architecture["action_horizon"], dtype=architecture["precision"],
            paligemma_variant=architecture["paligemma_variant"], action_expert_variant=architecture["action_expert_variant"],
            discrete_state_input=identity["discrete_state_input"], pytorch_compile_mode=None)
    model = PI0Pytorch(cfg)
    safetensors.torch.load_model(model, str(root / "model.safetensors"), strict=True)
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    model.requires_grad_(False).to(device).eval()
    stats = normalize.load(stats_path(root).parent)
    transform = transforms.compose([
        LiberoInputs(cfg.model_type), transforms.Normalize(stats, use_quantiles=True),
        transforms.ResizeImages(224, 224),
        transforms.TokenizePrompt(tokenizer.PaligemmaTokenizer(cfg.max_token_len), cfg.discrete_state_input),
        transforms.PadStatesAndActions(cfg.action_dim),
    ])
    with torch.inference_mode():
        while (req := read_packet(sys.stdin.buffer)) is not None:
            inp = transform({"observation/image": req["image"], "observation/wrist_image": req["wrist"],
                             "observation/state": req["state"], "prompt": str(req["prompt"].item())})
            inp = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(device)[None, ...], inp)
            obs = model_api.Observation.from_dict(inp)
            x = sample(model, obs, torch.from_numpy(req["noise"]).to(device), int(req["steps"]))
            write_packet(_output, actions=x[..., :len(stats["actions"].q01)].float().cpu().numpy())


if __name__ == "__main__":
    # Reserve stdout for packets, including when dependencies print during import.
    _output = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    main(Path(sys.argv[1]), sys.argv[2], json.loads(sys.argv[3]))
