"""Compare an actual two-rank Gloo Q update with a single full-batch update."""

from datetime import timedelta
import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot.scripts.lerobot_train")
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
from lerobot.scripts import lerobot_train
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook

from qplanning.grad_accumulation import offline_accumulation
from test_q_accumulation import TinyQ, capture_before_clipping


def max_difference(actual, expected):
    return max((a - b).abs().max().item() for a, b in zip(actual, expected, strict=True))


def compare_update(accelerator, micro, steps):
    import torch.distributed as dist

    rank = dist.get_rank()
    rows_per_rank = micro * steps
    count = 2 * rows_per_rank + 31
    torch.manual_seed(115)
    actions = torch.randn(count, 7)
    labels = torch.randint(0, 3, (count,))
    rows = [dict(action=a, label=l, row=i) for i, (a, l) in enumerate(zip(actions, labels, strict=True))]
    ref = TinyQ()
    initial_target = ref.target.clone()
    model = TinyQ()
    model.load_state_dict(ref.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    base_opt = torch.optim.AdamW(ref.parameters(), lr=2e-3)
    loader = torch.utils.data.DataLoader(rows, batch_size=rows_per_rank, shuffle=False)
    ddp, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    assert isinstance(ddp, torch.nn.parallel.DistributedDataParallel)
    loader.rng_types = None  # same epoch-boundary setting as the pinned trainer
    sync_flags = []
    comm_events = []

    def observe_forward(module, args):
        sync_flags.append(ddp.require_backward_grad_sync)
        assert len(args[0]["action"]) == micro

    def observe_reduce(group, bucket):
        comm_events.append(len(sync_flags))
        return allreduce_hook(group, bucket)

    model.register_forward_pre_hook(observe_forward)
    ddp.register_comm_hook(dist.group.WORLD, observe_reduce)
    captured = []
    base_grad = []
    with offline_accumulation(lerobot_train, micro, steps):
        iterator = lerobot_train.cycle(loader)
        first, batch = next(iterator), next(iterator)
        got = [len(first["action"]), len(batch["action"])]
        assert got == [rows_per_rank, rows_per_rank]
        expected = (list(range(2 * rows_per_rank, count)) + list(range(rows_per_rank - 31))
                    if rank == 0 else list(range(rows_per_rank, 2 * rows_per_rank)))
        assert batch["row"].tolist() == expected
        ids_by_rank = [torch.empty_like(batch["row"]) for _ in range(2)]
        dist.all_gather(ids_by_rank, batch["row"])
        ids = torch.cat(ids_by_rank)
        with patch.object(accelerator, "clip_grad_norm_",
                          capture_before_clipping(accelerator.clip_grad_norm_, captured)):
            lerobot_train.update_policy(SimpleNamespace(), ddp, batch, optimizer, 0.12, accelerator)

    with patch.object(accelerator, "clip_grad_norm_",
                      capture_before_clipping(accelerator.clip_grad_norm_, base_grad)):
        lerobot_train.update_policy(SimpleNamespace(), ref, {"action": actions[ids], "label": labels[ids]},
                                    base_opt, 0.12, accelerator)
    assert sync_flags == [False] * (steps - 1) + [True]
    assert comm_events and set(comm_events) == {steps}
    assert dist.get_backend() == "gloo" and model.updates == ref.updates == 1
    assert ref.batch_sizes == [2 * rows_per_rank]
    grad_error = max_difference(captured[0], base_grad[0])
    parameter_error = max_difference(model.parameters(), ref.parameters())
    ema_error = (model.target - ref.target).abs().max().item()
    reference_ema_change = (ref.target - initial_target).abs().max().item()
    assert reference_ema_change > 0 and max(grad_error, parameter_error, ema_error) < 1e-6
    for p, q in zip(model.parameters(), ref.parameters(), strict=True):
        torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-6)
    return dict(rank=rank, microbatch=micro, accumulation=steps, effective_batch=len(ids),
                local_rows_per_update=got, forward_sync_flags=sync_flags, all_reduce_microbatches=comm_events,
                max_gradient_difference=grad_error, max_parameter_difference=parameter_error,
                max_ema_difference=ema_error, reference_ema_change=reference_ema_change)


def distributed_worker(rank, port, output):
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE="2", RANK=str(rank),
                      LOCAL_WORLD_SIZE="2", LOCAL_RANK=str(rank), ACCELERATE_USE_CPU="true")
    torch.set_num_threads(1)
    accelerator = Accelerator(cpu=True, dataloader_config=DataLoaderConfiguration(even_batches=False),
                              kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True),
                                              InitProcessGroupKwargs(backend="gloo", timeout=timedelta(minutes=2))])
    try:
        assert accelerator.num_processes == 2 and accelerator.process_index == rank
        result = [compare_update(accelerator, micro, steps) for micro, steps in [(24, 4), (48, 2)]]
        Path(output, f"rank_{rank}.json").write_text(json.dumps(result))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_two_rank_update_matches_one_full_batch(tmp_path, record_testsuite_property):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    torch.multiprocessing.spawn(distributed_worker, args=(port, str(tmp_path)), nprocs=2, join=True)
    result = [r for rank in range(2) for r in json.loads((tmp_path / f"rank_{rank}.json").read_text())]
    assert len(result) == 4 and all(r["effective_batch"] == 192 for r in result)
    record_testsuite_property("two_rank_q_equivalence", json.dumps(result))


def test_scene24_launches_two_ranks_with_an_effective_batch_of_192(tmp_path):
    from qplanning.config.loader import load
    from qplanning.train_q import build_argv

    config = Path(__file__).resolve().parents[1] / "adapters" / "configs" / "scene24_l1.yaml"
    cfg = load(config)
    cfg.paths.dataset_root = str(tmp_path)
    train = cfg.train_q
    assert train.num_gpus == 2
    assert (train.batch_size, train.grad_accum_steps) in {(24, 4), (48, 2)}
    argv = build_argv(cfg)
    assert "--num_processes=2" in argv and "--batch_size=96" in argv
    assert "--mixed_precision=bf16" in argv
