"""Split one Q update across microbatches while retaining the trainer's effective-step clock."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from functools import partial
import sys
import time
from unittest.mock import patch


def _slice(batch, start, end):
    import torch

    def take(value):
        if isinstance(value, dict):
            return {key: take(v) for key, v in value.items()}
        if isinstance(value, (list, tuple)) or (isinstance(value, torch.Tensor) and value.ndim):
            return value[start:end]
        # The native preprocessor also supplies scalar done/reward defaults and an info dict.
        return value

    return take(batch)


def _join(pieces):
    import torch

    if len(pieces) == 1:
        return pieces[0]
    first = pieces[0]
    if isinstance(first, dict):
        return {key: _join([part[key] for part in pieces]) for key in first}
    if isinstance(first, torch.Tensor) and first.ndim:
        return torch.cat(pieces)
    if isinstance(first, (list, tuple)):
        return type(first)(item for part in pieces for item in part)
    return first


def full_batches(stream, size):
    """Carry a short final DataLoader batch across the epoch boundary without losing rows."""
    pieces = []
    buffered = 0
    for batch in stream:
        n = len(batch["action"])
        if buffered == 0 and n == size:
            yield batch
            continue
        start = 0
        while start < n:
            count = min(n - start, size - buffered)
            pieces.append(_slice(batch, start, start + count))
            start += count
            buffered += count
            if buffered == size:
                complete = _join(pieces)
                pieces = []
                buffered = 0
                yield complete


def _mean_metrics(rows):
    """Reconstruct scalar Q diagnostics, including extrema and per-bucket means."""
    merged = {}
    for key in set().union(*rows):
        present = [r for r in rows if key in r]
        if key.endswith("_min"):
            merged[key] = min(r[key] for r in present)
        elif key.endswith("_max"):
            merged[key] = max(r[key] for r in present)
        elif key.startswith("bucket_") and "/" in key and not key.endswith("/frac"):
            frac = key.split("/")[0] + "/frac"
            total = sum(r[frac] for r in present)
            merged[key] = sum(r[key] * r[frac] for r in present) / total
        else:
            merged[key] = sum(r[key] for r in present) / len(rows)
    return merged


def backward_microbatches(policy, batch, steps, *, backward, preprocess=None,
                          autocast=nullcontext, no_sync=None):
    """Backpropagate a mean Q loss. Leave clipping, optimizer and target updates to the caller."""
    size, remainder = divmod(len(batch["action"]), steps)
    if size == 0 or remainder:
        raise ValueError("the effective batch must split into equal non-empty microbatches")
    rows, losses = [], []
    for i in range(steps):
        micro = _slice(batch, i * size, (i + 1) * size)
        if preprocess is not None:
            micro = preprocess(micro)
        sync = no_sync(policy) if no_sync is not None and i < steps - 1 else nullcontext()
        with sync:
            with autocast():
                loss, metrics = policy.forward(micro)
            backward(loss / steps)
        losses.append(loss.detach().item())
        rows.append(metrics)
    return sum(losses) / steps, _mean_metrics(rows)


def _offline_update(train_metrics, policy, batch, optimizer, grad_clip_norm, accelerator,
                    lr_scheduler=None, lock=None, rabc_weights_provider=None, *, steps):
    import torch

    if rabc_weights_provider is not None:
        raise ValueError("Q training uses the Q-function's mean TD loss")
    started = time.perf_counter()
    policy.train()
    loss, metrics = backward_microbatches(policy, batch, steps, backward=accelerator.backward,
                                         autocast=accelerator.autocast, no_sync=accelerator.no_sync)
    if grad_clip_norm > 0:
        norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), float("inf"))
    with lock if lock is not None else nullcontext():
        optimizer.step()
    optimizer.zero_grad()
    if lr_scheduler is not None:
        lr_scheduler.step()
    accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()
    train_metrics.loss = loss
    train_metrics.grad_norm = norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - started
    return train_metrics, metrics


@contextmanager
def offline_accumulation(trainer, micro_size, steps):
    """Install only the two per-process callbacks needed by the pinned LeRobot entry point."""
    from accelerate.data_loader import DataLoaderDispatcher, DataLoaderShard

    native_cycle = trainer.cycle

    def cycle(loader):
        batches = native_cycle(loader)
        # The trainer prepares only its training loader with Accelerate; validation keeps
        # the ordinary finite PyTorch loader and its original batch semantics.
        if isinstance(loader, (DataLoaderShard, DataLoaderDispatcher)):
            return full_batches(batches, micro_size * steps)
        return batches

    with patch.object(trainer, "cycle", cycle), patch.object(trainer, "update_policy", partial(_offline_update, steps=steps)):
        yield


def main():
    from lerobot.scripts import lerobot_train

    micro_size, steps = map(int, sys.argv[1:3])
    del sys.argv[1:3]
    with offline_accumulation(lerobot_train, micro_size, steps):
        lerobot_train.main()


if __name__ == "__main__":
    main()
