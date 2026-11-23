"""Permit the CUDA bicubic backward kernel while keeping every other Q operation strict."""
from __future__ import annotations

from contextlib import contextmanager
import logging
import os
import re
import warnings

import torch
from torch.utils._python_dispatch import TorchDispatchMode


_WARNING = re.compile(r"^upsample_bicubic2d_backward_out_cuda does not have a deterministic implementation(?:[,.]|$)")
_BACKWARD = torch.ops.aten.upsample_bicubic2d_backward.default


class _BicubicDispatch(TorchDispatchMode):
    def __init__(self, tally):
        super().__init__()
        self.tally = tally

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is _BACKWARD and args[0].device.type == "cuda":
            self.tally.calls += 1
            # In particular, Flash Attention must still see strict mode in its backward.
            torch.use_deterministic_algorithms(True, warn_only=True)
            try:
                return func(*args, **(kwargs or {}))
            finally:
                torch.use_deterministic_algorithms(True, warn_only=False)
        return func(*args, **(kwargs or {}))


class BackwardWarnings:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.count = 0
        self.calls = 0

    @contextmanager
    def during_update(self):
        if not self.enabled or not torch.are_deterministic_algorithms_enabled():
            yield
            return

        previous_mode = torch.is_deterministic_algorithms_warn_only_enabled()
        previous_repeat = torch.is_warn_always_enabled()
        with warnings.catch_warnings():
            original = warnings.showwarning

            def record(message, category, filename, lineno, file=None, line=None):
                if _WARNING.match(str(message)):
                    # Autograd may deliver the CUDA warning when backward returns to Python.
                    if self.count >= self.calls:
                        raise RuntimeError("bicubic warning without a matching CUDA backward call")
                    self.count += 1
                else:
                    original(message, category, filename, lineno, file, line)

            warnings.filterwarnings("error", message=r"(?s).*determin", category=Warning)
            warnings.filterwarnings("always", message=_WARNING.pattern, category=UserWarning)
            warnings.showwarning = record
            try:
                torch.set_warn_always(True)
                if previous_mode:
                    torch.use_deterministic_algorithms(True, warn_only=False)
                with _BicubicDispatch(self):
                    yield
            finally:
                torch.use_deterministic_algorithms(True, warn_only=previous_mode)
                torch.set_warn_always(previous_repeat)


@contextmanager
def count_backward_warnings(enabled: bool):
    tally = BackwardWarnings(enabled)
    try:
        yield tally
    finally:
        if enabled:
            logging.getLogger(__name__).warning(
                "rank=%s upsample_bicubic2d_backward_out_cuda calls=%d warnings=%d",
                os.environ.get("RANK", "0"), tally.calls, tally.count,
            )
