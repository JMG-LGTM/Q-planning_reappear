import json
import os
from types import SimpleNamespace
import warnings

import pytest
import torch

from qplanning.config.schema import RunConfig
from qplanning.grad_accumulation import offline_updates
from qplanning.utils import backward_warnings
from qplanning.utils.backward_warnings import count_backward_warnings


BICUBIC = "upsample_bicubic2d_backward_out_cuda does not have a deterministic implementation, but you set warn_only=True"


@pytest.fixture
def strict_mode():
    mode = torch.get_deterministic_debug_mode()
    previous_repeat = torch.is_warn_always_enabled()
    backend = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
               torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.set_deterministic_debug_mode(mode)
        torch.set_warn_always(previous_repeat)
        (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
         torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark) = backend
        if workspace is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace


def test_only_exact_bicubic_warning_is_counted_and_errors_restore_settings(strict_mode, monkeypatch):
    flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
             torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
             os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
    repeat = torch.is_warn_always_enabled()
    cuda_arg = SimpleNamespace(device=SimpleNamespace(type="cuda"))

    def allowed(arg):
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        warnings.warn(BICUBIC, UserWarning)

    monkeypatch.setattr(backward_warnings, "_BACKWARD", allowed)
    with warnings.catch_warnings(record=True) as others, count_backward_warnings(True) as tally:
        warnings.simplefilter("always")
        with pytest.raises(UserWarning, match="another_kernel"), tally.during_update():
            assert torch.are_deterministic_algorithms_enabled()
            dispatch = backward_warnings._BicubicDispatch(tally)
            for _ in range(2):
                assert not torch.is_deterministic_algorithms_warn_only_enabled()
                dispatch.__torch_dispatch__(allowed, (), (cuda_arg,))
            assert not torch.is_deterministic_algorithms_warn_only_enabled()
            warnings.warn("ordinary unrelated warning", UserWarning)
            warnings.warn("another_kernel is non-deterministic on this device", UserWarning)
        assert tally.count == tally.calls == 2
        assert [str(w.message) for w in others] == ["ordinary unrelated warning"]
        with pytest.raises(RuntimeError, match="without a matching"), tally.during_update():
            warnings.warn(BICUBIC, UserWarning)
        with pytest.raises(UserWarning, match="upsample_bicubic2d_backward_out_cuda_extra"), tally.during_update():
            warnings.warn(BICUBIC.replace("cuda ", "cuda_extra "), UserWarning)
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    assert torch.is_warn_always_enabled() == repeat
    assert flags == (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                     torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
                     os.environ.get("CUBLAS_WORKSPACE_CONFIG"))


def test_other_native_pytorch_nondeterministic_operation_fails_before_next_line(strict_mode):
    with count_backward_warnings(True) as tally, tally.during_update():
        with pytest.raises((RuntimeError, UserWarning), match="deterministic"):
            torch.zeros(3).put_(torch.tensor([0, 0]), torch.tensor([1., 2.]), accumulate=False)
        assert tally.count == 0


@pytest.mark.parametrize("enabled,strict", [(False, True), (True, False)])
def test_disabled_policy_or_inactive_determinism_keeps_pytorch_mode(enabled, strict, strict_mode):
    torch.use_deterministic_algorithms(strict)
    with count_backward_warnings(enabled) as tally, tally.during_update():
        assert torch.are_deterministic_algorithms_enabled() == strict
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert tally.count == 0


def test_offline_native_update_enters_permitted_scope_only_when_enabled(strict_mode):
    def update():
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        calls.append("strict")
    calls = []
    trainer = SimpleNamespace(update_policy=update)
    with offline_updates(trainer, 24, 1, False):
        assert trainer.update_policy is update
    with offline_updates(trainer, 24, 1, True) as tally:
        trainer.update_policy()
        trainer.update_policy()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
    assert trainer.update_policy is update and tally.count == 0 and calls == ["strict", "strict"]


def test_scene_finetune_uses_determinism_from_prepare_and_writes_count(tmp_path, monkeypatch, strict_mode):
    from qplanning import self_improve
    cfg = RunConfig()
    cfg.allow_nondeterministic_bicubic_backward = True
    cfg.eval.deterministic = True
    cfg.paths.output_dir = str(tmp_path)
    torch.use_deterministic_algorithms(False)
    monkeypatch.setattr(self_improve, "_q_for_finetune", lambda *args: object())

    def finetune(*args):
        assert torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled()
        return 0.1, 1

    monkeypatch.setattr(self_improve, "finetune", finetune)
    assert self_improve.run(cfg, stage="finetune") == 0
    assert json.loads((tmp_path / "loop_summary.jsonl").read_text())["bicubic_backward_warnings"] == 0
    assert torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled()


def test_offline_cli_switch_reaches_both_accumulation_layouts(tmp_path):
    from qplanning.train_q import build_argv
    from qplanning.config.schema import Paths
    paths = Paths()
    cfg = RunConfig("test", 0, paths)
    assert cfg.paths is paths
    cfg.paths.dataset_root = str(tmp_path)
    cfg.train_q.dataset.repo_ids = ["demos"]
    cfg.train_q.num_gpus = 1
    assert not cfg.allow_nondeterministic_bicubic_backward
    assert "lerobot.scripts.lerobot_train" in build_argv(cfg)
    cfg.allow_nondeterministic_bicubic_backward = True
    for n in [1, 4]:
        cfg.train_q.grad_accum_steps = n
        argv = build_argv(cfg)
        assert "qplanning.grad_accumulation" in argv and "--allow-nondeterministic-bicubic-backward" in argv
        assert "--cudnn_deterministic=false" not in argv
