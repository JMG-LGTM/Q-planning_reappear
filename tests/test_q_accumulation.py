"""Effective-batch Q updates on CPU, including the native trainer and refinement paths."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot.scripts.lerobot_train")
from accelerate import Accelerator
from lerobot.scripts import lerobot_train

from qplanning import self_improve
from qplanning.config.schema import RunConfig
from qplanning.grad_accumulation import full_batches, offline_accumulation
from qplanning.processor_qfunction import q_batch_to_transition, q_transition_to_batch
from qplanning.train_q import build_argv


class TinyQ(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(7, 9)
        self.head = torch.nn.Linear(9, 3)
        self.register_buffer("target", self.head.weight.detach().clone())
        self.config = SimpleNamespace(h=2, camera_keys=[], optimizer_lr=2e-3, optimizer_lr_backbone=5e-4)
        self.batch_sizes = []
        self.updates = 0

    def forward(self, batch):
        self.batch_sizes.append(len(batch["action"]))
        pred = self.head(self.encoder(batch["action"]).tanh())
        loss = torch.nn.functional.cross_entropy(pred, batch["label"])
        return loss, {"td_ce_loss": loss.item(), "q_pred_min": pred.min().item(), "q_pred_max": pred.max().item()}

    def get_optim_params(self):
        return [{"params": list(self.head.parameters())}, {"params": list(self.encoder.parameters())}]

    @torch.no_grad()
    def update(self):
        self.target.lerp_(self.head.weight, 0.05)
        self.updates += 1

    def save_pretrained(self, directory):
        pass


def make_trial():
    torch.manual_seed(1729)
    policy = TinyQ()
    batch = {"action": torch.randn(48, 7), "label": torch.randint(0, 3, (48,)), "task": ["pick"] * 48}
    return policy, batch


def capture_before_clipping(real_clip, snapshots):
    def clip(parameters, *args, **kwargs):
        parameters = list(parameters)
        snapshots.append([param.grad.detach().clone() for param in parameters])
        return real_clip(parameters, *args, **kwargs)
    return clip


def compare_trials(reference, candidate, exact=False):
    expected_loss, expected_grads, expected_weights = reference
    actual_loss, actual_grads, actual_weights = candidate
    assert actual_loss == pytest.approx(expected_loss, abs=0 if exact else 1e-6, rel=0)
    assert len(actual_grads) == len(expected_grads) == 1
    assert expected_weights.keys() == actual_weights.keys()
    pairs = list(zip(expected_grads[0], actual_grads[0], strict=True))
    pairs.extend((v, actual_weights[k]) for k, v in expected_weights.items())
    for expected, actual in pairs:
        if exact:
            assert expected.numpy().tobytes() == actual.numpy().tobytes()
        else:
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def offline_trial(steps, native=False):
    policy, batch = make_trial()
    # Follow the stored Q data through the native converters; LeRobot adds scalar
    # metadata here, even when the original DataLoader batch only contains tensors.
    prepared = q_transition_to_batch(q_batch_to_transition({**batch, "q_bucket_index": batch["label"]}))
    prepared["label"] = prepared["q_bucket_index"]
    batch = prepared
    accelerator = Accelerator(cpu=True)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=2e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
    cfg = RunConfig()
    cfg.paths.dataset_root = "."
    cfg.train_q.dataset.repo_ids = ["example"]
    cfg.train_q.num_gpus = 1
    cfg.train_q.batch_size = 48 // steps
    cfg.train_q.grad_accum_steps = steps
    argv = build_argv(cfg)
    assert "--batch_size=48" in argv
    assert argv[2] == ("lerobot.scripts.lerobot_train" if steps == 1 else "qplanning.grad_accumulation")
    if steps > 1:
        assert argv[3:5] == [str(48 // steps), str(steps)]
    scope = nullcontext() if native or argv[2] == "lerobot.scripts.lerobot_train" else offline_accumulation(
        lerobot_train, 48 // steps, steps
    )
    snapshots = []
    with scope, patch.object(accelerator, "clip_grad_norm_", capture_before_clipping(accelerator.clip_grad_norm_, snapshots)):
        tracker, metrics = lerobot_train.update_policy(SimpleNamespace(), policy, batch, optimizer, 0.12,
                                                       accelerator, lr_scheduler=scheduler)
    assert policy.batch_sizes == [48 // steps] * steps
    assert policy.updates == scheduler.last_epoch == 1
    assert all(s["step"].item() == 1 for s in optimizer.state.values())
    assert tracker.loss == pytest.approx(metrics["td_ce_loss"])
    return tracker.loss, snapshots, policy.state_dict()


@pytest.mark.parametrize("steps", [1, 2, 4])
def test_offline_effective_update(steps):
    compare_trials(offline_trial(1, native=True), offline_trial(steps), exact=steps == 1)


def refinement_trial(module, steps, tmp_path):
    policy, batch = make_trial()
    cfg = RunConfig()
    cfg.self_improve.batch_size = 48 // steps
    cfg.self_improve.grad_accum_steps = steps
    cfg.self_improve.finetune_steps = 1
    cfg.self_improve.finetune_lr = 2e-3
    cfg.self_improve.grad_clip_norm = 0.12

    class Online(list):
        _sub_datasets = []

    def loader(offline, online, *, steps, batch_size, online_fraction):
        assert steps == 1 and batch_size == 48
        return [batch]

    snapshots = []
    real_clip = torch.nn.utils.clip_grad_norm_
    with (patch.object(module, "validate_shards", return_value=([tmp_path / "iter_000" / "online_episodes"], [])),
          patch.object(module, "OnlineQDataset", return_value=Online([0, 1])),
          patch("qplanning.data.mixture.load_demonstrations", return_value=[0, 1]),
          patch("qplanning.data.mixture.build_mixture_loader", loader),
          patch("qplanning.processor_qfunction.make_q_train_preprocessor", return_value=lambda x: x),
          patch.object(torch.nn.utils, "clip_grad_norm_", capture_before_clipping(real_clip, snapshots))):
        loss, global_step = module.finetune(cfg, policy, tmp_path, tmp_path / "ckpt", str(tmp_path), global_step=7)
    assert global_step == 8 and policy.updates == 1
    assert policy.batch_sizes == [48 // steps] * steps
    return loss, snapshots, policy.state_dict()


@pytest.mark.parametrize("steps", [2, 4])
def test_refinement_effective_update(steps, tmp_path):
    compare_trials(refinement_trial(self_improve, 1, tmp_path), refinement_trial(self_improve, steps, tmp_path))


def test_full_batches_preserve_epoch_tail_and_order():
    batches = [{"action": torch.tensor(values), "task": [str(i) for i in values]}
               for values in ([0, 1, 2, 3], [4], [5, 6, 7, 8], [9, 10, 11])]
    result = list(full_batches(iter(batches), 4))
    assert [r["action"].tolist() for r in result] == [list(range(i, i + 4)) for i in (0, 4, 8)]
    assert [v for r in result for v in r["task"]] == list(map(str, range(12)))
