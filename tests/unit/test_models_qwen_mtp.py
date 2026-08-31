from dataclasses import replace
import pytest
import torch
from torch import nn
import torch.nn.functional as F
from aster.core import LossTerm, LossBundle
from aster.models import QwenMTPConfig, Qwen35TextConfig, build_model, load_model
from aster.training import Trainer


class MTPObjective(nn.Module):
    def __init__(self, depth=2, detach=False):
        super().__init__()
        self.depth, self.detach = depth, detach

    def config_dict(self):
        return {"depth": self.depth, "detach_base": self.detach}

    def forward(self, model, batch):
        ids = batch["input_ids"]
        result = model(ids, mtp_depth=self.depth, detach_mtp_base=self.detach)
        terms = []
        for offset, logits in zip(result.auxiliary["mtp_offsets"], result.auxiliary["mtp_logits"]):
            targets = ids[:, offset + 1 :]
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum"
            )
            terms.append(
                LossTerm(
                    loss, loss.new_tensor(targets.numel()), "token", f"mtp_{offset}", 1 / self.depth
                )
            )
        return LossBundle(tuple(terms))


def mean(bundle):
    return sum(term.mean * term.weight for term in bundle.terms)


@pytest.mark.parametrize("share", [True, False])
def test_models_qwen_mtp_joint_training_sharing_and_reload(tmp_path, share):
    torch.set_num_threads(1)
    torch.manual_seed(211)
    c = QwenMTPConfig(num_mtp_layers=2, share_embeddings=share)
    model = build_model(c)
    objective = MTPObjective()
    batch = {"input_ids": torch.tensor([[1, 3, 5, 7, 9, 11, 2], [1, 4, 6, 8, 10, 12, 2]])}
    assert (model.mtp.embed_tokens.weight is model.backbone.get_input_embeddings().weight) == share
    assert (model.mtp.lm_head.weight is model.backbone.lm_head.weight) == share
    initial = float(mean(objective(model, batch)).detach())
    trainer = Trainer(model, objective, lr=0.006)
    for _ in range(20):
        trainer.step([batch])
    assert float(mean(objective(model, batch)).detach()) < initial * 0.6
    model.zero_grad(set_to_none=True)
    mean(objective(model, batch)).backward()
    assert model.backbone.model.layers[0].linear_attn.in_proj_qkv.weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    mean(MTPObjective(detach=True)(model, batch)).backward()
    assert model.backbone.model.layers[0].linear_attn.in_proj_qkv.weight.grad is None

    model.zero_grad(set_to_none=True)
    model.backbone.requires_grad_(False)
    mean(objective(model, batch)).backward()
    assert model.mtp.fc.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.backbone.parameters())
    model.eval()
    expected = model(batch["input_ids"], mtp_depth=3)
    model.save_pretrained(tmp_path / "mtp")
    restored = load_model(tmp_path / "mtp").eval()
    actual = restored(batch["input_ids"], mtp_depth=3)
    assert (
        restored.mtp.embed_tokens.weight is restored.backbone.get_input_embeddings().weight
    ) == share
    for p, q in zip(actual.auxiliary["mtp_logits"], expected.auxiliary["mtp_logits"]):
        torch.testing.assert_close(p, q, atol=0, rtol=0)


def test_models_qwen_mtp_selected_layer_and_independent_draft_state():
    torch.set_num_threads(1)
    torch.manual_seed(212)
    model = build_model(QwenMTPConfig(num_mtp_layers=2)).eval()
    tokens, hidden = torch.randint(1, 30, (2, 7)), torch.randn(2, 7, 32)
    full = model.mtp(tokens, hidden_states=hidden, spec_step_idx=0)

    with torch.no_grad():
        model.mtp.layers[1].mlp.down_proj.weight.normal_()
    torch.testing.assert_close(
        full.logits, model.mtp(tokens, hidden_states=hidden, spec_step_idx=2).logits, atol=0, rtol=0
    )
    prefix = model.mtp(tokens[:, :4], hidden_states=hidden[:, :4], use_cache=True)
    result = model.mtp(
        tokens[:, 4:], hidden_states=hidden[:, 4:], state=prefix.state, use_cache=True
    )
    torch.testing.assert_close(result.logits, full.logits[:, 4:], atol=2e-6, rtol=2e-5)
    rollback = result.state.truncate(4)
    torch.testing.assert_close(
        model.mtp(tokens[:, 4:], hidden_states=hidden[:, 4:], state=rollback).logits,
        result.logits,
        atol=0,
        rtol=0,
    )
    with pytest.raises(ValueError, match="selected layer"):
        model.mtp(tokens[:, 4:], hidden_states=hidden[:, 4:], state=prefix.state, spec_step_idx=1)
    base_state = model.backbone(tokens[:, :4], use_cache=True).state
    with pytest.raises(ValueError, match="base/hybrid"):
        model.mtp(tokens[:, 4:], hidden_states=hidden[:, 4:], state=base_state)


def test_models_qwen_mtp_exact_stochastic_resume(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(213)
    c = QwenMTPConfig(
        text_config=replace(Qwen35TextConfig(), attention_dropout=0.2), num_mtp_layers=2
    )
    model = build_model(c)
    objective = MTPObjective()
    batch = {"input_ids": torch.randint(1, 30, (2, 7))}
    trainer = Trainer(model, objective, lr=0.001)
    trainer.step([batch])
    trainer.save_checkpoint(tmp_path / "resume.json")
    expected = trainer.step([batch])
    weights = {n: v.detach().clone() for n, v in model.state_dict().items()}
    other = build_model(c)
    resumed = Trainer(other, MTPObjective(), lr=0.001)
    resumed.load_checkpoint(tmp_path / "resume.json")
    actual = resumed.step([batch])
    assert expected.loss == actual.loss
    for name, value in other.state_dict().items():
        torch.testing.assert_close(value, weights[name], atol=0, rtol=0)


def test_models_delta_decay_explicit_leaf_and_public_checkpoint_mapping():
    model = build_model(QwenMTPConfig())
    module = model.backbone.model.layers[0].linear_attn
    assert not list(module.parameters(recurse=False))
    assert set(dict(module.decay_gate.named_parameters())) == {"A_log", "dt_bias"}
    weights = model.state_dict()
    prefix = "backbone.model.layers.0.linear_attn."
    assert prefix + "A_log" in weights and prefix + "decay_gate.A_log" not in weights
    restored = build_model(model.config)
    restored.load_state_dict(weights, strict=True)
    conflicting = dict(weights)
    conflicting[prefix + "decay_gate.A_log"] = weights[prefix + "A_log"].clone()
    with pytest.raises(ValueError, match="both internal and public"):
        restored.load_state_dict(conflicting, strict=True)
