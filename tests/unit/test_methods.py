import copy
import torch
import torch.nn.functional as F
from aster.models import LlamaConfig, build_model
from aster.methods import (
    CrossEntropyObjective,
    DistillationObjective,
    PreferenceObjective,
    distribution_divergence,
    inject_lora,
    merge_lora,
)
from aster.training import Trainer


def batch():
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4], [1, 3, 4, 0]]),
        "labels": torch.tensor([[-100, 2, 3, 4], [-100, 3, 4, -100]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
    }


def test_kl_orientation_temperature_and_gradient():
    s = torch.tensor([[0.2, -0.5, 0.8]], requires_grad=True)
    t = torch.tensor([[0.8, 0.4, -0.3]])
    for kind in ("forward_kl", "reverse_kl", "js", "mixed_kl"):
        result = distribution_divergence(s, t, kind=kind, temperature=2.0)
        assert result.item() >= -1e-6
        result.sum().backward(retain_graph=True)
    expected = F.kl_div(F.log_softmax(s / 2, -1), F.softmax(t / 2, -1), reduction="sum") * 4
    torch.testing.assert_close(distribution_divergence(s, t, temperature=2.0).sum(), expected)
    assert s.grad.isfinite().all()


def test_native_ce_kd_shared_engine():
    torch.set_num_threads(1)
    torch.manual_seed(3)
    model = build_model(LlamaConfig())
    ce = CrossEntropyObjective()
    term = ce(model, batch())
    assert term.denominator.item() == 5
    trainer = Trainer(model, ce, lr=0.002, max_grad_norm=None)
    initial = term.mean.item()
    for _ in range(12):
        assert trainer.step([batch()]).updated
    assert ce(model, batch()).mean.item() < initial
    teacher = copy.deepcopy(model)
    student = build_model(LlamaConfig())
    objective = DistillationObjective(
        teacher, kd_weight=0.7, feature_weight=0.1, layer_pairs=((1, 1),)
    )
    trainer = Trainer(student, objective)
    assert trainer.step([batch()]).updated
    assert all(p.grad is None for p in teacher.parameters())


def test_lora_injection_and_merge_preserve_function():
    torch.manual_seed(2)
    model = build_model(LlamaConfig()).eval()
    original = model(**{"input_ids": batch()["input_ids"]}).logits
    targets = [
        name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)
    ][:2]
    inject_lora(model, targets=targets)
    torch.testing.assert_close(original, model(input_ids=batch()["input_ids"]).logits)
    for name, p in model.named_parameters():
        if name.endswith(".b"):
            with torch.no_grad():
                p.normal_(std=0.02)
    adapted = model(input_ids=batch()["input_ids"]).logits
    merged = merge_lora(model)
    torch.testing.assert_close(
        adapted, merged(input_ids=batch()["input_ids"]).logits, atol=2e-6, rtol=2e-5
    )


def test_preference_dpo_reference_zero_margin():
    model = build_model(LlamaConfig())
    objective = PreferenceObjective(copy.deepcopy(model), beta=0.1)
    pairs = {"chosen": batch(), "rejected": batch()}
    torch.testing.assert_close(objective(model, pairs).mean, torch.tensor(2.0).log())


def test_t5_supervision_uses_decoder_not_encoder_mask():
    from aster.models import T5Config

    model = build_model(T5Config())
    sample = {
        "input_ids": torch.tensor([[1, 2, 3, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0]]),
        "decoder_input_ids": torch.tensor([[0, 5, 6]]),
        "decoder_attention_mask": torch.tensor([[1, 1, 1]]),
        "labels": torch.tensor([[5, 6, 1]]),
    }
    objective = CrossEntropyObjective(causal=False)
    assert objective(model, sample).denominator == 3
    assert Trainer(model, objective).step([sample]).updated
