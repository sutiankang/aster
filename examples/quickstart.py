"""Train, merge, and verify a tiny native LoRA model without network access."""

import json

import torch

from aster.methods import CrossEntropyObjective, inject_lora, merge_lora
from aster.models import LlamaConfig, build_model
from aster.training import Trainer


def run():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    model = build_model(
        LlamaConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
    )
    model = inject_lora(model, targets=["lm_head"], rank=4, alpha=8.0)
    frozen = {
        name: value.detach().clone()
        for name, value in model.named_parameters()
        if not value.requires_grad
    }
    trainer = Trainer(model, CrossEntropyObjective(label_smoothing=0.05), lr=0.01)
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]])
    batch = {"input_ids": tokens, "labels": tokens.clone()}
    losses = [trainer.step([batch]).loss for _ in range(8)]
    for name, value in model.named_parameters():
        if name in frozen:
            torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)
    if not torch.count_nonzero(model.lm_head.b):
        raise RuntimeError("The adapter did not update")
    model.eval()
    merged = merge_lora(model)
    with torch.no_grad():
        actual = model(tokens).logits
        expected = merged(tokens).logits
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    return {
        "device": "cpu",
        "updates": len(losses),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "first_training_loss": losses[0],
        "last_training_loss": losses[-1],
        "base_unchanged": True,
        "merge_max_absolute_error": float((actual - expected).abs().max()),
        "scope": "synthetic workflow, not pretrained quality",
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, allow_nan=False))
