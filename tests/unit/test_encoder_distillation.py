import copy
import math

import pytest
import torch
import torch.nn.functional as F

from aster.models import BertConfig, build_model
from aster.methods.encoder_distillation import (
    relation_kl,
    TinyBERTStudent,
    TinyBERTObjective,
    EncoderDistillationMethod,
)
from aster.training import Trainer


def bert(width=12, heads=3, layers=1):
    return build_model(
        BertConfig(
            vocab_size=24,
            hidden_size=width,
            intermediate_size=width * 2,
            num_hidden_layers=layers,
            num_attention_heads=heads,
            max_position_embeddings=16,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
    )


def batch():
    return {
        "input_ids": torch.tensor([[1, 3, 7, 0], [1, 6, 8, 2]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]),
    }


def test_minilm_relation_probability_and_gradient_different_hidden_widths():
    torch.manual_seed(6)
    torch.set_num_threads(1)
    student = torch.randn(2, 4, 12, requires_grad=True)
    teacher = torch.randn(2, 4, 18, requires_grad=True)
    valid = batch()["attention_mask"].bool()
    term = relation_kl(student, student, teacher, teacher, valid, heads=3, name="qq")
    oracle = student.new_zeros(())
    count = 0

    for row in range(2):
        s = student[row, valid[row]].reshape(-1, 3, 4)
        t = teacher[row, valid[row]].detach().reshape(-1, 3, 6)
        for head in range(3):
            log_s = (s[:, head] @ s[:, head].T / math.sqrt(4)).log_softmax(-1)
            log_t = (t[:, head] @ t[:, head].T / math.sqrt(6)).log_softmax(-1)
            oracle += (log_t.exp() * (log_t - log_s)).sum()
            count += len(s)
    torch.testing.assert_close(term.mean, oracle / count)
    actual_grad = torch.autograd.grad(term.mean, student, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(oracle / count, student)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert teacher.grad is None and actual_grad[0, -1].eq(0).all()


def test_tinybert_raw_attention_and_projected_hidden_match_official_equations():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    student = TinyBERTStudent(bert(12, 3, 1), 18)
    teacher = bert(18, 3, 2)
    objective = TinyBERTObjective(teacher, attention_pairs=((0, 1),), hidden_pairs=((0, 0), (1, 2)))
    actual = objective(student, batch())
    left = student.student(**batch(), output_hidden_states=True)
    with torch.no_grad():
        right = teacher(**batch(), output_hidden_states=True)

    def score(model, layer, hidden):
        attention = model.bert.encoder.layer[layer].attention.self
        q = attention.query(hidden).reshape(2, 4, 3, -1).transpose(1, 2)
        k = attention.key(hidden).reshape(2, 4, 3, -1).transpose(1, 2)
        result = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
        result = result.masked_fill(~batch()["attention_mask"][:, None, None, :].bool(), -10000.0)
        return torch.where(result <= -100.0, torch.zeros_like(result), result)

    oracle = F.mse_loss(
        score(student.student, 0, left.hidden_states[0]), score(teacher, 1, right.hidden_states[1])
    )
    oracle += F.mse_loss(student.fit_dense(left.hidden_states[0]), right.hidden_states[0])
    oracle += F.mse_loss(student.fit_dense(left.hidden_states[1]), right.hidden_states[2])
    total = sum(term.mean for term in actual.terms)
    torch.testing.assert_close(total, oracle)
    total.backward()
    assert student.fit_dense.weight.grad is not None
    assert all(value.grad is None for value in teacher.parameters())


@pytest.mark.parametrize("kind", ["tinybert", "minilm"])
def test_encoder_kd_native_engine_zero3_export_and_resume(tmp_path, kind):
    torch.manual_seed(13)
    torch.set_num_threads(1)
    teacher = bert(18, 3, 2)
    student = bert(12, 3 if kind == "tinybert" else 2, 1)
    if kind == "tinybert":
        student = TinyBERTStudent(student, 18)
        settings = {"attention_pairs": ((0, 1),), "hidden_pairs": ((0, 0), (1, 2))}
    else:
        settings = {"student_layer": 0, "teacher_layer": 1, "version": 2, "relation_heads": 3}
    engine = Trainer(student, lr=0.002, zero_stage=3)
    method = EncoderDistillationMethod(
        engine, teacher, tokenizer_fingerprints=("same", "same"), kind=kind, **settings
    )
    assert method.update([batch()]).updated
    exported = method.export_student()
    assert exported.config.hidden_size == 12
    assert exported(**batch()).logits.shape == (2, 4, 24)
    engine.save_checkpoint(tmp_path / "checkpoint")
    method.update([batch()])
    expected = copy.deepcopy(method.export_student().state_dict())
    engine.load_checkpoint(tmp_path / "checkpoint", trusted=True)
    method.update([batch()])
    for key, value in method.export_student().state_dict().items():
        torch.testing.assert_close(expected[key], value, rtol=0, atol=0)
