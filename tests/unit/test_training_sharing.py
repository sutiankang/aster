from copy import deepcopy
import pytest
import torch
from torch import nn
from aster.core.contracts import LossTerm
from aster.training import Trainer


class TiedLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(7, 4)
        self.head = nn.Linear(4, 7, bias=False)
        self.head.weight = self.embedding.weight

    def forward(self, tokens):
        return self.head(self.embedding(tokens))


def objective(model, batch):
    tokens, targets = batch
    loss = nn.functional.cross_entropy(
        model(tokens).flatten(0, 1), targets.flatten(), reduction="sum"
    )
    return LossTerm(loss, torch.tensor(targets.numel()), "tokens")


def test_zero3_tied_embedding_head_shared_owner_polyak_and_resume(tmp_path):
    torch.manual_seed(917)
    model = TiedLanguageModel()
    reference = Trainer(deepcopy(model), objective, lr=0.03)
    trainer = Trainer(model, objective, lr=0.03, zero_stage=3)
    assert len(trainer.roles["model"].parameters) == 1
    assert trainer.model.embedding.shards[0] is trainer.model.head.shards[0]
    target = trainer.clone_target("model", "target", factory=TiedLanguageModel)
    old = target.embedding.weight.detach().clone()
    batch = (torch.tensor([[0, 1, 4], [6, 5, 2]]), torch.tensor([[1, 4, 3], [5, 2, 0]]))
    for _ in range(3):
        reference.step([batch])
        trainer.step([batch])
        torch.testing.assert_close(
            trainer.model(batch[0]), reference.model(batch[0]), rtol=1e-5, atol=2e-6
        )
    trainer.update_target("model", "target", 0.8)
    torch.testing.assert_close(
        target.embedding.weight, 0.8 * old + 0.2 * reference.model.embedding.weight
    )
    assert target.embedding.weight is target.head.weight
    path = trainer.save_checkpoint(tmp_path / "tied.json")
    trainer.step([batch])
    expected = trainer.model(batch[0]).detach().clone()
    trainer.load_checkpoint(path)
    trainer.step([batch])
    torch.testing.assert_close(trainer.model(batch[0]), expected, atol=0, rtol=0)


def test_meta_zero3_initializes_shards_without_materialized_model():
    calls = []

    def initialize(name, shape, dtype, offset, count, device):
        calls.append((name, shape, offset, count))
        return torch.arange(offset, offset + count, device=device, dtype=dtype) * 0.01

    model = nn.Sequential(nn.Linear(3, 5, device="meta"), nn.Tanh(), nn.Linear(5, 2, device="meta"))
    trainer = Trainer(model, zero_stage=3, sharded_initializer=initialize)
    assert len(calls) == 4 and all(
        not parameter.is_meta for parameter in trainer.model.parameters()
    )
    dense = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 2))
    with torch.no_grad():
        for parameter in dense.parameters():
            parameter.copy_(torch.arange(parameter.numel()).reshape(parameter.shape) * 0.01)
    x = torch.randn(4, 3)
    torch.testing.assert_close(trainer.model(x), dense(x))


def test_subtree_target_keeps_encoder_predictor_single_optimizer_and_prefix(tmp_path):
    class JEPA(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(3, 5), nn.Tanh())
            self.predictor = nn.Linear(5, 5)

        def forward(self, x):
            return self.predictor(self.encoder(x))

    torch.manual_seed(866)
    trainer = Trainer(JEPA(), zero_stage=3, lr=0.01)
    target = trainer.clone_target(
        "model",
        "target_encoder",
        source_path="encoder",
        factory=lambda: nn.Sequential(nn.Linear(3, 5), nn.Tanh()),
    )
    old = deepcopy(target.state_dict())

    def loss(model, x):
        error = (model(x) - target(x).detach()).square()
        return LossTerm(error.sum(), torch.tensor(error.numel()), "elements")

    trainer.phase("jepa", objective=loss, microbatches=[torch.ones(4, 3)])
    full = trainer.export_state_dict()
    trainer.update_target("model", "target_encoder", 0.9)
    for name, value in target.state_dict().items():
        torch.testing.assert_close(value, old[name] * 0.9 + full[f"encoder.{name}"] * 0.1)
    assert trainer.roles["model"].updates == 1 and not trainer.roles["target_encoder"].trainable
    path = trainer.save_checkpoint(tmp_path / "jepa.json")
    expected = deepcopy(target.state_dict())
    trainer.update_target("model", "target_encoder", 0.0)
    trainer.load_checkpoint(path)
    for name, value in expected.items():
        torch.testing.assert_close(value, target.state_dict()[name], rtol=0, atol=0)


def test_parameterlist_read_is_rejected_before_any_partial_conversion():
    class Bare(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 3)
            self.values = nn.ParameterList([nn.Parameter(torch.ones(3))])

        def forward(self, x):
            return self.linear(x) + self.values[0]

    model = Bare()
    original = deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="ParameterList"):
        Trainer(model, zero_stage=3)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)


class ContainerAlias(nn.Module):
    def __init__(self):
        super().__init__()
        self.predictions = nn.Module()
        self.predictions.decoder = nn.Linear(3, 2)
        self.predictions.bias = self.predictions.decoder.bias

    def forward(self, x):
        return self.predictions.decoder(x)


def test_zero3_plain_container_leaf_alias_preserves_single_owner_and_export(tmp_path):
    torch.manual_seed(197)
    model = ContainerAlias()
    expected = deepcopy(model)

    def loss(model, batch):
        value = (model(batch) - 0.5).square()
        return LossTerm(value.sum(), torch.tensor(value.numel()), "elements")

    reference = Trainer(expected, loss, lr=0.02)
    trainer = Trainer(model, loss, lr=0.02, zero_stage=3, ema_decay=0.9)
    assert len(trainer.roles["model"].parameters) == 2
    assert model.predictions._aster_zero3_parameter_aliases == {"bias": "decoder.bias"}
    target = trainer.clone_target("model", "target", factory=ContainerAlias)
    for _ in range(3):
        batch = torch.ones(3, 3)
        reference.step([batch])
        trainer.step([batch])
    exported = trainer.export_state_dict()
    assert set(exported) == set(expected.state_dict())
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(exported[name], value)
    assert torch.equal(exported["predictions.bias"], exported["predictions.decoder.bias"])
    trainer.update_target("model", "target", 0.0)
    torch.testing.assert_close(target.predictions.bias, expected.predictions.bias)
    assert target.predictions.bias is target.predictions.decoder.bias
    with pytest.raises(RuntimeError, match="not a Tensor"):
        torch.add(model.predictions.bias, 1.0)
    path = trainer.save_checkpoint(tmp_path / "alias")
    trainer.step([batch])
    next_state = trainer.export_state_dict()
    trainer.load_checkpoint(path)
    trainer.step([batch])
    for name, value in trainer.export_state_dict().items():
        torch.testing.assert_close(value, next_state[name], atol=0, rtol=0)
    assert "predictions.bias" in trainer.export_state_dict(ema=True)


def test_custom_forward_parameter_container_still_rejected_before_mutation():
    class Unsafe(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = nn.Linear(3, 2)
            self.bias = self.decoder.bias

        def forward(self, x):
            return self.decoder(x) + self.bias

    model = Unsafe()
    original = deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="叶子"):
        Trainer(model, zero_stage=3)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name])


@pytest.mark.parametrize(
    "options",
    [{"zero_stage": 1}, {"zero_stage": 2}, {"zero_stage": 3}, {"offload_optimizer": "cpu"}],
)
def test_max_norm_embedding_storage_owner_preflight_preserves_original(options):
    model = nn.Sequential(nn.Linear(3, 3), nn.Embedding(4, 3, max_norm=1.0))
    original = deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="max_norm"):
        Trainer(model, **options)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name])
    assert isinstance(model[0], nn.Linear) and isinstance(model[1], nn.Embedding)


def test_single_rank_unsharded_max_norm_preserves_torch_projection_and_update():
    model = nn.Embedding(4, 3, max_norm=1.0)
    with torch.no_grad():
        model.weight.fill_(2.0)
    reference = deepcopy(model)
    optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01)
    inputs = torch.tensor([0, 1, 0])

    def loss(module, tokens):
        values = module(tokens).square()
        return LossTerm(values.sum(), torch.tensor(values.numel()), "elements")

    trainer = Trainer(model, loss, lr=0.01, max_grad_norm=None)
    optimizer.zero_grad(set_to_none=True)
    loss(reference, inputs).mean.backward()
    optimizer.step()
    assert trainer.step([inputs]).updated
    torch.testing.assert_close(model.weight, reference.weight, atol=0, rtol=0)
