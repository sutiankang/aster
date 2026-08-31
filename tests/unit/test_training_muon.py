from copy import deepcopy
import ast
import hashlib
import os
import urllib.request

import pytest
import torch
from torch import nn

from aster.core import LossTerm
from aster.models import LlamaConfig, build_model
from aster.training import Trainer, MuonFactory, MuonWithAuxAdam
from aster.training.muon import SOURCES, newton_schulz
from aster.training.portable import gather_tensor, logical_tensors, optimizer_mapping


def test_explicit_groups_missing_gradient_semantics_and_rejections():
    matrix = nn.Parameter(torch.randn(4, 3))
    bias = nn.Parameter(torch.randn(4))
    with pytest.raises(ValueError, match="explicitly"):
        MuonWithAuxAdam([{"params": [matrix], "use_muon": True}])
    with pytest.raises(ValueError, match="one Muon/Adam"):
        MuonWithAuxAdam([dict(params=[matrix, matrix], use_muon=True, profile="keller")])
    with pytest.raises(ValueError, match="geometry"):
        MuonWithAuxAdam([dict(params=[bias], use_muon=True, profile="moonlight")])
    with pytest.raises(ValueError, match="Unknown"):
        MuonWithAuxAdam([dict(params=[matrix], use_muon=True, profile="keller", magic=True)])
    with pytest.raises(ValueError, match="source pin"):
        MuonWithAuxAdam(
            [dict(params=[matrix], use_muon=True, profile="keller", source_commit="0" * 40)]
        )
    for policy in ("skip", "zero"):
        parameter = nn.Parameter(torch.ones(3, 2))
        optimizer = MuonWithAuxAdam(
            [
                dict(
                    params=[parameter],
                    use_muon=True,
                    profile="keller",
                    lr=0.1,
                    weight_decay=0.1,
                    missing_grad=policy,
                )
            ]
        )
        optimizer.step()
        assert torch.equal(
            parameter,
            torch.ones_like(parameter) if policy == "skip" else torch.full_like(parameter, 0.99),
        )
        assert bool(optimizer.state) == (policy == "zero")
    with pytest.raises(FloatingPointError):
        newton_schulz(torch.full((3, 2), float("nan")))


def test_explicit_conv_and_batched_geometry_match_separate_matrix_oracles():
    torch.manual_seed(852)
    gradient = torch.randn(3, 7, 4)
    expected = torch.stack([newton_schulz(part) for part in gradient])
    assert torch.equal(newton_schulz(gradient), expected)
    parameter = nn.Parameter(torch.randn(3, 2, 2, 2))
    parameter.grad = torch.randn_like(parameter)
    optimizer = MuonWithAuxAdam(
        [dict(params=[parameter], use_muon=True, profile="keller", matrix_kind="conv2d")]
    )
    optimizer.step()
    assert torch.isfinite(parameter).all()
    with pytest.raises(ValueError, match="only 2D"):
        MuonWithAuxAdam(
            [dict(params=[parameter], use_muon=True, profile="moonlight", matrix_kind="conv2d")]
        )


def objective(model, ids):
    logits = model(ids).logits.float()
    value = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction="sum"
    )
    return LossTerm(value, torch.tensor(ids[:, 1:].numel(), dtype=torch.int64), "token")


def make(profile, precision="fp32", offload="none", directory=None, zero=0):
    torch.set_num_threads(1)
    torch.manual_seed(382)
    model = build_model(
        LlamaConfig(
            vocab_size=17,
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            tie_word_embeddings=True,
        )
    )
    factory = MuonFactory.from_model(
        model,
        auxiliary_modules=("lm_head",),
        profile=profile,
        muon_options={"lr": 0.001},
        auxiliary_options={"lr": 0.0005},
    )

    names = [name for group in factory.groups for name in group["names"]]
    assert len(names) == len(list(model.parameters())) and len(names) == len(set(names))
    assert all(
        "embed_tokens" not in name and "lm_head" not in name
        for group in factory.groups
        if group["use_muon"]
        for name in group["names"]
    )
    return Trainer(
        model,
        objective,
        optimizer_factory=factory,
        accumulation_steps=2,
        precision=precision,
        zero_stage=zero,
        offload_optimizer=offload,
        offload_directory=directory if offload == "nvme" else None,
        ema_decay=0.9,
    )


def inspect_gradients(engine, expected=None):

    optimizer, owners, sharded = optimizer_mapping(engine.roles["model"])
    entries = logical_tensors(engine.model, engine.parallel)
    captured = {}
    original = optimizer.step

    def step(*args, **kwargs):
        visited = set()
        for entry in entries:
            if not entry.parameter or id(entry.tensor) not in owners:
                continue
            owner = owners[id(entry.tensor)]
            if owner.grad is None:
                continue
            if id(owner) in visited:
                continue
            visited.add(id(owner))
            value = gather_tensor(owner.grad, entry, engine.parallel, optimizer_sharded=sharded)
            captured[entry.name] = value
            if expected is not None:
                torch.testing.assert_close(value, expected[entry.name], atol=5e-8, rtol=4e-6)

                assert not entry.dp_sharded and not sharded and entry.tp_dimension is None
                owner.grad.copy_(expected[entry.name].to(owner.device))
        return original(*args, **kwargs)

    optimizer.step = step
    return captured


@pytest.mark.parametrize("profile", ["keller", "moonlight"])
@pytest.mark.parametrize("precision,offload", [("fp32", "none"), ("bf16", "cpu"), ("bf16", "nvme")])
@pytest.mark.parametrize("zero", [0, 1, 2, 3])
def test_complete_model_named_ownership_native_portable_fresh_resume(
    profile, precision, offload, zero, tmp_path
):
    engine = make(profile, precision, offload, tmp_path / "offload", zero)
    batches = [torch.tensor([[1, 4, 5, 6]]), torch.tensor([[3, 7, 2], [4, 9, 11]])]
    assert engine.step(batches).updated
    native = engine.save_checkpoint(tmp_path / "native")
    portable = engine.save_portable_checkpoint(tmp_path / "portable")
    gradients = inspect_gradients(engine)
    engine.step(batches)
    expected = engine.export_state_dict()
    ema = engine.export_state_dict(ema=True)
    fresh = make(profile, precision, offload, tmp_path / "offload-fresh", zero)
    fresh.load_checkpoint(native)
    fresh.step(batches)
    for name, value in fresh.export_state_dict().items():
        assert torch.equal(value, expected[name])
    for name, value in fresh.export_state_dict(ema=True).items():
        assert torch.equal(value, ema[name])
    dense = make(profile, precision)
    dense.load_portable_checkpoint(portable, seed=9)
    inspect_gradients(dense, gradients)
    dense.step(batches)
    for name, value in dense.export_state_dict().items():
        assert torch.equal(value, expected[name])
    wrong = make(
        "keller" if profile == "moonlight" else "moonlight",
        precision,
        offload,
        tmp_path / "offload-wrong",
        zero,
    )
    with pytest.raises(ValueError, match="配置"):
        wrong.load_checkpoint(native)


@pytest.mark.parametrize("profile", ["keller", "moonlight"])
def test_pinned_unmodified_official_optimizer_equivalence(profile):
    if os.environ.get("ASTER_RUN_REMOTE_MUON_ORACLE") != "1":
        pytest.skip("Explicit pinned official optimizer source download")
    torch.set_num_threads(1)
    source = SOURCES[profile]
    url = (
        source["repository"].replace("github.com", "raw.githubusercontent.com")
        + "/"
        + source["commit"]
        + "/"
        + source["path"]
    )
    payload = urllib.request.urlopen(url, timeout=30).read()
    assert hashlib.sha256(payload).hexdigest() == source["sha256"]
    tree = ast.parse(payload)
    names = (
        {"zeropower_via_newtonschulz5", "muon_update", "adam_update", "SingleDeviceMuonWithAuxAdam"}
        if profile == "keller"
        else {"zeropower_via_newtonschulz5", "Muon"}
    )
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]

    for node in nodes:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    import math

    namespace = {"torch": torch, "math": math}
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), url, "exec"),
        namespace,
    )
    torch.manual_seed(840)
    actual = [
        nn.Parameter(torch.randn(7, 3)),
        nn.Parameter(torch.randn(3, 7)),
        nn.Parameter(torch.randn(7)),
    ]
    expected = [nn.Parameter(value.detach().clone()) for value in actual]
    groups = [
        dict(params=actual[:2], use_muon=True, profile=profile, lr=0.003, weight_decay=0.02),
        dict(
            params=actual[2:],
            use_muon=False,
            profile=profile,
            lr=0.003,
            weight_decay=0.02,
            eps=1e-5,
        ),
    ]
    optimizer = MuonWithAuxAdam(groups)
    if profile == "keller":
        reference = namespace["SingleDeviceMuonWithAuxAdam"](
            [
                dict(params=expected[:2], use_muon=True, lr=0.003, weight_decay=0.02),
                dict(params=expected[2:], use_muon=False, lr=0.003, weight_decay=0.02, eps=1e-5),
            ]
        )
    else:
        reference = namespace["Muon"](
            lr=0.003, wd=0.02, muon_params=expected[:2], adamw_params=expected[2:], adamw_eps=1e-5
        )
    for step in range(5):
        for one, two in zip(actual, expected):
            gradient = torch.randn_like(one) * (1e-6 if step == 0 else 1.0)
            one.grad = gradient.clone()
            two.grad = gradient.clone()
        gradients = [p.grad.clone() for p in actual]
        optimizer.step()
        reference.step()
        for one, two, gradient in zip(actual, expected, gradients):
            assert torch.equal(one, two)
            assert torch.equal(one.grad, gradient)


@pytest.mark.parametrize("zero", [0, 3])
@pytest.mark.parametrize("profile", ["keller", "moonlight"])
@pytest.mark.parametrize(
    "precision",
    [
        "bf16",
        pytest.param(
            "fp16",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="Real CUDA FP16 AMP validation requires CUDA torch and supported GPU",
            ),
        ),
    ],
)
def test_muon_multirole_amp_overflow_scheduler_and_resume(profile, zero, precision, tmp_path):
    def loss(model, inputs):
        value = model(inputs).float().square().sum() * 0.0001
        return LossTerm(value, torch.tensor(inputs.shape[0], dtype=torch.int64), "sample")

    def build():
        torch.manual_seed(609)
        model = nn.Sequential(nn.Linear(3, 4), nn.SiLU(), nn.Linear(4, 2))
        engine = Trainer(
            model,
            loss,
            precision=precision,
            device="cuda" if precision == "fp16" else "cpu",
            zero_stage=zero,
            offload_optimizer="cpu",
            ema_decay=0.9,
            optimizer_factory=MuonFactory.from_model(
                model, auxiliary_modules=("2",), profile=profile
            ),
        )
        critic = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 1))
        engine.add_role(
            "critic",
            critic,
            optimizer_factory=MuonFactory.from_model(
                critic, auxiliary_modules=("2",), profile=profile
            ),
        )
        engine.set_scheduler(
            lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        )
        return engine

    engine = build()
    inputs = torch.full((2, 3), 0.01, device=engine.device)
    assert engine.step([inputs]).updated
    assert engine.phase(
        "critic_step", role="critic", objective=loss, microbatches=[inputs], freeze_roles=("model",)
    ).updated
    assert not (
        {id(p) for p in engine.roles["model"].parameters}
        & {id(p) for p in engine.roles["critic"].parameters}
    )
    before = engine.export_state_dict()
    clock = engine.last_successful_update()
    scale = engine.loss_scale
    native, _, _ = optimizer_mapping(engine.roles["model"])
    options = [group["lr"] for group in native.param_groups]
    state = deepcopy(native.state_dict())

    def bad(model, value):
        term = loss(model, value)
        return LossTerm(term.numerator * float("inf"), term.denominator, term.unit)

    result = engine.phase("bad", objective=bad, microbatches=[inputs])
    expected_scale = scale / 2 if precision == "fp16" else scale
    assert result.overflow and not result.updated and engine.loss_scale == expected_scale
    assert engine.last_successful_update() == clock and engine.roles["model"].ema.updates == 1
    assert options == [group["lr"] for group in native.param_groups]
    for name, value in engine.export_state_dict().items():
        assert torch.equal(value, before[name])
    for key, saved in state["state"].items():
        for name, value in saved.items():
            actual = native.state_dict()["state"][key][name]
            assert (
                torch.equal(actual, value) if isinstance(value, torch.Tensor) else actual == value
            )
    checkpoint = engine.save_checkpoint(tmp_path / "checkpoint")
    assert engine.step([inputs]).updated
    expected = engine.export_state_dict()
    fresh = build()
    fresh.load_checkpoint(checkpoint)
    assert fresh.loss_scale == expected_scale
    assert fresh.step([inputs]).updated
    for name, value in fresh.export_state_dict().items():
        assert torch.equal(value, expected[name])
