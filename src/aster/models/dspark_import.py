"""Strict local DeepSpec state-dictionary conversion for supported draft layouts."""

from collections.abc import Mapping
import torch

from . import build_model, CausalLM, Qwen3Config
from .dspark import DSparkConfig
from .dspark_gemma4 import Gemma4DSparkConfig
from .gemma4 import Gemma4ForCausalLM, Gemma4TextConfig


def import_dspark_state(config, state_dict, *, target, embedding_head="required"):

    qwen = (
        type(config) is DSparkConfig
        and type(target) is CausalLM
        and type(target.config) is Qwen3Config
    )
    gemma = (
        type(config) is Gemma4DSparkConfig
        and type(target) is Gemma4ForCausalLM
        and type(target.config) is Gemma4TextConfig
    )
    if not (qwen or gemma) or config.target.to_dict() != target.config.to_dict():
        raise ValueError(
            "Official DSpark import requires an exactly matched native target/config family"
        )
    if not config.freeze_embedding_head or getattr(target, "_aster_training_owned", False):
        raise ValueError(
            "Import profile requires frozen vocabulary weights and an idle dense target"
        )
    if embedding_head not in {"required", "from_target"}:
        raise ValueError("Declare embedding_head as required or from_target")
    if (
        not isinstance(state_dict, Mapping)
        or not state_dict
        or any(
            not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in state_dict.items()
        )
    ):
        raise ValueError("Official DSpark weights must be a nonempty string-to-tensor mapping")

    with torch.random.fork_rng(devices=[]):
        model = build_model(config)
    expected = model.state_dict()
    native_only = {"teacher_weights_loaded", "teacher_fingerprint"}
    vocabulary = {"embed_tokens.weight", "lm_head.weight"}
    converted = {}
    for name, value in state_dict.items():
        if (
            name in native_only
            or name.startswith("confidence_head.")
            and not name.startswith("confidence_head.proj.")
        ):
            raise ValueError(
                "Expected official DSpark names, not native provenance/renamed confidence fields"
            )
        key = (
            name.replace("confidence_head.proj.", "confidence_head.", 1)
            if name.startswith("confidence_head.proj.")
            else name
        )
        if key not in expected or key in converted or value.shape != expected[key].shape:
            raise ValueError(f"Official DSpark tensor schema differs: {name}")
        if (
            value.layout != torch.strided
            or value.device.type == "meta"
            or not value.is_floating_point()
            or not torch.isfinite(value).all()
        ):
            raise ValueError(
                f"Official DSpark weights must be finite dense floating tensors: {name}"
            )
        converted[key] = value.detach().cpu().clone()
    required = set(expected) - native_only
    if embedding_head == "from_target":
        if vocabulary & set(converted):
            raise ValueError("from_target requires both vocabulary tensors to be omitted")
        required -= vocabulary
    if set(converted) != required:
        raise ValueError(
            f"Official DSpark missing/unexpected tensor keys: {sorted(required ^ set(converted))}"
        )
    target_vocabulary = {
        "embed_tokens.weight": target.get_input_embeddings().weight,
        "lm_head.weight": target.lm_head.weight,
    }
    for key, value in target_vocabulary.items():
        actual = value.detach().cpu()
        if not torch.isfinite(actual).all():
            raise ValueError("Target vocabulary contains nonfinite weights")
        if embedding_head == "required" and (
            converted[key].dtype != actual.dtype or not torch.equal(converted[key], actual)
        ):
            raise ValueError(
                "Frozen source vocabulary weights differ from the declared target identity"
            )
        if embedding_head == "from_target":
            converted[key] = actual.clone()
    model.initialize_from_target(target)
    converted.update(
        {key: value.clone() for key, value in model.state_dict().items() if key in native_only}
    )
    model.load_state_dict(converted, strict=True, assign=True)
    return model.eval()
