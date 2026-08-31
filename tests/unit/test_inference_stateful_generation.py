from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from aster.models import (
    build_model,
    LlamaConfig,
    MistralConfig,
    Qwen3NextConfig,
    Qwen35TextConfig,
    DeepSeekV3Config,
    DeepSeekV32Config,
    DeepSeekV4Config,
    GPT2Config,
    MambaConfig,
    Gemma4TextConfig,
    Gemma4Config,
    Qwen3VLConfig,
    pack_gemma4_images,
)
from aster.models.qwen_vl import pack_qwen_pixels
from aster.inference import StatefulTokenRunner, SamplingConfig


@pytest.fixture(autouse=True)
def small_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _oracle(model, prompt, count, inputs):

    tokens, logps = [], []
    with torch.no_grad():
        for index in range(count):
            arguments = dict(inputs)
            if arguments.get("attention_mask") is not None:
                arguments["attention_mask"] = torch.nn.functional.pad(
                    arguments["attention_mask"], (0, index), value=1
                )
            if arguments.get("mm_token_type_ids") is not None:
                arguments["mm_token_type_ids"] = torch.nn.functional.pad(
                    arguments["mm_token_type_ids"], (0, index)
                )
            logits = model(torch.tensor([prompt + tokens]), **arguments).logits[0, -1].float()
            token = int(logits.argmax())
            tokens.append(token)
            logps.append(float(logits.log_softmax(-1)[token]))
    return tokens, logps


@pytest.mark.parametrize(
    "family", ["llama", "window", "qwen_next", "qwen35", "mla", "dsa", "gpt2", "mamba", "gemma4"]
)
@pytest.mark.parametrize("values", [[0, 0, 1, 1, 1], [1, 0, 1, 0, 1]])
def test_stateful_complete_history_mask_logits_logprobs_and_no_caller_mutation(family, values):
    configs = {
        "llama": LlamaConfig(),
        "window": MistralConfig(sliding_window=3),
        "qwen_next": Qwen3NextConfig(
            num_hidden_layers=2, layer_types=("linear_attention", "full_attention"), num_experts=0
        ),
        "qwen35": Qwen35TextConfig(
            num_hidden_layers=2, layer_types=("linear_attention", "full_attention"), num_experts=0
        ),
        "mla": DeepSeekV3Config(),
        "dsa": DeepSeekV32Config(),
        "gpt2": GPT2Config(),
        "mamba": MambaConfig(),
        "gemma4": Gemma4TextConfig(),
    }
    torch.manual_seed(55)
    model = build_model(configs[family]).eval()
    runner = StatefulTokenRunner(model, policy_artifact_id="mask-audit:" + family)
    prompt, padding = [0, 0, 1, 4, 6], torch.tensor([values])
    original = padding.clone()
    calls, output_logits = [], []
    runner.model.register_forward_pre_hook(
        lambda module, args, kwargs: calls.append(deepcopy(kwargs["attention_mask"])),
        with_kwargs=True,
    )
    runner.model.register_forward_hook(
        lambda module, args, output: output_logits.append(output.logits[0, -1].clone())
    )
    result = runner.generate(
        prompt,
        SamplingConfig(max_new_tokens=4, temperature=0),
        modality_inputs={"attention_mask": padding},
    )
    expected_ids, expected_logp = _oracle(model, prompt, 4, {"attention_mask": padding})
    assert list(result.token_ids) == expected_ids
    torch.testing.assert_close(
        torch.tensor(result.raw_model_logprobs), torch.tensor(expected_logp), atol=3e-6, rtol=3e-5
    )
    assert result.behavior_logprobs == (0.0,) * 4
    assert torch.equal(padding, original) and result.prompt_token_ids == tuple(prompt)
    with torch.no_grad():
        for index, (actual_mask, logits) in enumerate(zip(calls, output_logits)):
            full_mask = torch.nn.functional.pad(original, (0, index), value=1)
            assert torch.equal(actual_mask, full_mask.bool())
            expected = model(
                torch.tensor([prompt + expected_ids[:index]]), attention_mask=full_mask
            ).logits[0, -1]
            torch.testing.assert_close(logits, expected, atol=5e-6, rtol=5e-5)


def test_generate_mask_and_prompt_are_frozen_before_external_token_callback():
    torch.manual_seed(55)
    model = build_model(
        Qwen3NextConfig(
            num_hidden_layers=2, layer_types=("linear_attention", "full_attention"), num_experts=0
        )
    ).eval()
    prompt, mask = [0, 0, 1, 4, 6], torch.tensor([[0, 0, 1, 1, 1]])
    initial_prompt, initial_mask = list(prompt), mask.clone()
    expected_ids, expected_logp = _oracle(
        model, initial_prompt, 4, {"attention_mask": initial_mask}
    )

    def mutate(_event):
        mask.fill_(1)
        prompt.append(7)

    runner = StatefulTokenRunner(model, policy_artifact_id="frozen")
    result = runner.generate(
        prompt,
        SamplingConfig(max_new_tokens=4, temperature=0),
        modality_inputs={"attention_mask": mask},
        on_token=mutate,
    )
    assert list(result.token_ids) == expected_ids and result.prompt_token_ids == tuple(
        initial_prompt
    )
    torch.testing.assert_close(
        torch.tensor(result.raw_model_logprobs), torch.tensor(expected_logp), atol=3e-6, rtol=3e-5
    )


@pytest.mark.parametrize("kind", ["qwen_image", "gemma_image", "gemma_video"])
def test_stateful_media_is_prefill_only_but_history_mask_persists(kind):
    torch.manual_seed(137)
    if kind == "qwen_image":
        c = Qwen3VLConfig()
        model = build_model(c).eval()
        pixels, grid = pack_qwen_pixels(torch.randn(1, 3, 8, 8), c.vision_config)
        prompt = (
            [0, 0, 1, c.vision_start_token_id]
            + [c.image_token_id] * 4
            + [c.vision_end_token_id, 3, 5]
        )
        types = torch.where(torch.tensor([prompt]) == c.image_token_id, 1, 0)
        inputs = {"pixel_values": pixels, "image_grid_thw": grid, "mm_token_type_ids": types}
    else:
        c = Gemma4Config(
            text_config=replace(Gemma4TextConfig(), use_bidirectional_attention="vision")
        )
        model = build_model(c).eval()
        video = kind == "gemma_video"
        packed = pack_gemma4_images(torch.rand(2 if video else 1, 3, 8, 8), c.vision_config)
        visual_id, count = (c.video_token_id, 8) if video else (c.image_token_id, 4)
        prompt = [0, 0, 1] + [visual_id] * count + [4, 5, 6]
        inputs = (
            {
                "pixel_values_videos": packed["pixel_values"][None],
                "video_position_ids": packed["pixel_position_ids"][None],
            }
            if video
            else {
                "pixel_values": packed["pixel_values"],
                "image_position_ids": packed["pixel_position_ids"],
            }
        )
        inputs["mm_token_type_ids"] = torch.tensor(
            [[0, 0, 0] + [2 if video else 1] * count + [0, 0, 0]]
        )
    inputs["attention_mask"] = torch.tensor([[0, 0] + [1] * (len(prompt) - 2)])
    runner = StatefulTokenRunner(model, policy_artifact_id=kind, processor_id="native-patches")
    keys = []
    runner.model.register_forward_pre_hook(
        lambda module, args, kwargs: keys.append(set(kwargs)), with_kwargs=True
    )
    result = runner.generate(
        prompt, SamplingConfig(max_new_tokens=3, temperature=0), modality_inputs=inputs
    )
    ids, logps = _oracle(model, prompt, 3, inputs)
    assert list(result.token_ids) == ids
    torch.testing.assert_close(
        torch.tensor(result.raw_model_logprobs), torch.tensor(logps), atol=4e-6, rtol=5e-5
    )
    assert keys[0] >= set(inputs)
    assert all(key == {"input_ids", "state", "use_cache", "attention_mask"} for key in keys[1:])


@pytest.mark.parametrize(
    "inputs",
    [
        {"attention_mask": torch.ones(5)},
        {"attention_mask": torch.ones(2, 5)},
        {"attention_mask": torch.tensor([[1, 1, 2, 1, 1]])},
        {"attention_mask": torch.tensor([[1.0, 1.0, float("nan"), 1.0, 1.0]])},
        {"attention_mask": torch.ones(1, 5, requires_grad=True)},
        {"attention_mask": torch.ones(1, 5, dtype=torch.complex64)},
        {"attention_mask": torch.tensor([[1, 1, 1, 1, 0]])},
        {"attention_mask": torch.zeros(1, 5)},
        {"position_ids": torch.arange(5)[None]},
        {"cache_position": torch.arange(5)},
        {"token_type_ids": torch.zeros(1, 5)},
        {"inputs_embeds": torch.ones(1, 5, 32)},
        {"pixel_values": torch.ones(1, 3, 8, 8)},
    ],
)
def test_invalid_or_unclassified_generation_semantics_fail_before_first_forward(inputs):
    runner = StatefulTokenRunner(build_model(LlamaConfig()), policy_artifact_id="bad-input")
    with pytest.raises(ValueError):
        runner.generate([1, 2, 3, 4, 5], modality_inputs=inputs)
    assert runner.calls == 0


def test_unknown_physical_layout_and_v4_padding_rejected_before_prefill():
    from aster.models.decoder import CausalLM

    class Unknown(CausalLM):
        pass

    runner = StatefulTokenRunner(Unknown(LlamaConfig()), policy_artifact_id="unknown")
    with pytest.raises(ValueError, match="physical-state"):
        runner.generate([1, 2], modality_inputs={"attention_mask": torch.ones(1, 2)})
    assert runner.calls == 0
    runner = StatefulTokenRunner(build_model(DeepSeekV4Config()), policy_artifact_id="v4")
    with pytest.raises(ValueError, match="unpadded"):
        runner.generate([0, 2], modality_inputs={"attention_mask": torch.tensor([[0, 1]])})
    assert runner.calls == 0
    actual = runner.generate(
        [1, 2],
        SamplingConfig(max_new_tokens=2, temperature=0),
        modality_inputs={"attention_mask": torch.ones(1, 2)},
    )
    expected = runner.generate([1, 2], SamplingConfig(max_new_tokens=2, temperature=0))
    assert (
        actual.token_ids == expected.token_ids
        and actual.raw_model_logprobs == expected.raw_model_logprobs
    )


def test_explicit_none_mask_and_position_match_existing_unmasked_generation():
    runner = StatefulTokenRunner(build_model(LlamaConfig()), policy_artifact_id="none")
    sampling = SamplingConfig(max_new_tokens=3, temperature=0)
    first = runner.generate([1, 2, 3], sampling)
    second = runner.generate(
        [1, 2, 3], sampling, modality_inputs={"attention_mask": None, "position_ids": None}
    )
    assert (
        first.token_ids == second.token_ids
        and first.raw_model_logprobs == second.raw_model_logprobs
    )
