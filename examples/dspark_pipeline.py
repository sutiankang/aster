import asyncio
from pathlib import Path
import torch

from aster.core import ArtifactStore, atomic_json
from aster.data.datasets import StatefulSampler
from aster.data.dspark import DSparkTeacherFeatures, DSparkFeatureCache, publish_dspark_features
from aster.models import CausalLM, Qwen3Config
from aster.models.dspark import DSparkConfig, DSparkDraft, target_state_identity
from aster.models.dspark_gemma4 import Gemma4DSparkConfig, Gemma4DSparkDraft
from aster.models.gemma4 import Gemma4TextConfig, Gemma4ForCausalLM
from aster.methods.dspark import DSparkMethod
from aster.methods.dspark_artifacts import publish_dspark_draft, load_dspark_draft
from aster.inference import ModelRunner, SamplingConfig
from aster.inference.gemma4 import Gemma4SnapshotRunner
from aster.inference.dspark import DSparkDecoder
from aster.evaluation.dspark import evaluate_dspark
from aster.training import Trainer


def run_demo(output_dir, *, family="qwen3", steps=12, zero_stage=3, seed=913):
    if family not in {"qwen3", "gemma4"} or type(steps) is not int or steps < 1:
        raise ValueError("Choose qwen3/gemma4 and a positive step count")
    if type(zero_stage) is not int or zero_stage not in {0, 1, 2, 3}:
        raise ValueError("ZeRO stage must be 0–3")
    output = Path(output_dir).absolute()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    if family == "qwen3":
        target = CausalLM(
            Qwen3Config(
                vocab_size=31,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=3,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=64,
            )
        ).eval()
        config = DSparkConfig(
            target.config,
            target_layer_ids=(-1, 1),
            block_size=3,
            num_anchors=2,
            markov_rank=4,
            markov_head_type="vanilla",
        )
        construct = DSparkDraft
    else:
        target = Gemma4ForCausalLM(
            Gemma4TextConfig(
                vocab_size=31,
                hidden_size=16,
                intermediate_size=32,
                head_dim=4,
                global_head_dim=8,
                hidden_size_per_layer_input=0,
                max_position_embeddings=64,
            )
        ).eval()
        config = Gemma4DSparkConfig(
            target.config,
            target_layer_ids=(-1, 1),
            block_size=3,
            num_anchors=2,
            markov_rank=4,
            markov_head_type="vanilla",
        )
        construct = Gemma4DSparkDraft
    target_id = target_state_identity(target)
    vocabulary = "synthetic_integer_vocab31_v1"
    teacher = DSparkTeacherFeatures(target, config, vocabulary_fingerprint=vocabulary)

    records = {}
    with torch.no_grad():
        for index in range(8):
            ids = torch.randint(1, 31, (1, 3))
            for _ in range(5):
                ids = torch.cat((ids, target(ids).logits[:, -1].argmax(-1)[:, None]), 1)
            mask = torch.ones_like(ids)
            mask[:, :3] = 0
            records[str(index)] = dict(input_ids=ids, loss_mask=mask)
    store = ArtifactStore(output / "store")
    features = publish_dspark_features(
        store,
        teacher,
        records,
        output / "features",
        dataset_id="synthetic_target_greedy",
        revision=f"seed_{seed}_v1",
        license_id="locally_generated_synthetic",
    )
    cache = DSparkFeatureCache(store, features.id)

    def training_objects():
        engine = Trainer(
            construct(config).initialize_from_target(target),
            accumulation_steps=2,
            zero_stage=zero_stage,
            lr=0.002,
            ema_decay=0.9,
        )
        method = DSparkMethod(
            engine,
            vocabulary_fingerprint=vocabulary,
            feature_cache_ids=[cache.artifact_id],
            feature_cache_store=store,
        )
        sampler = StatefulSampler(cache, seed=seed + 1)
        engine.register_state("feature_sampler", sampler)
        return engine, method, sampler

    def window(sampler):
        batches = []
        for _ in range(2):
            sample_ids = sampler.take(2)
            if not sample_ids:
                sampler.next_epoch()
                sample_ids = sampler.take(2)
            batches.append(cache.batch(sample_ids))
        return batches

    engine, method, sampler = training_objects()
    losses = []
    for _ in range(steps):
        losses.append(method.update(window(sampler)).loss)
    checkpoint = engine.save_checkpoint(output / "checkpoint")

    expected = method.update(window(sampler))
    expected_weights = {
        key: value.clone() for key, value in engine.export_state_dict(only_rank_zero=False).items()
    }
    resumed, resumed_method, resumed_sampler = training_objects()
    resumed.load_checkpoint(checkpoint, trusted=True)
    actual = resumed_method.update(window(resumed_sampler))
    assert actual.loss == expected.loss and actual.updated
    for key, value in resumed.export_state_dict(only_rank_zero=False).items():
        torch.testing.assert_close(value, expected_weights[key], atol=0, rtol=0)
    artifact = publish_dspark_draft(resumed_method, store, output / "deployment")
    draft, contract = load_dspark_draft(store, artifact.id)
    runner_cls = ModelRunner if family == "qwen3" else Gemma4SnapshotRunner
    runner = runner_cls(target, policy_artifact_id=target_id)
    decoder = DSparkDecoder(
        runner,
        draft,
        draft_policy_artifact_id=artifact.id,
        vocabulary_fingerprint=vocabulary,
        confidence_threshold=0.6,
    )

    report = asyncio.run(
        evaluate_dspark(
            decoder,
            {"heldout_0": [2, 5, 8], "heldout_1": [8, 2, 9]},
            SamplingConfig(max_new_tokens=8, temperature=0),
            protocol_id="synthetic_dspark_greedy_v1",
            dataset_revision=f"seed_{seed}_v1",
        )
    )
    report["training"] = dict(
        family=family,
        zero_stage=zero_stage,
        updates=steps + 1,
        first_loss=losses[0],
        last_loss=losses[-1],
        exact_next_step_resume=True,
        normalization_profile=contract["objective"]["normalization_profile"],
        feature_cache=features.id,
        checkpoint=str(checkpoint),
        draft_artifact=artifact.id,
    )
    atomic_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    parser.add_argument("--family", choices=("qwen3", "gemma4"), default="qwen3")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--zero-stage", type=int, default=3)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run_demo(
                arguments.output_dir,
                family=arguments.family,
                steps=arguments.steps,
                zero_stage=arguments.zero_stage,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
