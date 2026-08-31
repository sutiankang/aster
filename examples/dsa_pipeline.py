from dataclasses import asdict
from pathlib import Path
import torch

from aster.core import ArtifactStore, atomic_json, digest_json
from aster.data import ByteTokenizer, StatefulSampler, causal_collate
from aster.models import DeepSeekV32Config, build_model
from aster.methods.sparse_indexer import DSAIndexerObjective, prepare_dsa_stage
from aster.recipes import LanguageData, load_predictor_artifact
from aster.training import Trainer
from aster.training.runtime_state import apply_runtime_state


def run_pipeline(output_dir, *, source_directory=None, data=None, steps=2, zero_stage=3):
    if (
        type(steps) is not int
        or steps < 1
        or type(zero_stage) is not int
        or zero_stage not in {0, 1, 2, 3}
    ):
        raise ValueError("Choose positive steps and ZeRO0-3")
    output = Path(output_dir).absolute()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(431)
    store = ArtifactStore(output / "store")
    if source_directory is None:
        source_directory = output / "initial"
        tokenizer = ByteTokenizer()
        initial = build_model(DeepSeekV32Config(vocab_size=tokenizer.vocab_size, index_topk=3))
        initial.save_pretrained(source_directory / "model")
        tokenizer.save_pretrained(source_directory / "tokenizer")
    source = store.publish(
        source_directory, kind="token_predictor", metadata={"purpose": "dsa_pipeline_source"}
    )
    initial, tokenizer = load_predictor_artifact(source)
    if (
        not isinstance(initial.config, DeepSeekV32Config)
        or initial.config.vocab_size != tokenizer.vocab_size
    ):
        raise ValueError("Source must be native DeepSeekV32 with matching tokenizer vocabulary")
    data = Path(data) if data is not None else Path(__file__).parent / "data/tiny_text.jsonl"
    dataset = LanguageData(data, tokenizer, initial.config.max_position_embeddings)
    dataset.verify()
    stages = []

    def window(sampler):
        records = sampler.take(2)
        if not records:
            sampler.next_epoch()
            records = sampler.take(2)
        return [causal_collate(records, pad_token_id=tokenizer.pad_token_id)]

    for stage in ("dense_warmup", "sparse_training"):
        directory = output / stage
        directory.mkdir()
        objective = DSAIndexerObjective(stage)

        def construct():

            native, _ = load_predictor_artifact(source)
            engine = Trainer(
                prepare_dsa_stage(native, stage), objective, zero_stage=zero_stage, lr=0.001
            )
            sampler = StatefulSampler(dataset, seed=532)
            engine.register_state("training_sampler", sampler)
            return engine, sampler

        engine, sampler = construct()
        updates = []
        for _ in range(steps):
            result = engine.step(window(sampler))
            if not result.updated:
                raise RuntimeError("DSA update did not complete; do not publish")
            updates.append(asdict(result))
        checkpoint = engine.save_checkpoint(directory / "checkpoint.json")
        expected_result = engine.step(window(sampler))
        expected_weights = engine.export_state_dict()
        fresh, fresh_sampler = construct()
        fresh.load_checkpoint(checkpoint)
        actual = fresh.step(window(fresh_sampler))
        if actual != expected_result or not actual.updated:
            raise RuntimeError("Next-step resume differs")
        for name, value in fresh.export_state_dict().items():
            torch.testing.assert_close(value, expected_weights[name], atol=0, rtol=0)
        updates.append(asdict(actual))
        dataset.verify()
        deployed = build_model(initial.config)
        deployed.load_state_dict(fresh.export_state_dict(), strict=True)
        apply_runtime_state(deployed, fresh.export_runtime_state())
        export = directory / "deployment"
        deployed.save_pretrained(export / "model")
        tokenizer.save_pretrained(export / "tokenizer")
        metadata = {
            "stage": stage,
            "objective": objective.config_dict(),
            "updates": fresh.steps,
            "dataset_fingerprint": dataset.fingerprint,
            "tokenizer_fingerprint": digest_json(tokenizer.to_dict()),
            "exact_next_step_resume": True,
            "evidence_kind": "local_training_chain_not_public_quality",
        }
        artifact = store.publish(
            export, kind="token_predictor", metadata=metadata, parents=(source.id,)
        )
        stages.append(
            {**metadata, "artifact_id": artifact.id, "parent_id": source.id, "steps": updates}
        )
        source = artifact

    deployed, _ = load_predictor_artifact(source)
    deployed.eval()
    ids = torch.tensor([dataset[0]["input_ids"]])
    split = max(1, ids.shape[1] // 2)
    with torch.no_grad():
        prefix = deployed(ids[:, :split], use_cache=True)
        suffix = deployed(ids[:, split:], state=prefix.state, use_cache=True)
        torch.testing.assert_close(
            suffix.logits, deployed(ids).logits[:, split:], atol=3e-6, rtol=3e-5
        )
    report = dict(
        stages=stages,
        final_artifact_id=source.id,
        cache_equivalent=True,
        zero_stage=zero_stage,
        public_quality_validated=False,
    )
    atomic_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    parser.add_argument("--source-directory")
    parser.add_argument("--data")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--zero-stage", type=int, default=3)
    args = parser.parse_args()
    print(
        json.dumps(
            run_pipeline(
                args.output_dir,
                source_directory=args.source_directory,
                data=args.data,
                steps=args.steps,
                zero_stage=args.zero_stage,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
