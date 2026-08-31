"""Offline-first entry points for native training, generation, and evaluation."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import torch
from .core import ArtifactStore, read_json


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="aster", description="Native model / training / inference / evaluation framework"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Inspect installed runtime without modifying it")
    run = sub.add_parser("run", help="Execute a native artifact-based recipe DAG")
    run.add_argument("config")
    run.add_argument("--output", required=True)
    run.add_argument("--store", required=True)
    train = sub.add_parser("train", help="Train / distill on local language JSONL")
    train.add_argument("config")
    train.add_argument("--output", required=True)
    train.add_argument("--store", required=True)
    distributed = sub.add_parser(
        "distributed-train",
        help="Collective native DP/TP/ZeRO training under torchrun/native launcher; not one Workflow per rank",
    )
    distributed.add_argument("config")
    distributed.add_argument("--output", required=True)
    distributed.add_argument("--store", required=True)
    distributed.add_argument("--kind", choices=("language", "tensor"), default="language")
    distributed.add_argument("--backend", choices=("gloo", "nccl"), default=None)
    distributed.add_argument("--timeout-seconds", type=int, default=None)
    distributed.add_argument("--tensor-parallel", type=int, default=1)
    distributed.add_argument("--pipeline-parallel", type=int, default=1)
    distributed.add_argument(
        "--expert-parallel",
        type=int,
        default=1,
        help="Expert partition count within WORLD; native_moe provider only",
    )
    distributed.add_argument(
        "--expert-tensor-parallel",
        type=int,
        default=1,
        help="Expert tensor shards; WORLD = EP x ETP x EDP, attention TP=1 or ETP",
    )
    generate = sub.add_parser(
        "generate", help="Generate with a local artifact; no remote model loader"
    )
    generate.add_argument("--artifact", required=True)
    generate.add_argument("--store", required=True)
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--max-new-tokens", type=int, default=32)
    generate.add_argument("--max-length", type=int, default=128)
    generate.add_argument("--device", default="cpu")
    evaluate = sub.add_parser("evaluate", help="Evaluate a local language artifact")
    evaluate.add_argument("config")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--store", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command not in {"doctor", "distributed-train"} and (
            int(os.environ.get("WORLD_SIZE", "1")) > 1
            or (torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1)
        ):
            raise ValueError(
                "Do not run a single-writer Workflow on every rank; use distributed-train for collective training"
            )
        if args.command == "doctor":
            result = {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "torch_cuda_build": torch.version.cuda,
                "cuda_available_to_torch": torch.cuda.is_available(),
                "cuda_devices": torch.cuda.device_count(),
                "core_dependencies": ["torch", "numpy"],
                "network_access": "not_attempted",
            }
        elif args.command == "distributed-train":
            from .training.launch import distributed_session, DistributedEnvironment
            from .training.parallel import ParallelConfig
            from .training.recipes import collective_local, run_distributed_recipe

            environment = DistributedEnvironment.from_mapping(os.environ)
            model_parallel = args.tensor_parallel * args.pipeline_parallel
            if (
                min(args.tensor_parallel, args.pipeline_parallel) < 1
                or environment.world_size % model_parallel
            ):
                raise ValueError("TP/PP must be positive and their product must divide WORLD_SIZE")
            grid = ParallelConfig(
                tensor_parallel=args.tensor_parallel,
                pipeline_parallel=args.pipeline_parallel,
                data_parallel=environment.world_size // model_parallel,
                expert_parallel=args.expert_parallel,
                expert_tensor_parallel=args.expert_tensor_parallel,
            )
            with distributed_session(
                grid, backend=args.backend, timeout_seconds=args.timeout_seconds
            ) as context:
                config = collective_local(
                    context, lambda: read_json(args.config), "Read distributed training config"
                )
                store = collective_local(
                    context, lambda: ArtifactStore(args.store), "Open artifact store"
                )
                result = run_distributed_recipe(
                    config, kind=args.kind, directory=args.output, store=store, parallel=context
                )

                if context.rank == 0:
                    print(json.dumps(asdict(result), ensure_ascii=False, indent=2, allow_nan=False))
            return 0
        elif args.command == "run":
            from .core.workflow import Workflow, Stage
            from .recipes import BUILTIN_STAGES

            config = read_json(args.config)
            if set(config) != {"stages"}:
                raise ValueError("Workflow config only accepts stages")
            workflow = Workflow(
                [Stage(**stage) for stage in config["stages"]],
                BUILTIN_STAGES,
                artifact_store=ArtifactStore(args.store),
                directory=args.output,
            )
            result = workflow.run()
        elif args.command in {"train", "evaluate"}:
            from .core.workflow import Workflow, Stage
            from .recipes import BUILTIN_STAGES

            kind = "language_fit" if args.command == "train" else "language_evaluate"
            workflow = Workflow(
                [Stage(args.command, kind, read_json(args.config))],
                BUILTIN_STAGES,
                artifact_store=ArtifactStore(args.store),
                directory=args.output,
            )
            result = workflow.run()
        else:
            from .recipes import load_predictor_artifact
            from .evaluation.language import LanguageEvaluator

            artifact = ArtifactStore(args.store).get(args.artifact)
            model, tokenizer = load_predictor_artifact(artifact, device=args.device)
            evaluator = LanguageEvaluator(model, tokenizer, max_length=args.max_length)
            result = {
                "artifact": artifact.id,
                "text": evaluator.generate(args.prompt, max_new_tokens=args.max_new_tokens),
            }
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, OSError) as error:
        print(f"aster: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
