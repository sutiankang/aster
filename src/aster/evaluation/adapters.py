"""Opt-in official evaluators using native model computation and explicit execution grants."""

from __future__ import annotations
from dataclasses import dataclass
import hashlib
import importlib
from importlib import metadata
import inspect
import json
from pathlib import Path
import re
from types import SimpleNamespace
import uuid

import numpy as np

from ..core import atomic_json, digest_json, read_json
from .protocol import EvaluationRecord, EvaluationRun
from .suites import _json_value


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class OfficialModulePin:
    module: str
    source_file: str
    source_sha256: str
    distribution: str
    version: str
    revision: str

    def load(self, protocol, grant):
        grant.require(protocol, "official_evaluator")
        if self.module not in {
            "lmms_eval.evaluator",
            "libero.libero.envs.env_wrapper",
            "swebench.harness.run_evaluation",
        }:
            raise ValueError("No arbitrary external evaluator module execution")
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision) or not re.fullmatch(
            r"[0-9a-f]{64}", self.source_sha256
        ):
            raise ValueError("Official source requires a full commit and content digest")
        source = Path(self.source_file).resolve(strict=True)
        if (
            _file_hash(source) != self.source_sha256
            or metadata.version(self.distribution) != self.version
        ):
            raise ValueError("Installed evaluator differs from the approved source/version")
        if protocol.evaluator_version != self.revision:
            raise ValueError("Comparison protocol and evaluator source revision disagree")
        if protocol.controls.get("evaluator_source_sha256") != self.source_sha256:
            raise ValueError("Protocol does not bind the evaluator module content")
        module = importlib.import_module(self.module)
        if Path(module.__file__).resolve() != source:
            raise ValueError("Import resolution selected another evaluator implementation")
        return module


def _lmms_bridge(generate, score, *, candidate_artifact_id):
    from lmms_eval.api.model import lmms

    class NativeLMMS(lmms):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(_name_or_path=candidate_artifact_id)
            self.batch_size = 1

        def generate_until(self, requests):
            outputs = []
            for request in requests:
                if len(request.args) != 6:
                    raise ValueError("Pinned simple LMMS generate request schema changed")
                prompt, kwargs, doc_to_visual, doc_id, task, split = request.args
                doc = self.task_dict[task][split][doc_id]
                text = generate(
                    context=prompt,
                    visuals=doc_to_visual(doc),
                    generation_kwargs=dict(kwargs),
                    doc_id=str(doc_id),
                    task=task,
                    split=split,
                )
                if not isinstance(text, str):
                    raise ValueError("Native VLM callback must return actual generated text")
                outputs.append(text)
            return outputs

        def loglikelihood(self, requests):
            if score is None:
                raise NotImplementedError(
                    "This native VLM has no audited visual likelihood boundary"
                )
            results = []
            for request in requests:
                if len(request.args) != 6:
                    raise ValueError("Pinned simple LMMS likelihood request schema changed")
                context, target, visual, doc_id, task, split = request.args
                doc = self.task_dict[task][split][doc_id]
                results.append(
                    score(
                        context=context,
                        continuation=target(doc) if callable(target) else target,
                        visuals=visual(doc),
                        doc_id=str(doc_id),
                        task=task,
                        split=split,
                    )
                )
            return results

        def generate_until_multi_round(self, requests):
            raise NotImplementedError(
                "Multiround LMMS tasks require an explicit conversation adapter"
            )

    return NativeLMMS()


def evaluate_lmms(
    protocol,
    candidate_artifact_id,
    *,
    task,
    generate,
    source_pin,
    grant,
    environment,
    output_directory,
    score=None,
):

    if source_pin.module != "lmms_eval.evaluator":
        raise ValueError("Expected the official LMMS evaluator entry point")
    evaluator = source_pin.load(protocol, grant)
    task_name = task.get_config("task")
    split = protocol.controls["split"]
    revision = protocol.controls["dataset_revision"]
    if (
        revision in {"main", "master", "latest", ""}
        or task.get_config("dataset_kwargs").get("revision") != revision
    ):
        raise ValueError("LMMS task must pin the exact dataset revision")
    dataset = task.dataset[split]
    if getattr(dataset, "_fingerprint", None) != protocol.controls[
        "dataset_internal_fingerprint"
    ] or getattr(task.eval_docs, "_fingerprint", None) != getattr(dataset, "_fingerprint", None):
        raise ValueError("LMMS loaded dataset fingerprint differs")
    if tuple(f"{task_name}:{i}" for i in range(len(task.eval_docs))) != protocol.expected_ids:
        raise ValueError("LMMS must evaluate the exact complete frozen sample set")
    seed = protocol.controls["seed"]
    target = Path(output_directory)
    target.mkdir(parents=True, exist_ok=False)
    arguments = dict(
        model=_lmms_bridge(generate, score, candidate_artifact_id=candidate_artifact_id),
        tasks=[task],
        limit=None,
        offset=0,
        num_fewshot=protocol.controls["fewshot"],
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
        fewshot_random_seed=seed,
        log_samples=True,
        bootstrap_iters=protocol.controls.get("bootstrap_iters", 2000),
        batch_size=1,
    )
    inspect.signature(evaluator.simple_evaluate).bind(**arguments)
    try:
        raw = evaluator.simple_evaluate(**arguments)
    except Exception as error:
        failed = EvaluationRun(protocol, candidate_artifact_id, environment=environment)
        for identity in protocol.expected_ids:
            failed.add(
                EvaluationRecord(
                    identity, "error", error="official_evaluator_" + type(error).__name__
                )
            )
        failed.save(target)
        return failed

    atomic_json(target / "official-results.json", _json_value(raw))
    run = EvaluationRun(protocol, candidate_artifact_id, environment=environment)
    for sample in raw.get("samples", {}).get(task_name, []):
        identity = f"{task_name}:{sample['doc_id']}"
        value = sample.get(protocol.metric)
        if type(value) not in {float, int} or not np.isfinite(value):
            record = EvaluationRecord(
                identity, "error", error="missing_or_non_scalar_official_metric"
            )
        else:
            record = EvaluationRecord(
                identity,
                "ok",
                {protocol.metric: float(value)},
                details={
                    "doc_hash": sample.get("doc_hash"),
                    "raw_sample_sha256": digest_json(_json_value(sample)),
                },
            )
        run.add(record)
    run.finalize().save(target)
    return run


class GymnasiumFactory:
    def __init__(self, *, version, env_kwargs=None):
        self.version, self.env_kwargs = version, dict(env_kwargs or {})

    def __call__(self, case):
        if metadata.version("gymnasium") != self.version:
            raise ValueError("Gymnasium version differs from evaluation protocol")
        gymnasium = importlib.import_module("gymnasium")
        if ":" in case.task_id:
            raise ValueError("Implicit Python module environment registration is disabled")
        return gymnasium.make(case.task_id, **self.env_kwargs)


class LiberoFactory:
    def __init__(self, *, suite, source_pin, protocol, grant, camera_size=128):
        self.suite, self.source_pin, self.protocol, self.grant = suite, source_pin, protocol, grant
        self.camera_size = camera_size

    def __call__(self, case):
        if self.source_pin.module != "libero.libero.envs.env_wrapper":
            raise ValueError("Expected pinned LIBERO environment wrapper")
        environment_module = self.source_pin.load(self.protocol, self.grant)
        from libero.libero import benchmark, get_libero_path

        task_suite = benchmark.get_benchmark_dict()[self.suite]()
        task_index = int(case.task_id)
        task = task_suite.get_task(task_index)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        if _file_hash(bddl) != self.protocol.controls["bddl_hashes"][case.task_id]:
            raise ValueError("LIBERO BDDL differs from task manifest")
        index = (case.options or {})["init_state_index"]
        initial = task_suite.get_task_init_states(task_index)[index]
        if "sha256:" + digest_json(_json_value(initial)) != case.initial_state_id:
            raise ValueError("LIBERO initial state differs from frozen episode")
        native = environment_module.OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=self.camera_size,
            camera_widths=self.camera_size,
        )

        class Wrapper:
            def reset(self, *, seed, options=None):
                native.seed(seed)
                native.reset()
                return native.set_init_state(initial), {
                    "language": task.language,
                    "success": bool(native.check_success()),
                }

            def step(self, action):
                observation, reward, done, info = native.step(action)
                info = dict(info)
                info["success"] = bool(native.check_success())

                return observation, reward, bool(done), False, info

            def close(self):
                native.close()

        return Wrapper()


def normalize_swebench_report(protocol, candidate_artifact_id, report, *, environment):
    """Keep missing results, infrastructure errors, and empty patches in the complete denominator."""
    if protocol.metric != "resolved":
        raise ValueError("SWE-bench primary metric must be resolved")
    expected = set(protocol.expected_ids)
    groups = {
        name: set(report.get(name, ()))
        for name in (
            "resolved_ids",
            "unresolved_ids",
            "error_ids",
            "empty_patch_ids",
            "incomplete_ids",
        )
    }
    if any(values - expected for values in groups.values()) or groups["resolved_ids"] & set.union(
        *(values for name, values in groups.items() if name != "resolved_ids")
    ):
        raise ValueError("Official SWE-bench report has unknown or contradictory outcomes")
    run = EvaluationRun(protocol, candidate_artifact_id, environment=environment)
    for identity in protocol.expected_ids:
        if identity in groups["resolved_ids"]:
            record = EvaluationRecord(identity, "ok", {"resolved": 1.0})
        elif identity in groups["error_ids"] or identity in groups["incomplete_ids"]:
            record = EvaluationRecord(identity, "error", error="official_execution_incomplete")
        elif identity in groups["unresolved_ids"] or identity in groups["empty_patch_ids"]:
            record = EvaluationRecord(identity, "ok", {"resolved": 0.0})
        else:
            record = EvaluationRecord(identity, "error", error="missing_official_result")
        run.add(record)
    return run.finalize()


def evaluate_swebench(
    protocol,
    candidate_artifact_id,
    *,
    dataset_json,
    predictions_jsonl,
    source_pin,
    grant,
    environment,
    output_directory,
    max_workers=1,
    timeout_seconds=900,
):

    for effect in ("official_evaluator", "docker", "untrusted_code"):
        grant.require(protocol, effect)
    if source_pin.module != "swebench.harness.run_evaluation":
        raise ValueError("Expected official SWE-bench harness module")
    dataset, predictions = (
        Path(dataset_json).resolve(strict=True),
        Path(predictions_jsonl).resolve(strict=True),
    )
    if (
        dataset.suffix != ".json"
        or predictions.suffix != ".jsonl"
        or _file_hash(dataset) != protocol.dataset_fingerprint
    ):
        raise ValueError("SWE-bench requires the pinned local JSON dataset and JSONL predictions")
    records = read_json(dataset)
    if tuple(row["instance_id"] for row in records) != protocol.expected_ids:
        raise ValueError("SWE-bench dataset does not match the full ordered task manifest")

    if any("@sha256:" not in row.get("image", "") for row in records):
        raise ValueError(
            "Current SWE-bench evaluator requires explicitly pinned container image digests"
        )
    from ..agents.events import strict_loads

    proposed = [
        strict_loads(line) for line in predictions.read_text(encoding="utf-8").splitlines() if line
    ]
    ids = [item["instance_id"] for item in proposed]
    if (
        len(set(ids)) != len(ids)
        or set(ids) - set(protocol.expected_ids)
        or any(item.get("model_name_or_path") != candidate_artifact_id for item in proposed)
    ):
        raise ValueError("Prediction identity differs from candidate/full task manifest")
    if (
        type(max_workers) is not int
        or max_workers < 1
        or type(timeout_seconds) is not int
        or timeout_seconds < 1
    ):
        raise ValueError("Official evaluator resource limits must be explicit positive integers")
    target = Path(output_directory).resolve()
    target.mkdir(parents=True, exist_ok=False)
    module = source_pin.load(protocol, grant)
    arguments = {
        "dataset_name": str(dataset),
        "split": protocol.controls["split"],
        "instance_ids": list(protocol.expected_ids),
        "predictions_path": str(predictions),
        "max_workers": max_workers,
        "open_file_limit": 4096,
        "run_id": "aster-" + uuid.uuid4().hex,
        "timeout": timeout_seconds,
        "rewrite_reports": False,
        "modal": False,
        "report_dir": str(target),
    }
    inspect.signature(module.main).bind(**arguments)
    report_path = Path(module.main(**arguments)).resolve(strict=True)
    if report_path.parent != target:
        raise ValueError("Official report escaped the selected output directory")
    raw = read_json(report_path)
    run = normalize_swebench_report(protocol, candidate_artifact_id, raw, environment=environment)
    run.save(target)
    return run
