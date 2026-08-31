"""Native language-model scoring and optional official lm-eval task adapters."""

import torch
from ..inference.sampling import SamplingConfig, sample_token


class LanguageEvaluator:
    def __init__(self, model, tokenizer, *, max_length, prefix_token_id=None):
        if max_length < 2:
            raise ValueError("Evaluation context must contain at least two tokens")
        self.model, self.tokenizer, self.max_length = model, tokenizer, max_length
        self.prefix_token_id = (
            tokenizer.eos_token_id if prefix_token_id is None else prefix_token_id
        )

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def score_tokens(self, context, continuation):
        if not context:
            context = [self.prefix_token_id]
        sequence = list(context) + list(continuation)
        if len(continuation) > self.max_length:
            raise ValueError(
                "Continuation exceeds benchmark max length; use rolling likelihood for documents"
            )
        if not continuation:
            return 0.0, True

        start = max(0, len(sequence) - self.max_length - 1)
        inputs = torch.tensor([sequence[start:-1]], device=self.device)
        was_training = self.model.training
        self.model.eval()
        try:
            logits = (
                self.model(input_ids=inputs, use_cache=False)
                .logits[0, -len(continuation) :]
                .float()
            )
        finally:
            self.model.train(was_training)
        targets = torch.tensor(continuation, device=logits.device)
        logp = logits.log_softmax(-1).gather(-1, targets[:, None]).sum()
        return float(logp), bool((logits.argmax(-1) == targets).all())

    def score(self, context, continuation):
        spaces = len(context) - len(context.rstrip())
        if spaces:
            context, continuation = context[:-spaces], context[-spaces:] + continuation
        context_ids = self.tokenizer.encode(context, add_special_tokens=False)
        whole = self.tokenizer.encode(context + continuation, add_special_tokens=False)
        if whole[: len(context_ids)] != context_ids:
            raise ValueError(
                "Tokenizer merges across scoring boundary; provide audited token-level boundary instead of guessing"
            )
        return self.score_tokens(context_ids, whole[len(context_ids) :])

    def rolling(self, text):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        total = 0.0
        position = 0
        while position < len(ids):
            count = min(self.max_length, len(ids) - position)
            start = max(0, position + count - self.max_length - 1)
            context = (
                ([self.prefix_token_id] + ids[:position]) if start == 0 else ids[start:position]
            )
            total += self.score_tokens(context, ids[position : position + count])[0]
            position += count
        return total

    @torch.no_grad()
    def generate(
        self, context, *, max_new_tokens=32, until=(), temperature=0.0, top_p=1.0, top_k=0, seed=0
    ):
        ids = self.tokenizer.encode(context, add_special_tokens=False) or [self.prefix_token_id]
        if len(ids) + max_new_tokens > self.max_length:
            raise ValueError(
                "Generation exceeds configured evaluation context; set an explicit benchmark truncation policy"
            )
        settings = SamplingConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            eos_token_ids=(self.tokenizer.eos_token_id,),
        )
        rng = torch.Generator().manual_seed(seed)
        generated = []
        state = None
        inputs = torch.tensor([ids], device=self.device)
        was_training = self.model.training
        self.model.eval()
        try:
            for index in range(max_new_tokens):
                output = self.model(input_ids=inputs, state=state, use_cache=True)
                state = output.state
                sampled = sample_token(
                    output.logits[0, -1],
                    settings,
                    rng,
                    context_ids=ids + generated,
                    generated_count=index,
                )
                if sampled.token_id == self.tokenizer.eos_token_id:
                    break
                generated.append(sampled.token_id)
                text = self.tokenizer.decode(generated)
                endings = [text.find(stop) for stop in until if stop and stop in text]
                if endings:
                    return text[: min(endings)]

                inputs = torch.tensor(
                    [[sampled.token_id] if state is not None else ids + generated],
                    device=self.device,
                )
        finally:
            self.model.train(was_training)
        return self.tokenizer.decode(generated)


def lm_eval_adapter(evaluator):

    from lm_eval.api.model import LM

    class AsterLM(LM):
        def __init__(self):
            super().__init__()
            self._device = evaluator.device

        def loglikelihood(self, requests):
            return [evaluator.score(*request.args) for request in requests]

        def loglikelihood_rolling(self, requests):
            return [evaluator.rolling(request.args[0]) for request in requests]

        def generate_until(self, requests):
            outputs = []
            for request in requests:
                prompt, kwargs = request.args
                kwargs = dict(kwargs)
                maximum = kwargs.pop("max_gen_toks", 32)
                until = kwargs.pop("until", ())
                if isinstance(until, str):
                    until = (until,)
                do_sample = kwargs.pop("do_sample", False)
                temperature = kwargs.pop("temperature", 1.0 if do_sample else 0.0)
                if not do_sample and temperature != 0:
                    raise ValueError("Conflicting greedy/sampling settings")
                outputs.append(
                    evaluator.generate(
                        prompt,
                        max_new_tokens=maximum,
                        until=until,
                        temperature=temperature,
                        **kwargs,
                    )
                )
            return outputs

    return AsterLM()


def evaluate_official_language(
    evaluator, *, tasks, output_directory, limit=None, fewshot=0, seed=0, task_manager=None
):

    if (
        not isinstance(tasks, (list, tuple))
        or not tasks
        or any(not isinstance(task, str) or not task for task in tasks)
        or len(set(tasks)) != len(tasks)
    ):
        raise ValueError("Official tasks must be explicit, nonempty, distinct names")
    return _run_official(
        evaluator,
        tasks=tasks,
        names=tasks,
        output_directory=output_directory,
        limit=limit,
        fewshot=fewshot,
        seed=seed,
        task_manager=task_manager,
    )


def _validate_controls(limit, fewshot, seed):
    if type(fewshot) is not int or fewshot < 0 or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError(
            "fewshot and seed must be nonnegative integers; seed must fit NumPy uint32"
        )

    if limit is not None and not (
        (type(limit) is int and limit > 0) or (type(limit) is float and 0 < limit <= 1)
    ):
        raise ValueError("limit must be positive sample count or a fraction in (0,1]")


def _json_tree(value):

    import math
    import numpy as np

    if isinstance(value, np.generic):
        return _json_tree(value.item())
    if isinstance(value, np.ndarray):
        return _json_tree(value.tolist())
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_tree(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _json_tree(item) for key, item in value.items()}
    raise ValueError("Official language evidence contains an unsupported or nonfinite JSON value")


def _run_official(evaluator, *, tasks, names, output_directory, limit, fewshot, seed, task_manager):
    from pathlib import Path
    from lm_eval import simple_evaluate
    from importlib.metadata import version
    from ..core import atomic_json

    _validate_controls(limit, fewshot, seed)
    target = Path(output_directory)
    if target.exists():
        raise FileExistsError("Choose a fresh official evaluation run directory")
    result = simple_evaluate(
        model=lm_eval_adapter(evaluator),
        tasks=list(tasks),
        num_fewshot=fewshot,
        limit=limit,
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
        fewshot_random_seed=seed,
        log_samples=True,
        confirm_run_unsafe_code=False,
        task_manager=task_manager,
    )
    if not isinstance(result, dict) or not result.get("results") or "samples" not in result:
        raise ValueError("Official evaluator did not return both metrics and raw sample evidence")
    result = _json_tree(result)
    target.mkdir(parents=True)

    atomic_json(target / "official-results.json", result)
    atomic_json(
        target / "run.json",
        {
            "tasks": list(names),
            "limit": limit,
            "subset_only": limit is not None,
            "fewshot": fewshot,
            "seed": seed,
            "fewshot_random_seed": seed,
            "lm_eval_version": version("lm-eval"),
            "unsafe_task_code_authorized": False,
        },
    )
    return result


def evaluate_language_artifact(
    store,
    artifact_id,
    *,
    task_name,
    dataset_revision,
    output_directory,
    max_length,
    metric="acc",
    filter_name="none",
    limit=None,
    fewshot=0,
    seed=0,
    device="cpu",
    task_manager=None,
):

    from pathlib import Path
    from importlib.metadata import version
    import inspect
    from lm_eval.tasks import TaskManager
    from lm_eval.evaluator_utils import get_sample_size
    from ..core import digest_json, file_digest
    from ..recipes import load_predictor_artifact
    from .protocol import ComparisonProtocol, EvaluationRun, EvaluationRecord

    _validate_controls(limit, fewshot, seed)
    if (
        not isinstance(task_name, str)
        or not task_name
        or not isinstance(dataset_revision, str)
        or not dataset_revision
    ):
        raise ValueError("Name the exact task and dataset revision")
    if (
        metric not in {"acc", "acc_norm", "exact_match"}
        or not isinstance(filter_name, str)
        or not filter_name
    ):
        raise ValueError(
            "Artifact bridge requires a named filter and samplewise accuracy/EM metric"
        )
    target = Path(output_directory)
    if target.exists():
        raise FileExistsError("Choose a fresh artifact evaluation directory")
    candidate = store.get(artifact_id)
    if candidate.kind != "token_predictor":
        raise ValueError("Expected an immutable token_predictor artifact")
    model, tokenizer = load_predictor_artifact(candidate, device=device)
    evaluator = LanguageEvaluator(model, tokenizer, max_length=max_length)
    manager = task_manager if task_manager is not None else TaskManager()
    if type(manager) is not TaskManager:
        raise ValueError(
            "Use the actual official TaskManager, not a substituted evaluator callback"
        )
    loaded = manager.load([task_name])
    if loaded["groups"] or set(loaded["tasks"]) != {task_name}:
        raise ValueError(
            "Evaluate one leaf task with one complete sample denominator, not an expanded tag/group"
        )
    task = loaded["tasks"][task_name]
    if task.get_config("unsafe_code") or getattr(task, "UNSAFE_CODE", False):
        raise PermissionError(
            "Executable-code benchmarks require a separate sandbox/grant protocol"
        )
    sample_limit = get_sample_size(task, limit)
    documents = {
        str(index): digest_json(_json_tree(document))
        for index, document in task.doc_iterator(rank=0, limit=sample_limit, world_size=1)
    }
    if not documents:
        raise ValueError("No evaluation documents selected")
    controls = dict(
        task_name=task_name,
        metric=metric,
        filter=filter_name,
        requested_fewshot=fewshot,
        seed=seed,
        fewshot_random_seed=seed,
        limit=limit,
        subset_only=limit is not None,
        max_length=max_length,
        dataset_revision=dataset_revision,
        task_config=_json_tree(task.dump_config()),
        task_class=f"{type(task).__module__}.{type(task).__qualname__}",
        task_source_sha256=file_digest(inspect.getfile(type(task))),
        document_fingerprints=documents,
    )
    protocol = ComparisonProtocol(
        task_name,
        digest_json({"revision": dataset_revision, "documents": documents}),
        "lm-evaluation-harness",
        version("lm-eval"),
        controls,
        tuple(documents),
        metric,
    )
    result = _run_official(
        evaluator,
        tasks=[task],
        names=[task_name],
        output_directory=target / "official",
        limit=limit,
        fewshot=fewshot,
        seed=seed,
        task_manager=manager,
    )
    run = EvaluationRun(
        protocol,
        candidate.id,
        environment={
            "torch": torch.__version__,
            "device": str(evaluator.device),
            "lm_eval": version("lm-eval"),
            "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        },
    )
    for sample in result["samples"].get(task_name, []):
        if sample.get("filter") != filter_name:
            continue
        sample_id = str(sample["doc_id"])
        if sample_id not in documents or digest_json(sample["doc"]) != documents[sample_id]:
            raise ValueError("Official result document differs from the pre-evaluation manifest")
        value = sample.get(metric)
        if type(value) not in {float, int, bool} or not 0 <= value <= 1:
            run.add(EvaluationRecord(sample_id, "error", error="missing_or_invalid_sample_metric"))
        else:
            run.add(
                EvaluationRecord(
                    sample_id,
                    "ok",
                    {metric: float(value)},
                    details={
                        "doc_hash": sample.get("doc_hash"),
                        "prompt_hash": sample.get("prompt_hash"),
                        "target_hash": sample.get("target_hash"),
                        "official_sample": sample,
                    },
                )
            )
    run.finalize()

    store.get(candidate.id)
    report = run.save(target / "normalized")
    evidence = store.publish(
        target,
        kind="evaluation",
        metadata={
            "protocol_id": protocol.id,
            "candidate_artifact_id": candidate.id,
            "subset_only": limit is not None,
        },
        parents=(candidate.id,),
    )
    return {"artifact_id": evidence.id, "report": str(report), "summary": run.summary()}
