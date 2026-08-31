"""DSpark token CE, probability L1, and acceptance-confidence BCE with separate denominators."""

import math
import torch
from torch import nn
import torch.nn.functional as F

from ..core import LossTerm, LossBundle
from ..models.dspark import DSparkDraft
from ..models.dspark_gemma4 import Gemma4DSparkDraft
from ..training.parallel import Group


def dspark_loss_terms(
    output,
    *,
    ce_weight=0.1,
    l1_weight=0.9,
    confidence_weight=1.0,
    decay_gamma=4.0,
    denominator_offset=1e-6,
):
    logits = output.draft_logits.float()
    weights = output.eval_mask.float()
    if decay_gamma is not None and decay_gamma > 0:
        weights = weights * torch.exp(
            -torch.arange(logits.shape[-2], device=logits.device).float() / decay_gamma
        )
    denominator = weights.sum().double() + denominator_offset
    ce = F.cross_entropy(
        logits.flatten(0, -2), output.target_ids.flatten(), reduction="none"
    ).reshape_as(weights)
    terms = [LossTerm((ce * weights).sum(), denominator, "weighted_token", "dspark_ce", ce_weight)]
    if l1_weight > 0 or output.confidence_pred is not None:
        if output.aligned_target_logits is None:
            raise ValueError("DSpark L1/confidence supervision requires aligned target logits")
        target = output.aligned_target_logits.detach().float().softmax(-1)
        draft = logits.softmax(-1)
        distance = (draft - target).abs().sum(-1)
        if l1_weight > 0:
            terms.append(
                LossTerm(
                    (distance * weights).sum(),
                    denominator,
                    "weighted_token",
                    "dspark_l1",
                    l1_weight,
                )
            )
        if output.confidence_pred is not None:
            acceptance = (1 - 0.5 * distance.detach()).clamp(0, 1)
            confidence = F.binary_cross_entropy_with_logits(
                output.confidence_pred.float(), acceptance, reduction="none"
            )
            terms.append(
                LossTerm(
                    (confidence * weights).sum(),
                    denominator,
                    "weighted_token",
                    "dspark_confidence",
                    confidence_weight,
                )
            )
    return LossBundle(tuple(terms))


class DSparkObjective(nn.Module):
    def __init__(
        self,
        *,
        teacher_identity,
        vocabulary_fingerprint,
        feature_cache_ids=(),
        feature_cache_store=None,
        normalization_world_size=1,
        accumulation_steps=1,
        ce_weight=0.1,
        l1_weight=0.9,
        confidence_weight=1.0,
        decay_gamma=4.0,
        denominator_epsilon=1e-6,
        normalization_profile="official_microbatch_mean",
        empty_window_policy="official_step",
        normalization_group=None,
    ):
        super().__init__()
        if (
            not isinstance(teacher_identity, str)
            or len(teacher_identity) != 64
            or any(c not in "0123456789abcdef" for c in teacher_identity)
            or teacher_identity == "0" * 64
        ):
            raise ValueError("DSpark requires the initialized native teacher weight identity")
        if not isinstance(vocabulary_fingerprint, str) or not vocabulary_fingerprint:
            raise ValueError("DSpark requires explicit vocabulary identity")
        self.teacher_identity, self.vocabulary_fingerprint = (
            teacher_identity,
            vocabulary_fingerprint,
        )
        self.feature_cache_ids = tuple(feature_cache_ids)
        if len(set(self.feature_cache_ids)) != len(self.feature_cache_ids) or any(
            not isinstance(x, str) or len(x) != 64 or any(c not in "0123456789abcdef" for c in x)
            for x in self.feature_cache_ids
        ):
            raise ValueError("DSpark feature cache IDs must be distinct content hashes")
        if self.feature_cache_ids and feature_cache_store is None:
            raise ValueError(
                "Bound DSpark cache training requires the actual artifact store for data verification"
            )
        self.feature_cache_store = feature_cache_store
        if any(type(v) is not int or v < 1 for v in (normalization_world_size, accumulation_steps)):
            raise ValueError(
                "DSpark normalization topology/window must be explicit positive integers"
            )
        if normalization_profile not in {"official_microbatch_mean", "global_window"}:
            raise ValueError("Unknown DSpark normalization profile")
        if empty_window_policy not in {"official_step", "skip"}:
            raise ValueError("Unknown DSpark empty-window policy")
        if normalization_group is None:
            if normalization_world_size != 1:
                raise ValueError(
                    "Distributed DSpark normalization requires its existing explicit DP group"
                )
            normalization_group = Group()
        if (
            not isinstance(normalization_group, Group)
            or normalization_group.size != normalization_world_size
        ):
            raise ValueError("DSpark normalization group and declared world size differ")
        values = (ce_weight, l1_weight, confidence_weight, denominator_epsilon)
        if (
            any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values)
            or denominator_epsilon <= 0
            or sum(values[:3]) == 0
        ):
            raise ValueError("Invalid DSpark loss coefficients")
        if decay_gamma is not None and (
            type(decay_gamma) not in (int, float) or not math.isfinite(decay_gamma)
        ):
            raise ValueError("DSpark position decay must be finite or None")
        self.normalization_world_size, self.accumulation_steps = (
            normalization_world_size,
            accumulation_steps,
        )
        self.normalization_group = normalization_group
        self.normalization_profile, self.empty_window_policy = (
            normalization_profile,
            empty_window_policy,
        )

        self._window_has_targets = None
        self.settings = dict(
            ce_weight=ce_weight,
            l1_weight=l1_weight,
            confidence_weight=confidence_weight,
            decay_gamma=decay_gamma,
        )
        self.denominator_epsilon = denominator_epsilon

    def config_dict(self):
        return dict(
            type="dspark",
            **self.settings,
            denominator_epsilon=self.denominator_epsilon,
            teacher_identity=self.teacher_identity,
            vocabulary_fingerprint=self.vocabulary_fingerprint,
            feature_cache_ids=list(self.feature_cache_ids),
            normalization_world_size=self.normalization_world_size,
            accumulation_steps=self.accumulation_steps,
            reference="deepspec_005e03b",
            targets="detached_teacher_features_and_head",
            normalization_profile=self.normalization_profile,
            empty_window_policy=self.empty_window_policy,
            normalization_group_ranks=list(self.normalization_group.ranks),
            denominator_precision=(
                "fp32_dp_sum_epsilon_per_microbatch"
                if self.normalization_profile == "official_microbatch_mean"
                else "fp64_window_sum_single_global_epsilon"
            ),
        )

    def preflight_microbatches(self, model, batches):
        self._window_has_targets = None
        if (
            not isinstance(model, (DSparkDraft, Gemma4DSparkDraft))
            or len(batches) != self.accumulation_steps
        ):
            raise ValueError(
                "DSpark objective requires native draft and declared microbatch window"
            )
        c = model.config
        if model.teacher_identity != self.teacher_identity:
            raise ValueError("DSpark model teacher binding changed")
        if self.settings["confidence_weight"] > 0 and not c.enable_confidence_head:
            raise ValueError("Positive confidence weight requires the native confidence head")
        if (
            self.settings["l1_weight"] > 0 or c.enable_confidence_head
        ) and not c.freeze_embedding_head:
            raise ValueError("DSpark target supervision requires frozen target embedding/head")
        window_has_targets = False
        for batch in batches:
            allowed = {
                "input_ids",
                "target_hidden_states",
                "loss_mask",
                "target_last_hidden_states",
                "anchor_positions",
                "block_keep_mask",
                "teacher_identity",
                "vocabulary_fingerprint",
                "feature_cache_id",
                "cache_sample_ids",
            }
            if set(batch) - allowed:
                raise ValueError("Unknown DSpark batch fields")
            if (
                batch.get("teacher_identity") != self.teacher_identity
                or batch.get("vocabulary_fingerprint") != self.vocabulary_fingerprint
            ):
                raise ValueError("DSpark cached features belong to another teacher or vocabulary")
            if (
                self.feature_cache_ids
                and batch.get("feature_cache_id") not in self.feature_cache_ids
            ):
                raise ValueError("DSpark batch is not bound to an approved feature cache")
            if self.feature_cache_ids:
                from ..data.dspark import DSparkFeatureCache

                cache = DSparkFeatureCache(self.feature_cache_store, batch["feature_cache_id"])
                if (
                    cache.contract["teacher_identity"] != self.teacher_identity
                    or cache.contract["vocabulary_fingerprint"] != self.vocabulary_fingerprint
                    or cache.contract["draft_config"] != model.config.to_dict()
                ):
                    raise ValueError(
                        "DSpark feature cache contract differs from this draft/teacher"
                    )
                expected = cache.batch(batch.get("cache_sample_ids", ()))

                for key, value in expected.items():
                    if isinstance(value, torch.Tensor) and (
                        key not in batch
                        or batch[key].dtype != value.dtype
                        or not torch.equal(batch[key].detach().cpu(), value)
                    ):
                        raise ValueError(
                            "DSpark batch tensors differ from the immutable feature cache"
                        )
            model.validate_batch(
                batch["input_ids"],
                batch["target_hidden_states"],
                batch["loss_mask"],
                batch.get("target_last_hidden_states"),
            )
            if (self.settings["l1_weight"] > 0 or c.enable_confidence_head) and batch.get(
                "target_last_hidden_states"
            ) is None:
                raise ValueError("DSpark aligned final target features are required")
            if "anchor_positions" in batch:
                model.validate_anchors(
                    batch["input_ids"],
                    batch["loss_mask"],
                    batch["anchor_positions"],
                    batch.get("block_keep_mask"),
                )
                window_has_targets |= bool(batch["block_keep_mask"].any())
            elif "block_keep_mask" in batch:
                raise ValueError("Keep mask requires explicit anchors")
            else:
                valid = batch["loss_mask"].bool()
                window_has_targets |= bool((valid[:, :-1] & valid[:, 1:]).any())
        self._window_has_targets = window_has_targets
        return batches

    def forward(self, model, batch):
        if self.empty_window_policy == "skip" and self._window_has_targets is None:
            raise ValueError("DSpark skip policy requires complete-window preflight before forward")
        output = model(
            **{
                k: v
                for k, v in batch.items()
                if k
                not in {
                    "teacher_identity",
                    "vocabulary_fingerprint",
                    "feature_cache_id",
                    "cache_sample_ids",
                }
            }
        )
        terms = dspark_loss_terms(output, **self.settings, denominator_offset=0.0).terms
        window_active = True
        if (
            self.normalization_profile == "official_microbatch_mean"
            or self.empty_window_policy == "skip"
        ):
            totals = torch.stack(
                (
                    terms[0].denominator.float(),
                    terms[0].denominator.new_tensor(
                        float(bool(self._window_has_targets)), dtype=torch.float32
                    ),
                )
            )
            self.normalization_group.all_reduce(totals)
            if self.empty_window_policy == "skip":
                window_active = bool(totals[1] > 0)
        if self.normalization_profile == "official_microbatch_mean":
            denominator = totals[0] + self.denominator_epsilon
            return LossBundle(
                tuple(
                    LossTerm(
                        term.numerator / denominator * self.normalization_world_size,
                        term.denominator.new_tensor(int(window_active), dtype=torch.int64),
                        "global_microbatch",
                        term.name,
                        term.weight,
                    )
                    for term in terms
                )
            )

        offset = (
            self.denominator_epsilon / (self.normalization_world_size * self.accumulation_steps)
            if window_active
            else 0.0
        )
        return LossBundle(
            tuple(
                LossTerm(
                    term.numerator, term.denominator + offset, term.unit, term.name, term.weight
                )
                for term in terms
            )
        )


class DSparkMethod:
    def __init__(self, engine, *, vocabulary_fingerprint, state_name="dspark", **settings):
        error = None
        try:
            if any(
                getattr(engine.parallel.config, key, 1) != 1
                for key in (
                    "tensor_parallel",
                    "pipeline_parallel",
                    "context_parallel",
                    "gtp_remat",
                    "expert_parallel",
                    "expert_tensor_parallel",
                )
            ):
                raise ValueError("DSpark training currently admits pure DP/ZeRO profiles")
            if not isinstance(engine.model, (DSparkDraft, Gemma4DSparkDraft)):
                raise ValueError(
                    "DSpark Method requires an explicit native Qwen3/Gemma4 DSparkDraft"
                )
            if not isinstance(state_name, str) or not state_name or state_name in engine.states:
                raise ValueError("DSpark state name must be nonempty and unregistered")
            objective = DSparkObjective(
                teacher_identity=engine.model.teacher_identity,
                vocabulary_fingerprint=vocabulary_fingerprint,
                normalization_world_size=engine.parallel.world.size,
                normalization_group=engine.parallel.dp,
                accumulation_steps=engine.accumulation_steps,
                **settings,
            )
            self._validate_loss_domains(engine, objective)
            configuration = objective.config_dict()
        except Exception as exc:
            error, configuration = f"{type(exc).__name__}: {exc}", None
        declarations = engine.parallel.world.gather_objects((error, configuration))
        if any(x[0] for x in declarations) or any(x[1] != configuration for x in declarations):
            raise ValueError("DSpark setup differs across ranks: " + str(declarations))
        self.engine, self.objective, self.state_name, self.updates = (
            engine,
            objective,
            state_name,
            0,
        )
        engine.register_state(state_name, self)

    def update(self, batches):
        error = None
        try:
            self._validate_loss_domains(self.engine, self.objective)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        errors = self.engine.parallel.world.gather_objects(error)
        if any(errors):
            raise ValueError("DSpark loss normalization domain changed: " + str(errors))
        result = self.engine.phase(self.state_name, objective=self.objective, microbatches=batches)
        if result.updated:
            self.updates += 1
        return result

    @staticmethod
    def _validate_loss_domains(engine, objective):

        ranks = objective.normalization_group.ranks
        for name in ("dspark_ce", "dspark_l1", "dspark_confidence"):
            if engine.loss_groups.get(name, engine.replica_group).ranks != ranks:
                raise ValueError(
                    "DSpark loss terms require the declared complete DP normalization group"
                )

    def state_dict(self):
        return dict(configuration=self.objective.config_dict(), updates=self.updates)

    def load_state_dict(self, state):
        if (
            not isinstance(state, dict)
            or set(state) != {"configuration", "updates"}
            or state["configuration"] != self.objective.config_dict()
            or type(state["updates"]) is not int
            or state["updates"] < 0
        ):
            raise ValueError("DSpark method checkpoint differs")
        self.updates = state["updates"]
