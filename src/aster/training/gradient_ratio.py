"""Globally normalized gradient-ratio weighting without rerunning the model."""

from copy import deepcopy
import math

import torch

from .portable import logical_tensors


SOURCE = "https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/losses/vqperceptual.py"


class GradientRatioRegistry:
    def __init__(self, engine):
        self.engine = engine
        self.policies, self.records = {}, {}

    def register(
        self,
        name,
        *,
        role,
        reference_term,
        target_term,
        parameter,
        eps,
        min_ratio,
        max_ratio,
        multiplier,
    ):
        engine = self.engine
        error, policy = None, None
        try:
            if engine._busy or engine._failed:
                raise RuntimeError("Gradient ratio registration requires a valid idle Trainer")
            if not isinstance(name, str) or not name or name in self.policies:
                raise ValueError("Gradient ratio requires a unique nonempty name")
            if role not in engine.roles or not engine.roles[role].trainable:
                raise ValueError("Gradient ratio role must own an optimizer")
            for value in (reference_term, target_term, parameter):
                if not isinstance(value, str) or not value:
                    raise ValueError(
                        "Gradient ratio terms and parameter FQN must be nonempty strings"
                    )
            if reference_term == target_term:
                raise ValueError("Gradient ratio requires two different loss terms")
            if any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in (eps, min_ratio, max_ratio, multiplier)
            ):
                raise ValueError("Gradient ratio coefficients must be finite numbers")
            if eps <= 0 or min_ratio < 0 or max_ratio < min_ratio or multiplier < 0:
                raise ValueError("Invalid gradient ratio coefficient range")
            config = engine.parallel.config
            if any(
                getattr(config, key, 1) != 1
                for key in (
                    "tensor_parallel",
                    "pipeline_parallel",
                    "context_parallel",
                    "expert_parallel",
                    "expert_tensor_parallel",
                    "gtp_remat",
                )
            ):
                raise ValueError("Gradient ratio currently supports pure DP x ZeRO0-3 only")
            if any(
                value["role"] == role and value["target_term"] == target_term
                for value in self.policies.values()
            ):
                raise ValueError("Each target loss term has exactly one gradient ratio owner")
            policy = dict(
                name=name,
                role=role,
                reference_term=reference_term,
                target_term=target_term,
                parameter=parameter,
                eps=float(eps),
                min_ratio=float(min_ratio),
                max_ratio=float(max_ratio),
                multiplier=float(multiplier),
                formula="clip(norm(global_mean_reference_gradient)/(norm(global_mean_target_gradient)+eps))*multiplier",
                source=SOURCE,
                version=1,
            )
            self._entry(policy)
            self._domains(policy)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        engine._collective_error(error)
        declarations = engine.parallel.world.gather_objects(policy)
        if any(value != declarations[0] for value in declarations):
            raise ValueError("Gradient ratio policy differs across ranks")
        self.policies[name] = policy
        self.records[name] = None

    def _entry(self, policy):
        engine = self.engine
        entries = [
            entry
            for entry in logical_tensors(engine.roles[policy["role"]].model, engine.parallel)
            if entry.name == policy["parameter"] and entry.parameter
        ]
        if len(entries) != 1 or not entries[0].tensor.requires_grad:
            raise ValueError("Gradient ratio parameter must identify one trainable logical FQN")
        entry = entries[0]
        if (
            entry.group.ranks != engine.parallel.world.ranks
            or entry.tp_dimension is not None
            or entry.ep_dimension is not None
            or hasattr(entry.tensor, "_aster_extra_gradient_group")
        ):
            raise ValueError("Gradient ratio requires an unambiguous pure-DP parameter owner")
        return entry

    def _domains(self, policy):
        for term in (policy["reference_term"], policy["target_term"]):
            if (
                self.engine.loss_groups.get(term, self.engine.replica_group).ranks
                != self.engine.parallel.world.ranks
            ):
                raise ValueError(
                    "Gradient ratio loss terms must use the same full DP normalization domain"
                )

    def describe(self):
        return deepcopy([self.policies[name] for name in sorted(self.policies)])

    def probe_terms(self, role, schema):
        weights = {name: weight for name, _, weight in schema}
        needed = set()
        for policy in self.policies.values():
            if policy["role"] != role:
                continue
            self._domains(policy)
            if policy["reference_term"] not in weights or policy["target_term"] not in weights:
                raise ValueError("Registered gradient ratio loss terms are missing from this phase")
            if weights[policy["target_term"]] > 0 and policy["multiplier"] > 0:
                needed.update((policy["reference_term"], policy["target_term"]))
        return needed

    def resolve(self, role, parameters, buffers, terms, *, already_reduced):

        engine = self.engine
        weights = {key: value["weight"] for key, value in terms.items()}
        pending = {}
        indexes = {id(parameter): index for index, parameter in enumerate(parameters)}
        for name in sorted(self.policies):
            policy = self.policies[name]
            if policy["role"] != role:
                continue
            reference, target = policy["reference_term"], policy["target_term"]
            active = (
                weights[target] > 0
                and policy["multiplier"] > 0
                and terms[target]["denominator"] > 0
            )
            record = {
                "name": name,
                "role": role,
                "role_updates": engine.roles[role].updates + 1,
                "active": active,
                "reference_norm": None,
                "target_norm": None,
                "ratio": None,
                "multiplier": policy["multiplier"],
                "outer_weight": weights[target],
                "effective_weight": 0.0,
            }
            if active:
                entry = self._entry(policy)
                index = indexes[id(entry.tensor)]
                norms = []
                error = None
                for term in (reference, target):
                    if terms[term]["denominator"] <= 0 or buffers[term][index] is None:
                        error = f"Active gradient ratio has no global gradient/count for {term} at {entry.name}"
                engine._collective_error(error)
                for term in (reference, target):
                    gradient = buffers[term][index].detach().to(engine.device).clone()
                    if engine.zero_stage in (0, 1) and not already_reduced:
                        entry.group.all_reduce(gradient)
                    gradient.div_(terms[term]["denominator"])
                    if engine.zero_stage in (2, 3):
                        width = gradient.numel()
                        total = math.prod(entry.shape)
                        valid = max(0, min(width, total - entry.group.rank * width))
                        squared = gradient.flatten()[:valid].double().square().sum()
                        entry.group.all_reduce(squared)
                    else:
                        squared = gradient.double().square().sum()
                    norms.append(float(squared.sqrt()))
                if not all(math.isfinite(value) for value in norms):
                    return weights, {}, True
                ratio = min(
                    policy["max_ratio"],
                    max(policy["min_ratio"], norms[0] / (norms[1] + policy["eps"])),
                )
                coefficient = weights[target] * policy["multiplier"] * ratio
                if not math.isfinite(coefficient):
                    return weights, {}, True
                record.update(
                    reference_norm=norms[0],
                    target_norm=norms[1],
                    ratio=ratio,
                    effective_weight=coefficient,
                )
            weights[target] = record["effective_weight"]
            pending[name] = record
        return weights, pending, False

    def validate_records(self, state, updates):
        if not isinstance(state, dict) or set(state) != set(self.policies):
            raise ValueError("Gradient ratio checkpoint policy records differ")
        for name, record in state.items():
            if record is None:
                continue
            policy = self.policies[name]
            expected = {
                "name",
                "role",
                "role_updates",
                "active",
                "reference_norm",
                "target_norm",
                "ratio",
                "multiplier",
                "outer_weight",
                "effective_weight",
            }
            if (
                not isinstance(record, dict)
                or set(record) != expected
                or record["name"] != name
                or record["role"] != policy["role"]
                or type(record["role_updates"]) is not int
                or record["role_updates"] < 1
                or record["role_updates"] != updates[policy["role"]]
                or type(record["active"]) is not bool
                or record["multiplier"] != policy["multiplier"]
            ):
                raise ValueError("Invalid gradient ratio successful update record")
            values = (
                ("outer_weight", "effective_weight", "reference_norm", "target_norm", "ratio")
                if record["active"]
                else ("outer_weight", "effective_weight")
            )
            if any(
                type(record[key]) not in (int, float)
                or not math.isfinite(record[key])
                or record[key] < 0
                for key in values
            ):
                raise ValueError(
                    "Gradient ratio record must contain finite nonnegative coefficients"
                )
            if record["active"]:
                ratio = min(
                    policy["max_ratio"],
                    max(
                        policy["min_ratio"],
                        record["reference_norm"] / (record["target_norm"] + policy["eps"]),
                    ),
                )
                if (
                    ratio != record["ratio"]
                    or record["effective_weight"]
                    != record["outer_weight"] * record["multiplier"] * ratio
                ):
                    raise ValueError("Gradient ratio record disagrees with registered formula")
            elif record["effective_weight"] != 0 or any(
                record[key] is not None for key in ("reference_norm", "target_norm", "ratio")
            ):
                raise ValueError("Inactive gradient ratio must not invent probe norms")
        return deepcopy(state)
