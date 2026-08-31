"""Batched PUCT and Gumbel MuZero search over Torch tensors.

Formula and scheduling reference: google-deepmind/mctx (Apache-2.0).
Copyright 2021 DeepMind Technologies Limited.
https://github.com/google-deepmind/mctx/tree/main/mctx/_src

References:
https://github.com/google-deepmind/mctx/tree/main/mctx/_src"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch


@dataclass(frozen=True)
class RootOutput:
    prior_logits: torch.Tensor
    value: torch.Tensor
    embedding: torch.Tensor


@dataclass(frozen=True)
class RecurrentOutput:
    reward: torch.Tensor
    discount: torch.Tensor
    prior_logits: torch.Tensor
    value: torch.Tensor
    embedding: torch.Tensor


@dataclass
class SearchTree:
    node_visits: torch.Tensor
    raw_values: torch.Tensor
    node_values: torch.Tensor
    parents: torch.Tensor
    action_from_parent: torch.Tensor
    children_index: torch.Tensor
    children_prior_logits: torch.Tensor
    children_visits: torch.Tensor
    children_rewards: torch.Tensor
    children_discounts: torch.Tensor
    children_values: torch.Tensor
    embeddings: torch.Tensor
    root_invalid_actions: torch.Tensor
    root_gumbel: torch.Tensor

    def qvalues(self, indices=0):
        if isinstance(indices, int):
            return (
                self.children_rewards[:, indices]
                + self.children_discounts[:, indices] * self.children_values[:, indices]
            )
        rows = torch.arange(len(self.node_values), device=self.node_values.device)
        return (
            self.children_rewards[rows, indices]
            + self.children_discounts[rows, indices] * self.children_values[rows, indices]
        )

    def summary(self):
        visits = self.children_visits[:, 0]
        return {
            "visit_counts": visits.clone(),
            "visit_probs": visits.to(self.node_values.dtype)
            / visits.sum(-1, keepdim=True).clamp_min(1),
            "value": self.node_values[:, 0].clone(),
            "qvalues": self.qvalues(0),
        }


@dataclass(frozen=True)
class SearchOutput:
    action: torch.Tensor
    action_weights: torch.Tensor
    search_tree: SearchTree
    algorithm: str
    rng_state: torch.Tensor


def considered_visits(num_considered: int, num_simulations: int) -> tuple[int, ...]:

    if (
        type(num_considered) is not int
        or num_considered < 1
        or type(num_simulations) is not int
        or num_simulations < 0
    ):
        raise ValueError(
            "Sequential-halving dimensions must be nonnegative integers with at least one candidate"
        )
    if num_considered == 1:
        return tuple(range(num_simulations))
    levels = math.ceil(math.log2(num_considered))
    visit = [0] * num_considered
    active, sequence = num_considered, []
    while len(sequence) < num_simulations:
        repetitions = max(1, num_simulations // (levels * active))
        for _ in range(repetitions):
            sequence.extend(visit[:active])
            visit[:active] = [value + 1 for value in visit[:active]]
        active = max(2, active // 2)
    return tuple(sequence[:num_simulations])


def parent_and_siblings_q(qvalues, visits, node_value, *, epsilon=1e-8):

    known = visits > 0
    safe = torch.where(known, qvalues, node_value[..., None])
    minimum = torch.minimum(node_value, safe.min(-1).values)
    maximum = torch.maximum(node_value, safe.max(-1).values)
    completed = torch.where(known, qvalues, minimum[..., None])
    return (completed - minimum[..., None]) / (maximum - minimum).clamp_min(epsilon)[..., None]


def completed_qvalues(
    qvalues,
    visits,
    prior_logits,
    raw_value,
    *,
    value_scale=0.1,
    maxvisit_init=50.0,
    rescale_values=True,
    use_mixed_value=True,
    epsilon=1e-8,
):

    known = visits > 0
    prior = prior_logits.softmax(-1).clamp_min(torch.finfo(prior_logits.dtype).tiny)
    probability_mass = torch.where(known, prior, 0.0).sum(-1, keepdim=True)
    weighted = torch.where(
        known, prior * qvalues / torch.where(known, probability_mass, 1.0), 0.0
    ).sum(-1)
    counts = visits.sum(-1).to(qvalues.dtype)

    denominator = 1 + counts
    baseline = (
        raw_value / denominator + weighted * (counts / denominator)
        if use_mixed_value
        else raw_value
    )
    completed = torch.where(known, qvalues, baseline[..., None])
    if rescale_values:
        minimum, maximum = (
            completed.min(-1, keepdim=True).values,
            completed.max(-1, keepdim=True).values,
        )
        completed = (completed - minimum) / (maximum - minimum).clamp_min(epsilon)
    return completed * (maxvisit_init + visits.max(-1).values)[..., None] * value_scale


def _finite(name, value, shape, *, device):
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != tuple(shape)
        or value.device != device
    ):
        raise ValueError(f"{name} must have shape {tuple(shape)} on {device}")
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain finite floating values")


def _validate(
    root, recurrent_fn, num_simulations, generator, invalid_actions, max_depth, max_tree_bytes
):
    if not isinstance(root, RootOutput) or not callable(recurrent_fn):
        raise TypeError("RootOutput and a callable recurrent_fn are required")
    if not isinstance(root.prior_logits, torch.Tensor) or root.prior_logits.ndim != 2:
        raise ValueError("Root prior_logits must be B,A")
    batch, actions = root.prior_logits.shape
    if min(batch, actions) < 1:
        raise ValueError("Search needs nonempty root batch/action space")
    if type(num_simulations) is not int or num_simulations < 1:
        raise ValueError("Search simulation budget must be a positive integer")
    depth = num_simulations if max_depth is None else max_depth
    if type(depth) is not int or depth < 1:
        raise ValueError("Search max_depth must be a positive integer")
    device = root.prior_logits.device
    if (
        not isinstance(root.embedding, torch.Tensor)
        or root.embedding.ndim != 2
        or root.embedding.shape[1] < 1
    ):
        raise ValueError(
            "This native search requires a B,D tensor embedding, not an arbitrary PyTree"
        )
    _finite("root.prior_logits", root.prior_logits, (batch, actions), device=device)
    _finite("root.value", root.value, (batch,), device=device)
    _finite("root.embedding", root.embedding, (batch, root.embedding.shape[1]), device=device)
    if not isinstance(generator, torch.Generator) or torch.device(generator.device) != device:
        raise ValueError("An explicit torch.Generator on the search device is required")
    if invalid_actions is None:
        invalid_actions = torch.zeros((batch, actions), device=device, dtype=torch.bool)
    if (
        not isinstance(invalid_actions, torch.Tensor)
        or invalid_actions.shape != (batch, actions)
        or invalid_actions.dtype != torch.bool
        or invalid_actions.device != device
    ):
        raise ValueError("invalid_actions must be a boolean B,A mask on the search device")
    if invalid_actions.all(-1).any():
        raise ValueError("A root with no valid actions must be handled as terminal before search")

    scalar_bytes = (
        8 if root.value.dtype == torch.float64 or root.prior_logits.dtype == torch.float64 else 4
    )
    per_node = (
        2 * scalar_bytes
        + 3 * 8
        + actions * (4 * scalar_bytes + 2 * 8)
        + root.embedding.shape[1] * root.embedding.element_size()
    )
    estimated = batch * (num_simulations + 1) * per_node + batch * actions * (scalar_bytes + 1)
    if type(max_tree_bytes) is not int or max_tree_bytes < 1 or estimated > max_tree_bytes:
        raise ValueError(
            f"Search tree requires at least {estimated} bytes, exceeding the explicit storage budget"
        )
    dtype = torch.float64 if scalar_bytes == 8 else torch.float32
    return invalid_actions.clone(), depth, dtype


def _masked_logits(logits, invalid):
    return (logits - logits.max(-1, keepdim=True).values).masked_fill(
        invalid, torch.finfo(logits.dtype).min
    )


def _instantiate(root, prior, invalid, simulations, dtype, gumbel):
    batch, actions = prior.shape
    nodes = simulations + 1
    device = prior.device

    def zeros(shape):
        return torch.zeros(shape, dtype=dtype, device=device)

    def indices(shape, fill=0):
        return torch.full(shape, fill, dtype=torch.int64, device=device)

    bn, bna = (batch, nodes), (batch, nodes, actions)
    tree = SearchTree(
        indices(bn),
        zeros(bn),
        zeros(bn),
        indices(bn, -1),
        indices(bn, -1),
        indices(bna, -1),
        zeros(bna),
        indices(bna),
        zeros(bna),
        zeros(bna),
        zeros(bna),
        torch.zeros(
            (batch, nodes, root.embedding.shape[1]), dtype=root.embedding.dtype, device=device
        ),
        invalid,
        gumbel,
    )
    tree.node_visits[:, 0] = 1
    tree.raw_values[:, 0] = root.value
    tree.node_values[:, 0] = root.value
    tree.children_prior_logits[:, 0] = prior
    tree.embeddings[:, 0] = root.embedding
    return tree


def _scores_considered(considered, gumbel, logits, qvalues, visits):
    centered = logits - logits.max(-1, keepdim=True).values
    score = (gumbel + centered + qvalues).clamp_min(-1e9)
    return score.masked_fill(visits != considered[..., None], -torch.inf)


def _search(
    root,
    recurrent_fn,
    *,
    num_simulations,
    max_depth,
    generator,
    invalid,
    dtype,
    prior,
    algorithm,
    gumbel,
    pb_c_init=1.25,
    pb_c_base=19652.0,
    tie_break_epsilon=1e-7,
    max_num_considered_actions=16,
    value_scale=0.1,
    maxvisit_init=50.0,
    max_tree_bytes=512 * 1024 * 1024,
):
    del max_tree_bytes
    tree = _instantiate(root, prior, invalid, num_simulations, dtype, gumbel)
    batch, actions = prior.shape
    device = prior.device
    all_rows = torch.arange(batch, device=device)
    schedule = None
    if algorithm == "gumbel_muzero":
        considered = (~invalid).sum(-1).clamp_max(max_num_considered_actions).tolist()
        schedule = torch.tensor(
            [considered_visits(count, num_simulations) for count in considered], device=device
        )

    def choose(rows, nodes, depth, simulation):
        visits = tree.children_visits[rows, nodes]
        logits = tree.children_prior_logits[rows, nodes]
        qvalues = (
            tree.children_rewards[rows, nodes]
            + tree.children_discounts[rows, nodes] * tree.children_values[rows, nodes]
        )
        if algorithm == "muzero":
            count = tree.node_visits[rows, nodes].to(dtype)
            coefficient = pb_c_init + torch.log((count + pb_c_base + 1) / pb_c_base)
            exploration = (
                count.sqrt()[:, None] * coefficient[:, None] * logits.softmax(-1) / (1 + visits)
            )
            score = (
                parent_and_siblings_q(qvalues, visits, tree.node_values[rows, nodes]) + exploration
            )
            if tie_break_epsilon:
                score = score + tie_break_epsilon * torch.rand(
                    score.shape, device=device, dtype=dtype, generator=generator
                )
        else:
            transformed = completed_qvalues(
                qvalues,
                visits,
                logits,
                tree.raw_values[rows, nodes],
                value_scale=value_scale,
                maxvisit_init=maxvisit_init,
            )
            if depth == 0:
                score = _scores_considered(
                    schedule[rows, simulation], tree.root_gumbel[rows], logits, transformed, visits
                )
            else:
                score = (logits + transformed).softmax(-1) - visits / (
                    1 + visits.sum(-1, keepdim=True)
                )
        if depth == 0:
            score = score.masked_fill(invalid[rows], -torch.inf)
        if (
            torch.isnan(score).any()
            or torch.isposinf(score).any()
            or not torch.isfinite(score).any(-1).all()
        ):
            raise FloatingPointError(
                "Nonfinite/empty search action scores; no fallback action is selected"
            )
        return score.argmax(-1)

    for simulation in range(num_simulations):
        current = torch.zeros(batch, device=device, dtype=torch.int64)
        parent, action = current.clone(), current.clone()
        active = torch.ones(batch, device=device, dtype=torch.bool)
        for depth in range(max_depth):
            rows = active.nonzero(as_tuple=True)[0]
            selected = choose(rows, current[rows], depth, simulation)
            parent[rows], action[rows] = current[rows], selected
            following = tree.children_index[rows, current[rows], selected]
            continues = (following >= 0) & (depth + 1 < max_depth)
            active[rows] = continues
            current[rows[continues]] = following[continues]
            if not active.any():
                break

        next_index = tree.children_index[all_rows, parent, action]
        next_index = torch.where(next_index < 0, simulation + 1, next_index)
        output = recurrent_fn(action.clone(), tree.embeddings[all_rows, parent].clone())
        if not isinstance(output, RecurrentOutput):
            raise TypeError("recurrent_fn must return RecurrentOutput")
        for name in ("reward", "discount", "value"):
            _finite("recurrent." + name, getattr(output, name), (batch,), device=device)
        _finite("recurrent.prior_logits", output.prior_logits, (batch, actions), device=device)
        _finite("recurrent.embedding", output.embedding, root.embedding.shape, device=device)
        if output.embedding.dtype != root.embedding.dtype:
            raise ValueError("Recurrent embedding dtype must remain stable")
        if (output.discount.abs() > 1).any():
            raise ValueError("Recurrent discount must lie in [-1,1]")
        tree.children_prior_logits[all_rows, next_index] = output.prior_logits.to(dtype)
        tree.raw_values[all_rows, next_index] = output.value.to(dtype)
        tree.node_values[all_rows, next_index] = output.value.to(dtype)
        tree.node_visits[all_rows, next_index] += 1
        tree.embeddings[all_rows, next_index] = output.embedding
        tree.children_index[all_rows, parent, action] = next_index
        tree.children_rewards[all_rows, parent, action] = output.reward.to(dtype)
        tree.children_discounts[all_rows, parent, action] = output.discount.to(dtype)
        tree.parents[all_rows, next_index] = parent
        tree.action_from_parent[all_rows, next_index] = action

        current = next_index.clone()
        returned = tree.node_values[all_rows, current].clone()
        while (current != 0).any():
            rows = (current != 0).nonzero(as_tuple=True)[0]
            child = current[rows]
            parent = tree.parents[rows, child]
            action = tree.action_from_parent[rows, child]
            propagated = (
                tree.children_rewards[rows, parent, action]
                + tree.children_discounts[rows, parent, action] * returned[rows]
            )
            count = tree.node_visits[rows, parent].to(dtype)
            updated = tree.node_values[rows, parent] * (count / (count + 1)) + propagated / (
                count + 1
            )
            if not torch.isfinite(propagated).all() or not torch.isfinite(updated).all():
                raise FloatingPointError("Search backup overflow; no partial result is returned")
            tree.node_values[rows, parent] = updated
            tree.node_visits[rows, parent] += 1
            tree.children_values[rows, parent, action] = tree.node_values[rows, child]
            tree.children_visits[rows, parent, action] += 1
            returned[rows], current[rows] = propagated, parent
    return tree


@torch.no_grad()
def muzero_policy(
    root: RootOutput,
    recurrent_fn: Callable,
    *,
    num_simulations: int,
    generator: torch.Generator,
    invalid_actions=None,
    max_depth=None,
    dirichlet_fraction=0.25,
    dirichlet_alpha=0.3,
    pb_c_init=1.25,
    pb_c_base=19652.0,
    temperature=1.0,
    tie_break_epsilon=1e-7,
    max_tree_bytes=512 * 1024 * 1024,
) -> SearchOutput:

    settings = (
        dirichlet_fraction,
        dirichlet_alpha,
        pb_c_init,
        pb_c_base,
        temperature,
        tie_break_epsilon,
    )
    if (
        not all(math.isfinite(value) for value in settings)
        or not 0 <= dirichlet_fraction <= 1
        or min(dirichlet_alpha, pb_c_base) <= 0
        or min(pb_c_init, temperature, tie_break_epsilon) < 0
    ):
        raise ValueError("Invalid finite PUCT/noise/temperature settings")
    invalid, depth, dtype = _validate(
        root, recurrent_fn, num_simulations, generator, invalid_actions, max_depth, max_tree_bytes
    )
    prior = root.prior_logits.to(dtype)
    if dirichlet_fraction:
        noise = torch._standard_gamma(torch.full_like(prior, dirichlet_alpha), generator=generator)
        denominator = noise.sum(-1, keepdim=True)
        if not torch.isfinite(noise).all() or (denominator <= 0).any():
            raise RuntimeError("Dirichlet sampling produced invalid mass")
        probability = (1 - dirichlet_fraction) * prior.softmax(
            -1
        ) + dirichlet_fraction * noise / denominator
        prior = probability.clamp_min(torch.finfo(dtype).tiny).log()
    prior = _masked_logits(prior, invalid)
    tree = _search(
        root,
        recurrent_fn,
        num_simulations=num_simulations,
        max_depth=depth,
        generator=generator,
        invalid=invalid,
        dtype=dtype,
        prior=prior,
        algorithm="muzero",
        gumbel=torch.zeros_like(prior),
        pb_c_init=pb_c_init,
        pb_c_base=pb_c_base,
        tie_break_epsilon=tie_break_epsilon,
    )
    weights = tree.summary()["visit_probs"]
    logits = weights.clamp_min(torch.finfo(dtype).tiny).log()
    logits = (logits - logits.max(-1, keepdim=True).values) / max(
        torch.finfo(dtype).tiny, temperature
    )

    probability = logits.softmax(-1).masked_fill(invalid, 0.0)
    action = torch.multinomial(probability, 1, generator=generator).squeeze(-1)
    return SearchOutput(action, weights, tree, "muzero", generator.get_state().clone())


@torch.no_grad()
def gumbel_muzero_policy(
    root: RootOutput,
    recurrent_fn: Callable,
    *,
    num_simulations: int,
    generator: torch.Generator,
    invalid_actions=None,
    max_depth=None,
    max_num_considered_actions=16,
    gumbel_scale=1.0,
    value_scale=0.1,
    maxvisit_init=50.0,
    max_tree_bytes=512 * 1024 * 1024,
) -> SearchOutput:

    if type(max_num_considered_actions) is not int or max_num_considered_actions < 1:
        raise ValueError("max_num_considered_actions must be a positive integer")
    if not all(
        math.isfinite(value) and value >= 0 for value in (gumbel_scale, value_scale, maxvisit_init)
    ):
        raise ValueError("Finite nonnegative Gumbel/Q-transform settings required")
    invalid, depth, dtype = _validate(
        root, recurrent_fn, num_simulations, generator, invalid_actions, max_depth, max_tree_bytes
    )
    prior = _masked_logits(root.prior_logits.to(dtype), invalid)
    if gumbel_scale:
        uniform = torch.rand(prior.shape, device=prior.device, dtype=dtype, generator=generator)
        noise = -torch.log(
            -torch.log(uniform.clamp(min=torch.finfo(dtype).tiny, max=1 - torch.finfo(dtype).eps))
        )
        gumbel = noise * gumbel_scale
    else:
        gumbel = torch.zeros_like(prior)
    tree = _search(
        root,
        recurrent_fn,
        num_simulations=num_simulations,
        max_depth=depth,
        generator=generator,
        invalid=invalid,
        dtype=dtype,
        prior=prior,
        algorithm="gumbel_muzero",
        gumbel=gumbel,
        max_num_considered_actions=max_num_considered_actions,
        value_scale=value_scale,
        maxvisit_init=maxvisit_init,
    )
    visits = tree.children_visits[:, 0]
    complete = completed_qvalues(
        tree.qvalues(0),
        visits,
        prior,
        tree.raw_values[:, 0],
        value_scale=value_scale,
        maxvisit_init=maxvisit_init,
    )
    if not torch.isfinite(complete).all():
        raise FloatingPointError("Nonfinite completed search values")
    score = _scores_considered(visits.max(-1).values, gumbel, prior, complete, visits).masked_fill(
        invalid, -torch.inf
    )
    action = score.argmax(-1)
    weights = _masked_logits(prior + complete, invalid).softmax(-1)
    return SearchOutput(action, weights, tree, "gumbel_muzero", generator.get_state().clone())
