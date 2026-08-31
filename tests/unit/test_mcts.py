import math

import numpy as np
import pytest
import torch

from aster.planning import RootOutput, RecurrentOutput, gumbel_muzero_policy, muzero_policy
from aster.planning.mcts import considered_visits, completed_qvalues, parent_and_siblings_q


def _root(dtype=torch.float64):
    return RootOutput(
        torch.tensor([[0.3, -0.8, 0.9, 0.1], [-0.2, 1.2, 0.3, -0.4]], dtype=dtype),
        torch.tensor([0.2, -0.4], dtype=dtype),
        torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=dtype),
    )


def _recurrent(action, embedding):
    state, depth = embedding[:, 0], embedding[:, 1]
    following = 0.3 * state + 0.7 * (action + 1)
    prior = torch.stack(
        (0.1 * following, 0.2 - 0.3 * following, 0.5 + 0.1 * action, -0.2 + 0.02 * depth), -1
    )
    return RecurrentOutput(
        0.6 * (action == 2) + 0.1 * state,
        torch.full_like(state, 0.8),
        prior,
        0.8 * torch.tanh(following + 0.1 * action),
        torch.stack((following, depth + 1), -1),
    )


def _scalar_step(action, embedding):
    state, depth = embedding
    following = 0.3 * state + 0.7 * (action + 1)
    return (
        0.6 * int(action == 2) + 0.1 * state,
        0.8,
        [0.1 * following, 0.2 - 0.3 * following, 0.5 + 0.1 * action, -0.2 + 0.02 * depth],
        0.8 * math.tanh(following + 0.1 * action),
        [following, depth + 1],
    )


def _softmax(values):
    result = np.exp(np.asarray(values) - np.max(values))
    return result / result.sum()


def _scalar_qtransform(node, *, gumbel):
    counts = np.array([edge["count"] for edge in node["edges"]], dtype=np.int64)
    values = np.array([edge["reward"] + edge["discount"] * edge["value"] for edge in node["edges"]])
    if not gumbel:
        bounds = [node["value"], *values[counts > 0]]
        lo, hi = min(bounds), max(bounds)
        return np.array(
            [
                (value - lo) / max(hi - lo, 1e-8) if count else 0.0
                for value, count in zip(values, counts)
            ]
        )
    probabilities = _softmax(node["logits"])
    visited = [index for index, count in enumerate(counts) if count]
    normalizer = sum(probabilities[index] for index in visited)
    average = (
        sum(probabilities[index] / normalizer * values[index] for index in visited)
        if visited
        else 0.0
    )
    mixed = (node["raw"] + sum(counts) * average) / (1 + sum(counts))
    completed = np.array([value if count else mixed for value, count in zip(values, counts)])
    return (
        (completed - completed.min())
        / max(completed.max() - completed.min(), 1e-8)
        * (0.1 * (50 + max(counts)))
    )


def _scalar_schedule(candidates, budget):
    if candidates == 1:
        return list(range(budget))
    levels, current, records, result = (
        math.ceil(math.log2(candidates)),
        candidates,
        [0] * candidates,
        [],
    )
    while len(result) < budget:
        for _ in range(max(1, budget // (levels * current))):
            for index in range(current):
                result.append(records[index])
                records[index] += 1
        current = max(2, current // 2)
    return result[:budget]


def _scalar_search(logits, value, embedding, invalid, budget, depth, *, gumbel):

    def node(prior, estimate, state):
        return {
            "logits": np.asarray(prior),
            "value": estimate,
            "raw": estimate,
            "state": state,
            "count": 1,
            "edges": [
                {"child": None, "reward": 0.0, "discount": 0.0, "value": 0.0, "count": 0}
                for _ in prior
            ],
        }

    logits = np.asarray(logits) - max(logits)
    logits[invalid] = np.finfo(logits.dtype).min
    tree = {0: node(logits, value, embedding)}
    schedule = _scalar_schedule(min(3, int((~invalid).sum())), budget)
    for simulation in range(budget):
        current, path = 0, []
        for level in range(depth):
            n = tree[current]
            counts = np.array([edge["count"] for edge in n["edges"]])
            transformed = _scalar_qtransform(n, gumbel=gumbel)
            if not gumbel:
                bonus = math.sqrt(n["count"]) * (1.25 + math.log((n["count"] + 19653.0) / 19652.0))
                score = transformed + bonus * _softmax(n["logits"]) / (1 + counts)
            elif not level:
                score = np.maximum(-1e9, n["logits"] - max(n["logits"]) + transformed)
                score[counts != schedule[simulation]] = -np.inf
            else:
                score = _softmax(n["logits"] + transformed) - counts / (1 + sum(counts))
            if not level:
                score[invalid] = -np.inf
            action = int(np.argmax(score))
            edge = n["edges"][action]
            path.append((current, action))
            if edge["child"] is None or level + 1 == depth:
                break
            current = edge["child"]
        reward, discount, prior, estimate, state = _scalar_step(action, n["state"])
        child = simulation + 1 if edge["child"] is None else edge["child"]
        if child not in tree:
            tree[child] = node(prior, estimate, state)
        else:
            tree[child].update(logits=np.array(prior), value=estimate, raw=estimate, state=state)
            tree[child]["count"] += 1
        edge.update(child=child, reward=reward, discount=discount)
        propagated = estimate
        for parent, action in reversed(path):
            n, edge = tree[parent], tree[parent]["edges"][action]
            propagated = edge["reward"] + edge["discount"] * propagated
            n["value"] = (n["value"] * n["count"] + propagated) / (n["count"] + 1)
            n["count"] += 1
            edge["count"] += 1
            edge["value"] = tree[edge["child"]]["value"]
    return tree


@pytest.mark.parametrize("algorithm", ["muzero", "gumbel"])
@pytest.mark.parametrize("budget,depth", [(1, 1), (7, 1), (19, 3), (25, 25)])
def test_batched_search_every_edge_matches_independent_scalar_tree(algorithm, budget, depth):
    torch.set_num_threads(1)
    root = _root()
    invalid = torch.tensor([[False, True, False, False], [True, False, False, False]])
    calls = []

    def recurrent(action, embedding):
        calls.append((action.shape, embedding.shape))
        return _recurrent(action, embedding)

    function = muzero_policy if algorithm == "muzero" else gumbel_muzero_policy
    kwargs = (
        {"dirichlet_fraction": 0.0, "tie_break_epsilon": 0.0}
        if algorithm == "muzero"
        else {"gumbel_scale": 0.0, "max_num_considered_actions": 3}
    )
    actual = function(
        root,
        recurrent,
        generator=torch.Generator().manual_seed(31),
        num_simulations=budget,
        invalid_actions=invalid,
        max_depth=depth,
        **kwargs,
    )
    assert calls == [(torch.Size([2]), torch.Size([2, 2]))] * budget
    tree = actual.search_tree
    for row in range(2):
        reference = _scalar_search(
            root.prior_logits[row].numpy(),
            float(root.value[row]),
            root.embedding[row].tolist(),
            invalid[row].numpy(),
            budget,
            depth,
            gumbel=algorithm == "gumbel",
        )
        assert int(tree.node_visits[row, 0]) == budget + 1
        for index, node in reference.items():
            assert int(tree.node_visits[row, index]) == node["count"]
            assert abs(float(tree.node_values[row, index]) - node["value"]) < 2e-7
            for action, edge in enumerate(node["edges"]):
                assert int(tree.children_visits[row, index, action]) == edge["count"]
                assert int(tree.children_index[row, index, action]) == (
                    -1 if edge["child"] is None else edge["child"]
                )
                assert abs(float(tree.children_values[row, index, action]) - edge["value"]) < 2e-7
    assert torch.all(actual.action_weights[invalid] == 0)
    assert not invalid[torch.arange(2), actual.action].any()
    torch.testing.assert_close(actual.action_weights.sum(-1), torch.ones(2, dtype=torch.float64))
    if algorithm == "muzero":
        torch.testing.assert_close(actual.action_weights, tree.summary()["visit_probs"])
    else:
        complete = completed_qvalues(
            tree.qvalues(0),
            tree.children_visits[:, 0],
            tree.children_prior_logits[:, 0],
            tree.raw_values[:, 0],
        )
        expected = (tree.children_prior_logits[:, 0] + complete).softmax(-1)
        torch.testing.assert_close(actual.action_weights, expected)


def test_qtransform_mixed_raw_value_and_parent_range_have_distinct_formulas():
    q = torch.tensor([[2.0, 9.0, -3.0, 7.0]])
    visits = torch.tensor([[3, 0, 1, 0]])
    logits = torch.tensor([[0.0, 0.8, -0.7, 0.2]])
    raw = torch.tensor([1.5])
    result = completed_qvalues(q, visits, logits, raw, rescale_values=False)
    probabilities = logits.softmax(-1)[0]
    average = (probabilities[0] * 2 - probabilities[2] * 3) / (probabilities[0] + probabilities[2])
    mixed = (1.5 + 4 * average) / 5
    torch.testing.assert_close(result, torch.tensor([[2.0, mixed, -3.0, mixed]]) * 5.3)
    torch.testing.assert_close(
        parent_and_siblings_q(q, visits, raw), torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    )


@pytest.mark.parametrize(
    "candidates,budget,expected",
    [
        (1, 5, (0, 1, 2, 3, 4)),
        (4, 16, (0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5)),
        (3, 10, (0, 0, 0, 1, 1, 2, 2, 3, 3, 4)),
        (16, 5, (0, 0, 0, 0, 0)),
    ],
)
def test_sequential_halving_author_schedule_vectors(candidates, budget, expected):
    assert considered_visits(candidates, budget) == expected


def test_max_depth_reexpands_existing_leaf_and_backup_is_mean_not_max():
    root = RootOutput(torch.zeros(1, 1), torch.zeros(1), torch.zeros(1, 1))
    calls = 0

    def recurrent(action, embedding):
        nonlocal calls
        calls += 1
        return RecurrentOutput(
            torch.ones(1),
            torch.full((1,), 0.5),
            torch.zeros(1, 1),
            torch.tensor([float(calls)]),
            embedding + 1,
        )

    result = muzero_policy(
        root,
        recurrent,
        num_simulations=3,
        max_depth=1,
        generator=torch.Generator().manual_seed(1),
        dirichlet_fraction=0.0,
    )
    tree = result.search_tree
    assert calls == 3 and tree.node_visits[0, 1] == 3
    assert tree.node_values[0, 1] == 3 and tree.raw_values[0, 1] == 3
    assert tree.node_values[0, 0] == 1.5  # (0 + 1.5 + 2 + 2.5)/4
    assert tree.qvalues(0)[0, 0] == 2.5
    assert int((tree.node_visits > 0).sum()) == 2


@pytest.mark.parametrize("function", [muzero_policy, gumbel_muzero_policy])
def test_explicit_rng_state_reproduces_search_and_does_not_touch_global_rng(function):
    root = _root(torch.float32)
    generator = torch.Generator().manual_seed(974)
    saved, global_state = generator.get_state().clone(), torch.get_rng_state().clone()
    first = function(root, _recurrent, num_simulations=13, generator=generator)
    generator.set_state(saved)
    second = function(root, _recurrent, num_simulations=13, generator=generator)
    torch.testing.assert_close(first.action, second.action, atol=0, rtol=0)
    torch.testing.assert_close(first.action_weights, second.action_weights, atol=0, rtol=0)
    torch.testing.assert_close(
        first.search_tree.children_visits, second.search_tree.children_visits, atol=0, rtol=0
    )
    torch.testing.assert_close(first.rng_state, generator.get_state(), atol=0, rtol=0)
    torch.testing.assert_close(torch.get_rng_state(), global_state, atol=0, rtol=0)


@pytest.mark.parametrize(
    "invalid", ["mask", "shape", "value", "budget", "depth", "memory", "generator"]
)
def test_fail_fast_before_callback_or_random_consumption(invalid):
    root = _root()
    generator = torch.Generator().manual_seed(31)
    before = generator.get_state().clone()
    kwargs = dict(num_simulations=10, generator=generator)
    if invalid == "mask":
        kwargs["invalid_actions"] = torch.ones(2, 4, dtype=torch.bool)
    if invalid == "shape":
        root = RootOutput(root.prior_logits, root.value, root.embedding[:, :0])
    if invalid == "value":
        root = RootOutput(root.prior_logits, torch.tensor([float("nan"), 0.0]), root.embedding)
    if invalid == "budget":
        kwargs["num_simulations"] = 0
    if invalid == "depth":
        kwargs["max_depth"] = 0
    if invalid == "memory":
        kwargs["max_tree_bytes"] = 32
    if invalid == "generator":
        kwargs["generator"] = None

    def never(*args):
        raise AssertionError("Validation must precede recurrent execution")

    with pytest.raises((ValueError, TypeError)):
        muzero_policy(root, never, **kwargs)
    torch.testing.assert_close(before, generator.get_state(), atol=0, rtol=0)


def test_recurrent_errors_are_propagated_and_signed_discount_flips_value():
    root = RootOutput(torch.zeros(1, 1), torch.zeros(1), torch.zeros(1, 1))

    def recurrent(action, embedding):
        return RecurrentOutput(
            torch.zeros(1), -torch.ones(1), torch.zeros(1, 1), torch.tensor([2.0]), embedding
        )

    result = muzero_policy(
        root, recurrent, num_simulations=1, generator=torch.Generator().manual_seed(1)
    )
    assert result.search_tree.qvalues(0).item() == -2 and result.search_tree.node_values[0, 0] == -1

    def bad(*args):
        raise RuntimeError("recurrent failure")

    with pytest.raises(RuntimeError, match="recurrent failure"):
        gumbel_muzero_policy(root, bad, num_simulations=3, generator=torch.Generator())


def test_search_does_not_build_autograd_graph_or_modify_input_tensors():
    root = _root()
    root.prior_logits.requires_grad_()
    root.embedding.requires_grad_()
    root.value.requires_grad_()
    before = tuple(
        value.detach().clone() for value in (root.prior_logits, root.value, root.embedding)
    )
    result = gumbel_muzero_policy(root, _recurrent, num_simulations=4, generator=torch.Generator())
    assert (
        not result.action_weights.requires_grad and not result.search_tree.node_values.requires_grad
    )
    for value, expected in zip((root.prior_logits, root.value, root.embedding), before):
        torch.testing.assert_close(value, expected, atol=0, rtol=0)


def test_finite_inputs_that_overflow_backup_are_not_returned_as_valid_plan():
    root = RootOutput(torch.zeros(1, 1), torch.zeros(1), torch.zeros(1, 1))

    def recurrent(action, embedding):
        return RecurrentOutput(
            torch.tensor([3e38]), torch.ones(1), torch.zeros(1, 1), torch.tensor([3e38]), embedding
        )

    with pytest.raises(FloatingPointError, match="backup overflow"):
        muzero_policy(root, recurrent, num_simulations=1, generator=torch.Generator())
