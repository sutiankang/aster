import importlib.metadata

import numpy as np
import pytest
import torch

from aster.planning import RootOutput, RecurrentOutput, muzero_policy, gumbel_muzero_policy


@pytest.mark.parametrize("algorithm", ["muzero", "gumbel"])
def test_native_search_tree_against_installed_official_mctx(algorithm, record_property):
    mctx = pytest.importorskip(
        "mctx", reason="Official mctx/JAX oracle is an explicitly separate optional environment"
    )
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    record_property("mctx_version", importlib.metadata.version("mctx"))
    record_property("jax_version", importlib.metadata.version("jax"))
    record_property(
        "scope", "deterministic finite vector embedding, zero root noise, 13 simulations, depth 2"
    )
    prior = np.array([[0.3, -0.8, 0.9, 0.1], [-0.2, 1.2, 0.3, -0.4]], np.float32)
    value = np.array([0.2, -0.4], np.float32)
    embedding = np.array([[0.0, 0.0], [2.0, 0.0]], np.float32)
    invalid = np.array([[False, True, False, False], [True, False, False, False]])

    def native_step(action, state):
        x, depth = state[:, 0], state[:, 1]
        following = 0.3 * x + 0.7 * (action + 1)
        logits = torch.stack(
            (0.1 * following, 0.2 - 0.3 * following, 0.5 + 0.1 * action, -0.2 + 0.02 * depth), -1
        )
        return RecurrentOutput(
            0.6 * (action == 2) + 0.1 * x,
            torch.full_like(x, 0.8),
            logits,
            0.8 * torch.tanh(following + 0.1 * action),
            torch.stack((following, depth + 1), -1),
        )

    def official_step(params, key, action, state):
        del params, key
        x, depth = state[:, 0], state[:, 1]
        following = 0.3 * x + 0.7 * (action + 1)
        logits = jnp.stack(
            (0.1 * following, 0.2 - 0.3 * following, 0.5 + 0.1 * action, -0.2 + 0.02 * depth), -1
        )
        return mctx.RecurrentFnOutput(
            reward=0.6 * (action == 2) + 0.1 * x,
            discount=jnp.full_like(x, 0.8),
            prior_logits=logits,
            value=0.8 * jnp.tanh(following + 0.1 * action),
        ), jnp.stack((following, depth + 1), -1)

    native_root = RootOutput(
        torch.from_numpy(prior), torch.from_numpy(value), torch.from_numpy(embedding)
    )
    official_root = mctx.RootFnOutput(
        prior_logits=jnp.asarray(prior), value=jnp.asarray(value), embedding=jnp.asarray(embedding)
    )
    native_options = (
        {"dirichlet_fraction": 0.0, "tie_break_epsilon": 0.0}
        if algorithm == "muzero"
        else {"gumbel_scale": 0.0, "max_num_considered_actions": 3}
    )
    official_options = (
        {"dirichlet_fraction": 0.0}
        if algorithm == "muzero"
        else {"gumbel_scale": 0.0, "max_num_considered_actions": 3}
    )
    native_fn = muzero_policy if algorithm == "muzero" else gumbel_muzero_policy
    official_fn = mctx.muzero_policy if algorithm == "muzero" else mctx.gumbel_muzero_policy
    actual = native_fn(
        native_root,
        native_step,
        num_simulations=13,
        max_depth=2,
        generator=torch.Generator().manual_seed(313),
        invalid_actions=torch.from_numpy(invalid),
        **native_options,
    )
    expected = official_fn(
        None,
        jax.random.PRNGKey(313),
        official_root,
        official_step,
        num_simulations=13,
        max_depth=2,
        invalid_actions=jnp.asarray(invalid),
        **official_options,
    )
    for name in (
        "node_visits",
        "children_visits",
        "children_index",
        "parents",
        "action_from_parent",
    ):
        np.testing.assert_array_equal(
            getattr(actual.search_tree, name).numpy(),
            np.asarray(getattr(expected.search_tree, name)),
        )
    for name in (
        "raw_values",
        "node_values",
        "children_rewards",
        "children_discounts",
        "children_values",
    ):
        np.testing.assert_allclose(
            getattr(actual.search_tree, name).numpy(),
            np.asarray(getattr(expected.search_tree, name)),
            atol=3e-6,
            rtol=3e-6,
        )
    np.testing.assert_allclose(
        actual.action_weights.numpy(), np.asarray(expected.action_weights), atol=3e-6, rtol=3e-6
    )

    if algorithm == "gumbel":
        np.testing.assert_array_equal(actual.action.numpy(), np.asarray(expected.action))
