"""Minibatch optimal transport, flow inversion, and continuous density change."""

import math
from dataclasses import dataclass

import torch

from ..core import FieldOutput


@torch.no_grad()
def exact_assignment(cost):
    """Solve equal-weight square assignment by O(n**3) augmenting paths."""
    if (
        cost.ndim != 2
        or cost.shape[0] != cost.shape[1]
        or not len(cost)
        or not torch.isfinite(cost).all()
    ):
        raise ValueError("Exact assignment requires a finite nonempty square cost matrix")
    n = len(cost)
    values = cost.detach().double().cpu().tolist()
    row_potential, column_potential = [0.0] * (n + 1), [0.0] * (n + 1)
    owner, parent = [0] * (n + 1), [0] * (n + 1)
    for row in range(1, n + 1):
        owner[0], column = row, 0
        distance, visited = [float("inf")] * (n + 1), [False] * (n + 1)
        while True:
            visited[column] = True
            active_row, delta, next_column = owner[column], float("inf"), 0
            for candidate in range(1, n + 1):
                if visited[candidate]:
                    continue
                reduced = (
                    values[active_row - 1][candidate - 1]
                    - row_potential[active_row]
                    - column_potential[candidate]
                )
                if reduced < distance[candidate]:
                    distance[candidate], parent[candidate] = reduced, column
                if distance[candidate] < delta:
                    delta, next_column = distance[candidate], candidate
            for candidate in range(n + 1):
                if visited[candidate]:
                    row_potential[owner[candidate]] += delta
                    column_potential[candidate] -= delta
                else:
                    distance[candidate] -= delta
            column = next_column
            if owner[column] == 0:
                break
        while column:
            previous = parent[column]
            owner[column], column = owner[previous], previous
    assignment = [0] * n
    for column in range(1, n + 1):
        assignment[owner[column] - 1] = column - 1
    return torch.tensor(assignment, device=cost.device)


@torch.no_grad()
def sinkhorn_plan(cost, *, regularization=0.05, iterations=1000, tolerance=1e-7):
    if cost.ndim != 2 or min(cost.shape) < 1 or not torch.isfinite(cost).all():
        raise ValueError("Sinkhorn requires finite pairwise costs")
    if (
        not math.isfinite(regularization)
        or not math.isfinite(tolerance)
        or regularization <= 0
        or type(iterations) is not int
        or iterations < 1
        or tolerance <= 0
    ):
        raise ValueError("Invalid Sinkhorn regularization or convergence tolerance")
    log_kernel = -cost.double() / regularization
    n, m = cost.shape
    log_a, log_b = -math.log(n), -math.log(m)
    u, v = cost.new_zeros(n, dtype=torch.float64), cost.new_zeros(m, dtype=torch.float64)
    for step in range(iterations):
        u = log_a - torch.logsumexp(log_kernel + v[None], dim=1)
        v = log_b - torch.logsumexp(log_kernel + u[:, None], dim=0)
        if step % 10 == 0 or step == iterations - 1:
            plan = (log_kernel + u[:, None] + v[None]).exp()
            error = max(
                float((plan.sum(1) - 1 / n).abs().max()), float((plan.sum(0) - 1 / m).abs().max())
            )
            if error <= tolerance:
                return plan
    raise RuntimeError(
        f"Sinkhorn did not converge: marginal error={error}; no silent uniform fallback"
    )


@torch.no_grad()
def transport_pairing(source, target, *, method="exact", regularization=0.05, generator=None):
    """Return coupled source/target tensors and indices. Reorder conditioning with
    target_indices to preserve semantic alignment."""
    if (
        source.ndim < 2
        or target.shape[1:] != source.shape[1:]
        or not len(source)
        or not len(target)
    ):
        raise ValueError("Transport endpoints need equal event shapes and nonempty batches")
    cost = torch.cdist(source.flatten(1).double(), target.flatten(1).double()).square()
    if method == "exact":
        columns = exact_assignment(cost)
        rows = torch.arange(len(source), device=source.device)
    elif method == "sinkhorn":
        plan = sinkhorn_plan(cost, regularization=regularization)
        choices = torch.multinomial(
            plan.flatten(), len(source), replacement=True, generator=generator
        )
        rows, columns = choices // len(target), choices % len(target)
    else:
        raise ValueError("Transport method must be exact or sinkhorn")
    return {
        "source": source[rows],
        "target": target[columns],
        "source_indices": rows,
        "target_indices": columns,
        "mean_cost": cost[rows, columns].mean(),
        "method": method,
    }


def _grid(times, sample):
    if (
        not isinstance(sample, torch.Tensor)
        or sample.ndim < 2
        or min(sample.shape) < 1
        or not sample.is_floating_point()
        or not torch.isfinite(sample).all()
    ):
        raise ValueError(
            "Continuous ODE samples must be nonempty finite floating tensors with a batch axis"
        )
    values = torch.as_tensor(times, device=sample.device, dtype=torch.float64)
    if values.ndim != 1 or len(values) < 2 or not torch.isfinite(values).all():
        raise ValueError("ODE time grid must have at least two finite points")
    differences = values[1:] - values[:-1]
    if not ((differences > 0).all() or (differences < 0).all()):
        raise ValueError("ODE time grid must be strictly monotone")
    return values


def _field(field, sample, time, condition):
    output = field(sample, sample.new_full((len(sample),), float(time)), condition)
    if (
        not isinstance(output, FieldOutput)
        or output.prediction_type != "velocity"
        or output.prediction.shape != sample.shape
    ):
        raise ValueError("Continuous transport requires an aligned velocity field")
    if not torch.isfinite(output.prediction).all():
        raise ValueError("Nonfinite flow velocity")
    return output.prediction


def _advance(function, x, t, dt, method):
    k1, d1 = function(x, t)
    if method == "euler":
        return x + dt * k1, dt * d1
    if method == "heun":
        k2, d2 = function(x + dt * k1, t + dt)
        return x + dt * (k1 + k2) / 2, dt * (d1 + d2) / 2
    if method == "rk4":
        k2, d2 = function(x + dt * k1 / 2, t + dt / 2)
        k3, d3 = function(x + dt * k2 / 2, t + dt / 2)
        k4, d4 = function(x + dt * k3, t + dt)
        return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6, dt * (d1 + 2 * d2 + 2 * d3 + d4) / 6
    raise ValueError("Solver must be euler/heun/rk4")


@torch.no_grad()
def integrate_flow(field, sample, times, *, condition=None, method="rk4"):
    """Integrate in the requested time direction; the field defines which endpoint
    represents data."""
    if method not in {"euler", "heun", "rk4"}:
        raise ValueError("Solver must be euler/heun/rk4")
    grid, x = _grid(times, sample), sample.clone()

    def derivative(value, time):
        return _field(field, value, time, condition), 0.0

    for left, right in zip(grid[:-1], grid[1:]):
        x, _ = _advance(derivative, x, float(left), float(right - left), method)
    return x


@dataclass(frozen=True)
class FlowLikelihood:
    base_sample: torch.Tensor
    log_prob: torch.Tensor
    divergence_integral: torch.Tensor
    function_evaluations: int
    trace_estimator: str


def flow_log_likelihood(
    field,
    sample,
    times,
    *,
    condition=None,
    method="rk4",
    trace="hutchinson",
    probe=None,
    generator=None,
    base_log_prob=None,
):
    """Integrate from data time 1 to base time 0, using
    log p1 = log p0 + integral_1_to_0 div(v) dt."""
    if getattr(field, "training", False):
        raise ValueError("Density evaluation requires deterministic eval-mode velocity")
    if method not in {"euler", "heun", "rk4"}:
        raise ValueError("Solver must be euler/heun/rk4")
    grid = _grid(times, sample)
    if grid[0] != 1 or grid[-1] != 0 or trace not in {"exact", "hutchinson"}:
        raise ValueError("Likelihood needs an explicit 1-to-0 grid and declared trace estimator")
    if probe is None and trace == "hutchinson":
        probe = (
            torch.randint(0, 2, sample.shape, device=sample.device, generator=generator).to(sample)
            * 2
            - 1
        )
    if probe is not None and (probe.shape != sample.shape or not torch.isfinite(probe).all()):
        raise ValueError("Trace probe shape/range invalid")
    if trace == "hutchinson" and not ((probe == 1) | (probe == -1)).all():
        raise ValueError("This fixed-probe estimator requires Rademacher values +/-1")
    count = 0

    def derivative(value, time):
        nonlocal count
        count += 1
        with torch.enable_grad():
            value = value.detach().requires_grad_(True)
            velocity = _field(field, value, time, condition)
            if trace == "hutchinson":
                inner = (velocity * probe).sum()
                gradient = (
                    torch.autograd.grad(inner, value, allow_unused=True)[0]
                    if inner.requires_grad
                    else None
                )
                divergence = (
                    value.new_zeros(len(value))
                    if gradient is None
                    else (gradient * probe).flatten(1).sum(-1)
                )
            else:
                flat = velocity.flatten(1)
                divergence = value.new_zeros(len(value))
                for index in range(flat.shape[-1]):
                    scalar = flat[:, index].sum()
                    gradient = (
                        torch.autograd.grad(scalar, value, retain_graph=True, allow_unused=True)[0]
                        if scalar.requires_grad
                        else None
                    )
                    if gradient is not None:
                        divergence += gradient.flatten(1)[:, index]
            return velocity.detach(), divergence.detach()

    x, integral = sample.detach().clone(), sample.new_zeros(len(sample))
    for left, right in zip(grid[:-1], grid[1:]):
        x, increment = _advance(derivative, x, float(left), float(right - left), method)
        integral += increment
    base = (
        (-0.5 * (x.square() + math.log(2 * math.pi))).flatten(1).sum(-1)
        if base_log_prob is None
        else base_log_prob(x)
    )
    if base.shape != (len(sample),) or not torch.isfinite(base + integral).all():
        raise ValueError("Base density must return one finite log probability per example")
    return FlowLikelihood(x, base + integral, integral, count, trace)
