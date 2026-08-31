"""Positive-weight CMA-ES following the author's purecma formulation."""

from copy import deepcopy
import math
import torch


class CMAES:
    def __init__(
        self, mean, *, sigma=0.1, population=None, seed=0, max_covariance_bytes=512 * 1024 * 1024
    ):
        mean = torch.as_tensor(mean, dtype=torch.float64, device="cpu").detach().clone()
        if (
            mean.ndim != 1
            or not len(mean)
            or not torch.isfinite(mean).all()
            or not math.isfinite(sigma)
            or sigma <= 0
        ):
            raise ValueError("CMA-ES needs a finite vector and positive sigma")
        n = len(mean)
        if (
            type(max_covariance_bytes) is not int
            or max_covariance_bytes < 1
            or 4 * n * n * 8 > max_covariance_bytes
        ):
            raise ValueError("CMA-ES covariance workspace exceeds the explicit memory budget")
        population = 4 + int(3 * math.log(n)) if population is None else population
        if type(population) is not int or population < 2 or type(seed) is not int:
            raise ValueError("CMA-ES population/seed must be integers")
        self.settings = dict(
            dimension=n, population=population, algorithm="purecma_positive_weights_v3"
        )
        self.mean, self.sigma = mean, float(sigma)
        self.population, self.mu = population, population // 2
        weights = (
            math.log(population / 2 + 0.5) - torch.arange(1, self.mu + 1, dtype=torch.float64).log()
        )
        self.weights = weights / weights.sum()
        self.mueff = 1 / float(self.weights.square().sum())
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(
            1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff)
        )
        self.damps = 2 * self.mueff / population + 0.3 + self.cs
        self.lazy_gap = 0.5 * population / (n * (self.c1 + self.cmu))
        self.covariance = torch.eye(n, dtype=torch.float64)
        self.basis = self.covariance.clone()
        self.eigenvalues = torch.ones(n, dtype=torch.float64)
        self.invsqrt = self.covariance.clone()
        self.pc, self.ps = torch.zeros_like(mean), torch.zeros_like(mean)
        self.evaluations = self.eigen_evaluations = 0
        self.generator = torch.Generator().manual_seed(seed)
        self.pending = None
        self.best_value, self.best = None, mean.clone()

    def ask(self):
        if self.pending is not None:
            raise RuntimeError("CMA-ES already has an unevaluated population")
        if self.evaluations > self.eigen_evaluations + self.lazy_gap:
            eigenvalues, basis = torch.linalg.eigh((self.covariance + self.covariance.T) / 2)
            if not torch.isfinite(eigenvalues).all() or eigenvalues.min() <= 0:
                raise RuntimeError("CMA-ES covariance is not positive definite")
            self.eigenvalues, self.basis = eigenvalues, basis
            self.invsqrt = (basis / eigenvalues.sqrt()[None, :]) @ basis.T
            self.eigen_evaluations = self.evaluations
        normals = torch.randn(
            self.population, len(self.mean), dtype=torch.float64, generator=self.generator
        )
        self.pending = self.mean + self.sigma * (normals * self.eigenvalues.sqrt()) @ self.basis.T

        return self.pending.clone()

    def tell(self, values):

        values = torch.as_tensor(values, dtype=torch.float64, device="cpu")
        if (
            self.pending is None
            or values.shape != (self.population,)
            or not torch.isfinite(values).all()
        ):
            raise ValueError(
                "CMA-ES tell requires finite scores for exactly the pending population"
            )
        order = values.argsort(stable=True)
        candidates = self.pending[order]
        mean = (candidates[: self.mu] * self.weights[:, None]).sum(0)
        difference = mean - self.mean
        ps = (1 - self.cs) * self.ps + math.sqrt(
            self.cs * (2 - self.cs) * self.mueff
        ) / self.sigma * (self.invsqrt @ difference)
        evaluations = self.evaluations + self.population
        n = len(mean)
        hsig = float(ps.square().sum()) / n / (
            1 - (1 - self.cs) ** (2 * evaluations / self.population)
        ) < 2 + 4 / (n + 1)
        pc = (1 - self.cc) * self.pc + math.sqrt(
            self.cc * (2 - self.cc) * self.mueff
        ) / self.sigma * hsig * difference
        c1a = self.c1 * (1 - (1 - float(hsig) ** 2) * self.cc * (2 - self.cc))
        differences = candidates[: self.mu] - self.mean
        covariance = (1 - c1a - self.cmu) * self.covariance + self.c1 * torch.outer(pc, pc)
        covariance = covariance + self.cmu / self.sigma**2 * (
            differences.T @ (differences * self.weights[:, None])
        )
        sigma = self.sigma * math.exp(
            min(1.0, self.cs / self.damps * (float(ps.square().sum()) / n - 1) / 2)
        )
        if (
            not math.isfinite(sigma)
            or sigma <= 0
            or not all(torch.isfinite(v).all() for v in (mean, pc, ps, covariance))
        ):
            raise RuntimeError("CMA-ES numerical failure; pending generation was not committed")

        self.mean, self.pc, self.ps, self.covariance, self.sigma = mean, pc, ps, covariance, sigma
        self.evaluations = evaluations
        best_value = float(values[order[0]])
        if self.best_value is None or best_value < self.best_value:
            self.best_value, self.best = best_value, candidates[0].clone()
        self.pending = None
        return dict(evaluations=evaluations, best_value=self.best_value, sigma=self.sigma)

    def state_dict(self):
        return dict(
            settings=deepcopy(self.settings),
            sigma=self.sigma,
            evaluations=self.evaluations,
            eigen_evaluations=self.eigen_evaluations,
            best_value=self.best_value,
            tensors={
                key: getattr(self, key).clone()
                for key in (
                    "mean",
                    "pc",
                    "ps",
                    "covariance",
                    "basis",
                    "eigenvalues",
                    "invsqrt",
                    "best",
                )
            },
            pending=None if self.pending is None else self.pending.clone(),
            rng=self.generator.get_state().clone(),
        )

    def load_state_dict(self, state):
        if (
            state["settings"] != self.settings
            or not math.isfinite(state["sigma"])
            or state["sigma"] <= 0
        ):
            raise ValueError("CMA-ES checkpoint algorithm/dimensions/sigma differ")
        for key, value in state["tensors"].items():
            if (
                key
                not in {"mean", "pc", "ps", "covariance", "basis", "eigenvalues", "invsqrt", "best"}
                or value.shape != getattr(self, key).shape
                or value.dtype != torch.float64
                or not torch.isfinite(value).all()
            ):
                raise ValueError("Invalid CMA-ES checkpoint tensors")
        if set(state["tensors"]) != {
            "mean",
            "pc",
            "ps",
            "covariance",
            "basis",
            "eigenvalues",
            "invsqrt",
            "best",
        }:
            raise ValueError("Incomplete CMA-ES checkpoint tensors")
        for key in ("evaluations", "eigen_evaluations"):
            if type(state[key]) is not int or state[key] < 0 or state[key] % self.population:
                raise ValueError("Invalid CMA-ES checkpoint generation clock")
        if (
            state["eigen_evaluations"] > state["evaluations"]
            or (state["tensors"]["eigenvalues"] <= 0).any()
        ):
            raise ValueError("Invalid CMA-ES checkpoint eigensystem")
        if state["best_value"] is not None and (
            type(state["best_value"]) not in (float, int) or not math.isfinite(state["best_value"])
        ):
            raise ValueError("Invalid CMA-ES checkpoint best score")
        pending = state["pending"]
        if pending is not None and (
            pending.shape != (self.population, len(self.mean))
            or pending.dtype != torch.float64
            or not torch.isfinite(pending).all()
        ):
            raise ValueError("Invalid pending CMA-ES population")
        trial = torch.Generator()
        trial.set_state(state["rng"])
        for key, value in state["tensors"].items():
            setattr(self, key, value.detach().cpu().clone())
        self.sigma, self.evaluations, self.eigen_evaluations = (
            state["sigma"],
            state["evaluations"],
            state["eigen_evaluations"],
        )
        self.best_value = state["best_value"]
        self.pending = None if pending is None else pending.detach().cpu().clone()
        self.generator.set_state(state["rng"])
