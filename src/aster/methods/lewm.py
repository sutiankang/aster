"""End-to-end LeWorldModel prediction and SIGReg over complete logical batches."""

from copy import deepcopy
import math
import torch
from torch import nn
from ..core import LossTerm, LossBundle
from ..models.lewm import LeWMConfig


class SIGReg(nn.Module):
    """Test the batch distribution independently at each time, then average across
    time/projections. Merging time into the batch changes the statistic."""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        if type(knots) is not int or knots < 2 or type(num_proj) is not int or num_proj < 1:
            raise ValueError("Invalid SIGReg quadrature/projection count")
        self.knots, self.num_proj = knots, num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, embeddings, projections=None):
        if (
            embeddings.ndim != 3
            or min(embeddings.shape) < 1
            or not torch.isfinite(embeddings).all()
        ):
            raise ValueError("SIGReg expects finite [T,B,D] embeddings")
        if projections is None:
            projections = torch.randn(embeddings.shape[-1], self.num_proj, device=embeddings.device)
            projections = projections / projections.norm(dim=0)
        if (
            projections.shape != (embeddings.shape[-1], self.num_proj)
            or projections.device != embeddings.device
            or not torch.isfinite(projections).all()
        ):
            raise ValueError("SIGReg projections must match explicit [D,M] and device")
        if not torch.allclose(
            projections.norm(dim=0),
            torch.ones(self.num_proj, device=embeddings.device, dtype=projections.dtype),
            atol=2e-5,
            rtol=2e-5,
        ):
            raise ValueError("SIGReg projection columns must be unit-norm")
        value = (embeddings @ projections).unsqueeze(-1) * self.t.to(embeddings.device)
        error = (value.cos().mean(-3) - self.phi.to(embeddings.device)).square() + value.sin().mean(
            -3
        ).square()
        return ((error @ self.weights.to(embeddings.device)) * embeddings.shape[-2]).mean()


class LeWMObjective(nn.Module):
    def __init__(self, *, regularization_weight=0.09, knots=17, num_proj=1024):
        super().__init__()
        if not math.isfinite(regularization_weight) or regularization_weight < 0:
            raise ValueError("Invalid LeWM regularization weight")
        self.regularization_weight = regularization_weight
        self.sigreg = SIGReg(knots, num_proj)

    def config_dict(self):
        return dict(
            type="lewm",
            regularization_weight=self.regularization_weight,
            knots=self.sigreg.knots,
            num_proj=self.sigreg.num_proj,
            target_gradient="joint_no_stopgrad",
            statistics="one_complete_logical_batch",
        )

    def validate(self, model, batch):
        if not isinstance(model.config, LeWMConfig):
            raise ValueError("LeWM objective requires its genuine native model")
        if (
            not isinstance(batch, dict)
            or not {"pixels", "actions"} <= set(batch)
            or set(batch) - {"pixels", "actions", "projections"}
        ):
            raise ValueError(
                "LeWM batches require explicit pixels/actions and optional fixed projections"
            )
        pixels, action = batch["pixels"], batch["actions"]
        c = model.config
        if (
            not isinstance(pixels, torch.Tensor)
            or pixels.ndim != 5
            or pixels.shape[1:3] != (c.history_size + 1, c.encoder.num_channels)
            or len(pixels) < 2
        ):
            raise ValueError("LeWM complete batch requires B>=2, H+1 normalized image frames")
        if (
            not pixels.is_floating_point()
            or not torch.isfinite(pixels).all()
            or min(pixels.shape[-2:]) < c.encoder.patch_size
        ):
            raise ValueError("Invalid LeWM normalized pixels")
        if (
            not isinstance(action, torch.Tensor)
            or action.shape != (len(pixels), c.history_size, c.action_dim)
            or action.device != pixels.device
            or not action.is_floating_point()
            or not torch.isfinite(action).all()
        ):
            raise ValueError(
                "LeWM needs finite aligned transition actions; terminal NaNs are not silently erased"
            )
        projection = batch.get("projections")
        if projection is not None and (
            projection.shape != (c.embed_dim, self.sigreg.num_proj)
            or projection.device != pixels.device
            or not torch.isfinite(projection).all()
        ):
            raise ValueError("Invalid explicit LeWM projection matrix")

    def preflight_microbatches(self, model, batches):
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
            and not getattr(self, "_method_active", False)
        ):
            raise ValueError(
                "Distributed LeWM nonlinear statistics require LeWMMethod, not local-batch objective averaging"
            )
        if len(batches) != 1:
            raise ValueError(
                "SIGReg/BN require one complete logical batch; use LeWMMethod.update for input chunks"
            )
        for batch in batches:
            self.validate(model, batch)
        return batches

    def forward(self, model, batch):
        self.validate(model, batch)
        output = model(batch["pixels"], batch["actions"])

        difference = (output.predictions - output.embeddings[:, 1:]).square()
        regularization = self.sigreg(output.embeddings.transpose(0, 1), batch.get("projections"))
        return LossBundle(
            (
                LossTerm(
                    difference.sum(),
                    difference.new_tensor(difference.numel(), dtype=torch.int64),
                    "latent_scalar",
                    "prediction",
                ),
                LossTerm(
                    regularization,
                    regularization.new_tensor(1, dtype=torch.int64),
                    "logical_batch",
                    "sigreg",
                    self.regularization_weight,
                ),
            )
        )


class LeWMMethod:
    """Treat input chunks as data-transfer chunks; global SIGReg statistics still require
    the full logical batch and do not imply activation-memory gradient accumulation."""

    def __init__(self, engine, *, objective=None, seed=0, max_batch_bytes=256 * 1024 * 1024):
        self.engine = engine
        self.objective = objective or LeWMObjective()
        error = None
        try:
            if (
                not isinstance(engine.model.config, LeWMConfig)
                or not isinstance(self.objective, LeWMObjective)
                or engine.accumulation_steps != 1
            ):
                raise ValueError("LeWMMethod needs LeWM and a fixed one-logical-batch Trainer")
            if any(
                getattr(engine.parallel.config, key) != 1
                for key in (
                    "tensor_parallel",
                    "pipeline_parallel",
                    "context_parallel",
                    "gtp_remat",
                    "expert_parallel",
                    "expert_tensor_parallel",
                )
            ):
                raise ValueError(
                    "LeWM reference supports DP/ZeRO, not implicit model parallel layouts"
                )
            if (
                type(seed) is not int
                or not 0 <= seed < 2**63
                or type(max_batch_bytes) is not int
                or max_batch_bytes < 1
            ):
                raise ValueError("Invalid LeWM RNG/budget")
            self.settings = dict(
                model=engine.model.config.to_dict(),
                objective=self.objective.config_dict(),
                seed=seed,
                max_batch_bytes=max_batch_bytes,
                layout="replicated_global_logical_batch",
            )
        except Exception as exc:
            error = str(exc)
        errors = engine.parallel.world.gather_objects(error)
        if any(errors):
            raise ValueError("LeWM declaration preflight failed: " + str(errors))
        declarations = engine.parallel.world.gather_objects(self.settings)
        if any(item != self.settings for item in declarations):
            raise ValueError("LeWM global batch declarations differ")
        self.rng = torch.Generator().manual_seed(seed)
        self.updates = 0
        self.incomplete = False
        engine.register_state("lewm_method", self)

    def state_dict(self):
        return dict(
            schema_version=1,
            settings=deepcopy(self.settings),
            rng=self.rng.get_state(),
            updates=self.updates,
            incomplete=self.incomplete,
        )

    def load_state_dict(self, value):
        if (
            set(value) != {"schema_version", "settings", "rng", "updates", "incomplete"}
            or value["schema_version"] != 1
            or value["settings"] != self.settings
            or value["incomplete"]
            or type(value["updates"]) is not int
            or value["updates"] < 0
        ):
            raise ValueError(
                "LeWM checkpoint settings/state differ or describe an incomplete update"
            )
        probe = torch.Generator()
        probe.set_state(value["rng"])
        self.rng.set_state(value["rng"])
        self.updates = value["updates"]
        self.incomplete = False

    def _gather(self, chunks):
        error, local = None, None
        engine = self.engine
        try:
            if self.incomplete:
                raise ValueError("Restore the last completed LeWM checkpoint before retry")
            chunks = list(chunks)
            if not chunks:
                raise ValueError("LeWM needs nonempty local data chunks")
            rows = []
            for chunk in chunks:
                if not isinstance(chunk, dict) or set(chunk) != {"pixels", "actions"}:
                    raise ValueError("LeWM data chunks need pixels/actions only")
                pixels, actions = chunk["pixels"], chunk["actions"]
                c = engine.model.config
                if (
                    not isinstance(pixels, torch.Tensor)
                    or pixels.ndim != 5
                    or len(pixels) < 1
                    or pixels.shape[1:3] != (c.history_size + 1, c.encoder.num_channels)
                ):
                    raise ValueError("Invalid LeWM raw chunk image sequence")
                if not isinstance(actions, torch.Tensor) or actions.shape != (
                    len(pixels),
                    c.history_size,
                    c.action_dim,
                ):
                    raise ValueError("LeWM chunk transition actions are misaligned")
                if any(
                    value.dtype != torch.float32 or not torch.isfinite(value).all()
                    for value in (pixels, actions)
                ):
                    raise ValueError(
                        "LeWM reference input chunks require finite normalized FP32 tensors"
                    )
                rows.append(
                    {key: value.detach().cpu().contiguous() for key, value in chunk.items()}
                )
            local = {key: torch.cat([row[key] for row in rows]) for key in ("pixels", "actions")}
            if (
                sum(value.numel() * value.element_size() for value in local.values())
                > self.settings["max_batch_bytes"]
            ):
                raise ValueError("LeWM local raw data exceeds explicit batch budget")
        except Exception as exc:
            error = str(exc)
        records = engine.parallel.world.gather_objects(
            (
                error,
                None
                if local is None
                else sum(value.numel() * value.element_size() for value in local.values()),
            )
        )
        if any(item[0] for item in records):
            raise ValueError("LeWM input preflight failed: " + str(records))
        if sum(item[1] for item in records) > self.settings["max_batch_bytes"]:
            raise ValueError("LeWM global raw data exceeds explicit batch budget")
        gathered = engine.parallel.dp.gather_objects(local)

        result = {
            key: torch.cat([item[key] for item in gathered]).to(engine.device)
            for key in ("pixels", "actions")
        }
        self.objective.validate(engine.model, result)
        return result

    def update(self, chunks):
        batch = self._gather(chunks)
        self.incomplete = True
        projection = torch.randn(
            self.engine.model.config.embed_dim, self.objective.sigreg.num_proj, generator=self.rng
        )
        batch["projections"] = (projection / projection.norm(dim=0)).to(self.engine.device)

        drop_seed = int(torch.randint(2**31, (), generator=self.rng))
        device = self.engine.device
        devices = [device.index] if device.type == "cuda" else []
        self.objective._method_active = True
        try:
            with torch.random.fork_rng(devices=devices):
                torch.random.set_rng_state(torch.Generator().manual_seed(drop_seed).get_state())
                if device.type == "cuda":
                    with torch.cuda.device(device):
                        torch.cuda.manual_seed(drop_seed)
                result = self.engine.phase("lewm", objective=self.objective, microbatches=[batch])
        finally:
            self.objective._method_active = False
        if not result.updated:
            raise RuntimeError("LeWM phase did not update; restore the last completed checkpoint")
        self.updates += 1
        self.incomplete = False
        return result
