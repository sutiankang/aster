"""Content-bound video tokenization and jointly trained Genie world artifacts."""

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path

import torch
from torch import nn

from ..core import atomic_json, digest_json, file_digest, read_json
from ..core.update_provenance import validate_successful_update_record
from ..models.genie import GenieTokenizer, GenieWorld
from .genie import GenieVQObjective, GenieWorldObjective, encode_genie_video


def tensor_identity(value):

    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
    ):
        raise ValueError("Expected a real dense tensor")
    value = value.detach().cpu().contiguous()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
    }


def _state_identity(config, tensors):

    return digest_json(
        {
            "configuration": config,
            "tensors": {key: tensor_identity(value) for key, value in sorted(tensors.items())},
        }
    )


def _sources():
    root = Path(__file__).resolve().parents[1]
    return {
        name: file_digest(root / name)
        for name in (
            "methods/genie_artifact_training.py",
            "methods/genie.py",
            "models/genie.py",
            "models/serialization.py",
            "models/config.py",
            "models/__init__.py",
            "core/update_provenance.py",
        )
    }


def _load_model(artifact):
    from ..models import load_model

    layouts = [
        root
        for root in (artifact.path, artifact.path / "model")
        if (root / "config.json").is_file()
    ]
    if len(layouts) != 1:
        raise ValueError("Exactly one native model layout is required")

    with torch.random.fork_rng(devices=[]):
        model = load_model(layouts[0])
    return model.eval()


def _descriptor(configuration, world=False):
    return {
        "class": (
            "aster.methods.genie_artifact_training.BoundGenieWorldObjective"
            if world
            else "aster.methods.genie.GenieVQObjective"
        ),
        "codec": "config_dict",
        "configuration": configuration,
    }


def _publish(engine, store, directory, *, ema, world, parents):
    from ..models import build_model
    from ..training.sharding import Zero3Unit
    from ..training.recipes import collective_local, agree, leader_call
    from ..training.trainer import _objective_configuration

    context = engine.parallel

    def preflight():
        if type(ema) is not bool or engine._busy or engine._failed:
            raise ValueError("Genie publication requires an idle successful Trainer boundary")
        if (
            any(getattr(context, name).size != 1 for name in ("tp", "pp", "cp", "gtp_remat"))
            or getattr(context.config, "expert_parallel", 1) != 1
            or getattr(context.config, "expert_tensor_parallel", 1) != 1
        ):
            raise ValueError("Genie publication currently supports dense DP/ZeRO only")
        role = engine.roles["model"]
        model = role.model.module if isinstance(role.model, Zero3Unit) else role.model
        if type(model) is not (GenieWorld if world else GenieTokenizer) or any(
            p.dtype != torch.float32 for p in model.parameters()
        ):
            raise ValueError(
                "Publish FP32-stored native Genie model weights, not a foreign runtime or partial shard"
            )
        expected = BoundGenieWorldObjective if world else GenieVQObjective
        if type(engine.objective) is not expected or (ema and role.ema is None):
            raise ValueError(
                "Genie publication requires its actual objective and requested sampling role"
            )
        if world:
            engine.objective.verify()
        configuration = engine.objective.config_dict()
        receipt = validate_successful_update_record(
            engine.last_successful_update(),
            _objective_configuration(engine.objective),
            role_updates=role.updates,
        )
        lineage = tuple(
            dict.fromkeys(tuple(parents) + ((configuration["trace_artifact_id"],) if world else ()))
        )
        for parent in lineage:
            store.get(parent, verify=True)
        return model.config, configuration, receipt, lineage

    config, objective, receipt, lineage = collective_local(
        context, preflight, "Validate native Genie publication"
    )
    declaration = {
        "config": config.to_dict(),
        "objective": objective,
        "receipt": receipt,
        "parents": lineage,
        "ema": ema,
        "directory": str(Path(directory).absolute()),
        "sources": _sources(),
    }
    agree(context, declaration, "Genie deployment identity")
    tensors = engine.export_state_dict(ema=ema)

    def publish():
        root = Path(directory).absolute()
        root.mkdir(parents=True, exist_ok=False)
        with torch.random.fork_rng(devices=[]):
            model = build_model(config)
        model.load_state_dict(tensors, strict=True, assign=True)
        model.save_pretrained(root / "model")
        contract = {
            "schema_version": 1,
            "model": config.to_dict(),
            "objective": objective,
            "successful_update": receipt,
            "sampling_role": "ema" if ema else "model",
            "weight_identity": _state_identity(config.to_dict(), tensors),
            "native_sources": declaration["sources"],
            "proof_scope": "last_successful_objective_and_current_deployment_weights_not_full_history",
        }
        atomic_json(root / "genie_contract.json", contract)
        return store.publish(
            root,
            kind="native_genie_world" if world else "native_genie_tokenizer",
            metadata={"contract_id": digest_json(contract), "training_checkpoint_included": False},
            parents=lineage,
        ).id

    artifact_id = leader_call(context, publish, "Publish native Genie weights")
    return collective_local(
        context, lambda: store.get(artifact_id, verify=True), "Verify native Genie weights"
    )


def publish_genie_tokenizer(engine, store, directory, *, ema=False, parents=()):
    return _publish(engine, store, directory, ema=ema, world=False, parents=parents)


def publish_genie_world(engine, store, directory, *, ema=False, parents=()):
    return _publish(engine, store, directory, ema=ema, world=True, parents=parents)


def load_trained_genie(store, artifact_id, *, world=False):

    artifact = store.get(artifact_id, verify=True)
    if artifact.kind != ("native_genie_world" if world else "native_genie_tokenizer"):
        raise ValueError("Expected a provenance-bound native Genie deployment")
    contract = read_json(artifact.path / "genie_contract.json")
    if (
        artifact.metadata.get("contract_id") != digest_json(contract)
        or contract.get("schema_version") != 1
        or contract.get("sampling_role") not in {"model", "ema"}
    ):
        raise ValueError("Genie deployment contract identity differs")
    receipt = contract["successful_update"]
    validate_successful_update_record(
        receipt, _descriptor(contract["objective"], world), role_updates=receipt.get("role_updates")
    )
    model = _load_model(artifact)
    if (
        type(model) is not (GenieWorld if world else GenieTokenizer)
        or model.config.to_dict() != contract["model"]
        or any(p.dtype != torch.float32 for p in model.parameters())
    ):
        raise ValueError("Genie deployment class/config/precision differs from its contract")
    if _state_identity(model.config.to_dict(), model.state_dict()) != contract["weight_identity"]:
        raise ValueError("Genie current weights differ from the published role")
    if world:
        trace_id = contract["objective"].get("trace_artifact_id")
        if trace_id not in artifact.parents:
            raise ValueError("Genie world omitted its actual tokenization parent")
    return model, contract


@dataclass(frozen=True)
class GenieVideoSpec:
    dataset_id: str
    revision: str
    split: str
    license_id: str
    fps: float
    normalization: str = "float32_zero_one_TCHW_no_resize"

    def __post_init__(self):
        if any(
            not isinstance(value, str) or not value
            for value in (self.dataset_id, self.revision, self.split, self.license_id)
        ) or self.revision.lower() in {"main", "master", "latest"}:
            raise ValueError("Declare fixed dataset revision, split and licensing terms")
        if (
            type(self.fps) not in {float, int}
            or not math.isfinite(self.fps)
            or self.fps <= 0
            or self.normalization != "float32_zero_one_TCHW_no_resize"
        ):
            raise ValueError("Declare finite FPS and the exact implemented pixel normalization")


def _video_row(row):
    if not isinstance(row, dict) or set(row) != {"video", "valid"}:
        raise ValueError("A video source row contains exactly video and valid")
    video, valid = row["video"], row["valid"]
    if (
        not isinstance(video, torch.Tensor)
        or video.ndim != 4
        or min(video.shape) < 1
        or video.shape[0] < 2
        or video.dtype != torch.float32
        or not torch.isfinite(video).all()
        or video.min() < 0
        or video.max() > 1
    ):
        raise ValueError("Video source is finite float32 [T,C,H,W] in [0,1]")
    if (
        not isinstance(valid, torch.Tensor)
        or valid.dtype != torch.bool
        or valid.shape != video.shape[:1]
        or ((~valid[:-1]) & valid[1:]).any()
    ):
        raise ValueError("Valid source frames must form an aligned boolean prefix")
    return {key: value.detach().cpu().contiguous().clone() for key, value in row.items()}


def publish_genie_videos(store, cases, directory, *, spec, parents=()):

    if (
        not isinstance(spec, GenieVideoSpec)
        or not cases
        or any(not isinstance(key, str) or not key for key in cases)
    ):
        raise ValueError("Provide an explicit nonempty source case mapping and GenieVideoSpec")
    rows = {key: _video_row(value) for key, value in cases.items()}
    if len({tuple(row["video"].shape) for row in rows.values()}) != 1:
        raise ValueError("This bounded Genie corpus requires one explicit sequence/geometry bucket")
    root = Path(directory).absolute()
    root.mkdir(parents=True, exist_ok=False)
    entries = {}
    for key, row in rows.items():
        name = digest_json(key) + ".pt"
        with (root / name).open("xb") as stream:
            torch.save(row, stream)
        entries[key] = {
            "path": name,
            "sha256": file_digest(root / name),
            "tensors": {k: tensor_identity(v) for k, v in row.items()},
        }
    manifest = {
        "schema_version": 1,
        "spec": asdict(spec),
        "expected_ids": list(rows),
        "entries": entries,
    }
    atomic_json(root / "videos.json", manifest)
    return store.publish(
        root,
        kind="native_genie_video_corpus",
        metadata={"manifest_id": digest_json(manifest)},
        parents=parents,
    )


class GenieVideoCorpus:
    def __init__(self, store, artifact_id):
        self.store, self.artifact_id = store, artifact_id
        artifact = store.get(artifact_id, verify=True)
        if artifact.kind != "native_genie_video_corpus":
            raise ValueError("Expected fixed numeric video corpus")
        self.manifest = read_json(artifact.path / "videos.json")
        self.root = artifact.path
        if (
            artifact.metadata.get("manifest_id") != digest_json(self.manifest)
            or self.manifest.get("schema_version") != 1
        ):
            raise ValueError("Video source manifest differs")
        self.spec = GenieVideoSpec(**self.manifest["spec"])
        self.ids = tuple(self.manifest["expected_ids"])
        if (
            not self.ids
            or len(set(self.ids)) != len(self.ids)
            or set(self.ids) != set(self.manifest["entries"])
        ):
            raise ValueError("Video source sample population differs")
        shapes = {tuple(self.load(key)["video"].shape) for key in self.ids}
        if len(shapes) != 1:
            raise ValueError("Mixed video source geometry")
        self.shape = next(iter(shapes))

    def verify(self):
        artifact = self.store.get(self.artifact_id, verify=True)
        if artifact.metadata.get("manifest_id") != digest_json(self.manifest):
            raise ValueError("Video manifest mutated")

    def load(self, key):
        entry = self.manifest["entries"][key]
        if entry["path"] != digest_json(key) + ".pt":
            raise ValueError("Invalid source tensor path")
        path = self.root / entry["path"]
        if file_digest(path) != entry["sha256"]:
            raise ValueError("Video source bytes changed")
        row = _video_row(torch.load(path, map_location="cpu", weights_only=True))
        if {k: tensor_identity(v) for k, v in row.items()} != entry["tensors"]:
            raise ValueError("Video source numeric identity changed")
        return row


def tokenize_genie_artifact(store, tokenizer_artifact_id, video_artifact_id, directory):

    tokenizer, contract = load_trained_genie(store, tokenizer_artifact_id)
    corpus = GenieVideoCorpus(store, video_artifact_id)
    sources = _sources()
    root = Path(directory).absolute()
    root.mkdir(parents=True, exist_ok=False)
    entries = {}
    for key in corpus.ids:
        row = corpus.load(key)
        encoded = encode_genie_video(tokenizer, row["video"][None], valid=row["valid"][None])[
            "tokens"
        ][0].cpu()
        name = digest_json(key) + ".pt"
        with (root / name).open("xb") as stream:
            torch.save({"tokens": encoded}, stream)
        entries[key] = {
            "path": name,
            "sha256": file_digest(root / name),
            "tokens": tensor_identity(encoded),
            "video": tensor_identity(row["video"]),
            "valid": tensor_identity(row["valid"]),
        }
    corpus.verify()
    store.get(tokenizer_artifact_id, verify=True)
    if sources != _sources():
        raise RuntimeError("Native tokenization source changed during encoding")
    trace = {
        "schema_version": 1,
        "tokenizer_artifact_id": tokenizer_artifact_id,
        "video_artifact_id": video_artifact_id,
        "tokenizer_weight_identity": contract["weight_identity"],
        "spec": asdict(corpus.spec),
        "expected_ids": list(corpus.ids),
        "entries": entries,
        "encoding": "native_cpu_float32_eval_full_causal_clip",
        "torch_version": str(torch.__version__),
        "native_sources": sources,
    }
    atomic_json(root / "tokenization.json", trace)
    return store.publish(
        root,
        kind="native_genie_tokenization",
        metadata={"trace_id": digest_json(trace)},
        parents=(tokenizer_artifact_id, video_artifact_id),
    )


class TokenizedGenieData:
    def __init__(self, store, artifact_id):
        self.store, self.artifact_id = store, artifact_id
        artifact = store.get(artifact_id, verify=True)
        self.root = artifact.path
        if artifact.kind != "native_genie_tokenization":
            raise ValueError("Expected a native tokenization trace")
        self.trace = read_json(artifact.path / "tokenization.json")
        if (
            artifact.metadata.get("trace_id") != digest_json(self.trace)
            or self.trace.get("schema_version") != 1
        ):
            raise ValueError("Tokenization trace identity differs")
        self.tokenizer_artifact_id, self.video_artifact_id = (
            self.trace["tokenizer_artifact_id"],
            self.trace["video_artifact_id"],
        )
        if (
            artifact.parents != (self.tokenizer_artifact_id, self.video_artifact_id)
            or self.trace["native_sources"] != _sources()
            or self.trace["encoding"] != "native_cpu_float32_eval_full_causal_clip"
            or self.trace.get("torch_version") != str(torch.__version__)
        ):
            raise ValueError("Tokenization lineage/source/encoding protocol differs")
        self.corpus = GenieVideoCorpus(store, self.video_artifact_id)
        self.ids = self.corpus.ids
        if (
            self.trace["expected_ids"] != list(self.ids)
            or set(self.trace["entries"]) != set(self.ids)
            or self.trace["spec"] != asdict(self.corpus.spec)
        ):
            raise ValueError("Tokenization corpus population/normalization differs")
        tokenizer, contract = load_trained_genie(store, self.tokenizer_artifact_id)
        self.tokenizer_config = tokenizer.config
        if contract["weight_identity"] != self.trace["tokenizer_weight_identity"]:
            raise ValueError("Tokenization used another codec weight identity")
        for key in self.ids:
            row = self.load(key)
            actual = encode_genie_video(tokenizer, row["video"][None], valid=row["valid"][None])[
                "tokens"
            ][0]
            if not torch.equal(row["tokens"], actual):
                raise ValueError("Stored token targets are not the actual pinned codec output")
        self.verify()

    def verify(self):
        artifact = self.store.get(self.artifact_id, verify=True)
        self.corpus.verify()
        self.store.get(self.tokenizer_artifact_id, verify=True)
        if (
            artifact.metadata.get("trace_id") != digest_json(self.trace)
            or self.trace["native_sources"] != _sources()
        ):
            raise ValueError("Tokenization trace or producing source changed")

    def load(self, key):
        row = self.corpus.load(key)
        entry = self.trace["entries"][key]
        if entry["path"] != digest_json(key) + ".pt" or any(
            entry[name] != tensor_identity(row[name]) for name in ("video", "valid")
        ):
            raise ValueError("Tokenization source row differs")
        path = self.root / entry["path"]
        if file_digest(path) != entry["sha256"]:
            raise ValueError("Tokenization tensor bytes changed")
        saved = torch.load(path, weights_only=True, map_location="cpu")
        if (
            not isinstance(saved, dict)
            or set(saved) != {"tokens"}
            or saved["tokens"].dtype != torch.int64
            or tensor_identity(saved["tokens"]) != entry["tokens"]
        ):
            raise ValueError("Tokenization tensor numeric identity differs")
        return {**row, "tokens": saved["tokens"].clone()}


class BoundGenieWorldObjective(nn.Module):
    def __init__(
        self, store, trace_artifact_id, *, commitment_cost=0.25, dynamics_weight=1.0, parallel=None
    ):
        super().__init__()

        def build():
            data = TokenizedGenieData(store, trace_artifact_id)
            objective = GenieWorldObjective(
                sequence_length=data.corpus.shape[0],
                commitment_cost=commitment_cost,
                dynamics_weight=dynamics_weight,
            )
            return data, objective

        if parallel is None:
            self.data, self.objective = build()
        else:
            from ..training.recipes import collective_local

            self.data, self.objective = collective_local(
                parallel, build, "Load bound Genie tokenization and objective"
            )

    def config_dict(self):
        return {
            "type": "bound_genie_world",
            "trace_artifact_id": self.data.artifact_id,
            "tokenizer_artifact_id": self.data.tokenizer_artifact_id,
            "video_artifact_id": self.data.video_artifact_id,
            "normalization": self.data.corpus.spec.normalization,
            "objective": self.objective.config_dict(),
        }

    def verify(self):
        self.data.verify()

    def batch(self, indices, *, device="cpu"):
        indices = tuple(indices)
        if not indices or any(
            type(index) is not int or not 0 <= index < len(self.data.ids) for index in indices
        ):
            raise ValueError("Source row indices must select a nonempty fixed corpus batch")
        rows = [self.data.load(self.data.ids[index]) for index in indices]
        return {
            **{
                key: torch.stack([row[key] for row in rows]).to(device)
                for key in ("video", "tokens", "valid")
            },
            "source_indices": torch.tensor(indices, dtype=torch.int64, device=device),
        }

    def _validate(self, model, batch):
        if (
            not isinstance(batch, dict)
            or set(batch) - {"video", "tokens", "valid", "mask", "source_indices"}
            or not {"video", "tokens", "valid", "source_indices"} <= set(batch)
        ):
            raise ValueError("Bound Genie batch needs actual pixels/tokens/valid/source indices")
        indices = batch["source_indices"]
        if (
            not isinstance(indices, torch.Tensor)
            or indices.dtype != torch.int64
            or indices.shape != (len(batch["video"]),)
        ):
            raise ValueError("Source row indices do not align with batch rows")
        tc, wc = self.data.tokenizer_config, model.config
        if (tc.num_codes, tc.spatial_tokens, tc.max_frames) != (
            wc.dynamics.vocab_size,
            wc.dynamics.spatial_tokens,
            wc.dynamics.max_frames,
        ):
            raise ValueError(
                "Training world is incompatible with the actual frozen video tokenizer"
            )
        stripped = {key: value for key, value in batch.items() if key != "source_indices"}
        self.objective._validate(model, stripped)
        for position, index in enumerate(indices.detach().cpu().tolist()):
            if not 0 <= index < len(self.data.ids):
                raise ValueError("Source row index outside fixed corpus")
            row = self.data.load(self.data.ids[index])
            for name in ("video", "tokens", "valid"):
                if tensor_identity(batch[name][position]) != tensor_identity(row[name]):
                    raise ValueError(
                        "Training row is not the bound video/token/normalization trace"
                    )
        return stripped

    def preflight_microbatches(self, model, batches):
        self.verify()

        for batch in batches:
            self._validate(model, batch)
        return batches

    def forward(self, model, batch):
        return self.objective(model, self._validate(model, batch))
