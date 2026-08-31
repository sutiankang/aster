"""Extract target-model features into version-bound local DSpark datasets."""

from copy import deepcopy
from pathlib import Path
import re

import torch

from ..core import atomic_json, read_json, digest_json
from ..models import CausalLM, Qwen3Config
from ..models.dspark import DSparkConfig, target_state_identity
from ..models.serialization import LocalModelMixin


class DSparkTeacherFeatures:
    def __init__(self, target, config, *, vocabulary_fingerprint):
        from ..models.gemma4 import Gemma4ForCausalLM, Gemma4TextConfig
        from ..models.dspark_gemma4 import Gemma4DSparkConfig

        qwen = (
            type(target) is CausalLM
            and type(target.config) is Qwen3Config
            and type(config) is DSparkConfig
        )
        gemma = (
            type(target) is Gemma4ForCausalLM
            and type(target.config) is Gemma4TextConfig
            and type(config) is Gemma4DSparkConfig
        )
        if not (qwen or gemma):
            raise ValueError(
                "DSpark feature extraction requires a matched native Qwen3 or Gemma4 target/draft pair"
            )
        if target.config.to_dict() != config.target.to_dict():
            raise ValueError("DSpark teacher config differs")
        if not isinstance(vocabulary_fingerprint, str) or not vocabulary_fingerprint:
            raise ValueError("Declare token-ID vocabulary semantics")
        if getattr(target, "_aster_training_owned", False):
            raise ValueError("Export an idle dense target before feature extraction")
        self.model = deepcopy(target).eval().requires_grad_(False)
        self.config, self.vocabulary_fingerprint = config, vocabulary_fingerprint
        self.teacher_identity = target_state_identity(self.model)
        self.device = next(self.model.parameters()).device

        self.extraction_profile = (
            "native_gemma4_eval_no_autocast_unpadded"
            if gemma
            else "native_qwen3_eval_no_autocast_unpadded"
        )

    @torch.no_grad()
    def extract(self, input_ids, loss_mask):
        c = self.config
        if (
            input_ids.ndim != 2
            or min(input_ids.shape) < 1
            or input_ids.dtype != torch.int64
            or (input_ids < 0).any()
            or (input_ids >= c.target.vocab_size).any()
        ):
            raise ValueError("Teacher input must contain native vocabulary int64 IDs")
        if input_ids.shape[1] + c.block_size > c.target.max_position_embeddings:
            raise ValueError("Teacher context leaves no room for declared DSpark block")
        if loss_mask.shape != input_ids.shape or not ((loss_mask == 0) | (loss_mask == 1)).all():
            raise ValueError("Teacher loss mask must be aligned binary values")
        ids = input_ids.detach().to(self.device)

        with torch.autocast(self.device.type, enabled=False):
            output = self.model(ids, output_hidden_states=True)

        chosen = torch.cat([output.hidden_states[i + 1] for i in c.target_layer_ids], -1)
        return dict(
            input_ids=ids.cpu().clone(),
            loss_mask=loss_mask.detach().cpu().clone(),
            target_hidden_states=chosen.cpu().clone(),
            target_last_hidden_states=output.hidden_states[-1].cpu().clone(),
            teacher_identity=self.teacher_identity,
            vocabulary_fingerprint=self.vocabulary_fingerprint,
        )


def publish_dspark_features(
    store, teacher, records, directory, *, dataset_id, revision, license_id, parents=()
):

    if (
        not isinstance(teacher, DSparkTeacherFeatures)
        or not records
        or any(not isinstance(k, str) or not k for k in records)
    ):
        raise ValueError("Expected an explicit teacher and nonempty named input records")
    if any(
        not isinstance(v, str) or not v for v in (dataset_id, revision, license_id)
    ) or revision.lower() in {"main", "master", "latest"}:
        raise ValueError("Feature cache requires a fixed dataset revision and license declaration")
    if target_state_identity(teacher.model) != teacher.teacher_identity:
        raise ValueError("Teacher changed after feature extractor construction")
    for record in records.values():
        if (
            set(record) != {"input_ids", "loss_mask"}
            or record["input_ids"].ndim != 2
            or record["input_ids"].shape[0] != 1
        ):
            raise ValueError("Each cache record is exactly one unpadded source sequence")
    root = Path(directory).absolute()
    root.mkdir(parents=True, exist_ok=False)
    entries = {}
    for index, (sample_id, record) in enumerate(records.items()):
        result = teacher.extract(**record)
        tensors = {k: v for k, v in result.items() if isinstance(v, torch.Tensor)}
        filename = f"feature-{index:08d}.pt"
        LocalModelMixin._atomic_tensors(root / filename, tensors)
        entries[sample_id] = filename
    contract = dict(
        schema_version=1,
        teacher_identity=teacher.teacher_identity,
        vocabulary_fingerprint=teacher.vocabulary_fingerprint,
        draft_config=teacher.config.to_dict(),
        dataset_id=dataset_id,
        revision=revision,
        license_id=license_id,
        records=entries,
        extraction=teacher.extraction_profile,
        quality_claim="not_evaluated",
    )
    atomic_json(root / "features.json", contract)
    return store.publish(
        root,
        kind="native_dspark_features",
        metadata={"contract_id": digest_json(contract)},
        parents=parents,
    )


class DSparkFeatureCache:
    def __init__(self, store, artifact_id):
        self.store, self.artifact_id = store, artifact_id
        artifact = store.get(artifact_id, verify=True)
        contract = read_json(artifact.path / "features.json")
        if (
            artifact.kind != "native_dspark_features"
            or contract.get("schema_version") != 1
            or artifact.metadata.get("contract_id") != digest_json(contract)
        ):
            raise ValueError("Invalid DSpark feature-cache contract")
        if not contract["records"] or any(
            not isinstance(k, str) or not k or not re.fullmatch(r"feature-\d{8}\.pt", v)
            for k, v in contract["records"].items()
        ):
            raise ValueError("DSpark feature index paths must be fixed local leaf names")
        if len(set(contract["records"].values())) != len(contract["records"]):
            raise ValueError("Aliased DSpark feature index")
        self.contract = contract
        self.sample_ids, self.fingerprint = tuple(sorted(contract["records"])), artifact_id

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError(index)
        return self.sample_ids[index]

    def verify(self):
        self.store.get(self.artifact_id, verify=True)

    def batch(self, sample_ids, *, device="cpu"):
        artifact = self.store.get(self.artifact_id, verify=True)
        if not sample_ids or any(key not in self.contract["records"] for key in sample_ids):
            raise ValueError("Unknown or empty DSpark cache batch")
        rows = [
            torch.load(
                artifact.path / self.contract["records"][key], map_location="cpu", weights_only=True
            )
            for key in sample_ids
        ]
        fields = {"input_ids", "loss_mask", "target_hidden_states", "target_last_hidden_states"}
        if any(
            not isinstance(row, dict)
            or set(row) != fields
            or any(not isinstance(t, torch.Tensor) for t in row.values())
            for row in rows
        ):
            raise ValueError("DSpark cache tensor schema differs")
        if len({row["input_ids"].shape[1] for row in rows}) != 1:
            raise ValueError(
                "DSpark cache batch must have equal sequence lengths; no implicit padding"
            )
        return dict(
            {key: torch.cat([row[key] for row in rows]).to(device) for key in fields},
            teacher_identity=self.contract["teacher_identity"],
            vocabulary_fingerprint=self.contract["vocabulary_fingerprint"],
            feature_cache_id=self.artifact_id,
            cache_sample_ids=tuple(sample_ids),
        )
