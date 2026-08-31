"""Export trained DSpark drafts into independent deployment artifacts."""

from pathlib import Path
import torch

from ..core import atomic_json, read_json, digest_json
from ..core.update_provenance import validate_successful_update_record
from ..data.dspark import DSparkFeatureCache
from ..models import load_model, build_model
from ..models.dspark import DSparkDraft, target_state_identity
from ..models.dspark_gemma4 import Gemma4DSparkDraft
from ..training.recipes import collective_local, agree, leader_call
from ..training.runtime_state import apply_runtime_state
from .dspark import DSparkMethod


def publish_dspark_draft(method, store, directory, *, ema=False):
    if not isinstance(method, DSparkMethod):
        raise ValueError("Publish through the native DSpark method lifecycle")
    engine = method.engine
    context = engine.parallel

    def preflight():
        if (
            engine._busy
            or engine._failed
            or type(ema) is not bool
            or (ema and engine.roles["model"].ema is None)
        ):
            raise ValueError("DSpark publication requires an idle successful requested model role")
        configuration = method.objective.config_dict()
        descriptor = {
            "class": "aster.methods.dspark.DSparkObjective",
            "codec": "config_dict",
            "configuration": configuration,
        }
        receipt = validate_successful_update_record(
            engine.last_successful_update(), descriptor, role_updates=engine.roles["model"].updates
        )
        parents = tuple(configuration["feature_cache_ids"])
        if not parents:
            raise ValueError(
                "Audited DSpark publication requires bound immutable feature-cache training"
            )
        for parent in parents:
            cache = DSparkFeatureCache(store, parent)
            if (
                cache.contract["teacher_identity"] != configuration["teacher_identity"]
                or cache.contract["draft_config"] != engine.model.config.to_dict()
            ):
                raise ValueError("DSpark publication feature lineage differs")
        return dict(
            objective=configuration,
            receipt=receipt,
            parents=parents,
            config=engine.model.config.to_dict(),
            ema=ema,
        )

    declaration = collective_local(context, preflight, "Validate native DSpark publication")
    agree(context, declaration, "DSpark deployment identity")
    weights = engine.export_state_dict(ema=ema)
    runtime = engine.export_runtime_state()

    def publish():
        root = Path(directory).absolute()
        root.mkdir(parents=True, exist_ok=False)
        with torch.random.fork_rng(devices=[]):
            model = build_model(engine.model.config)
        model.load_state_dict(weights, strict=True, assign=True)
        apply_runtime_state(model, runtime)
        model.save_pretrained(root / "model")
        contract = dict(
            schema_version=1,
            **declaration,
            weight_identity=target_state_identity(model),
            proof_scope="last_successful_objective_approved_cache_set_current_role_weights_not_full_history",
            quality_claim="not_evaluated",
        )
        atomic_json(root / "dspark_contract.json", contract)
        return store.publish(
            root,
            kind="native_dspark_draft",
            metadata={"contract_id": digest_json(contract)},
            parents=declaration["parents"],
        ).id

    artifact_id = leader_call(context, publish, "Publish native DSpark draft")
    return collective_local(
        context, lambda: store.get(artifact_id, verify=True), "Verify native DSpark deployment"
    )


def load_dspark_draft(store, artifact_id):
    artifact = store.get(artifact_id, verify=True)
    if artifact.kind != "native_dspark_draft":
        raise ValueError("Expected native DSpark deployment artifact")
    contract = read_json(artifact.path / "dspark_contract.json")
    if contract.get("schema_version") != 1 or artifact.metadata.get("contract_id") != digest_json(
        contract
    ):
        raise ValueError("DSpark deployment contract differs")
    receipt = contract["receipt"]
    validate_successful_update_record(
        receipt,
        {
            "class": "aster.methods.dspark.DSparkObjective",
            "codec": "config_dict",
            "configuration": contract["objective"],
        },
        role_updates=receipt.get("role_updates"),
    )
    if (
        tuple(contract["parents"]) != artifact.parents
        or tuple(contract["objective"]["feature_cache_ids"]) != artifact.parents
    ):
        raise ValueError("DSpark deployment omitted its actual approved feature-cache lineage")
    with torch.random.fork_rng(devices=[]):
        model = load_model(artifact.path / "model")
    if (
        not isinstance(model, (DSparkDraft, Gemma4DSparkDraft))
        or model.config.to_dict() != contract["config"]
        or target_state_identity(model) != contract["weight_identity"]
    ):
        raise ValueError("DSpark deployment weights/configuration differ from the publishing role")
    return model.eval(), contract
