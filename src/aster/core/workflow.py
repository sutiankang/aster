"""Reentrant artifact-based DAG execution with content-bound stage identities."""

from dataclasses import dataclass, field
import inspect
from pathlib import Path
from .contracts import StageResult
from .serialization import RunLock, atomic_json, digest_json, read_json, file_digest


@dataclass(frozen=True)
class Stage:
    name: str
    kind: str
    config: dict = field(default_factory=dict)
    needs: tuple[str, ...] = ()


class Workflow:
    def __init__(self, stages, handlers, *, artifact_store, directory):
        self.stages, self.handlers, self.artifact_store = (
            tuple(stages),
            dict(handlers),
            artifact_store,
        )
        self.directory = Path(directory)
        names = [stage.name for stage in self.stages]
        if len(set(names)) != len(names) or not names:
            raise ValueError("Workflow requires uniquely named stages")
        known = set()
        for stage in self.stages:
            if (
                not stage.name.replace("_", "").replace("-", "").isalnum()
                or stage.kind not in handlers
                or not set(stage.needs) <= known
            ):
                raise ValueError(
                    "Stages must be path-safe and topologically ordered without missing dependencies"
                )
            known.add(stage.name)

    def run(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        outputs = {}
        package_root = Path(__file__).resolve().parents[1]
        code_fingerprint = digest_json(
            {
                path.relative_to(package_root).as_posix(): file_digest(path)
                for path in sorted(package_root.rglob("*.py"))
            }
        )
        with RunLock(self.directory / "run.lock"):
            for stage in self.stages:
                handler = self.handlers[stage.kind]
                inputs = {name: outputs[name] for name in stage.needs}
                signature = digest_json(
                    {
                        "kind": stage.kind,
                        "config": stage.config,
                        "inputs": inputs,
                        "handler_source": inspect.getsource(handler),
                        "package_source": code_fingerprint,
                    }
                )
                target = self.directory / stage.name
                manifest = target / "stage.json"
                if manifest.exists():
                    state = read_json(manifest)
                    if state["signature"] != signature:
                        raise ValueError(f"Stage {stage.name} changed; use a new run directory")
                    if state["status"] == "complete":
                        for artifact_id in state["result"]["artifacts"].values():
                            self.artifact_store.get(artifact_id)
                        outputs[stage.name] = state["result"]
                        continue

                    raise RuntimeError(
                        f"Stage {stage.name} is {state['status']}; inspect receipts and start an explicit recovery run"
                    )
                target.mkdir(parents=True, exist_ok=True)
                atomic_json(manifest, {"signature": signature, "status": "started"})
                try:
                    result = handler(stage.config, inputs, target, self.artifact_store)
                    if not isinstance(result, StageResult):
                        raise TypeError("Workflow handler must return StageResult")
                    for artifact_id in result.artifacts.values():
                        self.artifact_store.get(artifact_id)
                    payload = {
                        "artifacts": result.artifacts,
                        "metrics": result.metrics,
                        "details": result.details,
                    }
                    atomic_json(
                        manifest, {"signature": signature, "status": "complete", "result": payload}
                    )
                    outputs[stage.name] = payload
                except Exception as error:
                    atomic_json(
                        manifest,
                        {
                            "signature": signature,
                            "status": "failed",
                            "error": f"{type(error).__name__}: {error}",
                        },
                    )
                    raise
        return outputs
