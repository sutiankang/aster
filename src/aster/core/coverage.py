"""Validate implementation and test paths in a capability manifest."""

from pathlib import Path
from .serialization import read_json


def inspect_coverage(path, *, repository_root):
    manifest = read_json(path)
    root = Path(repository_root).resolve(strict=True)
    if manifest.get("schema_version") != 1:
        raise ValueError("Unknown capability manifest version")
    capabilities = manifest["capabilities"]
    seen = set()
    counts = {}
    findings = []
    for capability in capabilities:
        identifier = capability["id"]
        if identifier in seen:
            raise ValueError("Duplicate capability ID")
        seen.add(identifier)
        state = capability["implementation"]
        counts[state] = counts.get(state, 0) + 1
        for field in ("paths", "tests"):
            for relative in capability[field]:
                candidate = (root / relative).resolve()
                if not candidate.is_relative_to(root) or not candidate.is_file():
                    findings.append(
                        {"id": identifier, "problem": f"Missing/invalid {field}: {relative}"}
                    )
        if state != "planned" and not capability["paths"]:
            findings.append(
                {"id": identifier, "problem": "Implementation state without source paths"}
            )
        if (
            capability["equivalence"] not in {"unverified", "not_applicable"}
            and not capability["tests"]
        ):
            findings.append({"id": identifier, "problem": "Equivalence label lacks test paths"})
    incomplete = [
        c["id"] for c in capabilities if c.get("required") and c["implementation"] != "complete"
    ]
    return {
        "required": sum(bool(c.get("required")) for c in capabilities),
        "implementation_counts": counts,
        "incomplete": incomplete,
        "findings": findings,
        "complete": not incomplete and not findings,
    }
