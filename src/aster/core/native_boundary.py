"""Static import-boundary checks for native modules and optional evaluation dependencies."""

import ast
from dataclasses import dataclass, asdict
from pathlib import Path
import sys


@dataclass(frozen=True)
class ImportFinding:
    path: str
    line: int
    expression: str
    reason: str


def audit_native_boundary(package_root):
    root = Path(package_root).resolve(strict=True)
    if not root.is_dir() or root.name != "aster":
        raise ValueError(
            "Audit the actual aster package directory, not an arbitrary filesystem root"
        )
    native = set(sys.stdlib_module_names) | {"aster", "torch", "numpy", "__future__"}

    evaluation = {"PIL", "cleanfid", "lm_eval", "lmms_eval", "libero", "swebench", "gymnasium"}

    local_formats = {"inference/checkpoint.py": {"safetensors"}, "data/qwen_vl.py": {"PIL"}}

    native_kernels = {"optimization/_triton_attention.py": {"triton"}}
    findings, imports, files = [], [], []
    for path in sorted(root.rglob("*.py")):
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise ValueError("Source symlink escapes the audited package")
        relative = path.relative_to(root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=relative)
        files.append(relative)
        is_evaluation = relative.startswith("evaluation/")
        allowed = (
            native
            | (evaluation if is_evaluation else set())
            | local_formats.get(relative, set())
            | native_kernels.get(relative, set())
        )
        dynamic_aliases = {"__import__", "importlib.import_module"}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "importlib":
                dynamic_aliases.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "import_module"
                )
            elif isinstance(node, ast.Import):
                dynamic_aliases.update(
                    (alias.asname or "importlib") + ".import_module"
                    for alias in node.names
                    if alias.name == "importlib"
                )
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                modules = [node.module or ""]
            elif isinstance(node, ast.Call):
                expression = ast.unparse(node.func)
                if expression in dynamic_aliases:
                    if (
                        node.args
                        and isinstance(node.args[0], ast.Constant)
                        and isinstance(node.args[0].value, str)
                    ):
                        modules = [node.args[0].value]
                    elif not is_evaluation:
                        findings.append(
                            ImportFinding(
                                relative,
                                node.lineno,
                                expression,
                                "Unresolved dynamic import outside official evaluation boundary",
                            )
                        )
                if (
                    expression in {"torch.hub.load", "torch.hub.load_state_dict_from_url"}
                    and not is_evaluation
                ):
                    findings.append(
                        ImportFinding(
                            relative,
                            node.lineno,
                            expression,
                            "Implicit remote repository/weight loading is not a native runtime operation",
                        )
                    )
            for module in modules:
                imports.append({"path": relative, "line": node.lineno, "module": module})
                if module.split(".")[0] not in allowed:
                    findings.append(
                        ImportFinding(
                            relative,
                            node.lineno,
                            module,
                            "Undeclared dependency outside the native/evaluation allowlist",
                        )
                    )
    return {
        "schema_version": 1,
        "files": files,
        "imports": imports,
        "findings": [asdict(finding) for finding in findings],
        "passed": not findings,
        "evidence": "static_import_syntax_not_a_security_sandbox_or_full_equivalence_proof",
    }
