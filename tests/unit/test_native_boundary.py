from pathlib import Path

from aster.core.native_boundary import audit_native_boundary


def test_runtime_keeps_models_trainers_inference_and_agents_native():
    root = Path(__file__).resolve().parents[2] / "src" / "aster"
    report = audit_native_boundary(root)
    assert report["passed"], report["findings"]
    assert len(report["files"]) > 70


def test_lazy_high_level_wrappers_and_dynamic_imports_are_detected(tmp_path):
    root = tmp_path / "aster"
    root.mkdir()
    (root / "evaluation").mkdir()
    (root / "bad.py").write_text(
        "def forward():\n    from transformers import AutoModel\n    return AutoModel\n",
        encoding="utf-8",
    )
    (root / "lazy.py").write_text(
        "from importlib import import_module as load\ndef f(name):\n    return load(name)\n",
        encoding="utf-8",
    )
    (root / "evaluation" / "ok.py").write_text("from cleanfid import fid\n", encoding="utf-8")
    report = audit_native_boundary(root)
    assert not report["passed"]
    assert {item["path"] for item in report["findings"]} == {"bad.py", "lazy.py"}


def test_media_codec_and_native_kernel_exceptions_are_file_scoped(tmp_path):
    root = tmp_path / "aster"
    root.mkdir()
    (root / "data").mkdir()
    (root / "models").mkdir()
    (root / "optimization").mkdir()
    (root / "data/qwen_vl.py").write_text("from PIL import Image\n", encoding="utf-8")
    (root / "optimization/_triton_attention.py").write_text(
        "import triton\nimport triton.language as tl\n", encoding="utf-8"
    )
    (root / "data/unrelated.py").write_text("from PIL import Image\n", encoding="utf-8")
    (root / "models/wrapper.py").write_text(
        "import triton\nfrom transformers import AutoModel\n", encoding="utf-8"
    )
    report = audit_native_boundary(root)
    assert not report["passed"]
    assert {item["path"] for item in report["findings"]} == {
        "data/unrelated.py",
        "models/wrapper.py",
    }
