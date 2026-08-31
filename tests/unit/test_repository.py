from pathlib import Path
import ast
import re
import runpy
import tomllib


ROOT = Path(__file__).resolve().parents[2]


def test_release_version_matches_package_and_public_status():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8-sig"))
    version = metadata["project"]["version"]
    module = ast.parse((ROOT / "src/aster/__init__.py").read_text(encoding="utf-8-sig"))
    versions = [
        ast.literal_eval(node.value)
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets
        )
    ]
    assert versions == [version]
    status = (ROOT / "docs/STATUS.md").read_text(encoding="utf-8-sig")
    assert f"# Release status — v{version}" in status
    for name in ("README.md", "README.zh-CN.md"):
        homepage = (ROOT / name).read_text(encoding="utf-8-sig")
        assert f"v{version}" in homepage and "docs/STATUS.md" in homepage


def test_test_extra_declares_fixture_dependencies_without_expanding_core_runtime():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8-sig"))
    project = metadata["project"]

    def names(requirements):
        return {re.split(r"[<>=!~;\s\[]", item, maxsplit=1)[0].lower() for item in requirements}

    assert {"pytest", "pillow", "safetensors"} <= names(project["optional-dependencies"]["test"])
    assert not {"pillow", "safetensors"} & names(project["dependencies"])


def test_homepages_expose_roadmap_in_primary_navigation():
    for name in ("README.md", "README.zh-CN.md"):
        homepage = (ROOT / name).read_text(encoding="utf-8-sig")
        navigation = homepage.split("\n---", 1)[0]
        assert 'href="docs/ROADMAP.md"' in navigation
    roadmap = (ROOT / "docs/ROADMAP.md").read_text(encoding="utf-8-sig")
    assert "## Milestones and acceptance criteria" in roadmap
    assert "STATUS.md" in roadmap and "scope/capabilities.json" in roadmap


def test_repository_documentation_and_source_hygiene():
    checker = runpy.run_path(str(ROOT / "tools/check_repository.py"))
    result = checker["check"](ROOT)
    assert not result["findings"], result["findings"]


def test_quickstart_updates_only_adapter_and_merges():
    example = runpy.run_path(str(ROOT / "examples/quickstart.py"))
    result = example["run"]()
    assert result["base_unchanged"]
    assert result["updates"] == 8
    assert 0 < result["trainable_parameters"] < result["total_parameters"]
    assert result["merge_max_absolute_error"] < 1e-5


def test_repository_checker_allows_unicode_data_but_rejects_prose(tmp_path):
    checker = runpy.run_path(str(ROOT / "tools/check_repository.py"))["check"]
    (tmp_path / "src").mkdir()
    source = tmp_path / "src/sample.py"
    source.write_text("value = '\\u4e2d\\u6587'\n", encoding="utf-8")
    assert not checker(tmp_path)["findings"]
    source.write_text("# " + chr(0x4E2D) + chr(0x6587) + "\nvalue = 1\n", encoding="utf-8")
    assert checker(tmp_path)["findings"]


def test_repository_checker_ignores_code_but_checks_real_links(tmp_path):
    checker = runpy.run_path(str(ROOT / "tools/check_repository.py"))["check"]
    (tmp_path / "README.md").write_text(
        "`[B,H](K,V)`\n~~~python\n[B,H](K,V)\n~~~\n[missing](absent.md)\n",
        encoding="utf-8",
    )
    findings = checker(tmp_path)["findings"]
    assert len(findings) == 1
    assert "absent.md" in findings[0]


def test_repository_checker_checks_html_links_and_image_alt(tmp_path):
    checker = runpy.run_path(str(ROOT / "tools/check_repository.py"))["check"]
    (tmp_path / "present.md").write_text("# Present\n", encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text(
        '<a href="present.md">Present</a>\n'
        '<a href="missing.md">Missing</a>\n'
        '<img src="missing.svg">\n'
        '<a href="https://example.com/">External</a>\n'
        '~~~html\n<img src="ignored.svg">\n~~~\n',
        encoding="utf-8",
    )
    findings = checker(tmp_path)["findings"]
    assert len(findings) == 3
    assert any("missing.md" in item for item in findings)
    assert any("missing.svg" in item for item in findings)
    assert any("alt text" in item for item in findings)
    readme.write_text(
        '<a href="present.md#present">Present</a>\n'
        '<img src="present.md" alt="A descriptive example">\n',
        encoding="utf-8",
    )
    assert not checker(tmp_path)["findings"]


def test_repository_checker_rejects_html_path_escape(tmp_path):
    checker = runpy.run_path(str(ROOT / "tools/check_repository.py"))["check"]
    (tmp_path / "README.md").write_text('<a href="../outside.md">Outside</a>\n', encoding="utf-8")
    assert "escaping link" in checker(tmp_path)["findings"][0]


def test_repository_checker_rejects_case_mismatches_on_any_platform(tmp_path):
    checker = runpy.run_path(str(ROOT / "tools/check_repository.py"))["check"]
    (tmp_path / "Guide.md").write_text("# Guide\n", encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text("[Guide](guide.md)\n", encoding="utf-8")
    assert checker(tmp_path)["findings"]
    readme.write_text("[Guide](Guide.md)\n", encoding="utf-8")
    assert not checker(tmp_path)["findings"]


def test_repository_checker_checks_static_accessible_svg(tmp_path):
    checker = runpy.run_path(str(ROOT / "tools/check_repository.py"))["check"]
    assets = tmp_path / "docs/assets"
    assets.mkdir(parents=True)
    svg = assets / "example.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg">'
        "<title>Example</title><desc>A simple mark</desc>"
        '<path id="mark" d="M0 0L1 1"/><use href="#mark"/></svg>',
        encoding="utf-8",
    )
    result = checker(tmp_path)
    assert result["svg_assets"] == 1 and not result["findings"]
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" onload="alert(1)">'
        '<script>bad()</script><image href="https://example.com/image.png"/></svg>',
        encoding="utf-8",
    )
    findings = checker(tmp_path)["findings"]
    assert len(findings) == 5
    svg.write_text("<svg>", encoding="utf-8")
    assert "invalid SVG" in checker(tmp_path)["findings"][0]
