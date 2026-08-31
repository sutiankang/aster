"""Offline source/documentation hygiene checks; no third-party dependencies."""

import argparse
import ast
from html.parser import HTMLParser
import io
from pathlib import Path, PurePosixPath
import posixpath
import re
import sys
import tokenize
from urllib.parse import unquote
import xml.etree.ElementTree as ET


HAN = re.compile(r"[\u3400-\u9fff]")
PRIVATE = re.compile(
    r"[A-Za-z]:[/\\]+Users[/\\]+|https?://[^\s<>)]*(?:feishu\.cn|larksuite\.com)/(?:wiki|docx)/|aster_validation_\d+"
)
LINK = re.compile(r"\[[^\]\n]*\]\(([^\s)]+)(?:\s+\"[^\"]*\")?\)")


class DocumentHTML(HTMLParser):
    """Collect navigation and image references that Markdown regexes cannot see."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links = []
        self.missing_alt = 0

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        key = {"a": "href", "img": "src"}.get(tag)
        if key and attributes.get(key):
            self.links.append(attributes[key])
        if tag == "img" and not (attributes.get("alt") or "").strip():
            self.missing_alt += 1


def exact_path_case(root, relative):
    """Check URL spelling even on a case-insensitive development filesystem."""
    current = root
    for part in PurePosixPath(relative).parts:
        if part not in {entry.name for entry in current.iterdir()}:
            return False
        current = current / part
    return True


def check(root):
    root = Path(root).resolve(strict=True)
    findings = []
    source_files = []
    for folder in ("src", "tests", "examples", "tools"):
        source_files.extend(sorted((root / folder).rglob("*.py")))
    for path in source_files:
        relative = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8-sig")
        try:
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(
                    node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    prose = ast.get_docstring(node)
                    if prose and HAN.search(prose):
                        findings.append(f"{relative}: use English for docstrings")
            for token in tokenize.generate_tokens(io.StringIO(text).readline):
                if token.type == tokenize.COMMENT and HAN.search(token.string):
                    findings.append(f"{relative}:{token.start[0]}: use English for comments")
        except (SyntaxError, tokenize.TokenError) as error:
            findings.append(f"{relative}: {error}")

    documents = sorted(root.glob("*.md")) + sorted((root / "docs").rglob("*.md"))
    for path in documents:
        relative = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8-sig")
        if PRIVATE.search(text):
            findings.append(f"{relative}: personal path or private document reference")
        prose = []
        fence = None
        for line in text.splitlines():
            marker = re.match(r"^\s*(`{3,}|~{3,})", line)
            if marker:
                if fence is None:
                    fence = marker.group(1)
                elif marker.group(1)[0] == fence[0] and len(marker.group(1)) >= len(fence):
                    fence = None
                continue
            if fence is None:
                prose.append(re.sub(r"`[^`\n]*`", "", line))
        visible = "\n".join(prose)
        html = DocumentHTML()
        html.feed(visible)
        if html.missing_alt:
            findings.append(f"{relative}: HTML images require descriptive alt text")
        for raw in LINK.findall(visible) + html.links:
            target = unquote(raw.strip("<>"))
            if re.match(r"(?:https?://|mailto:|#)", target):
                continue
            target = target.split("#", 1)[0]
            if not target:
                continue
            destination = (path.parent / target).resolve()
            if not destination.is_relative_to(root) or not destination.exists():
                findings.append(f"{relative}: broken or escaping link: {target}")
            else:
                lexical = posixpath.normpath(
                    posixpath.join(path.relative_to(root).parent.as_posix(), target)
                )
                if not exact_path_case(root, lexical):
                    findings.append(f"{relative}: link path case differs from the file: {target}")
    assets = sorted((root / "docs" / "assets").rglob("*.svg"))
    for path in assets:
        relative = path.relative_to(root).as_posix()
        try:
            asset = ET.fromstring(path.read_text(encoding="utf-8-sig"))
        except ET.ParseError as error:
            findings.append(f"{relative}: invalid SVG: {error}")
            continue
        namespace = "{http://www.w3.org/2000/svg}"
        if asset.tag != namespace + "svg":
            findings.append(f"{relative}: expected an SVG root")
        for tag in ("title", "desc"):
            element = asset.find(namespace + tag)
            if element is None or not "".join(element.itertext()).strip():
                findings.append(f"{relative}: SVG requires a descriptive {tag}")
        for element in asset.iter():
            tag = element.tag.rsplit("}", 1)[-1]
            if tag in ("script", "foreignObject"):
                findings.append(f"{relative}: SVG must be a static image")
            for name, value in element.attrib.items():
                local_name = name.rsplit("}", 1)[-1]
                if local_name.lower().startswith("on"):
                    findings.append(f"{relative}: SVG event handlers are not allowed")
                if local_name == "href" and not value.startswith("#"):
                    findings.append(f"{relative}: SVG references must be self-contained")
    return {
        "python_files": len(source_files),
        "documents": len(documents),
        "svg_assets": len(assets),
        "findings": findings,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=Path(__file__).resolve().parents[1])
    result = check(parser.parse_args().root)
    for finding in result["findings"]:
        print(finding)
    print(
        f"Checked {result['python_files']} Python files and {result['documents']} documents; "
        f"{result['svg_assets']} SVG assets; {len(result['findings'])} findings."
    )
    return bool(result["findings"])


if __name__ == "__main__":
    sys.exit(main())
