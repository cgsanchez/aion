"""Verify the authenticated Chapter 13 NQ9 synthesis and report package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _verify(path: Path, expected: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(f"hash mismatch for {path}: {actual} != {expected}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/chapter13_nq9_report_manifest.json"),
    )
    arguments = parser.parse_args()
    repo = arguments.repo.resolve()
    manifest_path = (repo / arguments.manifest).resolve()
    manifest = _load(manifest_path)

    checked: list[Path] = []
    synthesis = manifest["synthesis"]
    synthesis_root = Path(synthesis["root"])
    for name in ("result", "provenance", "completed"):
        path = synthesis_root / f"{name}.json"
        _verify(path, synthesis[f"{name}_sha256"])
        checked.append(path)

    completed = _load(synthesis_root / "completed.json")
    _verify(synthesis_root / "result.json", completed["result_sha256"])
    _verify(synthesis_root / "provenance.json", completed["provenance_sha256"])
    for relative, expected in completed["artifact_sha256"].items():
        path = synthesis_root / relative
        _verify(path, expected)
        checked.append(path)

    report = manifest["report"]
    for path_key, hash_key in (
        ("entrypoint", "entrypoint_sha256"),
        ("chapter14_outline", "chapter14_outline_sha256"),
        ("latex_source", "latex_source_sha256"),
        ("pdf", "pdf_sha256"),
        ("figure_manifest", "figure_manifest_sha256"),
    ):
        path = repo / report[path_key]
        _verify(path, report[hash_key])
        checked.append(path)

    figure_manifest = _load(repo / report["figure_manifest"])
    figure_root = (repo / report["figure_manifest"]).parent
    for name, expected in figure_manifest["figures"].items():
        path = figure_root / name
        _verify(path, expected)
        checked.append(path)

    for path_key, hash_key in (
        ("synthesis", "synthesis_sha256"),
        ("report_assets", "report_assets_sha256"),
    ):
        path = repo / manifest["tools"][path_key]
        _verify(path, manifest["tools"][hash_key])
        checked.append(path)

    print(
        json.dumps(
            {
                "status": f"verified_{manifest['status']}",
                "manifest": str(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
                "verified_file_count": len(set(checked)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
