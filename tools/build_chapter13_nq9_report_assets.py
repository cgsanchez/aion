"""Copy authenticated NQ9 figures into the portable report asset directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthesis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    synthesis = arguments.synthesis.resolve()
    output = arguments.output.resolve()
    completed_path = synthesis / "completed.json"
    with completed_path.open(encoding="utf-8") as handle:
        completed = json.load(handle)
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}
    for source in sorted((synthesis / "figures").glob("*.png")):
        relative = f"figures/{source.name}"
        expected = completed["artifact_sha256"][relative]
        actual = _sha256(source)
        if actual != expected:
            raise RuntimeError(f"synthesis figure hash mismatch: {source}")
        destination = output / source.name
        shutil.copyfile(source, destination)
        copied = _sha256(destination)
        if copied != expected:
            raise RuntimeError(f"copied figure hash mismatch: {destination}")
        manifest[source.name] = copied
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "aion.chapter13.nq9-report-figures.v1",
                "synthesis_completed_sha256": _sha256(completed_path),
                "figures": manifest,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "figures": manifest}, indent=2))


if __name__ == "__main__":
    main()
