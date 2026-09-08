"""Thin command-line front end to the same typed Aion API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

from aion.config import ReferenceConfig, SimulationConfig, load_config
from aion.errors import AionError, ConfigurationError, FeatureNotImplementedError, SchemaError
from aion.io import validate_artifact
from aion.workflows import prepare_reference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aion")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command, help_text in (
        ("prepare", "validate and prepare an immutable electronic reference"),
        ("run", "validate and execute one simulation"),
    ):
        subparser = subparsers.add_parser(command, help=help_text)
        subparser.add_argument("configuration", type=Path)
        subparser.add_argument(
            "--validate-only",
            action="store_true",
            help="print the resolved configuration and scientific ID without numerical work",
        )

    resume_parser = subparsers.add_parser("resume", help="resume an immutable checkpoint")
    resume_parser.add_argument("checkpoint", type=Path)

    inspect_parser = subparsers.add_parser("inspect", help="validate an Aion HDF5 header")
    inspect_parser.add_argument("artifact", type=Path)

    export_parser = subparsers.add_parser("export", help="export a completed artifact")
    export_parser.add_argument("artifact", type=Path)
    export_parser.add_argument("output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = cast(str, args.command)
    try:
        if command in {"prepare", "run"}:
            return _configuration_command(command, args)
        if command == "inspect":
            header = validate_artifact(cast(Path, args.artifact))
            print(
                json.dumps(
                    {
                        "artifact_id": header.artifact_id,
                        "artifact_kind": header.schema.kind.value,
                        "complete": header.complete,
                        "schema_name": header.schema.name,
                        "schema_version": str(header.artifact_version),
                    },
                    sort_keys=True,
                )
            )
            return 0
        if command == "resume":
            raise FeatureNotImplementedError("checkpoint restart is scheduled for WP5")
        if command == "export":
            raise FeatureNotImplementedError("artifact export is scheduled for WP5")
        raise AssertionError(f"unhandled command {command!r}")
    except (ConfigurationError, SchemaError) as exc:
        print(f"aion: {exc}", file=sys.stderr)
        return 2
    except FeatureNotImplementedError as exc:
        print(f"aion: {exc}", file=sys.stderr)
        return 3
    except AionError as exc:
        print(f"aion: {exc}", file=sys.stderr)
        return 1


def _configuration_command(command: str, args: argparse.Namespace) -> int:
    resolved = load_config(cast(Path, args.configuration))
    print(resolved.normalized_toml, end="")
    print(f"scientific_id = {resolved.scientific_id}")
    if cast(bool, args.validate_only):
        return 0
    if command == "prepare":
        if not isinstance(resolved.config, ReferenceConfig):
            raise ConfigurationError("prepare requires an aion.reference-input document")
        prepare_reference(resolved.config)
        return 0
    if not isinstance(resolved.config, SimulationConfig):
        raise ConfigurationError("run requires an aion.simulation-input document")
    raise FeatureNotImplementedError("simulation execution is scheduled for WP5")


if __name__ == "__main__":
    raise SystemExit(main())
