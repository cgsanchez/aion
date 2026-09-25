"""Thin command-line front end to the same typed Aion API."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import cast

from aion.config import (
    AionConfig,
    CompiledSourceConfig,
    ReferenceConfig,
    ReferenceOutputConfig,
    SimulationConfig,
    WilsonSimulationConfig,
    WilsonStationaryConfig,
    WilsonStationaryOutputConfig,
    dumps_config,
    load_config,
)
from aion.errors import AionError, ConfigurationError, RunCancelledError, SchemaError
from aion.io import validate_artifact
from aion.io.export import export_trajectory_csv
from aion.workflows import (
    build_simulation,
    load_reference,
    prepare_reference,
    resume,
    run,
)


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
    resume_parser.add_argument("--output", type=Path)

    inspect_parser = subparsers.add_parser("inspect", help="validate an Aion HDF5 header")
    inspect_parser.add_argument("artifact", type=Path)

    export_parser = subparsers.add_parser("export", help="export a completed artifact")
    export_parser.add_argument("artifact", type=Path)
    export_parser.add_argument("output", type=Path)
    export_parser.add_argument(
        "--observable",
        action="append",
        dest="observables",
        help="export only this definition ID (repeatable)",
    )
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
            trajectory = resume(
                cast(Path, args.checkpoint),
                output=cast(Path | None, args.output),
            )
            print(f"run_id = {trajectory.run_id}")
            print(f"trajectory = {trajectory.path}")
            return 0
        if command == "export":
            selected = cast(list[str] | None, args.observables)
            outputs = export_trajectory_csv(
                cast(Path, args.artifact),
                cast(Path, args.output),
                observable_ids=None if selected is None else tuple(selected),
            )
            for output in outputs:
                print(output)
            return 0
        raise AssertionError(f"unhandled command {command!r}")
    except (ConfigurationError, SchemaError) as exc:
        print(f"aion: {exc}", file=sys.stderr)
        return 2
    except RunCancelledError as exc:
        print(f"aion: {exc}", file=sys.stderr)
        return 130
    except AionError as exc:
        print(f"aion: {exc}", file=sys.stderr)
        return 1


def _configuration_command(command: str, args: argparse.Namespace) -> int:
    configuration_path = cast(Path, args.configuration).expanduser().resolve()
    resolved = load_config(configuration_path)
    config = _resolve_operational_paths(resolved.config, configuration_path.parent)
    print(dumps_config(config), end="")
    print(f"scientific_id = {config.scientific_id}")
    if cast(bool, args.validate_only):
        return 0
    if command == "prepare":
        if not isinstance(config, ReferenceConfig):
            raise ConfigurationError("prepare requires an aion.reference-input document")
        reference = prepare_reference(config)
        reference.save()
        print(f"reference = {config.output.artifact_path}")
        return 0
    if not isinstance(config, SimulationConfig):
        raise ConfigurationError("run requires an aion.simulation-input document")
    reference = load_reference(config.reference.path, backend=config.backend)
    simulation = build_simulation(
        config,
        reference,
        original_toml=resolved.original_toml,
    )
    trajectory = run(simulation)
    print(f"run_id = {trajectory.run_id}")
    print(f"trajectory = {trajectory.path}")
    return 0


def _resolve_operational_paths(
    config: AionConfig,
    base: Path,
) -> AionConfig:
    """Resolve only path-like execution fields relative to the input document."""

    def resolve(path: Path) -> Path:
        expanded = path.expanduser()
        return expanded.resolve() if expanded.is_absolute() else (base / expanded).resolve()

    if isinstance(config, ReferenceConfig):
        return replace(
            config,
            output=ReferenceOutputConfig(resolve(config.output.artifact_path)),
        )
    if isinstance(config, WilsonStationaryConfig):
        return replace(
            config,
            reference=replace(config.reference, path=resolve(config.reference.path)),
            output=WilsonStationaryOutputConfig(resolve(config.output.artifact_path)),
        )
    if isinstance(config, WilsonSimulationConfig):
        return replace(
            config,
            reference=replace(config.reference, path=resolve(config.reference.path)),
            stationary_state=replace(
                config.stationary_state,
                path=resolve(config.stationary_state.path),
            ),
            output=replace(config.output, directory=resolve(config.output.directory)),
        )
    source = config.source
    if isinstance(source, CompiledSourceConfig):
        source = replace(source, path=resolve(source.path))
    return replace(
        config,
        reference=replace(config.reference, path=resolve(config.reference.path)),
        source=source,
        output=replace(config.output, directory=resolve(config.output.directory)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
