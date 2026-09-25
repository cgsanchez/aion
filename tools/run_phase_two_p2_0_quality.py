#!/usr/bin/env python3
"""Run and record the complete Phase Two P2-0 software-quality baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _run_gate(
    *,
    identifier: str,
    command: tuple[str, ...],
    repo: Path,
    output: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    started = perf_counter()
    completed = subprocess.run(
        command,
        cwd=repo,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = perf_counter() - started
    log_path = output / f"{identifier}.log"
    log_path.write_text(
        f"command: {' '.join(command)}\n"
        f"exit_code: {completed.returncode}\n"
        f"elapsed_seconds: {elapsed:.9f}\n"
        "\n--- stdout ---\n"
        f"{completed.stdout}"
        "\n--- stderr ---\n"
        f"{completed.stderr}",
        encoding="utf-8",
    )
    print(
        f"{identifier}: exit={completed.returncode} elapsed={elapsed:.2f}s",
        flush=True,
    )
    return {
        "identifier": identifier,
        "command": list(command),
        "exit_code": completed.returncode,
        "elapsed_seconds": elapsed,
        "passed": completed.returncode == 0,
        "log": log_path.name,
        "log_sha256": _sha256(log_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-gpu", action="store_true")
    arguments = parser.parse_args()
    output = arguments.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)

    repo = Path(__file__).resolve().parents[1]
    executable = Path(sys.executable)
    executable_directory = executable.parent
    one_thread = os.environ.copy()
    one_thread.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    eight_threads = os.environ.copy()
    eight_threads.update(
        {
            "OMP_NUM_THREADS": "8",
            "OPENBLAS_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
            "NUMEXPR_NUM_THREADS": "8",
        }
    )
    gates: list[tuple[str, tuple[str, ...], dict[str, str]]] = [
        ("ruff_check", (str(executable_directory / "ruff"), "check", "src", "tests"), one_thread),
        (
            "ruff_format_check",
            (str(executable_directory / "ruff"), "format", "--check", "src", "tests"),
            one_thread,
        ),
        ("mypy", (str(executable_directory / "mypy"),), one_thread),
        (
            "pytest_fast",
            (str(executable), "-m", "pytest", "-n", "8", "-m", "fast"),
            one_thread,
        ),
        (
            "pytest_integration",
            (str(executable), "-m", "pytest", "-q", "-m", "integration"),
            eight_threads,
        ),
    ]
    if arguments.include_gpu:
        gates.append(
            (
                "pytest_physical_gpu",
                (str(repo / "tools/gpu-python"), "-m", "pytest", "-q", "-m", "gpu"),
                one_thread,
            )
        )

    records = [
        _run_gate(
            identifier=identifier,
            command=command,
            repo=repo,
            output=output,
            environment=environment,
        )
        for identifier, command, environment in gates
    ]
    result = {
        "schema": "aion.phase-two.p2-0.quality-result",
        "schema_version": "1.0.0",
        "status": "executed_unreviewed",
        "all_passed": all(record["passed"] for record in records),
        "physical_gpu_included": arguments.include_gpu,
        "gates": records,
    }
    result_path = output / "result.json"
    _write_json(result_path, result)
    provenance = {
        "schema": "aion.phase-two.provenance",
        "schema_version": "1.0.0",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "repository": str(repo),
        "git_head": _git(repo, "rev-parse", "HEAD"),
        "git_status_porcelain": _git(
            repo,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ).splitlines(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "host": platform.node(),
        "thread_policy": {
            "fast_and_gpu": 1,
            "fast_workers": 8,
            "molecular_integration": 8,
        },
        "artifacts_sha256": {path.name: _sha256(path) for path in sorted(output.glob("*.log"))},
    }
    provenance_path = output / "provenance.json"
    _write_json(provenance_path, provenance)
    completed = {
        "schema": "aion.phase-two.completed",
        "schema_version": "1.0.0",
        "status": "executed_unreviewed",
        "result_sha256": _sha256(result_path),
        "provenance_sha256": _sha256(provenance_path),
    }
    _write_json(output / "completed.json", completed)
    print(json.dumps({"output": str(output), **completed}, indent=2), flush=True)
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
