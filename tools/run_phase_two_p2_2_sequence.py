#!/usr/bin/env python3
"""Run or resume the fixed Phase Two P2-2 bridge without host oversubscription."""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
from pathlib import Path
from typing import TextIO

_STAGES = (
    "prepare-co",
    "run-co-cpu",
    "run-co-gpu",
    "prepare-nh3",
    "run-nh3-cpu",
    "run-nh3-gpu",
    "run-nh3-bare-cpu",
    "run-nh3-bare-gpu",
    "analyze",
)
_GPU_STAGES = frozenset(("run-co-gpu", "run-nh3-gpu", "run-nh3-bare-gpu"))
_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _completed(output: Path, stage: str) -> bool:
    return (output / "stages" / f"{stage}.json").is_file()


def _acquire_sequence_lock(output: Path) -> TextIO:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "sequence.lock"
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f"another P2-2 sequence owns {path}") from None
    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return handle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host-threads", type=int, default=8)
    arguments = parser.parse_args()
    if arguments.host_threads < 1:
        parser.error("--host-threads must be positive")

    output = arguments.output.expanduser().resolve()
    with _acquire_sequence_lock(output):
        repo = Path(__file__).resolve().parents[1]
        driver = repo / "tools/run_phase_two_p2_2_bridge.py"
        gpu_python = repo / "tools/gpu-python"
        environment = os.environ.copy()
        thread_count = str(arguments.host_threads)
        for name in _THREAD_VARIABLES:
            environment[name] = thread_count
        environment["AION_HOST_THREADS"] = thread_count

        for stage in _STAGES:
            if _completed(output, stage):
                print(f"P2-2: skipping completed stage {stage}", flush=True)
                continue
            if stage in _GPU_STAGES:
                command = (
                    str(gpu_python),
                    str(driver),
                    "--output",
                    str(output),
                    "--stage",
                    stage,
                )
            else:
                command = (
                    sys.executable,
                    str(driver),
                    "--output",
                    str(output),
                    "--stage",
                    stage,
                )
            print(f"P2-2: starting stage {stage}", flush=True)
            subprocess.run(command, cwd=repo, env=environment, check=True)
            print(f"P2-2: completed stage {stage}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
