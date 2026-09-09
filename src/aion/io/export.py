"""Explicit one-way exports from canonical trajectory artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from aion.errors import TrajectoryError
from aion.io.trajectory import Trajectory, load_trajectory
from aion.io.util import publish_text


def _column_names(shape: tuple[int, ...], *, complex_values: bool) -> list[str]:
    indices = list(np.ndindex(shape)) if shape else [()]
    labels = ["_".join(str(index) for index in item) for item in indices]
    bases = ["value" if not label else f"value_{label}" for label in labels]
    if not complex_values:
        return bases
    return [component for base in bases for component in (f"{base}_real", f"{base}_imag")]


def _rows(values: np.ndarray) -> np.ndarray:
    flat = values.reshape(values.shape[0], -1)
    if not np.iscomplexobj(flat):
        return np.asarray(flat, dtype=np.float64)
    result = np.empty((flat.shape[0], 2 * flat.shape[1]), dtype=np.float64)
    result[:, 0::2] = flat.real
    result[:, 1::2] = flat.imag
    return result


def export_trajectory_csv(
    trajectory: Trajectory | str | Path,
    directory: str | Path,
    *,
    observable_ids: tuple[str, ...] | None = None,
) -> tuple[Path, ...]:
    """Export each selected independently sampled observable to its own CSV."""

    source = load_trajectory(trajectory) if isinstance(trajectory, str | Path) else trajectory
    selected = source.observable_ids if observable_ids is None else tuple(observable_ids)
    unknown = sorted(set(selected) - set(source.observable_ids))
    if unknown:
        raise TrajectoryError("unknown observable(s) for CSV export: " + ", ".join(unknown))
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    records: list[dict[str, object]] = []
    for definition_id in selected:
        series = source.read_observable(definition_id)
        path = target / f"{definition_id.replace('.', '__')}.csv"
        if path.exists():
            raise TrajectoryError(f"refusing to overwrite CSV export {path}")
        columns = _column_names(
            series.definition.shape,
            complex_values=np.iscomplexobj(series.values),
        )
        temporary = path.with_name(f".{path.name}.partial")
        try:
            with temporary.open("x", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("global_step", "time_au", *columns))
                numeric = _rows(series.values)
                for index in range(series.steps.size):
                    writer.writerow(
                        (
                            int(series.steps[index]),
                            repr(float(series.times_au[index])),
                            *(repr(float(value)) for value in numeric[index]),
                        )
                    )
            try:
                path.hardlink_to(temporary)
            except FileExistsError:
                raise TrajectoryError(f"refusing to overwrite CSV export {path}") from None
        finally:
            if temporary.exists():
                temporary.unlink()
        outputs.append(path)
        records.append(
            {
                "definition_id": definition_id,
                "file": path.name,
                "mathematical_definition": series.definition.mathematical_definition,
                "originating_formulation": series.definition.originating_formulation,
                "physical_dimension": series.definition.physical_dimension.value,
                "sampling_location": series.definition.sampling_location.value,
                "shape": list(series.definition.shape),
                "unit": series.definition.unit.value,
            }
        )
    manifest = target / "export_manifest.json"
    try:
        publish_text(
            manifest,
            json.dumps(
                {
                    "schema": "aion.csv-export",
                    "schema_version": "1.0.0",
                    "source_trajectory": str(source.path.resolve()),
                    "source_trajectory_sha256": source.sha256,
                    "streams": records,
                },
                sort_keys=True,
                indent=2,
            )
            + "\n",
        )
    except FileExistsError as exc:
        raise TrajectoryError(str(exc)) from exc
    return (*outputs, manifest)
