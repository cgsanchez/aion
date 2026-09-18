#!/usr/bin/env python3
"""Authenticate and validate one WP6 exact linear-dynamics execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from aion.config import canonical_sha256


def _arguments() -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("execution_directory", type=Path)
    value: Path = parser.parse_args().execution_directory
    return value.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"not a JSON object: {path}")
    return dict(value)


def _identity(value: dict[str, Any], key: str) -> str:
    core = dict(value)
    identity = str(core.pop(key))
    if canonical_sha256(core) != identity:
        raise ValueError(f"canonical {key} mismatch")
    return identity


def main() -> None:
    execution = _arguments()
    plan_path = execution / "execution_plan.json"
    result_path = execution / "result.json"
    arrays_path = execution / "arrays.npz"
    completed_path = execution / "completed.json"
    plan = _json(plan_path)
    plan_id = _identity(plan, "plan_id")
    result = _json(result_path)
    completed = _json(completed_path)
    if result["plan_id"] != plan_id or completed["plan_id"] != plan_id:
        raise ValueError("WP6 artifacts do not reference one execution plan")
    for path, field in (
        (plan_path, "execution_plan_sha256"),
        (result_path, "result_sha256"),
        (arrays_path, "arrays_sha256"),
    ):
        if _sha256(path) != completed[field]:
            raise ValueError(f"WP6 artifact hash mismatch: {path}")
    if result["status"] != completed["status"]:
        raise ValueError("WP6 result and completion statuses differ")
    qualification = result["status"] == "executed_unreviewed"
    if result["status"] not in {"executed_unreviewed", "smoke_not_qualification"}:
        raise ValueError("WP6 execution has an incomplete status")

    semantic = result["array_semantic_sha256"]
    required_history = {
        f"h2_matrix_history__{location}__{family}"
        for location in ("endpoint", "midpoint")
        for family in (
            "times_au",
            "magnetic_field_au",
            "magnetic_field_dot_au",
            "electric_field_origin_au",
            "metric",
            "mechanical",
            "connection",
            "metric_dot",
            "ordinary_derivative_matrix",
        )
    }
    with np.load(arrays_path, allow_pickle=False) as arrays:
        if set(arrays.files) != set(semantic):
            raise ValueError("WP6 array inventory mismatch")
        if not required_history.issubset(arrays.files):
            raise ValueError("WP6 exact matrix-history fixture is incomplete")
        for name in arrays.files:
            value = np.asarray(arrays[name])
            if not np.all(np.isfinite(value)):
                raise ValueError(f"non-finite WP6 array: {name}")
            if canonical_sha256(value) != semantic[name]:
                raise ValueError(f"WP6 semantic array hash mismatch: {name}")
        for location in ("endpoint", "midpoint"):
            mechanical = arrays[f"h2_matrix_history__{location}__mechanical"]
            connection = arrays[f"h2_matrix_history__{location}__connection"]
            ordinary = arrays[
                f"h2_matrix_history__{location}__ordinary_derivative_matrix"
            ]
            if not np.allclose(ordinary, mechanical - 1j * connection, atol=0.0, rtol=0.0):
                raise ValueError("WP6 ordinary-derivative matrix identity failed")
        field_dot = arrays["h2_matrix_history__midpoint__magnetic_field_dot_au"]
        if not np.any(np.linalg.norm(field_dot, axis=1) > 0.0):
            raise ValueError("WP6 history does not contain a time-dependent magnetic field")

    tolerances = plan["fixture"]["acceptance_tolerances"]
    checks = result["matrix_checks"]
    if max(
        float(checks["direct_factorized_connection_residual"]),
        float(checks["direct_factorized_metric_dot_residual"]),
    ) > float(tolerances["direct_factorized_relative"]):
        raise ValueError("WP6 direct/factorized temporal identity failed")
    if float(checks["metric_compatibility_residual"]) > float(
        tolerances["metric_compatibility_relative"]
    ):
        raise ValueError("WP6 exact metric compatibility failed")
    if float(checks["metric_rate_finite_difference_residual"]) > float(
        tolerances["metric_rate_finite_difference_relative"]
    ):
        raise ValueError("WP6 analytic metric rate failed its finite-difference check")
    if float(checks["maximum_matrix_gauge_covariance_residual"]) > float(
        tolerances["matrix_gauge_covariance_relative"]
    ):
        raise ValueError("WP6 matrix gauge covariance failed")

    static_orders = [
        float(row["measured_order_to_next"])
        for row in result["static_spectral_convergence"]
        if row["measured_order_to_next"] is not None
    ]
    if not static_orders or min(static_orders) < float(tolerances["minimum_static_pade_order"]):
        raise ValueError("WP6 static Padé convergence order is too low")

    trajectory_rows = [*result["h2_trajectory_rows"], *result["h3_trajectory_rows"]]
    maximum_norm_drift = max(
        float(row["maximum_corrected_norm_drift"]) for row in trajectory_rows
    )
    minimum_metric = min(
        float(row["minimum_endpoint_metric_eigenvalue"]) for row in trajectory_rows
    )
    maximum_compatibility = max(
        [float(checks["metric_compatibility_residual"]),]
        + [
            float(row["maximum_metric_compatibility_residual"])
            for row in result["h3_trajectory_rows"]
        ]
    )
    if maximum_norm_drift > float(tolerances["corrected_norm_drift"]):
        raise ValueError("WP6 corrected physical norm drift is too large")
    if minimum_metric <= float(tolerances["minimum_metric_eigenvalue"]):
        raise ValueError("WP6 trajectory crossed a metric-positivity boundary")
    if maximum_compatibility > float(tolerances["metric_compatibility_relative"]):
        raise ValueError("WP6 trajectory metric compatibility failed")

    time_orders = [
        float(row["measured_order_to_next"])
        for row in result["h2_timestep_convergence"]
        if row["measured_order_to_next"] is not None
    ]
    gauge_orders = [
        float(row["measured_order_to_next"])
        for row in result["h2_gauge_trajectory_rows"]
        if row["measured_order_to_next"] is not None
    ]
    if qualification and (
        not time_orders
        or min(time_orders) < float(tolerances["minimum_time_convergence_order"])
    ):
        raise ValueError("WP6 time-dependent trajectory convergence order is too low")
    if qualification and (
        not gauge_orders
        or min(gauge_orders) < float(tolerances["minimum_time_convergence_order"])
    ):
        raise ValueError("WP6 trajectory gauge residual does not converge fast enough")

    restart = result["h2_restart_check"]
    if float(restart["coefficient_residual"]) != 0.0 or not bool(restart["bitwise_equal"]):
        raise ValueError("WP6 in-memory restart differs from uninterrupted propagation")
    if restart["persistence_status"] != "deferred_to_stage_c_after_g6":
        raise ValueError("WP6 persistence-stage boundary is not explicit")

    expected_h3_rows = 108 if qualification else 36
    if len(result["h3_trajectory_rows"]) != expected_h3_rows:
        raise ValueError("WP6 three-centre model/case inventory is incomplete")
    if qualification and not result["h3_quadrature_stability"]:
        raise ValueError("WP6 lacks three-centre quadrature-refinement comparisons")

    print(f"execution={execution}")
    print(f"status=authenticated_{result['status']}")
    print(f"arrays_verified={len(semantic)}")
    print(f"maximum_norm_drift={maximum_norm_drift:.6e}")
    print(f"minimum_metric_eigenvalue={minimum_metric:.6e}")
    print(f"maximum_metric_compatibility_residual={maximum_compatibility:.6e}")
    print(f"minimum_static_pade_order={min(static_orders):.6f}")
    if time_orders:
        print(f"minimum_time_convergence_order={min(time_orders):.6f}")
    if gauge_orders:
        print(f"minimum_gauge_convergence_order={min(gauge_orders):.6f}")


if __name__ == "__main__":
    main()
