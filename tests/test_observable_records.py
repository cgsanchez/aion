from __future__ import annotations

import numpy as np
import pytest

from aion.config import AtomicUnit, PhysicalDimension
from aion.errors import SchemaError
from aion.observables import ObservableDefinition, ObservableRecord, SamplingLocation

pytestmark = pytest.mark.fast


def definition() -> ObservableDefinition:
    return ObservableDefinition(
        definition_id="current.mechanical.total",
        originating_formulation="bare_velocity_gauge",
        physical_dimension=PhysicalDimension.CURRENT,
        unit=AtomicUnit.CURRENT,
        shape=(3,),
        sampling_location=SamplingLocation.ENDPOINT,
        mathematical_definition="q Re Tr[P (p - q a S)] / m",
        decomposition_labels=("x", "y", "z"),
    )


def test_observable_record_copies_and_freezes_values() -> None:
    values = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    record = ObservableRecord(definition(), 2, 0.1, values)
    values[0] = 99.0
    assert record.values[0] == 1.0
    assert not record.values.flags.writeable
    assert record.definition.hdf5_path == "/observables/current/mechanical/total"


def test_observable_record_rejects_ambiguous_id_shape_and_precision() -> None:
    with pytest.raises(SchemaError, match="namespaced"):
        ObservableDefinition(
            definition_id="current",
            originating_formulation="bare_velocity_gauge",
            physical_dimension=PhysicalDimension.CURRENT,
            unit=AtomicUnit.CURRENT,
            shape=(3,),
            sampling_location=SamplingLocation.ENDPOINT,
            mathematical_definition="J",
        )
    with pytest.raises(SchemaError, match="shape"):
        ObservableRecord(definition(), 0, 0.0, np.zeros(2, dtype=np.float64))
    with pytest.raises(SchemaError, match="float64 or complex128"):
        ObservableRecord(definition(), 0, 0.0, np.zeros(3, dtype=np.float32))
