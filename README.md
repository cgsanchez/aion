# aion

Prototype code for real-time TDDFT experiments connected to the gauge-covariant
finite-basis formalism in `../Docs`.

The first target is deliberately narrow:

- pure length gauge,
- spatially uniform electric field,
- molecular Gaussian AO basis,
- PySCF AO density matrix propagation,
- adiabatic PySCF DFT matrix rebuilt at every time step.

The benzene example separates the expensive PySCF LR-TDDFT/Casida step from
the real-time propagation. The Casida script writes the low-lying singlet roots
to JSON, including the selected dipole-active transition used to tune the laser.

The real-time records include electron number, closed-shell idempotency error,
field-free molecular energy, laser coupling energy, total instantaneous energy,
electronic dipole, and electric field.

Two real-time propagators are currently available:

- `LengthGaugeRTTDDFT`: direct density-matrix leapfrog baseline.
- `LengthGaugeCNRTTDDFT`: occupied-orbital generalized Crank-Nicolson
  propagator for closed-shell RKS references.

Run the expensive Casida step once from the repository root:

```bash
python examples/benzene_length_gauge/run_casida.py
```

Then run the dynamics as often as needed without recomputing Casida:

```bash
python examples/benzene_length_gauge/run_benzene.py
```

The Crank-Nicolson version is:

```bash
python examples/benzene_length_gauge/run_benzene_cn.py
```

If PySCF is not installed in the active environment, the example attempts to use
the sibling source checkout at `../pyscf`.
