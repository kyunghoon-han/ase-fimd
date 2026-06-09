"""Regression tests for the specific bugs fixed during development.

Each test here corresponds to a concrete defect; if one of these fails, a known
issue has resurfaced. They run on EMT / Lennard-Jones only.
"""

from __future__ import annotations

import numpy as np
import pytest

from fimd.core import FIMDBasis, FIMDynamics

from .conftest import make_lj


# --------------------------------------------------------------------------- #
# Band mask: no zero-frequency (rigid/imaginary) mode may be active.
# Bug: rigid modes were removed by array index on a signed eigenvalue sort, so
# imaginary modes could leak into the active band and diverge.
# --------------------------------------------------------------------------- #
def test_band_mask_excludes_zero_frequency_modes(lj_reference, lj_trajectory):
    basis = FIMDBasis.from_trajectory(
        trajectory=lj_trajectory, reference=lj_reference, calculator=make_lj(),
        band=(0.0, 400.0), timestep_reference=2.0, verbose=False,
    )
    active_omega = basis.omega[basis.band_mask]
    # No active mode may have (near) zero angular frequency.
    assert np.all(np.abs(active_omega) > 1e-9)


def test_fimdynamics_guards_zero_frequency_active_mode(lj_reference, lj_trajectory):
    """Hand-building a basis with a zero-frequency active mode must fail loudly."""
    basis = FIMDBasis.from_trajectory(
        trajectory=lj_trajectory, reference=lj_reference, calculator=make_lj(),
        band=(0.0, 400.0), timestep_reference=2.0, verbose=False,
        max_initial_displacement=0.2,
    )
    # Force a zero-frequency mode into the active set.
    idx = np.argmin(np.abs(basis.omega))
    basis.omega[idx] = 0.0
    basis.band_mask[idx] = True

    atoms = lj_trajectory[-1].copy()
    atoms.calc = make_lj()
    with pytest.raises(ValueError):
        FIMDynamics(atoms, basis=basis, timestep_fs=1.0)


# --------------------------------------------------------------------------- #
# Step counter: FIMDynamics.nsteps must advance by exactly 1 per step.
# Bug: step() incremented nsteps on top of ASE's base class -> +2 per step,
# which doubled time labels and halved the effective save rate.
# --------------------------------------------------------------------------- #
def test_nsteps_increments_by_one(lj_reference, lj_trajectory):
    basis = FIMDBasis.from_trajectory(
        trajectory=lj_trajectory, reference=lj_reference, calculator=make_lj(),
        band=(0.0, 300.0), timestep_reference=2.0, verbose=False,
        max_initial_displacement=0.2,
    )
    atoms = lj_trajectory[-1].copy()
    atoms.calc = make_lj()
    dyn = FIMDynamics(atoms, basis=basis, timestep_fs=1.0)

    seen = []
    dyn.attach(lambda: seen.append(int(dyn.nsteps)), interval=1)
    n = 20
    dyn.run(n)
    # Observer fires at step 0 then after each of n steps -> consecutive integers.
    assert dyn.nsteps == n
    assert seen[0] == 0 and seen[-1] == n
    # Strictly consecutive (no skipping by 2).
    assert seen == list(range(0, n + 1))


# --------------------------------------------------------------------------- #
# Energy conservation: the band Hamiltonian H_B is the conserved quantity,
# not the full Cartesian potential energy.
# --------------------------------------------------------------------------- #
def test_band_energy_conserved_nve(lj_reference, lj_trajectory):
    basis = FIMDBasis.from_trajectory(
        trajectory=lj_trajectory, reference=lj_reference, calculator=make_lj(),
        band=(0.0, 200.0), timestep_reference=2.0, verbose=False,
        max_initial_displacement=0.2,
    )
    atoms = lj_trajectory[-1].copy()
    atoms.calc = make_lj()
    dyn = FIMDynamics(atoms, basis=basis, timestep_fs=1.0)

    HB = [dyn.get_band_energy()]
    for _ in range(500):
        dyn.step()
        HB.append(dyn.get_band_energy())
    HB = np.array(HB)

    assert np.all(np.isfinite(HB))
    drift = abs(HB[-1] - HB[0])
    span = HB.max() - HB.min()
    # Smooth analytic potential + symplectic step -> tight conservation.
    assert drift < 1e-2, f"H_band drift too large: {drift}"
    assert span < 5e-2, f"H_band span too large: {span}"


def test_band_energy_methods_present_and_consistent(lj_reference, lj_trajectory):
    basis = FIMDBasis.from_trajectory(
        trajectory=lj_trajectory, reference=lj_reference, calculator=make_lj(),
        band=(0.0, 300.0), timestep_reference=2.0, verbose=False,
        max_initial_displacement=0.2,
    )
    atoms = lj_trajectory[-1].copy()
    atoms.calc = make_lj()
    dyn = FIMDynamics(atoms, basis=basis, timestep_fs=1.0)

    breakdown = dyn.get_modal_energy_breakdown()
    # band_energy from the breakdown must equal get_band_energy().
    assert np.isclose(breakdown["band_energy"], dyn.get_band_energy(), atol=1e-9)
    # harmonic per-mode = kinetic + potential.
    assert np.allclose(breakdown["harmonic"],
                       breakdown["kinetic"] + breakdown["potential"], atol=1e-9)
    # reference_energy was recorded.
    assert basis.reference_energy is not None


# --------------------------------------------------------------------------- #
# A run started from a sane initial condition must not blow up.
# --------------------------------------------------------------------------- #
def test_trajectory_stays_bounded(lj_reference, lj_trajectory):
    basis = FIMDBasis.from_trajectory(
        trajectory=lj_trajectory, reference=lj_reference, calculator=make_lj(),
        band=(0.0, 300.0), timestep_reference=2.0, verbose=False,
        max_initial_displacement=0.2,
    )
    atoms = lj_trajectory[-1].copy()
    atoms.calc = make_lj()
    dyn = FIMDynamics(atoms, basis=basis, timestep_fs=1.0)

    p0 = atoms.get_positions().copy()
    for _ in range(400):
        dyn.step()
    disp = np.sqrt(((atoms.get_positions() - p0) ** 2).sum(-1)).max()
    assert np.all(np.isfinite(atoms.get_positions()))
    assert disp < 5.0, f"atoms drifted unphysically far: {disp} A"


# --------------------------------------------------------------------------- #
# Basis save/load round-trip, including the reference_energy field.
# --------------------------------------------------------------------------- #
def test_basis_save_load_roundtrip(tmp_path, lj_reference, lj_trajectory):
    basis = FIMDBasis.from_trajectory(
        trajectory=lj_trajectory, reference=lj_reference, calculator=make_lj(),
        band=(0.0, 300.0), timestep_reference=2.0, verbose=False,
    )
    path = tmp_path / "basis.npz"
    basis.save(str(path))
    loaded = FIMDBasis.load(str(path))

    assert np.allclose(loaded.omega, basis.omega)
    assert np.array_equal(loaded.band_mask, basis.band_mask)
    assert loaded.reference_energy is not None
    assert np.isclose(loaded.reference_energy, basis.reference_energy)
