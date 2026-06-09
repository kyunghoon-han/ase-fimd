"""Tests for band parsing helpers and the trajectory-covariance basis route."""

from __future__ import annotations

import numpy as np
import pytest

from fimd.core import FIMDBasis, FIMDynamics
from fimd.protocols import parse_bands, ps_to_steps

from .conftest import make_lj


def test_parse_bands_colon_and_dash():
    assert parse_bands("0:200,200:600") == [(0.0, 200.0), (200.0, 600.0)]
    assert parse_bands("0-200,200-600") == [(0.0, 200.0), (200.0, 600.0)]
    assert parse_bands("0:200;200:600") == [(0.0, 200.0), (200.0, 600.0)]


def test_parse_bands_passthrough_sequence():
    assert parse_bands([(0, 200), (200, 600)]) == [(0.0, 200.0), (200.0, 600.0)]


def test_parse_bands_rejects_inverted():
    with pytest.raises(ValueError):
        parse_bands("600:200")


def test_parse_bands_rejects_empty():
    with pytest.raises(ValueError):
        parse_bands("")


def test_ps_to_steps():
    assert ps_to_steps(1.0, 0.5) == 2000
    assert ps_to_steps(0.0, 0.5) == 0
    with pytest.raises(ValueError):
        ps_to_steps(1.0, 0.0)
    with pytest.raises(ValueError):
        ps_to_steps(-1.0, 0.5)


# --------------------------------------------------------------------------- #
# Covariance basis route: W from covariance, omega from velocity spectra.
# --------------------------------------------------------------------------- #
def test_covariance_basis_builds_and_runs(lj_reference, lj_trajectory):
    basis = FIMDBasis.from_trajectory_covariance(
        trajectory=lj_trajectory, reference=lj_reference, calculator=make_lj(),
        band=(0.0, 400.0), timestep_reference=2.0, verbose=False,
        max_initial_displacement=0.2,
    )
    # Should produce some active modes, none at zero frequency.
    assert basis.n_active_modes >= 1
    active = basis.omega[basis.band_mask]
    assert np.all(np.abs(active) > 1e-9)
    # metadata records the route.
    assert basis.metadata.get("basis_source") == "covariance"

    atoms = lj_trajectory[-1].copy()
    atoms.calc = make_lj()
    dyn = FIMDynamics(atoms, basis=basis, timestep_fs=1.0)
    HB = [dyn.get_band_energy()]
    for _ in range(300):
        dyn.step()
        HB.append(dyn.get_band_energy())
    HB = np.array(HB)
    assert np.all(np.isfinite(HB))


def test_covariance_centres_on_trajectory_mean(lj_reference, lj_trajectory):
    """The covariance route must centre on <r> (the trajectory mean), not an
    external reference. With rigid-motion removal disabled, the reconstructed
    reference equals the raw trajectory mean."""
    basis = FIMDBasis.from_trajectory_covariance(
        trajectory=lj_trajectory, reference=lj_reference, calculator=make_lj(),
        band=(0.0, 400.0), timestep_reference=2.0, verbose=False,
        remove_rigid_motion=False,
    )
    mean_pos = np.mean([f.get_positions() for f in lj_trajectory], axis=0)
    assert np.allclose(basis.reference_positions, mean_pos, atol=1e-6)
    # And it must differ from the external (minimised) reference it was given.
    assert not np.allclose(basis.reference_positions, lj_reference.get_positions(), atol=1e-3)
