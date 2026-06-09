"""Optional tests for heavy calculators (tblite, MACE, SO3LR).

These are skipped automatically when the backend isn't installed, so the core
suite still passes on a bare CI runner. Run them locally where the backends
exist to validate the full stack.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import units
from ase.build import molecule
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from ase.optimize import BFGS

from fimd.core import FIMDBasis, FIMDynamics


def _short_traj(calc_factory, name):
    m = molecule("H2O")
    m.center(vacuum=5.0)
    m.calc = calc_factory()
    BFGS(m, logfile=None).run(fmax=0.05, steps=200)
    ref = m.copy()
    ref.calc = calc_factory()
    MaxwellBoltzmannDistribution(m, temperature_K=300)
    Stationary(m)
    frames = []

    def save():
        f = m.copy()
        f.set_velocities(m.get_velocities())
        f.info["fimd_dt_fs"] = 1.0
        frames.append(f)

    save()
    dyn = VelocityVerlet(m, timestep=0.5 * units.fs)
    dyn.attach(save, interval=2)
    dyn.run(200)
    return ref, frames


def _conserves(ref, frames, calc_factory, band=(0.0, 2000.0)):
    basis = FIMDBasis.from_trajectory(
        trajectory=frames, reference=ref, calculator=calc_factory(),
        band=band, timestep_reference=1.0, verbose=False,
        max_initial_displacement=0.1,
    )
    atoms = frames[-1].copy()
    atoms.calc = calc_factory()
    dyn = FIMDynamics(atoms, basis=basis, timestep_fs=0.5, precision="float64")
    HB = [dyn.get_band_energy()]
    for _ in range(100):
        dyn.step()
        HB.append(dyn.get_band_energy())
    return np.array(HB)


def test_tblite_fimd():
    pytest.importorskip("tblite")
    from tblite.ase import TBLite

    def mk():
        return TBLite(method="GFN2-xTB", verbosity=0)

    ref, frames = _short_traj(mk, "tblite")
    HB = _conserves(ref, frames, mk)
    assert np.all(np.isfinite(HB))


def test_mace_fimd():
    pytest.importorskip("mace")
    from mace.calculators import mace_mp

    def mk():
        return mace_mp(model="small", default_dtype="float64", dispersion=False)

    try:
        ref, frames = _short_traj(mk, "mace")
    except Exception as exc:  # model download may be unavailable offline
        pytest.skip(f"MACE model unavailable: {exc}")
    HB = _conserves(ref, frames, mk)
    assert np.all(np.isfinite(HB))


def test_so3lr_fimd():
    pytest.importorskip("so3lr")
    import numpy as _np
    from so3lr import So3lrCalculator

    def mk():
        return So3lrCalculator(lr_cutoff=1000.0, dtype=_np.float64)

    ref, frames = _short_traj(mk, "so3lr")
    HB = _conserves(ref, frames, mk)
    assert _np.all(_np.isfinite(HB))
