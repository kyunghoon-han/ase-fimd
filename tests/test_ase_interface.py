"""Tests that FIMD is usable as an ASE-native integrator (Langevin-style).

These lock in the property that ``FIMDynamics`` behaves like any other
``ase.md`` integrator: it subclasses ``MolecularDynamics``, accepts an ASE-style
positional timestep, supports ``attach``/``run``/``logfile``/``trajectory``, and
takes a temperature to switch from NVE to an NVT band thermostat.
"""

from __future__ import annotations

import numpy as np
from ase import units
from ase.calculators.lj import LennardJones
from ase.io import Trajectory, read
from ase.md.md import MolecularDynamics
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet

from fimd.core import FIMDBasis, FIMDynamics
from fimd.reference_systems import lj_cluster


def _basis_and_start(band=(0.0, 120.0)):
    atoms, lj_kw = lj_cluster(noshells=2)

    def calc():
        return LennardJones(**lj_kw)

    work = atoms.copy()
    work.calc = calc()
    MaxwellBoltzmannDistribution(work, temperature_K=30)
    Stationary(work)
    frames = []

    def save():
        f = work.copy()
        f.set_velocities(work.get_velocities())
        f.info["fimd_dt_fs"] = 2.0
        frames.append(f)

    save()
    md = VelocityVerlet(work, timestep=1.0 * units.fs)
    md.attach(save, interval=1)
    md.run(250)

    ref = atoms.copy()
    ref.calc = calc()
    basis = FIMDBasis.from_trajectory(
        trajectory=frames, reference=ref, calculator=calc(),
        band=band, timestep_reference=2.0, verbose=False,
        max_initial_displacement=0.2,
    )
    start = frames[-1].copy()
    start.calc = calc()
    return start, basis


def test_fimdynamics_is_ase_molecular_dynamics():
    assert issubclass(FIMDynamics, MolecularDynamics)


def test_ase_style_positional_timestep_and_run():
    start, basis = _basis_and_start()
    md = FIMDynamics(start, basis, 2 * units.fs, friction=0.01)
    md.run(50)
    assert md.nsteps == 50
    # ASE stores the timestep in ASE time units on .dt
    assert np.isclose(md.dt, 2 * units.fs)


def test_attach_observer_and_logfile_and_trajectory(tmp_path):
    start, basis = _basis_and_start()
    logfile = tmp_path / "run.log"
    trajfile = tmp_path / "run.traj"

    md = FIMDynamics(
        start, basis, 2 * units.fs,
        logfile=str(logfile), loginterval=10,
    )
    traj = Trajectory(str(trajfile), "w", start)
    md.attach(traj.write, interval=10)

    hits = []
    md.attach(lambda: hits.append(md.get_band_energy()), interval=10)
    md.run(100)
    traj.close()

    assert logfile.exists() and logfile.stat().st_size > 0
    frames = read(str(trajfile), index=":")
    assert len(frames) >= 2
    assert len(hits) >= 2 and all(np.isfinite(h) for h in hits)


def test_nve_band_energy_conserved_via_ase_interface():
    start, basis = _basis_and_start(band=(0.0, 100.0))
    md = FIMDynamics(start, basis, 2 * units.fs)  # NVE (no temperature)
    HB = [md.get_band_energy()]
    for _ in range(300):
        md.step()
        HB.append(md.get_band_energy())
    HB = np.array(HB)
    assert np.all(np.isfinite(HB))
    assert (HB.max() - HB.min()) < 5e-2


def test_temperature_enables_nvt_band_thermostat():
    start, basis = _basis_and_start(band=(0.0, 150.0))
    md = FIMDynamics(start, basis, 2 * units.fs, temperature=30.0, friction=0.01)
    md.run(100)
    assert np.isfinite(md.get_band_energy())
