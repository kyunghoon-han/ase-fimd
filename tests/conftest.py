"""Shared fixtures for the FIMD test suite.

The core suite runs entirely on calculators that ship with ASE (EMT and
Lennard-Jones) so it works on a bare CI runner with no heavy quantum-chemistry
or ML dependencies. Tests that need tblite / MACE / SO3LR are guarded with
importorskip in their own modules.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms, units
from ase.calculators.lj import LennardJones
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from ase.optimize import BFGS

# Lennard-Jones parameters tuned for a small, well-bound argon-like cluster.
LJ_KW = dict(epsilon=0.30, sigma=2.5, rc=8.0)


def make_lj():
    return LennardJones(**LJ_KW)


@pytest.fixture
def lj_cluster():
    """A relaxed 7-atom LJ cluster (a real local minimum)."""
    rng = np.random.default_rng(0)
    atoms = Atoms("Ar7", positions=rng.standard_normal((7, 3)) * 1.8 + 2.5)
    atoms.calc = make_lj()
    BFGS(atoms, logfile=None).run(fmax=1e-4, steps=3000)
    return atoms


@pytest.fixture
def lj_reference(lj_cluster):
    ref = lj_cluster.copy()
    ref.calc = make_lj()
    return ref


@pytest.fixture
def lj_trajectory(lj_cluster):
    """A short NVE trajectory around the minimum, with velocities stored."""
    atoms = lj_cluster.copy()
    atoms.calc = make_lj()
    MaxwellBoltzmannDistribution(atoms, temperature_K=30)
    Stationary(atoms)
    frames = []

    def save():
        f = atoms.copy()
        f.set_velocities(atoms.get_velocities())
        f.info["fimd_dt_fs"] = 2.0
        frames.append(f)

    save()
    dyn = VelocityVerlet(atoms, timestep=1.0 * units.fs)
    dyn.attach(save, interval=1)
    dyn.run(400)
    return frames
