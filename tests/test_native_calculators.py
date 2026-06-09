"""Reference structures appropriate to each native ASE calculator.

The native ASE potentials are only physically meaningful on specific systems:

* ``emt``    - metallic embedded-atom: small transition-metal clusters
               (Al, Cu, Ag, Au, Ni, Pd, Pt). Default here: a Cu icosahedron.
* ``lj``     - pairwise Lennard-Jones: noble-gas / van-der-Waals clusters.
               Default here: an argon icosahedron with real Ar parameters.
* ``morse``  - a bond (pair) potential: a diatomic or a compact cluster.
               Default here: an H2 diatomic at r0, or a small cluster.

Using a calculator on the wrong system (e.g. EMT on an organic molecule)
produces an energy-conserving but physically meaningless trajectory, so these
builders exist to give each calculator a system it can actually describe.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ase.calculators.morse import MorsePotential
from ase.cluster import Icosahedron
from ase.optimize import BFGS

# Real argon Lennard-Jones parameters (eV, Angstrom).
ARGON_LJ = dict(epsilon=0.0103, sigma=3.40, rc=10.0)


def emt_cluster(noshells: int = 2, element: str = "Cu", relax: bool = True) -> Atoms:
    """A small metallic cluster suited to the EMT calculator."""
    atoms = Icosahedron(element, noshells=noshells)
    atoms.calc = EMT()
    if relax:
        BFGS(atoms, logfile=None).run(fmax=0.01, steps=500)
    return atoms


def lj_cluster(noshells: int = 2, relax: bool = True, **lj_kwargs) -> Tuple[Atoms, dict]:
    """A noble-gas cluster suited to the Lennard-Jones calculator.

    Returns ``(atoms, lj_kwargs)`` so the caller can build matching calculators.
    Geometry is an argon icosahedron; parameters default to real argon.
    """
    kw = dict(ARGON_LJ)
    kw.update(lj_kwargs)
    atoms = Icosahedron("Ar", noshells=noshells)
    atoms.calc = LennardJones(**kw)
    if relax:
        BFGS(atoms, logfile=None).run(fmax=1e-3, steps=2000)
    return atoms, kw


def morse_diatomic(relax: bool = True, **morse_kwargs) -> Atoms:
    """An H2-style diatomic at the Morse minimum (single vibrational mode)."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.0]])
    atoms.calc = MorsePotential(**morse_kwargs)
    if relax:
        BFGS(atoms, logfile=None).run(fmax=1e-6, steps=500)
    return atoms


def morse_cluster(n: int = 7, seed: int = 3, relax: bool = True, **morse_kwargs) -> Atoms:
    """A small compact cluster for the Morse potential (richer spectrum)."""
    rng = np.random.default_rng(seed)
    atoms = Atoms(f"H{n}", positions=rng.standard_normal((n, 3)) * 0.5 + 1.0)
    atoms.calc = MorsePotential(**morse_kwargs)
    if relax:
        BFGS(atoms, logfile=None).run(fmax=1e-4, steps=3000)
    return atoms


__all__ = [
    "ARGON_LJ",
    "emt_cluster",
    "lj_cluster",
    "morse_diatomic",
    "morse_cluster",
]