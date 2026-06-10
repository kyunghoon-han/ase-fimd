#!/usr/bin/env python3
"""Demonstrate that FIMD is usable as an ASE-native integrator.

This mirrors the standard ASE molecular-dynamics idiom -- the same shape as
``ase.md.Langevin`` -- to show that FIMD slots into existing ASE workflows:

    from ase.md import Langevin
    md = Langevin(atoms, 2 * units.fs, temperature_K=300, friction=0.01,
                  logfile="run.log")
    md.attach(traj.write, interval=50)
    md.run(10)

The FIMD equivalent is identical except for one extra argument -- the modal
``basis`` -- which is intrinsic to the method (FIMD propagates a chosen
vibrational band in a modal frame, so it needs the modal basis W, omega and a
reference). Everything else (``run``, ``attach``, ``logfile``, ``loginterval``,
``trajectory``, observers) is the ordinary ASE ``MolecularDynamics`` interface,
because ``FIMDynamics`` subclasses ``ase.md.md.MolecularDynamics``.

Run directly to execute a small self-checking demonstration:

    python examples/ase_native_interface.py
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
from ase import units
from ase.calculators.lj import LennardJones
from ase.io import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet

from fimd.core import FIMDBasis, FIMDynamics
from fimd.reference_systems import lj_cluster


def build_basis_and_start(band=(0.0, 100.0)):
    """Prepare an LJ argon cluster, a short reference trajectory, and a basis.

    This is the FIMD-specific setup step (the analogue of choosing a thermostat
    for Langevin). It returns a starting Atoms object and the modal basis.
    """
    atoms, lj_kw = lj_cluster(noshells=2)          # relaxed Ar13, real Ar params

    def calc():
        return LennardJones(**lj_kw)

    # A short NVE reference trajectory used to identify the vibrational modes.
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
    ref_md = VelocityVerlet(work, timestep=1.0 * units.fs)
    ref_md.attach(save, interval=1)
    ref_md.run(300)

    reference = atoms.copy()
    reference.calc = calc()
    basis = FIMDBasis.from_trajectory(
        trajectory=frames, reference=reference, calculator=calc(),
        band=band, timestep_reference=2.0, verbose=False,
        max_initial_displacement=0.2,
    )

    start = frames[-1].copy()
    start.calc = calc()
    return start, basis, lj_kw


def demo_ase_native_interface():
    """Run FIMD through the ASE-native MolecularDynamics interface and verify it."""
    print("=" * 70)
    print("FIMD as an ASE-native integrator -- Langevin-style usage")
    print("=" * 70)

    start, basis, _ = build_basis_and_start(band=(0.0, 100.0))

    # ---- Confirm FIMDynamics really is an ASE MolecularDynamics subclass ----
    from ase.md.md import MolecularDynamics
    assert issubclass(FIMDynamics, MolecularDynamics), \
        "FIMDynamics must subclass ase.md.md.MolecularDynamics"
    print("\n[1] FIMDynamics subclasses ase.md.md.MolecularDynamics: OK")

    tmp = tempfile.mkdtemp()
    logfile = os.path.join(tmp, "fimd_run.log")
    trajfile = os.path.join(tmp, "fimd_run.traj")

    # ---------------------------------------------------------------------
    # The ASE idiom, mirroring Josh's Langevin example almost line-for-line:
    #
    #     md = Langevin(atoms, 2*units.fs, temperature_K=300, friction=0.01,
    #                   logfile=tag+'.log')
    #     traj = Trajectory(tag+'.traj', 'w', atoms)
    #     md.attach(traj.write, interval=50)
    #     md.run(10)
    #
    # FIMD differs only by the modal `basis` argument.
    # ---------------------------------------------------------------------
    md = FIMDynamics(
        start,                       # atoms
        basis,                       # <-- the one FIMD-specific argument
        2 * units.fs,                # timestep (ASE-style positional, in ASE units)
        temperature=None,            # None -> NVE; set a value (K) for an NVT band thermostat
        friction=0.01,               # band Langevin friction (used only for NVT)
        logfile=logfile,
        loginterval=10,
    )

    traj = Trajectory(trajfile, "w", start)
    md.attach(traj.write, interval=10)

    # An extra observer, to prove the ASE observer mechanism works.
    band_energies = []
    md.attach(lambda: band_energies.append(md.get_band_energy()), interval=10)

    n_steps = 200
    md.run(n_steps)                  # <-- identical call shape to Langevin.run()
    traj.close()

    print(f"[2] md.run({n_steps}) completed via the ASE interface: OK")
    print(f"    final nsteps = {md.nsteps} (expected {n_steps})")
    assert md.nsteps == n_steps

    # ---- The logfile and trajectory were written (ASE I/O machinery) ----
    assert os.path.exists(logfile) and os.path.getsize(logfile) > 0
    from ase.io import read
    written = read(trajfile, index=":")
    print(f"[3] logfile written and trajectory has {len(written)} frames: OK")
    assert len(written) >= 2

    # ---- Observers fired; the band Hamiltonian H_B is conserved (NVE) ----
    HB = np.array(band_energies)
    assert HB.size > 0 and np.all(np.isfinite(HB))
    drift = float(abs(HB[-1] - HB[0]))
    span = float(HB.max() - HB.min())
    print(f"[4] observer fired {HB.size} times; H_band drift={drift:.2e} eV, "
          f"span={span:.2e} eV: OK")
    assert drift < 1e-2 and span < 5e-2

    print("\nAll ASE-native interface checks passed.\n")
    return md


def demo_nvt_band_thermostat():
    """Show the NVT variant: just pass a temperature, exactly like Langevin."""
    print("=" * 70)
    print("FIMD NVT band thermostat -- pass temperature like Langevin")
    print("=" * 70)
    start, basis, _ = build_basis_and_start(band=(0.0, 150.0))

    md = FIMDynamics(
        start, basis, 2 * units.fs,
        temperature=30.0,            # K -- enables the band Ornstein-Uhlenbeck thermostat
        friction=0.01,
    )
    md.run(200)
    HB = md.get_band_energy()
    print(f"\n[NVT] ran 200 steps with a 30 K band thermostat; "
          f"H_band finite: {np.isfinite(HB)}")
    assert np.isfinite(HB)
    print("NVT band-thermostat check passed.\n")


if __name__ == "__main__":
    demo_ase_native_interface()
    demo_nvt_band_thermostat()
    print("Done -- FIMD behaves as an ASE-native integrator.")
