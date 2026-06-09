"""High-level workflows for running FIMD from XYZ files."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from ase import Atoms, units
from ase.io import read, write

from .calculators import get_calculator
from .core import (
    FIMDBasis,
    FIMDynamics,
    FloatDTypeLike,
    get_velocities_ang_per_fs,
    load_trajectory,
    resolve_precision,
)
from .md import minimise_atoms, run_reference_md


@dataclass
class FIMDRunResult:
    atoms: Any
    basis: FIMDBasis
    dynamics: FIMDynamics
    output_dir: str
    results_file: str
    basis_file: str
    trajectory_file: str
    final_xyz: str


def run_fimd_from_xyz(
    xyz_file: str,
    calculator: Union[str, Any] = "emt",
    calculator_kwargs: Optional[Dict[str, Any]] = None,
    band: Tuple[float, float] = (0.0, 500.0),
    output_dir: str = "fimd_output",
    reference_file: Optional[str] = None,
    trajectory_file: Optional[str] = None,
    minimise: bool = True,
    minimise_fmax: float = 0.01,
    minimise_steps: int = 1000,
    thermalise_steps: int = 0,
    thermalise_temperature_K: float = 300.0,
    thermalise_friction_per_fs: float = 0.01,
    reference_md_steps: int = 1000,
    reference_md_dt_fs: float = 1.0,
    reference_temperature_K: Optional[float] = 300.0,
    reference_ensemble: str = "nve",
    reference_save_interval: int = 1,
    fimd_steps: int = 1000,
    fimd_dt_fs: float = 1.0,
    fimd_temperature_K: Optional[float] = None,
    fimd_friction_per_fs: float = 0.01,
    save_interval: int = 10,
    hessian_step: float = 0.001,
    basis_source: str = "hessian",
    remove_rigid_motion: bool = True,
    max_initial_displacement: Optional[float] = None,
    initial_temperature_K: Optional[float] = None,
    precision: FloatDTypeLike = "float64",
    velocity_unit: str = "angstrom/ps",
    trajectory_format: str = "xyz",
    keep_intermediate_traj: bool = False,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> FIMDRunResult:
    """
    Complete package-level workflow from an XYZ file.

    The calculator can be a string understood by ``get_calculator`` or a ready
    ASE calculator object. If ``trajectory_file`` is omitted, a short ordinary
    ASE MD pre-run is generated for band identification.
    """
    dtype = resolve_precision(precision)
    os.makedirs(output_dir, exist_ok=True)
    calculator_kwargs = dict(calculator_kwargs or {})
    # Thread the run precision to calculators that can follow it (e.g. MACE),
    # without overriding an explicit dtype the user set in calculator_kwargs.
    calc = get_calculator(calculator, _precision_hint=dtype.name, **calculator_kwargs)

    atoms0 = read(xyz_file)
    atoms0.calc = calc

    if reference_file:
        reference = read(reference_file)
        reference.calc = calc
    elif minimise:
        if verbose:
            print("Minimising input structure for FIMD reference...")
        reference = minimise_atoms(
            atoms0.copy(),
            calculator=calc,
            fmax=minimise_fmax,
            steps=minimise_steps,
            logfile=os.path.join(output_dir, "minimise.log"),
            trajectory=os.path.join(output_dir, "minimise.traj"),
        )
        write(os.path.join(output_dir, "reference.xyz"), reference)
    else:
        reference = atoms0.copy()
        reference.calc = calc
        write(os.path.join(output_dir, "reference.xyz"), reference)

    if trajectory_file:
        if verbose:
            print(f"Loading external reference trajectory: {trajectory_file}")
        trajectory = load_trajectory(
            trajectory_file,
            dt_fs=reference_md_dt_fs,
            symbols=reference.get_chemical_symbols(),
            masses=reference.get_masses(),
            precision=dtype,
            velocity_unit=velocity_unit,
            verbose=verbose,
        )
        if str(basis_source).strip().lower() in {"covariance", "cov", "trajectory", "spectral"}:
            if verbose:
                print("Building trajectory-derived (covariance) basis from external trajectory.")
            basis = FIMDBasis.from_trajectory_covariance(
                trajectory=trajectory,
                reference=reference,
                calculator=calc,
                band=band,
                timestep_reference=reference_md_dt_fs,
                precision=dtype,
                remove_rigid_motion=remove_rigid_motion,
                max_initial_displacement=max_initial_displacement,
                initial_temperature_K=initial_temperature_K,
                seed=seed,
                verbose=verbose,
            )
        else:
            basis = FIMDBasis.from_trajectory(
                trajectory=trajectory,
                reference=reference,
                calculator=calc,
                band=band,
                hessian_step=hessian_step,
                timestep_reference=reference_md_dt_fs,
                precision=dtype,
                remove_rigid_motion=remove_rigid_motion,
                max_initial_displacement=max_initial_displacement,
                initial_temperature_K=initial_temperature_K,
                seed=seed,
                verbose=verbose,
            )
        atoms_fimd = trajectory[-1].copy()
    elif reference_md_steps > 0:
        atoms_ref = reference.copy()
        atoms_ref.calc = calc
        thermalised = False
        if thermalise_steps and thermalise_steps > 0:
            if verbose:
                print(f"Thermalising at {thermalise_temperature_K:g} K for "
                      f"{thermalise_steps} steps (NVT Langevin) before reference MD...")
            run_reference_md(
                atoms_ref,
                timestep_fs=reference_md_dt_fs,
                nsteps=thermalise_steps,
                temperature_K=thermalise_temperature_K,
                ensemble="nvt",
                friction_per_fs=thermalise_friction_per_fs,
                save_interval=max(thermalise_steps, 1),  # we only need the final state
                trajectory_path=None,
                xyz_path=None,
                logfile=os.path.join(output_dir, "thermalise.log"),
                precision=dtype.name,
                initialise=True,
                seed=seed,
            )
            thermalised = True  # atoms_ref now carries equilibrated positions+velocities
        if verbose:
            print("Running ordinary ASE reference MD for band identification...")
        trajectory = run_reference_md(
            atoms_ref,
            timestep_fs=reference_md_dt_fs,
            nsteps=reference_md_steps,
            temperature_K=reference_temperature_K,
            ensemble=reference_ensemble,
            save_interval=reference_save_interval,
            trajectory_path=os.path.join(output_dir, "reference_md.traj"),
            xyz_path=os.path.join(output_dir, "reference_md.xyz"),
            logfile=os.path.join(output_dir, "reference_md.log"),
            precision=dtype.name,
            initialise=not thermalised,  # keep equilibrated velocities if we thermalised
            seed=seed,
        )
        basis = FIMDBasis.from_trajectory(
            trajectory=trajectory,
            reference=reference,
            calculator=calc,
            band=band,
            hessian_step=hessian_step,
            timestep_reference=reference_md_dt_fs * reference_save_interval,
            precision=dtype,
            remove_rigid_motion=remove_rigid_motion,
            max_initial_displacement=max_initial_displacement,
            initial_temperature_K=initial_temperature_K,
            seed=seed,
            verbose=verbose,
        )
        atoms_fimd = trajectory[-1].copy()
    else:
        if verbose:
            print("Building FIMD basis from reference only; initial q,p will be zero unless atoms carry velocities.")
        basis = FIMDBasis.from_reference_only(
            reference=reference,
            calculator=calc,
            band=band,
            hessian_step=hessian_step,
            precision=dtype,
            verbose=verbose,
        )
        atoms_fimd = atoms0.copy()

    basis_file = os.path.join(output_dir, "fimd_basis.npz")
    basis.save(basis_file)

    atoms_fimd.calc = calc
    fmt = str(trajectory_format).strip().lower()
    if fmt not in {"xyz", "extxyz", "traj", "both", "none"}:
        raise ValueError("trajectory_format must be 'xyz', 'extxyz', 'traj', 'both', or 'none'.")
    # Only let the integrator stream a binary .traj when the user wants one.
    trajectory_out = (os.path.join(output_dir, "fimd_trajectory.traj")
                      if fmt in {"traj", "both"} else None)
    logfile = os.path.join(output_dir, "fimd.log")
    dyn = FIMDynamics(
        atoms_fimd,
        basis=basis,
        timestep_fs=fimd_dt_fs,
        temperature=fimd_temperature_K,
        friction=fimd_friction_per_fs,
        trajectory=trajectory_out,
        logfile=logfile,
        loginterval=save_interval,
        precision=dtype,
    )

    times: list = []
    energies: list = []
    band_energies: list = []
    positions: list = []
    velocities: list = []
    _logged_steps: set = set()

    def log_frame() -> None:
        # ASE fires observers at step 0 and after each step; guard against the
        # initial frame being recorded twice when the manual seed and the
        # interval-0 fire coincide.
        n = int(dyn.nsteps)
        if n in _logged_steps:
            return
        _logged_steps.add(n)
        times.append(float(n * fimd_dt_fs))
        energies.append(float(atoms_fimd.get_potential_energy()))
        band_energies.append(float(dyn.get_band_energy()))
        positions.append(np.asarray(atoms_fimd.get_positions(), dtype=dtype).copy())
        # Velocities in Angstrom/fs (for velocity autocorrelation / VDOS).
        vel = get_velocities_ang_per_fs(atoms_fimd, dtype)
        if vel is None:
            vel = np.zeros((len(atoms_fimd), 3), dtype=dtype)
        velocities.append(np.asarray(vel, dtype=dtype).copy())
        if verbose and n % max(save_interval * 20, 1) == 0:
            print(f"  FIMD step {n}: E_full={energies[-1]:.8f} eV  "
                  f"H_band={band_energies[-1]:.8f} eV")

    # Seed the initial frame, then attach so subsequent steps are appended.
    log_frame()

    dyn.attach(log_frame, interval=save_interval)
    if verbose:
        print(f"Running FIMD: steps={fimd_steps}, dt={fimd_dt_fs:g} fs")
    dyn.run(fimd_steps)

    band_energies_arr = np.asarray(band_energies, dtype=dtype)
    if verbose and np.all(np.isfinite(band_energies_arr)) and band_energies_arr.size > 1:
        span = float(band_energies_arr.max() - band_energies_arr.min())
        drift = float(abs(band_energies_arr[-1] - band_energies_arr[0]))
        print(f"Band energy conservation: span={span:.6f} eV  drift={drift:.6f} eV")

    breakdown = dyn.get_modal_energy_breakdown()

    results_file = os.path.join(output_dir, "fimd_results.npz")
    np.savez_compressed(
        results_file,
        times_fs=np.asarray(times, dtype=dtype),
        energies_eV=np.asarray(energies, dtype=dtype),
        band_energies_eV=band_energies_arr,
        positions=np.asarray(positions, dtype=dtype),
        velocities_ang_per_fs=np.asarray(velocities, dtype=dtype),
        velocity_unit=np.asarray("angstrom/fs", dtype=object),
        symbols=np.asarray(atoms_fimd.get_chemical_symbols(), dtype=object),
        masses=np.asarray(atoms_fimd.get_masses(), dtype=dtype),
        band=np.asarray(band, dtype=dtype),
        fimd_dt_fs=np.asarray(fimd_dt_fs, dtype=dtype),
        precision=np.asarray(dtype.name, dtype=object),
        active_mode_index=np.asarray(breakdown["mode_index"], dtype=int),
        active_frequencies_cm1=np.asarray(breakdown["frequencies_cm1"], dtype=dtype),
        final_mode_kinetic_eV=np.asarray(breakdown["kinetic"], dtype=dtype),
        final_mode_potential_eV=np.asarray(breakdown["potential"], dtype=dtype),
        final_mode_harmonic_eV=np.asarray(breakdown["harmonic"], dtype=dtype),
        final_residual_dV_eV=np.asarray(breakdown["residual_dV"], dtype=dtype),
    )

    final_xyz = os.path.join(output_dir, "fimd_final.xyz")
    write(final_xyz, atoms_fimd)

    # Full trajectory as XYZ (every logged frame), with time and energies in the
    # per-frame comment line. This is the primary trajectory output unless the
    # user asked for traj-only.
    trajectory_xyz = None
    if fmt in {"xyz", "both", "extxyz"}:
        symbols = atoms_fimd.get_chemical_symbols()
        masses = atoms_fimd.get_masses()
        want_ext = (fmt == "extxyz")
        frames = []
        for t, E, Eb, pos, vel in zip(times, energies, band_energies, positions, velocities, strict=False):
            frame = Atoms(symbols=symbols, positions=pos)
            frame.info["time_fs"] = float(t)
            frame.info["energy_eV"] = float(E)
            frame.info["band_energy_eV"] = float(Eb)
            if want_ext:
                # Rich extended-XYZ: also embed masses + velocities per atom.
                # NOTE: many viewers (e.g. VMD's plain XYZ reader) cannot parse
                # the extra columns and will appear to "freeze" mid-trajectory.
                # Use this only with extended-XYZ-aware tools (ASE, OVITO).
                frame.set_masses(masses)
                frame.set_velocities(np.asarray(vel, dtype=dtype) * units.fs)
            frames.append(frame)
        trajectory_xyz = os.path.join(output_dir, "fimd_trajectory.xyz")
        # Plain XYZ (element + xyz only) is the default and is what VMD/most
        # viewers read reliably; the comment line still carries the scalar info.
        write(trajectory_xyz, frames, format=("extxyz" if want_ext else "xyz"))

    # Optionally clean up intermediate .traj files from minimise / reference MD.
    if not keep_intermediate_traj:
        for name in ("minimise.traj", "reference_md.traj"):
            p = os.path.join(output_dir, name)
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

    primary_trajectory = (trajectory_xyz if trajectory_xyz is not None
                          else trajectory_out)

    return FIMDRunResult(
        atoms=atoms_fimd,
        basis=basis,
        dynamics=dyn,
        output_dir=output_dir,
        results_file=results_file,
        basis_file=basis_file,
        trajectory_file=primary_trajectory,
        final_xyz=final_xyz,
    )