"""Ordinary ASE MD/minimisation helpers used before FIMD."""

from __future__ import annotations

import os
from typing import Any, List, Optional, Sequence

import numpy as np

from ase import Atoms, units
from ase.io import write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.verlet import VelocityVerlet
from ase.optimize import BFGS, FIRE

try:
    from ase.md.langevin import Langevin
except Exception:  # pragma: no cover
    Langevin = None

from .core import resolve_precision, set_velocities_ang_per_fs


def attach_calculator(atoms: Atoms, calculator: Any) -> Atoms:
    atoms.calc = calculator
    return atoms


def minimise_atoms(
    atoms: Atoms,
    calculator: Any = None,
    fmax: float = 0.01,
    steps: int = 1000,
    optimizer: str = "bfgs",
    logfile: Optional[str] = None,
    trajectory: Optional[str] = None,
) -> Atoms:
    """Minimise an ASE ``Atoms`` object and return a copy of the minimised structure."""
    if calculator is not None:
        atoms.calc = calculator
    if atoms.calc is None:
        raise ValueError("atoms.calc must be set before minimisation.")
    key = optimizer.strip().lower()
    opt_cls = FIRE if key == "fire" else BFGS
    opt = opt_cls(atoms, logfile=logfile, trajectory=trajectory)
    opt.run(fmax=fmax, steps=steps)
    ref = atoms.copy()
    ref.calc = atoms.calc
    return ref


def initialise_velocities(
    atoms: Atoms,
    temperature_K: float,
    remove_translation: bool = True,
    remove_rotation: bool = True,
    seed: Optional[int] = None,
) -> None:
    """Initialise Maxwell-Boltzmann velocities for an ASE ``Atoms`` object."""
    if seed is not None:
        rng = np.random.default_rng(seed)
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=rng)
    else:
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
    if remove_translation:
        Stationary(atoms)
    if remove_rotation:
        try:
            ZeroRotation(atoms)
        except Exception:
            pass


def run_reference_md(
    atoms: Atoms,
    timestep_fs: float,
    nsteps: int,
    temperature_K: Optional[float] = 300.0,
    ensemble: str = "nve",
    friction_per_fs: float = 0.01,
    save_interval: int = 1,
    trajectory_path: Optional[str] = None,
    xyz_path: Optional[str] = None,
    logfile: Optional[str] = None,
    precision: str = "float64",
    initialise: bool = True,
    seed: Optional[int] = None,
) -> List[Atoms]:
    """Run ordinary ASE MD and return saved frames for FIMD band identification."""
    dtype = resolve_precision(precision)
    if atoms.calc is None:
        raise ValueError("atoms.calc must be set before reference MD.")
    if nsteps < 0:
        raise ValueError("nsteps must be non-negative.")
    if save_interval <= 0:
        raise ValueError("save_interval must be positive.")

    if initialise and temperature_K is not None and temperature_K > 0:
        initialise_velocities(atoms, temperature_K, seed=seed)
    elif atoms.get_velocities() is None:
        set_velocities_ang_per_fs(atoms, np.zeros((len(atoms), 3), dtype=dtype), dtype)

    frames: List[Atoms] = []

    def save_frame() -> None:
        frame = atoms.copy()
        frame.set_velocities(atoms.get_velocities())
        frame.info["fimd_dt_fs"] = float(timestep_fs * save_interval)
        frame.info["fimd_precision"] = dtype.name
        frames.append(frame)

    save_frame()

    key = ensemble.strip().lower()
    if key in {"nvt", "langevin"}:
        if Langevin is None:
            raise ImportError("ase.md.langevin.Langevin is unavailable in this ASE installation.")
        dyn = Langevin(
            atoms,
            timestep_fs * units.fs,
            temperature_K=float(temperature_K or 300.0),
            friction=float(friction_per_fs) / units.fs,
            logfile=logfile,
        )
    elif key in {"nve", "verlet", "velocityverlet"}:
        dyn = VelocityVerlet(atoms, timestep=timestep_fs * units.fs, logfile=logfile)
    else:
        raise ValueError("ensemble must be 'nve' or 'nvt'.")

    dyn.attach(save_frame, interval=save_interval)
    dyn.run(nsteps)

    if trajectory_path:
        os.makedirs(os.path.dirname(os.path.abspath(trajectory_path)), exist_ok=True)
        write(trajectory_path, frames)
    if xyz_path:
        os.makedirs(os.path.dirname(os.path.abspath(xyz_path)), exist_ok=True)
        write(xyz_path, frames)
    return frames
