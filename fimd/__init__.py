"""Modular Fourier-Integrator Molecular Dynamics package for ASE."""

__version__ = "0.1.2"

from .core import (
    FIMDBasis,
    FIMDynamics,
    ase_velocity_to_ang_per_fs,
    ang_per_fs_to_ase_velocity,
    cm1_to_radfs,
    radfs_to_cm1,
    load_trajectory,
    resolve_precision,
    run_fimd_from_trajectory,
)
from .calculators import get_calculator, available_calculators
from .md import minimise_atoms, run_reference_md
from .workflow import FIMDRunResult, run_fimd_from_xyz
from .protocols import (
    MultiBandFIMDResult,
    PreparedReference,
    parse_bands,
    prepare_equilibrated_reference,
    ps_to_steps,
    run_fimd_bands_from_prepared_reference,
    run_multiband_fimd_from_xyz,
)

__all__ = [
    "FIMDBasis",
    "FIMDynamics",
    "FIMDRunResult",
    "PreparedReference",
    "MultiBandFIMDResult",
    "get_calculator",
    "available_calculators",
    "load_trajectory",
    "run_reference_md",
    "minimise_atoms",
    "run_fimd_from_xyz",
    "run_multiband_fimd_from_xyz",
    "prepare_equilibrated_reference",
    "run_fimd_bands_from_prepared_reference",
    "parse_bands",
    "ps_to_steps",
    "run_fimd_from_trajectory",
    "resolve_precision",
    "cm1_to_radfs",
    "radfs_to_cm1",
    "ase_velocity_to_ang_per_fs",
    "__version__",
    "ang_per_fs_to_ase_velocity",
]
