"""Reusable multi-stage FIMD protocols.

This module keeps long workflows out of the low-level FIMD integrator:

1. minimise an input structure,
2. thermalise/equilibrate it with ordinary ASE MD,
3. reuse the equilibrated trajectory for several independent FIMD bands.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from ase.io import read, write

from .calculators import get_calculator
from .core import FloatDTypeLike
from .md import minimise_atoms, run_reference_md
from .workflow import FIMDRunResult, run_fimd_from_xyz


BandLike = Tuple[float, float]


def ps_to_steps(duration_ps: float, timestep_fs: float) -> int:
    """Convert a duration in ps and a timestep in fs to an integer step count."""
    if timestep_fs <= 0:
        raise ValueError("timestep_fs must be positive")
    if duration_ps < 0:
        raise ValueError("duration_ps must be non-negative")
    return int(round(float(duration_ps) * 1000.0 / float(timestep_fs)))


def parse_bands(text: Union[str, Sequence[BandLike]]) -> List[BandLike]:
    """Parse bands from ``"0:200,200:600"`` or return a sequence unchanged."""
    if isinstance(text, str):
        bands: List[BandLike] = []
        for item in text.replace(";", ",").split(","):
            item = item.strip()
            if not item:
                continue
            if ":" in item:
                lo, hi = item.split(":", 1)
            elif "-" in item:
                lo, hi = item.split("-", 1)
            else:
                raise ValueError("Bands must look like '0:200,200:600' or '0-200,200-600'.")
            band = (float(lo), float(hi))
            if band[1] <= band[0]:
                raise ValueError(f"Invalid band {band}: upper edge must exceed lower edge.")
            bands.append(band)
        if not bands:
            raise ValueError("No bands were parsed.")
        return bands
    return [(float(lo), float(hi)) for lo, hi in text]


def band_slug(band: BandLike) -> str:
    """Filesystem-safe directory name for a wavenumber band."""
    lo, hi = band
    def fmt(x: float) -> str:
        if float(x).is_integer():
            return f"{int(x):04d}"
        return str(x).replace(".", "p").replace("-", "m")
    return f"band_{fmt(lo)}_{fmt(hi)}cm-1"


@dataclass
class PreparedReference:
    """Files produced by the conventional pre-FIMD stage."""

    output_dir: str
    input_file: str
    reference_file: str
    thermalisation_traj: str
    equilibration_traj: str
    equilibration_xyz: str
    metadata_file: str
    minimise_steps: int
    thermalisation_steps: int
    equilibration_steps: int
    md_dt_fs: float
    equilibration_save_interval: int


@dataclass
class MultiBandFIMDResult:
    """Result bundle for one reference preparation plus several FIMD bands."""

    prepared_reference: PreparedReference
    band_results: Dict[str, FIMDRunResult] = field(default_factory=dict)
    metadata_file: Optional[str] = None


def prepare_equilibrated_reference(
    xyz_file: str,
    calculator: Union[str, Any] = "mace_mp",
    calculator_kwargs: Optional[Dict[str, Any]] = None,
    output_dir: str = "fimd_200ps_multiband/reference",
    minimise_fmax: float = 0.02,
    minimise_steps: int = 2000,
    md_dt_fs: float = 0.5,
    temperature_K: float = 300.0,
    thermalisation_ps: float = 20.0,
    equilibration_ps: float = 20.0,
    thermalisation_ensemble: str = "nvt",
    equilibration_ensemble: str = "nvt",
    friction_per_fs: float = 0.01,
    thermalisation_save_interval: int = 200,
    equilibration_save_interval: int = 20,
    precision: FloatDTypeLike = "float64",
    seed: Optional[int] = 12345,
    verbose: bool = True,
) -> PreparedReference:
    """Minimise, thermalise, and equilibrate an XYZ structure with ordinary ASE MD.

    The saved equilibration trajectory is intended to be reused by multiple FIMD
    band runs.  The default save interval stores one frame every 10 fs when
    ``md_dt_fs=0.5``.
    """
    os.makedirs(output_dir, exist_ok=True)
    calculator_kwargs = dict(calculator_kwargs or {})
    input_copy = os.path.join(output_dir, "input.xyz")
    shutil.copyfile(xyz_file, input_copy)

    calc = get_calculator(calculator, **calculator_kwargs)
    atoms = read(xyz_file)
    atoms.calc = calc

    if verbose:
        print("=" * 72)
        print("Preparing conventional MD reference for FIMD")
        print("=" * 72)
        print(f"Input:       {xyz_file}")
        print(f"Calculator:  {calculator}")
        print(f"Atoms:       {len(atoms)}")

    reference = minimise_atoms(
        atoms,
        calculator=calc,
        fmax=minimise_fmax,
        steps=minimise_steps,
        logfile=os.path.join(output_dir, "minimise.log"),
        trajectory=os.path.join(output_dir, "minimise.traj"),
    )
    reference_file = os.path.join(output_dir, "reference.xyz")
    write(reference_file, reference)

    thermalisation_steps = ps_to_steps(thermalisation_ps, md_dt_fs)
    equilibration_steps = ps_to_steps(equilibration_ps, md_dt_fs)

    if verbose:
        print(f"Thermalising:  {thermalisation_ps:g} ps = {thermalisation_steps} steps")
    thermalisation_traj = os.path.join(output_dir, "thermalisation.traj")
    run_reference_md(
        reference,
        timestep_fs=md_dt_fs,
        nsteps=thermalisation_steps,
        temperature_K=temperature_K,
        ensemble=thermalisation_ensemble,
        friction_per_fs=friction_per_fs,
        save_interval=thermalisation_save_interval,
        trajectory_path=thermalisation_traj,
        xyz_path=os.path.join(output_dir, "thermalisation.xyz"),
        logfile=os.path.join(output_dir, "thermalisation.log"),
        precision=str(precision),
        initialise=True,
        seed=seed,
    )

    if verbose:
        print(f"Equilibrating: {equilibration_ps:g} ps = {equilibration_steps} steps")
    equilibration_traj = os.path.join(output_dir, "equilibration.traj")
    equilibration_xyz = os.path.join(output_dir, "equilibration.xyz")
    run_reference_md(
        reference,
        timestep_fs=md_dt_fs,
        nsteps=equilibration_steps,
        temperature_K=temperature_K,
        ensemble=equilibration_ensemble,
        friction_per_fs=friction_per_fs,
        save_interval=equilibration_save_interval,
        trajectory_path=equilibration_traj,
        xyz_path=equilibration_xyz,
        logfile=os.path.join(output_dir, "equilibration.log"),
        precision=str(precision),
        initialise=False,
        seed=None,
    )

    metadata_file = os.path.join(output_dir, "reference_protocol.json")
    metadata = {
        "input_file": os.path.abspath(xyz_file),
        "calculator": str(calculator),
        "calculator_kwargs": calculator_kwargs,
        "minimise_fmax": minimise_fmax,
        "minimise_steps_requested": minimise_steps,
        "md_dt_fs": md_dt_fs,
        "temperature_K": temperature_K,
        "thermalisation_ps": thermalisation_ps,
        "thermalisation_steps": thermalisation_steps,
        "thermalisation_ensemble": thermalisation_ensemble,
        "thermalisation_save_interval": thermalisation_save_interval,
        "equilibration_ps": equilibration_ps,
        "equilibration_steps": equilibration_steps,
        "equilibration_ensemble": equilibration_ensemble,
        "equilibration_save_interval": equilibration_save_interval,
        "precision": str(precision),
        "seed": seed,
    }
    with open(metadata_file, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return PreparedReference(
        output_dir=output_dir,
        input_file=input_copy,
        reference_file=reference_file,
        thermalisation_traj=thermalisation_traj,
        equilibration_traj=equilibration_traj,
        equilibration_xyz=equilibration_xyz,
        metadata_file=metadata_file,
        minimise_steps=minimise_steps,
        thermalisation_steps=thermalisation_steps,
        equilibration_steps=equilibration_steps,
        md_dt_fs=md_dt_fs,
        equilibration_save_interval=equilibration_save_interval,
    )


def run_fimd_bands_from_prepared_reference(
    prepared: PreparedReference,
    calculator: Union[str, Any] = "mace_mp",
    calculator_kwargs: Optional[Dict[str, Any]] = None,
    bands: Union[str, Sequence[BandLike]] = "0:200,200:600,600:1200,1450:1750",
    output_dir: str = "fimd_200ps_multiband/bands",
    production_ps: float = 200.0,
    fimd_dt_fs: float = 0.5,
    fimd_temperature_K: Optional[float] = None,
    fimd_friction_per_fs: float = 0.01,
    save_interval: int = 200,
    hessian_step: float = 0.001,
    max_initial_displacement: Optional[float] = None,
    precision: FloatDTypeLike = "float64",
    verbose: bool = True,
) -> Dict[str, FIMDRunResult]:
    """Run independent FIMD production trajectories for several bands."""
    os.makedirs(output_dir, exist_ok=True)
    calculator_kwargs = dict(calculator_kwargs or {})
    parsed_bands = parse_bands(bands)
    fimd_steps = ps_to_steps(production_ps, fimd_dt_fs)
    reference_md_dt_between_saved_frames = prepared.md_dt_fs * prepared.equilibration_save_interval

    if verbose:
        print("=" * 72)
        print("Running multi-band FIMD production")
        print("=" * 72)
        print(f"Production: {production_ps:g} ps = {fimd_steps} steps per band")
        print(f"Bands:      {parsed_bands}")
        print(f"Reference trajectory frame spacing: {reference_md_dt_between_saved_frames:g} fs")

    results: Dict[str, FIMDRunResult] = {}
    for band in parsed_bands:
        slug = band_slug(band)
        band_dir = os.path.join(output_dir, slug)
        if verbose:
            print("-" * 72)
            print(f"Band {band[0]:g}-{band[1]:g} cm^-1 -> {band_dir}")
        result = run_fimd_from_xyz(
            xyz_file=prepared.input_file,
            calculator=calculator,
            calculator_kwargs=calculator_kwargs,
            band=band,
            output_dir=band_dir,
            reference_file=prepared.reference_file,
            trajectory_file=prepared.equilibration_traj,
            minimise=False,
            reference_md_steps=0,
            reference_md_dt_fs=reference_md_dt_between_saved_frames,
            fimd_steps=fimd_steps,
            fimd_dt_fs=fimd_dt_fs,
            fimd_temperature_K=fimd_temperature_K,
            fimd_friction_per_fs=fimd_friction_per_fs,
            save_interval=save_interval,
            hessian_step=hessian_step,
            max_initial_displacement=max_initial_displacement,
            precision=precision,
            velocity_unit="ase",
            verbose=verbose,
        )
        results[slug] = result
    return results


def run_multiband_fimd_from_xyz(
    xyz_file: str,
    calculator: Union[str, Any] = "mace_mp",
    calculator_kwargs: Optional[Dict[str, Any]] = None,
    bands: Union[str, Sequence[BandLike]] = "0:200,200:600,600:1200,1450:1750",
    output_dir: str = "fimd_200ps_multiband",
    minimise_fmax: float = 0.02,
    minimise_steps: int = 2000,
    conventional_md_dt_fs: float = 0.5,
    temperature_K: float = 300.0,
    thermalisation_ps: float = 20.0,
    equilibration_ps: float = 20.0,
    thermalisation_ensemble: str = "nvt",
    equilibration_ensemble: str = "nvt",
    friction_per_fs: float = 0.01,
    thermalisation_save_interval: int = 200,
    equilibration_save_interval: int = 20,
    production_ps: float = 200.0,
    fimd_dt_fs: float = 0.5,
    fimd_temperature_K: Optional[float] = None,
    fimd_friction_per_fs: float = 0.01,
    fimd_save_interval: int = 200,
    hessian_step: float = 0.001,
    max_initial_displacement: Optional[float] = None,
    precision: FloatDTypeLike = "float64",
    seed: Optional[int] = 12345,
    verbose: bool = True,
) -> MultiBandFIMDResult:
    """Full minimisation -> conventional MD -> multi-band FIMD protocol."""
    os.makedirs(output_dir, exist_ok=True)
    prepared = prepare_equilibrated_reference(
        xyz_file=xyz_file,
        calculator=calculator,
        calculator_kwargs=calculator_kwargs,
        output_dir=os.path.join(output_dir, "reference"),
        minimise_fmax=minimise_fmax,
        minimise_steps=minimise_steps,
        md_dt_fs=conventional_md_dt_fs,
        temperature_K=temperature_K,
        thermalisation_ps=thermalisation_ps,
        equilibration_ps=equilibration_ps,
        thermalisation_ensemble=thermalisation_ensemble,
        equilibration_ensemble=equilibration_ensemble,
        friction_per_fs=friction_per_fs,
        thermalisation_save_interval=thermalisation_save_interval,
        equilibration_save_interval=equilibration_save_interval,
        precision=precision,
        seed=seed,
        verbose=verbose,
    )
    band_results = run_fimd_bands_from_prepared_reference(
        prepared=prepared,
        calculator=calculator,
        calculator_kwargs=calculator_kwargs,
        bands=bands,
        output_dir=os.path.join(output_dir, "bands"),
        production_ps=production_ps,
        fimd_dt_fs=fimd_dt_fs,
        fimd_temperature_K=fimd_temperature_K,
        fimd_friction_per_fs=fimd_friction_per_fs,
        save_interval=fimd_save_interval,
        hessian_step=hessian_step,
        max_initial_displacement=max_initial_displacement,
        precision=precision,
        verbose=verbose,
    )
    metadata_file = os.path.join(output_dir, "multiband_protocol.json")
    metadata = {
        "prepared_reference": asdict(prepared),
        "bands": list(parse_bands(bands)),
        "band_output_dirs": {slug: result.output_dir for slug, result in band_results.items()},
        "production_ps": production_ps,
        "fimd_dt_fs": fimd_dt_fs,
        "fimd_save_interval": fimd_save_interval,
        "hessian_step": hessian_step,
        "max_initial_displacement": max_initial_displacement,
        "precision": str(precision),
    }
    with open(metadata_file, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return MultiBandFIMDResult(
        prepared_reference=prepared,
        band_results=band_results,
        metadata_file=metadata_file,
    )


__all__ = [
    "BandLike",
    "PreparedReference",
    "MultiBandFIMDResult",
    "band_slug",
    "parse_bands",
    "ps_to_steps",
    "prepare_equilibrated_reference",
    "run_fimd_bands_from_prepared_reference",
    "run_multiband_fimd_from_xyz",
]
