#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core Fourier-Integrator Molecular Dynamics classes.

This module deliberately keeps the ASE separation of responsibilities:
calculators provide energies/forces, while ``FIMDynamics`` is an ASE
``MolecularDynamics`` integrator.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from ase import Atoms, units
    from ase.io import read
    from ase.md.md import MolecularDynamics
except Exception as exc:  # pragma: no cover - import-time user guidance
    raise ImportError(
        "The fimd package requires ASE. Install it with: pip install ase "
        "or conda install -c conda-forge ase."
    ) from exc


FloatDTypeLike = Union[str, np.dtype, type]


def resolve_precision(precision: FloatDTypeLike = "float64") -> np.dtype:
    """Resolve user-facing precision aliases to ``np.float32`` or ``np.float64``."""
    if isinstance(precision, np.dtype):
        dtype = precision
    elif isinstance(precision, str):
        key = precision.strip().lower()
        aliases = {
            "f": np.float32,
            "float": np.float32,
            "single": np.float32,
            "float32": np.float32,
            "fp32": np.float32,
            "d": np.float64,
            "double": np.float64,
            "float64": np.float64,
            "fp64": np.float64,
        }
        if key not in aliases:
            raise ValueError(
                f"Unsupported precision {precision!r}. Use 'float32'/'float' "
                "or 'float64'/'double'."
            )
        dtype = np.dtype(aliases[key])
    else:
        dtype = np.dtype(precision)

    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(f"Precision must be float32 or float64, not {dtype}.")
    return dtype


def complex_dtype_for(dtype: np.dtype) -> np.dtype:
    return np.dtype(np.complex64 if dtype == np.dtype(np.float32) else np.complex128)


def ase_velocity_to_ang_per_fs(v_ase: np.ndarray, precision: FloatDTypeLike = "float64") -> np.ndarray:
    """Convert ASE internal velocities to physical velocities in Å/fs."""
    dtype = resolve_precision(precision)
    return np.asarray(v_ase, dtype=dtype) * dtype.type(units.fs)


def ang_per_fs_to_ase_velocity(v_ang_fs: np.ndarray, precision: FloatDTypeLike = "float64") -> np.ndarray:
    """Convert physical velocities in Å/fs to ASE internal velocity units."""
    dtype = resolve_precision(precision)
    return np.asarray(v_ang_fs, dtype=dtype) / dtype.type(units.fs)


def set_velocities_ang_per_fs(atoms: Atoms, velocities_ang_fs: np.ndarray, precision: FloatDTypeLike = "float64") -> None:
    atoms.set_velocities(ang_per_fs_to_ase_velocity(velocities_ang_fs, precision))


def get_velocities_ang_per_fs(atoms: Atoms, precision: FloatDTypeLike = "float64") -> Optional[np.ndarray]:
    v = atoms.get_velocities()
    if v is None:
        return None
    return ase_velocity_to_ang_per_fs(v, precision)


def _normalise_velocity_unit(unit: str) -> str:
    key = str(unit).strip().lower().replace(" ", "").replace("_", "-")
    aliases = {
        "ase": "ase",
        "ase-internal": "ase",
        "angstrom/fs": "angstrom/fs",
        "a/fs": "angstrom/fs",
        "å/fs": "angstrom/fs",
        "ang/fs": "angstrom/fs",
        "angperfs": "angstrom/fs",
        "angstrom/ps": "angstrom/ps",
        "a/ps": "angstrom/ps",
        "å/ps": "angstrom/ps",
        "ang/ps": "angstrom/ps",
        "angperps": "angstrom/ps",
    }
    if key not in aliases:
        raise ValueError(
            f"Unsupported velocity_unit={unit!r}. Use 'angstrom/fs', "
            "'angstrom/ps', or 'ase'."
        )
    return aliases[key]


def velocities_to_ang_per_fs(
    velocities: np.ndarray,
    velocity_unit: str,
    precision: FloatDTypeLike = "float64",
) -> np.ndarray:
    """Convert an array of velocities to Å/fs."""
    dtype = resolve_precision(precision)
    unit = _normalise_velocity_unit(velocity_unit)
    v = np.asarray(velocities, dtype=dtype)
    if unit == "angstrom/fs":
        return v
    if unit == "angstrom/ps":
        return v * dtype.type(1.0e-3)
    if unit == "ase":
        return ase_velocity_to_ang_per_fs(v, dtype)
    raise AssertionError("unreachable")


def _tag_frame(frame: Atoms, dt_fs: float, precision: np.dtype) -> None:
    frame.info["fimd_dt_fs"] = float(dt_fs)
    frame.info["fimd_precision"] = precision.name


# =============================================================================
# Trajectory loading
# =============================================================================


def load_trajectory(
    filename: str,
    dt_fs: float = 1.0,
    symbols: Optional[Sequence[str]] = None,
    masses: Optional[np.ndarray] = None,
    precision: FloatDTypeLike = "float64",
    velocity_unit: str = "angstrom/ps",
    mmap_mode: Optional[str] = None,
    verbose: bool = True,
) -> List[Atoms]:
    """
    Load a trajectory into a list of ASE ``Atoms`` frames.

    ``.npz`` files must contain one of ``positions``, ``coords``, or ``xyz``;
    velocities are optional and may be stored as ``velocities``, ``vels``,
    ``vel``, or ``v``. ASE-readable formats are delegated to ``ase.io.read``.

    Velocities inside ASE frames are always stored in ASE internal units, but
    all FIMD internals use physical Å/fs.
    """
    dtype = resolve_precision(precision)
    ext = os.path.splitext(filename)[1].lower()
    if verbose:
        print(f"Loading trajectory: {filename} (format={ext}, precision={dtype.name})")

    if ext == ".npz":
        trajectory = _load_npz_trajectory(
            filename, dt_fs, symbols, masses, dtype, velocity_unit, mmap_mode, verbose
        )
    elif ext == ".npy":
        trajectory = _load_npy_trajectory(filename, dt_fs, symbols, masses, dtype, mmap_mode, verbose)
    else:
        try:
            frames = read(filename, index=":")
        except Exception as exc:
            raise ValueError(
                f"Unsupported trajectory format {ext!r}; ASE also failed to read it."
            ) from exc
        if isinstance(frames, Atoms):
            trajectory = [frames]
        else:
            trajectory = list(frames)
        _apply_symbols_masses(trajectory, symbols, masses)
        ensure_velocities(trajectory, dt_fs=dt_fs, precision=dtype, verbose=verbose)

    for frame in trajectory:
        _tag_frame(frame, dt_fs, dtype)

    if verbose and trajectory:
        print(f"  Loaded {len(trajectory)} frames, {len(trajectory[0])} atoms")
    return trajectory


def _get_first_array(data: Any, names: Sequence[str]) -> Optional[np.ndarray]:
    for name in names:
        if name in data:
            return data[name]
    return None


def _load_npz_trajectory(
    filename: str,
    dt_fs: float,
    symbols: Optional[Sequence[str]],
    masses: Optional[np.ndarray],
    dtype: np.dtype,
    velocity_unit: str,
    mmap_mode: Optional[str],
    verbose: bool,
) -> List[Atoms]:
    data = np.load(filename, allow_pickle=True, mmap_mode=mmap_mode)
    positions = _get_first_array(data, ("positions", "coords", "xyz"))
    if positions is None:
        raise ValueError("NPZ trajectory must contain 'positions', 'coords', or 'xyz'.")
    positions = np.asarray(positions, dtype=dtype)
    if positions.ndim == 2:
        positions = positions[np.newaxis, :, :]
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("Positions must have shape (n_frames, n_atoms, 3) or (n_atoms, 3).")

    velocities = _get_first_array(data, ("velocities", "vels", "vel", "v"))
    if velocities is None:
        if verbose:
            print("  Warning: no velocities in NPZ; estimating them from positions.")
        velocities_ang_fs = estimate_velocities(positions, dt_fs=dt_fs, precision=dtype)
    else:
        velocities = np.asarray(velocities, dtype=dtype)
        if velocities.ndim == 2:
            velocities = velocities[np.newaxis, :, :]
        velocities_ang_fs = velocities_to_ang_per_fs(velocities, velocity_unit, dtype)

    if symbols is None:
        sym = _get_first_array(data, ("symbols", "elements"))
        if sym is None:
            raise ValueError("NPZ has no symbols; pass symbols=... from a topology or reference file.")
        symbols = [str(x) for x in list(sym)]
    if masses is None:
        m = _get_first_array(data, ("masses",))
        masses = None if m is None else np.asarray(m, dtype=dtype)
    else:
        masses = np.asarray(masses, dtype=dtype)

    trajectory: List[Atoms] = []
    for pos, vel in zip(positions, velocities_ang_fs):
        atoms = Atoms(symbols=list(symbols), positions=np.asarray(pos, dtype=dtype))
        if masses is not None:
            atoms.set_masses(masses)
        set_velocities_ang_per_fs(atoms, vel, dtype)
        trajectory.append(atoms)
    return trajectory


def _load_npy_trajectory(
    filename: str,
    dt_fs: float,
    symbols: Optional[Sequence[str]],
    masses: Optional[np.ndarray],
    dtype: np.dtype,
    mmap_mode: Optional[str],
    verbose: bool,
) -> List[Atoms]:
    positions = np.asarray(np.load(filename, mmap_mode=mmap_mode), dtype=dtype)
    if positions.ndim == 2:
        positions = positions[np.newaxis, :, :]
    if symbols is None:
        raise ValueError("NPY trajectories contain positions only; pass symbols=....")
    if verbose:
        print("  Warning: NPY has no velocities; estimating them from positions.")
    velocities_ang_fs = estimate_velocities(positions, dt_fs=dt_fs, precision=dtype)
    masses_arr = None if masses is None else np.asarray(masses, dtype=dtype)

    trajectory: List[Atoms] = []
    for pos, vel in zip(positions, velocities_ang_fs):
        atoms = Atoms(symbols=list(symbols), positions=np.asarray(pos, dtype=dtype))
        if masses_arr is not None:
            atoms.set_masses(masses_arr)
        set_velocities_ang_per_fs(atoms, vel, dtype)
        trajectory.append(atoms)
    return trajectory


def _apply_symbols_masses(
    trajectory: Sequence[Atoms],
    symbols: Optional[Sequence[str]],
    masses: Optional[np.ndarray],
) -> None:
    for atoms in trajectory:
        if symbols is not None:
            atoms.set_chemical_symbols(list(symbols))
        if masses is not None:
            atoms.set_masses(np.asarray(masses, dtype=float))


def estimate_velocities(
    positions: np.ndarray,
    dt_fs: float,
    precision: FloatDTypeLike = "float64",
) -> np.ndarray:
    """Estimate velocities from positions, returning physical Å/fs."""
    dtype = resolve_precision(precision)
    positions = np.asarray(positions, dtype=dtype)
    if len(positions) <= 1:
        return np.zeros_like(positions, dtype=dtype)
    return np.gradient(positions, dtype.type(dt_fs), axis=0).astype(dtype, copy=False)


def ensure_velocities(
    trajectory: Sequence[Atoms],
    dt_fs: float,
    precision: FloatDTypeLike = "float64",
    verbose: bool = True,
) -> None:
    """Ensure each frame has velocities; estimate them if missing."""
    dtype = resolve_precision(precision)
    missing = False
    try:
        missing = trajectory[0].get_velocities() is None
    except Exception:
        missing = True
    if not missing:
        return
    if verbose:
        print("  Warning: no ASE velocities found; estimating from positions.")
    positions = np.asarray([atoms.get_positions() for atoms in trajectory], dtype=dtype)
    velocities_ang_fs = estimate_velocities(positions, dt_fs=dt_fs, precision=dtype)
    for atoms, vel in zip(trajectory, velocities_ang_fs):
        set_velocities_ang_per_fs(atoms, vel, dtype)


def mass_weighted_com(positions: np.ndarray, masses: np.ndarray, precision: FloatDTypeLike = "float64") -> np.ndarray:
    """Mass-weighted centre of mass for one frame."""
    dtype = resolve_precision(precision)
    pos = np.asarray(positions, dtype=dtype)
    m = np.asarray(masses, dtype=dtype).reshape(-1, 1)
    return np.sum(m * pos, axis=0) / np.sum(m)


def mass_weighted_kabsch_rotation(positions: np.ndarray, reference: np.ndarray, masses: np.ndarray, precision: FloatDTypeLike = "float64") -> np.ndarray:
    """Return R such that (positions-COM) @ R aligns to (reference-COM)."""
    dtype = resolve_precision(precision)
    pos = np.asarray(positions, dtype=dtype)
    ref = np.asarray(reference, dtype=dtype)
    m = np.asarray(masses, dtype=dtype)
    p0 = pos - mass_weighted_com(pos, m, dtype)
    q0 = ref - mass_weighted_com(ref, m, dtype)
    sw = np.sqrt(m).reshape(-1, 1)
    H = (sw * p0).T @ (sw * q0)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]).astype(dtype) @ U.T
    return R.astype(dtype, copy=False)


def align_trajectory_to_reference(
    positions: np.ndarray,
    reference: np.ndarray,
    masses: np.ndarray,
    velocities_ang_fs: Optional[np.ndarray] = None,
    precision: FloatDTypeLike = "float64",
):
    """Remove COM translation and rigid rotation relative to a reference geometry.

    Returns aligned positions and rotations.  If velocities are supplied, the
    centre-of-mass velocity is removed and the same Kabsch rotation is applied.
    """
    dtype = resolve_precision(precision)
    pos = np.asarray(positions, dtype=dtype)
    ref = np.asarray(reference, dtype=dtype)
    m = np.asarray(masses, dtype=dtype)
    ref_com = mass_weighted_com(ref, m, dtype)
    aligned = np.empty_like(pos, dtype=dtype)
    rotations = np.empty((len(pos), 3, 3), dtype=dtype)
    vel_aligned = None if velocities_ang_fs is None else np.empty_like(np.asarray(velocities_ang_fs, dtype=dtype))
    mcol = m.reshape(-1, 1)
    mtot = np.sum(m)
    for i, frame in enumerate(pos):
        R = mass_weighted_kabsch_rotation(frame, ref, m, dtype)
        rotations[i] = R
        com = mass_weighted_com(frame, m, dtype)
        aligned[i] = (frame - com) @ R + ref_com
        if vel_aligned is not None:
            v = np.asarray(velocities_ang_fs[i], dtype=dtype)
            v_com = np.sum(mcol * v, axis=0) / mtot
            vel_aligned[i] = (v - v_com) @ R
    if vel_aligned is None:
        return aligned, rotations
    return aligned, vel_aligned, rotations


def limit_modal_displacement(
    q: np.ndarray,
    p: np.ndarray,
    W: np.ndarray,
    masses: np.ndarray,
    max_displacement: float,
    precision: FloatDTypeLike = "float64",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Scale q,p together if the reconstructed modal displacement is too large."""
    dtype = resolve_precision(precision)
    if max_displacement <= 0:
        raise ValueError("max_displacement must be positive")
    inv_sqrt_m = np.repeat(dtype.type(1.0) / np.sqrt(np.asarray(masses, dtype=dtype)), 3)
    dx = inv_sqrt_m * (np.asarray(W, dtype=dtype) @ np.asarray(q, dtype=dtype))
    max_abs = float(np.max(np.abs(dx))) if dx.size else 0.0
    if max_abs <= max_displacement or max_abs == 0.0:
        return np.asarray(q, dtype=dtype), np.asarray(p, dtype=dtype), 1.0
    scale = float(max_displacement / max_abs)
    return (np.asarray(q, dtype=dtype) * dtype.type(scale),
            np.asarray(p, dtype=dtype) * dtype.type(scale),
            scale)


# =============================================================================
# Physical constants
# =============================================================================


C_CM_PER_S = 2.99792458e10
CM1_TO_RADFS = 2.0 * np.pi * C_CM_PER_S * 1.0e-15
RADFS_TO_CM1 = 1.0 / CM1_TO_RADFS

# 1 eV = 9.648533...e-3 amu Å^2 fs^-2.  The old code used 9.648e-3;
# keep a slightly more precise value while preserving the same convention.
EV_TO_AMU_A2_FS2 = 9.648533212331001e-3
EIG_TO_RADFS2 = EV_TO_AMU_A2_FS2
FORCE_CONV = EV_TO_AMU_A2_FS2


def cm1_to_radfs(cm1: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return np.asarray(cm1) * CM1_TO_RADFS


def radfs_to_cm1(radfs: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return np.asarray(radfs) * RADFS_TO_CM1


# =============================================================================
# FIMDBasis
# =============================================================================


@dataclass
class FIMDBasis:
    reference_positions: np.ndarray
    masses: np.ndarray
    symbols: List[str]
    W: np.ndarray
    omega: np.ndarray
    K: np.ndarray
    reference_force: np.ndarray
    band: Tuple[float, float]
    band_mask: np.ndarray
    q0: np.ndarray
    p0: np.ndarray
    fft_frequencies: np.ndarray
    fft_amplitudes: np.ndarray
    n_atoms: int
    n_modes: int
    n_active_modes: int
    timestep_reference: float
    precision: str = "float64"
    n_rigid_modes: int = 6
    reference_energy: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_trajectory(
        cls,
        trajectory: Sequence[Atoms],
        reference: Atoms,
        calculator: Any,
        band: Tuple[float, float],
        hessian_step: float = 0.001,
        window: str = "hann",
        calibration_frames: Optional[int] = None,
        timestep_reference: Optional[float] = None,
        precision: FloatDTypeLike = "float64",
        n_rigid_modes: Optional[int] = None,
        remove_rigid_motion: bool = True,
        max_initial_displacement: Optional[float] = None,
        initial_temperature_K: Optional[float] = None,
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> "FIMDBasis":
        dtype = resolve_precision(precision)
        cdtype = complex_dtype_for(dtype)

        if len(trajectory) == 0:
            raise ValueError("trajectory must contain at least one frame.")
        n_atoms = len(reference)
        n_modes = 3 * n_atoms
        if len(trajectory[0]) != n_atoms:
            raise ValueError(f"Trajectory atoms ({len(trajectory[0])}) != reference atoms ({n_atoms}).")

        if n_rigid_modes is None:
            n_rigid_modes = 5 if n_atoms == 2 else min(6, n_modes)

        masses = np.asarray(reference.get_masses(), dtype=dtype)
        symbols = list(reference.get_chemical_symbols())
        reference_positions = np.asarray(reference.get_positions(), dtype=dtype).copy()

        if verbose:
            print("=" * 60)
            print("FIMDBasis: analysing trajectory for band-limited FIMD")
            print("=" * 60)
            print(f"System: {n_atoms} atoms, {n_modes} Cartesian DOF")
            print(f"Band: {band[0]:.3f} - {band[1]:.3f} cm^-1")
            print("[1/4] Computing numerical Hessian at reference geometry...")

        ref_with_calc = reference.copy()
        ref_with_calc.calc = calculator
        K = _compute_numerical_hessian(ref_with_calc, h=hessian_step, precision=dtype)
        reference_force = _compute_reference_force(ref_with_calc, precision=dtype)
        try:
            reference_energy = float(ref_with_calc.get_potential_energy())
        except Exception:
            reference_energy = None
        if verbose:
            f0_rms = float(np.sqrt(np.mean(reference_force**2)))
            if f0_rms > 1e-6:
                print(f"      Reference-force RMS: {f0_rms:.3e} eV/Å (linear Taylor term retained)")
        H_mw = _mass_weight_hessian(K, masses, precision=dtype)
        eig, W = _diagonalize_hessian(H_mw, precision=dtype)
        omega = np.sqrt(np.maximum(eig * dtype.type(EIG_TO_RADFS2), dtype.type(0.0))).astype(dtype)
        freq_cm1 = np.asarray(radfs_to_cm1(omega), dtype=dtype)
        n_imaginary = int(np.sum(eig < dtype.type(-1.0e-6)))
        if verbose:
            start = min(n_rigid_modes, len(freq_cm1) - 1)
            print(f"      Frequency range: {freq_cm1[start]:.3f} - {freq_cm1[-1]:.3f} cm^-1")
            print(f"      Imaginary modes: {n_imaginary}")

        frames = list(trajectory[-calibration_frames:] if calibration_frames is not None else trajectory)
        n_frames = len(frames)
        if n_frames == 0:
            raise ValueError("No frames left after applying calibration_frames.")

        if timestep_reference is None:
            timestep_reference = float(frames[0].info.get("fimd_dt_fs", 1.0))
        dt_fs = float(timestep_reference)

        if verbose:
            print("[2/4] Projecting trajectory and computing FFT diagnostics...")
            print(f"      Using {n_frames} frames; dt={dt_fs:g} fs")

        positions_raw = np.asarray([atoms.get_positions() for atoms in frames], dtype=dtype)
        velocities_list: List[np.ndarray] = []
        missing_vel = False
        for atoms in frames:
            vel = get_velocities_ang_per_fs(atoms, dtype)
            if vel is None:
                missing_vel = True
                break
            velocities_list.append(vel)

        if remove_rigid_motion:
            if missing_vel:
                positions, rotations = align_trajectory_to_reference(
                    positions_raw, reference_positions, masses, velocities_ang_fs=None, precision=dtype
                )
                if verbose:
                    print("      Rigid-body motion removed; estimating velocities after alignment.")
                velocities = estimate_velocities(positions, dt_fs=dt_fs, precision=dtype)
            else:
                positions, velocities, rotations = align_trajectory_to_reference(
                    positions_raw, reference_positions, masses,
                    velocities_ang_fs=np.asarray(velocities_list, dtype=dtype), precision=dtype
                )
                if verbose:
                    print("      Removed centre-of-mass and rotational motion before modal projection.")
        else:
            positions = positions_raw
            if missing_vel:
                if verbose:
                    print("      Warning: missing velocities; estimating from positions.")
                velocities = estimate_velocities(positions, dt_fs=dt_fs, precision=dtype)
            else:
                velocities = np.asarray(velocities_list, dtype=dtype)

        sqrt_M = np.repeat(np.sqrt(masses), 3).astype(dtype, copy=False)
        x_mw = np.empty((n_frames, n_modes), dtype=dtype)
        v_mw = np.empty((n_frames, n_modes), dtype=dtype)
        for i in range(n_frames):
            dx = (positions[i] - reference_positions).reshape(-1)
            x_mw[i] = sqrt_M * dx
            v_mw[i] = sqrt_M * velocities[i].reshape(-1)

        win = _window(n_frames, window, dtype)
        x_fft = np.fft.rfft((x_mw * win[:, None]).astype(dtype, copy=False), axis=0).astype(cdtype)
        fft_freq_hz = np.fft.rfftfreq(n_frames, d=dt_fs * 1.0e-15)
        fft_freq_cm1 = np.asarray(fft_freq_hz / C_CM_PER_S, dtype=dtype)
        if verbose and len(fft_freq_cm1) > 1:
            print(f"      FFT frequency resolution: {fft_freq_cm1[1]:.3f} cm^-1")

        if verbose:
            print("[3/4] Selecting active normal modes...")
        band_mask = _create_band_mask(omega, band[0], band[1], n_rigid_modes=n_rigid_modes, eig=eig)
        n_active = int(np.sum(band_mask))
        if n_active == 0:
            warnings.warn(f"No modes found in band {band[0]}-{band[1]} cm^-1.")
        elif verbose:
            active_freq = freq_cm1[band_mask]
            print(f"      Active modes: {n_active} / {max(n_modes - n_rigid_modes, 0)} vibrational")
            print(f"      Active frequency range: {active_freq.min():.3f} - {active_freq.max():.3f} cm^-1")

        if verbose:
            print("[4/4] Extracting band-limited initial conditions...")
        x_last = (positions[-1] - reference_positions).reshape(-1)
        v_last = velocities[-1]
        q0 = (W.T @ (sqrt_M * x_last)).astype(dtype, copy=False)
        p0 = (W.T @ (sqrt_M * v_last.reshape(-1))).astype(dtype, copy=False)
        q0[~band_mask] = dtype.type(0.0)
        p0[~band_mask] = dtype.type(0.0)

        if initial_temperature_K is not None and n_active > 0:
            rng = np.random.default_rng(seed)
            kT_modal = dtype.type(units.kB * float(initial_temperature_K)) * dtype.type(EV_TO_AMU_A2_FS2)
            p0[band_mask] = rng.normal(scale=float(np.sqrt(kT_modal)), size=n_active).astype(dtype)

        if max_initial_displacement is not None:
            q0, p0, scale = limit_modal_displacement(
                q0, p0, W, masses, max_displacement=float(max_initial_displacement), precision=dtype
            )
            if verbose and scale < 1.0:
                print(f"      Scaled initial in-band amplitudes by {scale:.6g} to keep max |dx| <= {max_initial_displacement:g} Å")

        if verbose:
            print("=" * 60)
            print("FIMDBasis ready")
            print("=" * 60)

        return cls(
            reference_positions=reference_positions,
            masses=masses,
            symbols=symbols,
            W=W,
            omega=omega,
            K=K,
            reference_force=reference_force,
            reference_energy=reference_energy,
            band=(float(band[0]), float(band[1])),
            band_mask=band_mask,
            q0=q0,
            p0=p0,
            fft_frequencies=fft_freq_cm1,
            fft_amplitudes=x_fft.T,
            n_atoms=n_atoms,
            n_modes=n_modes,
            n_active_modes=n_active,
            timestep_reference=dt_fs,
            precision=dtype.name,
            n_rigid_modes=int(n_rigid_modes),
            metadata={"window": window, "hessian_step": float(hessian_step), "remove_rigid_motion": bool(remove_rigid_motion), "reference_force_correction": True},
        )

    @classmethod
    def from_trajectory_covariance(
        cls,
        trajectory: Sequence[Atoms],
        reference: Atoms,
        calculator: Any,
        band: Tuple[float, float],
        window: str = "hann",
        calibration_frames: Optional[int] = None,
        timestep_reference: Optional[float] = None,
        precision: FloatDTypeLike = "float64",
        n_rigid_modes: Optional[int] = None,
        remove_rigid_motion: bool = True,
        n_modes_keep: Optional[int] = None,
        max_initial_displacement: Optional[float] = None,
        initial_temperature_K: Optional[float] = None,
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> "FIMDBasis":
        """Build the FIMD basis from a conventional MD trajectory (no Hessian).

        This is the trajectory/spectral route of the paper (the ``*`` curves in
        Fig. 1): the mode matrix ``W`` is the eigenbasis of the windowed
        mass-weighted covariance ``Ĉ ≈ ⟨x xᵀ⟩``, and each mode frequency ``ω_ν``
        is read off as the peak of the projected-velocity power spectrum
        ``|FFT(w_νᵀ M^{1/2} ṙ)|²``.  An effective Cartesian Hessian consistent
        with ``(W, Ω)`` is synthesised so the runtime residual-force kick uses
        the same harmonic baseline as the drift.

        The calculator is still required: it provides the runtime forces for the
        residual-force kick and the reference force ``F(r0)`` for the linear
        Taylor term, but it is *not* used to build ``W`` or ``Ω``.
        """
        dtype = resolve_precision(precision)
        cdtype = complex_dtype_for(dtype)

        if len(trajectory) == 0:
            raise ValueError("trajectory must contain at least one frame.")
        n_atoms = len(reference)
        n_modes = 3 * n_atoms
        if len(trajectory[0]) != n_atoms:
            raise ValueError(f"Trajectory atoms ({len(trajectory[0])}) != reference atoms ({n_atoms}).")
        if n_rigid_modes is None:
            n_rigid_modes = 5 if n_atoms == 2 else min(6, n_modes)

        masses = np.asarray(reference.get_masses(), dtype=dtype)
        symbols = list(reference.get_chemical_symbols())
        reference_positions = np.asarray(reference.get_positions(), dtype=dtype).copy()

        if verbose:
            print("=" * 60)
            print("FIMDBasis: trajectory-derived (covariance) basis")
            print("=" * 60)
            print(f"System: {n_atoms} atoms, {n_modes} Cartesian DOF")
            print(f"Band: {band[0]:.3f} - {band[1]:.3f} cm^-1")

        # NOTE: reference_force is computed later, at r0 = <r> (the trajectory
        # mean), once the frames are aligned.

        frames = list(trajectory[-calibration_frames:] if calibration_frames is not None else trajectory)
        n_frames = len(frames)
        if n_frames < 4:
            raise ValueError("Covariance basis needs at least ~4 frames; supply a longer trajectory.")
        if timestep_reference is None:
            timestep_reference = float(frames[0].info.get("fimd_dt_fs", 1.0))
        dt_fs = float(timestep_reference)

        positions_raw = np.asarray([a.get_positions() for a in frames], dtype=dtype)
        velocities_list: List[np.ndarray] = []
        missing_vel = False
        for a in frames:
            vel = get_velocities_ang_per_fs(a, dtype)
            if vel is None:
                missing_vel = True
                break
            velocities_list.append(vel)

        if remove_rigid_motion:
            if missing_vel:
                positions, _ = align_trajectory_to_reference(
                    positions_raw, reference_positions, masses, velocities_ang_fs=None, precision=dtype
                )
                velocities = estimate_velocities(positions, dt_fs=dt_fs, precision=dtype)
            else:
                positions, velocities, _ = align_trajectory_to_reference(
                    positions_raw, reference_positions, masses,
                    velocities_ang_fs=np.asarray(velocities_list, dtype=dtype), precision=dtype
                )
            if verbose:
                print("      Removed centre-of-mass and rotational motion before covariance estimate.")
        else:
            positions = positions_raw
            velocities = (estimate_velocities(positions, dt_fs=dt_fs, precision=dtype)
                          if missing_vel else np.asarray(velocities_list, dtype=dtype))

        sqrt_M = np.repeat(np.sqrt(masses), 3).astype(dtype, copy=False)

        # Paper convention for the trajectory route: r0 = <r>. Re-centre the
        # reference on the (aligned) trajectory mean so the covariance measures
        # fluctuations about the actual sampled geometry, not about an unrelated
        # externally supplied minimum. Everything downstream (q0, runtime
        # reconstruction, reference force) then uses this consistent r0.
        traj_mean = positions.mean(axis=0)
        reference_positions = np.asarray(traj_mean, dtype=dtype).copy()
        ref_with_calc = reference.copy()
        ref_with_calc.set_positions(reference_positions)
        ref_with_calc.calc = calculator
        reference_force = _compute_reference_force(ref_with_calc, precision=dtype)
        try:
            reference_energy = float(ref_with_calc.get_potential_energy())
        except Exception:
            reference_energy = None

        x_mw = np.empty((n_frames, n_modes), dtype=dtype)
        v_mw = np.empty((n_frames, n_modes), dtype=dtype)
        for i in range(n_frames):
            x_mw[i] = sqrt_M * (positions[i] - reference_positions).reshape(-1)
            v_mw[i] = sqrt_M * velocities[i].reshape(-1)

        win = _window(n_frames, window, dtype)

        if verbose:
            print("[1/3] Estimating mode matrix W from windowed covariance...")
        cov_evals, W = _covariance_eigenbasis(x_mw, win, precision=dtype)

        if verbose:
            print("[2/3] Assigning frequencies from projected-velocity spectra...")
        omega = _frequencies_from_velocity_spectra(v_mw, W, dt_fs, win, precision=dtype)
        freq_cm1 = np.asarray(radfs_to_cm1(omega), dtype=dtype)

        # Optionally keep only the dominant covariance modes; the rest are frozen
        # by giving them zero variance contribution (they stay out of any band via
        # the velocity-spectrum frequencies and explicit masking below).
        keep_mask = np.ones(n_modes, dtype=bool)
        if n_modes_keep is not None and 0 < int(n_modes_keep) < n_modes:
            keep_mask[int(n_modes_keep):] = False

        # Synthesise an effective Hessian consistent with (W, Ω) for the kick.
        K = _effective_hessian_from_modes(W, omega, masses, precision=dtype)

        # Band selection: covariance modes are not eigenvalue-signed like a
        # Hessian, so exclude (a) modes with omega below a small floor (drift /
        # rigid) and (b) the lowest-variance modes if a keep count was given.
        eps_omega = float(cm1_to_radfs(1.0))
        band_mask = (freq_cm1 >= band[0]) & (freq_cm1 <= band[1])
        band_mask &= (omega > dtype.type(eps_omega))
        band_mask &= keep_mask
        # Drop the n_rigid_modes lowest-frequency surviving modes as rigid/soft drift.
        if n_rigid_modes > 0:
            surviving = np.where(band_mask)[0]
            if surviving.size:
                lowest = surviving[np.argsort(freq_cm1[surviving])[: min(n_rigid_modes, surviving.size)]]
                # Only treat as rigid those genuinely near zero (< ~10 cm^-1).
                lowest = lowest[freq_cm1[lowest] < 10.0]
                band_mask[lowest] = False
        n_active = int(np.sum(band_mask))
        if n_active == 0:
            warnings.warn(f"No covariance modes found in band {band[0]}-{band[1]} cm^-1.")
        elif verbose:
            af = freq_cm1[band_mask]
            print(f"      Active modes: {n_active}; range {af.min():.2f} - {af.max():.2f} cm^-1")

        if verbose:
            print("[3/3] Extracting band-limited initial conditions...")
        x_last = (positions[-1] - reference_positions).reshape(-1)
        v_last = velocities[-1].reshape(-1)
        q0 = (W.T @ (sqrt_M * x_last)).astype(dtype, copy=False)
        p0 = (W.T @ (sqrt_M * v_last)).astype(dtype, copy=False)
        q0[~band_mask] = dtype.type(0.0)
        p0[~band_mask] = dtype.type(0.0)

        if initial_temperature_K is not None and n_active > 0:
            rng = np.random.default_rng(seed)
            kT_modal = dtype.type(units.kB * float(initial_temperature_K)) * dtype.type(EV_TO_AMU_A2_FS2)
            p0[band_mask] = rng.normal(scale=float(np.sqrt(kT_modal)), size=n_active).astype(dtype)

        if max_initial_displacement is not None:
            q0, p0, scale = limit_modal_displacement(
                q0, p0, W, masses, max_displacement=float(max_initial_displacement), precision=dtype
            )
            if verbose and scale < 1.0:
                print(f"      Scaled initial in-band amplitudes by {scale:.6g}")

        # FFT diagnostics (same convention as the Hessian route).
        x_fft = np.fft.rfft((x_mw * win[:, None]).astype(dtype, copy=False), axis=0).astype(cdtype)
        fft_freq_hz = np.fft.rfftfreq(n_frames, d=dt_fs * 1.0e-15)
        fft_freq_cm1 = np.asarray(fft_freq_hz / C_CM_PER_S, dtype=dtype)

        if verbose:
            print("=" * 60)
            print("FIMDBasis ready (covariance route)")
            print("=" * 60)

        return cls(
            reference_positions=reference_positions,
            masses=masses,
            symbols=symbols,
            W=W,
            omega=omega,
            K=K,
            reference_force=reference_force,
            reference_energy=reference_energy,
            band=(float(band[0]), float(band[1])),
            band_mask=band_mask,
            q0=q0,
            p0=p0,
            fft_frequencies=fft_freq_cm1,
            fft_amplitudes=x_fft.T,
            n_atoms=n_atoms,
            n_modes=n_modes,
            n_active_modes=n_active,
            timestep_reference=dt_fs,
            precision=dtype.name,
            n_rigid_modes=int(n_rigid_modes),
            metadata={
                "basis_source": "covariance",
                "window": window,
                "remove_rigid_motion": bool(remove_rigid_motion),
                "reference_force_correction": True,
                "covariance_eigenvalues": cov_evals.tolist(),
            },
        )

    @classmethod
    def from_reference_only(
        cls,
        reference: Atoms,
        calculator: Any,
        band: Optional[Tuple[float, float]] = None,
        hessian_step: float = 0.001,
        precision: FloatDTypeLike = "float64",
        n_rigid_modes: Optional[int] = None,
        verbose: bool = True,
    ) -> "FIMDBasis":
        dtype = resolve_precision(precision)
        n_atoms = len(reference)
        n_modes = 3 * n_atoms
        if n_rigid_modes is None:
            n_rigid_modes = 5 if n_atoms == 2 else min(6, n_modes)
        masses = np.asarray(reference.get_masses(), dtype=dtype)
        symbols = list(reference.get_chemical_symbols())
        reference_positions = np.asarray(reference.get_positions(), dtype=dtype).copy()

        if verbose:
            print("=" * 60)
            print("FIMDBasis: building from reference geometry")
            print("=" * 60)
        ref_with_calc = reference.copy()
        ref_with_calc.calc = calculator
        K = _compute_numerical_hessian(ref_with_calc, h=hessian_step, precision=dtype)
        reference_force = _compute_reference_force(ref_with_calc, precision=dtype)
        try:
            reference_energy = float(ref_with_calc.get_potential_energy())
        except Exception:
            reference_energy = None
        H_mw = _mass_weight_hessian(K, masses, precision=dtype)
        eig, W = _diagonalize_hessian(H_mw, precision=dtype)
        omega = np.sqrt(np.maximum(eig * dtype.type(EIG_TO_RADFS2), dtype.type(0.0))).astype(dtype)
        freq_cm1 = np.asarray(radfs_to_cm1(omega), dtype=dtype)

        if band is None:
            band = (0.0, float(freq_cm1[-1]))
            band_mask = (eig > dtype.type(1.0e-6))
            if n_rigid_modes > 0:
                rigid_idx = np.argsort(np.abs(eig))[: min(int(n_rigid_modes), len(eig))]
                band_mask[rigid_idx] = False
        else:
            band_mask = _create_band_mask(omega, band[0], band[1], n_rigid_modes=n_rigid_modes, eig=eig)
        n_active = int(np.sum(band_mask))

        if verbose:
            start = min(n_rigid_modes, len(freq_cm1) - 1)
            print(f"Frequency range: {freq_cm1[start]:.3f} - {freq_cm1[-1]:.3f} cm^-1")
            print(f"Active modes: {n_active}")

        return cls(
            reference_positions=reference_positions,
            masses=masses,
            symbols=symbols,
            W=W,
            omega=omega,
            K=K,
            reference_force=reference_force,
            reference_energy=reference_energy,
            band=(float(band[0]), float(band[1])),
            band_mask=band_mask,
            q0=np.zeros(n_modes, dtype=dtype),
            p0=np.zeros(n_modes, dtype=dtype),
            fft_frequencies=np.array([], dtype=dtype),
            fft_amplitudes=np.array([], dtype=complex_dtype_for(dtype)),
            n_atoms=n_atoms,
            n_modes=n_modes,
            n_active_modes=n_active,
            timestep_reference=1.0,
            precision=dtype.name,
            n_rigid_modes=int(n_rigid_modes),
            metadata={"hessian_step": float(hessian_step)},
        )

    @property
    def dtype(self) -> np.dtype:
        return resolve_precision(self.precision)

    def get_frequencies_cm1(self) -> np.ndarray:
        return np.asarray(radfs_to_cm1(self.omega), dtype=self.dtype)

    def get_active_frequencies_cm1(self) -> np.ndarray:
        return np.asarray(radfs_to_cm1(self.omega[self.band_mask]), dtype=self.dtype)

    def summary(self) -> str:
        freq = self.get_frequencies_cm1()
        active = self.get_active_frequencies_cm1()
        start = min(self.n_rigid_modes, len(freq) - 1)
        out = [
            "FIMDBasis Summary:",
            f"  Atoms: {self.n_atoms}",
            f"  Total Cartesian modes: {self.n_modes}",
            f"  Active modes: {self.n_active_modes}",
            f"  Band: {self.band[0]:.3f} - {self.band[1]:.3f} cm^-1",
            f"  Precision: {self.precision}",
            f"  Reference dt: {self.timestep_reference:g} fs",
            f"  Full frequency range: {freq[start]:.3f} - {freq[-1]:.3f} cm^-1",
        ]
        if len(active):
            out.append(f"  Active frequency range: {active.min():.3f} - {active.max():.3f} cm^-1")
        return "\n".join(out)

    def save(self, filename: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        np.savez_compressed(
            filename,
            reference_positions=self.reference_positions,
            masses=self.masses,
            symbols=np.array(self.symbols, dtype=object),
            W=self.W,
            omega=self.omega,
            K=self.K,
            reference_force=self.reference_force,
            reference_energy=np.array([] if self.reference_energy is None else float(self.reference_energy), dtype=float),
            band=np.array(self.band, dtype=self.dtype),
            band_mask=self.band_mask,
            q0=self.q0,
            p0=self.p0,
            fft_frequencies=self.fft_frequencies,
            fft_amplitudes=self.fft_amplitudes,
            timestep_reference=np.array(self.timestep_reference, dtype=float),
            precision=np.array(self.precision, dtype=object),
            n_rigid_modes=np.array(self.n_rigid_modes, dtype=int),
            metadata=np.array(self.metadata, dtype=object),
        )
        print(f"Saved FIMDBasis to {filename}")

    @classmethod
    def load(cls, filename: str) -> "FIMDBasis":
        data = np.load(filename, allow_pickle=True)
        precision = str(data["precision"].item()) if "precision" in data else "float64"
        metadata = data["metadata"].item() if "metadata" in data else {}
        n_rigid = int(data["n_rigid_modes"].item()) if "n_rigid_modes" in data else 6
        timestep = float(data["timestep_reference"].item()) if "timestep_reference" in data else 1.0
        band_mask = data["band_mask"].astype(bool)
        reference_force = data["reference_force"] if "reference_force" in data else np.zeros(len(data["omega"]), dtype=resolve_precision(precision))
        if "reference_energy" in data:
            _re = data["reference_energy"]
            reference_energy = None if _re.size == 0 else float(_re.item())
        else:
            reference_energy = None
        return cls(
            reference_positions=data["reference_positions"],
            masses=data["masses"],
            symbols=[str(x) for x in list(data["symbols"])],
            W=data["W"],
            omega=data["omega"],
            K=data["K"],
            reference_force=reference_force,
            reference_energy=reference_energy,
            band=(float(data["band"][0]), float(data["band"][1])),
            band_mask=band_mask,
            q0=data["q0"],
            p0=data["p0"],
            fft_frequencies=data["fft_frequencies"],
            fft_amplitudes=data["fft_amplitudes"],
            n_atoms=len(data["masses"]),
            n_modes=len(data["omega"]),
            n_active_modes=int(np.sum(band_mask)),
            timestep_reference=timestep,
            precision=precision,
            n_rigid_modes=n_rigid,
            metadata=metadata if isinstance(metadata, dict) else {},
        )


# =============================================================================
# FIMDynamics
# =============================================================================


class FIMDynamics(MolecularDynamics):
    """ASE-compatible FIMD integrator using a pre-computed ``FIMDBasis``."""

    def __init__(
        self,
        atoms: Atoms,
        basis: FIMDBasis,
        timestep: Optional[float] = None,
        timestep_fs: Optional[float] = None,
        temperature: Optional[float] = None,
        friction: float = 0.01,
        trajectory: Optional[str] = None,
        logfile: Optional[str] = None,
        loginterval: int = 1,
        use_basis_init: bool = True,
        append_trajectory: bool = False,
        precision: Optional[FloatDTypeLike] = None,
    ):
        dtype = resolve_precision(precision or basis.precision)
        if timestep is None:
            if timestep_fs is None:
                timestep_fs = 1.0
            timestep = float(timestep_fs) * units.fs
        MolecularDynamics.__init__(
            self,
            atoms,
            timestep,
            trajectory,
            logfile,
            loginterval,
            append_trajectory=append_trajectory,
        )

        if len(atoms) != basis.n_atoms:
            raise ValueError(f"Atoms ({len(atoms)}) do not match basis ({basis.n_atoms}).")
        if atoms.calc is None:
            raise ValueError("atoms.calc must be set before constructing FIMDynamics.")

        self.basis = basis
        self.temperature = temperature
        self.friction = float(friction)
        self.precision = dtype.name
        self.reference_energy = (None if basis.reference_energy is None
                                 else float(basis.reference_energy))
        self.dtype = dtype

        self.r_ref = np.asarray(basis.reference_positions, dtype=dtype).copy()
        self.masses = np.asarray(basis.masses, dtype=dtype).copy()
        self.W = np.asarray(basis.W, dtype=dtype).copy()
        self.omega = np.asarray(basis.omega, dtype=dtype).copy()
        self.K = np.asarray(basis.K, dtype=dtype).copy()
        self.reference_force = np.asarray(basis.reference_force, dtype=dtype).reshape(-1).copy()
        self.band_mask = np.asarray(basis.band_mask, dtype=bool).copy()

        # Guard: a zero-frequency mode left active would be propagated as a free
        # drift while still receiving the full residual-force kick, producing an
        # unbounded, self-amplifying trajectory.  This should be impossible given
        # the eigenvalue-based band selection, but a hand-built or legacy basis
        # could violate it, so fail loudly rather than diverge silently.
        active_omega = self.omega[self.band_mask]
        if active_omega.size and np.any(np.abs(active_omega) < dtype.type(1.0e-12)):
            raise ValueError(
                "FIMDynamics received an active mode with omega == 0 (a rigid or "
                "imaginary mode). Rebuild the basis so that band_mask excludes "
                "non-positive-curvature modes by eigenvalue."
            )

        if use_basis_init and (np.any(basis.q0 != 0) or np.any(basis.p0 != 0)):
            self.q = np.asarray(basis.q0, dtype=dtype).copy()
            self.p = np.asarray(basis.p0, dtype=dtype).copy()
            print("FIMD: initialised from trajectory-derived basis state")
        else:
            x = np.asarray(atoms.get_positions(), dtype=dtype)
            v = get_velocities_ang_per_fs(atoms, dtype)
            if v is None:
                v = np.zeros_like(x, dtype=dtype)
            self.q, self.p = _cartesian_to_modal(x, v, self.r_ref, self.masses, self.W, dtype)
            self.q[~self.band_mask] = dtype.type(0.0)
            self.p[~self.band_mask] = dtype.type(0.0)
            print("FIMD: initialised from current atoms")

        x_new, v_new = _modal_to_cartesian(self.q, self.p, self.r_ref, self.masses, self.W, dtype)
        atoms.set_positions(x_new)
        set_velocities_ang_per_fs(atoms, v_new, dtype)

        self.S = _compute_source_term(self.q, self.W, self.masses, self.r_ref, self.K, self.reference_force, self.atoms, self.band_mask, dtype)
        self.nsteps = 0
        print(f"FIMD: ready with {basis.n_active_modes} active modes")
        print(f"      Band: {basis.band[0]:.3f} - {basis.band[1]:.3f} cm^-1")

    def step(self, forces=None):
        dt = self.dtype.type(self.dt / units.fs)
        self.p[self.band_mask] += self.dtype.type(0.5) * dt * self.S[self.band_mask]
        self.q, self.p = _harmonic_step(self.q, self.p, self.omega, dt, self.band_mask, self.dtype)

        x_new, _ = _modal_to_cartesian(self.q, self.p, self.r_ref, self.masses, self.W, self.dtype)
        self.atoms.set_positions(x_new)
        S_new = _compute_source_term(self.q, self.W, self.masses, self.r_ref, self.K, self.reference_force, self.atoms, self.band_mask, self.dtype)

        self.p[self.band_mask] += self.dtype.type(0.5) * dt * S_new[self.band_mask]
        self.S = S_new

        if self.temperature is not None:
            self.p = _langevin_thermostat(
                self.p,
                dt,
                gamma=self.friction,
                temperature=float(self.temperature),
                band_mask=self.band_mask,
                dtype=self.dtype,
            )

        _, v_new = _modal_to_cartesian(self.q, self.p, self.r_ref, self.masses, self.W, self.dtype)
        set_velocities_ang_per_fs(self.atoms, v_new, self.dtype)

    def get_modal_energies(self) -> np.ndarray:
        """Per-mode *harmonic* energy E_ν = 1/2 (π_ν² + ω_ν² q_ν²), in eV.

        This is the quadratic reference energy of each mode and is NOT, on its
        own, the conserved quantity: energy is continually exchanged between
        this harmonic part and the residual potential ΔV(r_B).  Use
        ``get_band_energy`` for the conserved band Hamiltonian.
        """
        conv = self.dtype.type(1.0 / EIG_TO_RADFS2)
        return (self.dtype.type(0.5) * self.p**2
                + self.dtype.type(0.5) * (self.omega * self.q) ** 2) * conv

    def get_residual_potential(self) -> float:
        """Residual potential ΔV(r_B) at the current band geometry, in eV.

        Consistent with the source-term Taylor split used by the integrator:

            ΔV(r) = V(r) - V(r0) + F(r0)·dx - 1/2 dxᵀ K dx,    dx = r - r0,

        whose negative gradient is the residual force F_Δ that drives the kick.
        Returns ``nan`` if the reference energy V(r0) was not recorded.
        """
        if self.reference_energy is None:
            return float("nan")
        sqrt_M_inv = np.repeat(self.dtype.type(1.0) / np.sqrt(self.masses), 3)
        dx = sqrt_M_inv * (self.W @ self.q)
        V_full = float(self.atoms.get_potential_energy())
        lin = float(self.reference_force @ dx)            # F(r0)·dx
        quad = 0.5 * float(dx @ (self.K @ dx))            # 1/2 dxᵀ K dx
        return V_full - float(self.reference_energy) + lin - quad

    def get_band_energy(self) -> float:
        """Conserved band Hamiltonian H_B (eV).

        H_B = Σ_{ν∈B} 1/2 (π_ν² + ω_ν² q_ν²) · conv  +  ΔV(r_B)

        In NVE with a fixed reference this is the quantity the symplectic
        kick–drift–kick step conserves (cf. Supplementary Fig. S2), and is the
        correct diagnostic for band-limited energy conservation — unlike the
        full Cartesian potential, which also moves with out-of-band content.
        """
        conv = self.dtype.type(1.0 / EIG_TO_RADFS2)
        mask = self.band_mask
        harm = float(np.sum(self.dtype.type(0.5) * self.p[mask] ** 2
                            + self.dtype.type(0.5) * (self.omega[mask] * self.q[mask]) ** 2) * conv)
        dV = self.get_residual_potential()
        return harm + dV

    def get_modal_energy_breakdown(self) -> Dict[str, Any]:
        """Per-mode energy decomposition for the active band.

        Returns a dict with, for the active modes only:
          - ``mode_index``    : indices into the full mode array
          - ``frequencies_cm1``: mode frequencies (cm^-1)
          - ``kinetic``       : 1/2 π_ν² · conv (eV)
          - ``potential``     : 1/2 ω_ν² q_ν² · conv (eV)  [harmonic part]
          - ``harmonic``      : kinetic + potential (eV)
          - ``residual_dV``   : the shared ΔV(r_B) (eV, not per-mode separable)
          - ``band_energy``   : H_B = Σ harmonic + residual_dV (eV)
        """
        conv = self.dtype.type(1.0 / EIG_TO_RADFS2)
        mask = self.band_mask
        idx = np.where(mask)[0]
        ke = (self.dtype.type(0.5) * self.p[mask] ** 2) * conv
        pe = (self.dtype.type(0.5) * (self.omega[mask] * self.q[mask]) ** 2) * conv
        dV = self.get_residual_potential()
        return {
            "mode_index": idx,
            "frequencies_cm1": np.asarray(radfs_to_cm1(self.omega[mask]), dtype=self.dtype),
            "kinetic": np.asarray(ke, dtype=self.dtype),
            "potential": np.asarray(pe, dtype=self.dtype),
            "harmonic": np.asarray(ke + pe, dtype=self.dtype),
            "residual_dV": float(dV),
            "band_energy": float(np.sum(ke + pe) + dV),
        }

    def get_modal_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.q.copy(), self.p.copy()


# =============================================================================
# Internal numerical helpers
# =============================================================================


def _window(n_frames: int, name: str, dtype: np.dtype) -> np.ndarray:
    key = str(name).strip().lower()
    if key in ("hann", "hanning"):
        return np.hanning(n_frames).astype(dtype)
    if key == "hamming":
        return np.hamming(n_frames).astype(dtype)
    if key == "blackman":
        return np.blackman(n_frames).astype(dtype)
    if key in ("none", "rect", "rectangular", "boxcar"):
        return np.ones(n_frames, dtype=dtype)
    raise ValueError("window must be 'hann', 'hamming', 'blackman', or 'none'.")


def _compute_reference_force(atoms: Atoms, precision: FloatDTypeLike = "float64") -> np.ndarray:
    """Compute F(r_ref) in eV/Å and return a flattened vector.

    The FIMD theory is usually written at a stationary reference, where this
    is zero.  Keeping it explicitly makes the Taylor split correct also for
    trajectory-centred references.
    """
    dtype = resolve_precision(precision)
    return np.asarray(atoms.get_forces(), dtype=dtype).reshape(-1).copy()


def _compute_numerical_hessian(atoms: Atoms, h: float = 0.001, precision: FloatDTypeLike = "float64") -> np.ndarray:
    dtype = resolve_precision(precision)
    n_atoms = len(atoms)
    dim = 3 * n_atoms
    H = np.zeros((dim, dim), dtype=dtype)
    pos0 = np.asarray(atoms.get_positions(), dtype=dtype).copy()
    flat0 = pos0.reshape(-1)

    for j in range(dim):
        plus = flat0.copy()
        plus[j] += dtype.type(h)
        atoms.set_positions(plus.reshape(n_atoms, 3))
        f_plus = np.asarray(atoms.get_forces(), dtype=dtype).reshape(-1)

        minus = flat0.copy()
        minus[j] -= dtype.type(h)
        atoms.set_positions(minus.reshape(n_atoms, 3))
        f_minus = np.asarray(atoms.get_forces(), dtype=dtype).reshape(-1)

        H[:, j] = -(f_plus - f_minus) / dtype.type(2.0 * h)

    atoms.set_positions(pos0)
    return (dtype.type(0.5) * (H + H.T)).astype(dtype, copy=False)


def _mass_weight_hessian(H: np.ndarray, masses: np.ndarray, precision: FloatDTypeLike = "float64") -> np.ndarray:
    dtype = resolve_precision(precision)
    sqrt_m_inv = np.repeat(dtype.type(1.0) / np.sqrt(np.asarray(masses, dtype=dtype)), 3)
    return (np.outer(sqrt_m_inv, sqrt_m_inv) * np.asarray(H, dtype=dtype)).astype(dtype, copy=False)


def _diagonalize_hessian(H_mw: np.ndarray, precision: FloatDTypeLike = "float64") -> Tuple[np.ndarray, np.ndarray]:
    dtype = resolve_precision(precision)
    eig, W = np.linalg.eigh(np.asarray(H_mw, dtype=dtype))
    idx = np.argsort(eig)
    return eig[idx].astype(dtype, copy=False), W[:, idx].astype(dtype, copy=False)


def _cartesian_to_modal(
    x: np.ndarray,
    v_ang_fs: np.ndarray,
    r_ref: np.ndarray,
    masses: np.ndarray,
    W: np.ndarray,
    precision: FloatDTypeLike = "float64",
) -> Tuple[np.ndarray, np.ndarray]:
    dtype = resolve_precision(precision)
    sqrt_M = np.repeat(np.sqrt(np.asarray(masses, dtype=dtype)), 3)
    dx = (np.asarray(x, dtype=dtype) - np.asarray(r_ref, dtype=dtype)).reshape(-1)
    vel = np.asarray(v_ang_fs, dtype=dtype).reshape(-1)
    q = W.T @ (sqrt_M * dx)
    p = W.T @ (sqrt_M * vel)
    return q.astype(dtype, copy=False), p.astype(dtype, copy=False)


def _modal_to_cartesian(
    q: np.ndarray,
    p: np.ndarray,
    r_ref: np.ndarray,
    masses: np.ndarray,
    W: np.ndarray,
    precision: FloatDTypeLike = "float64",
) -> Tuple[np.ndarray, np.ndarray]:
    dtype = resolve_precision(precision)
    n_atoms = len(masses)
    sqrt_M_inv = np.repeat(dtype.type(1.0) / np.sqrt(np.asarray(masses, dtype=dtype)), 3)
    dx = sqrt_M_inv * (W @ np.asarray(q, dtype=dtype))
    vel = sqrt_M_inv * (W @ np.asarray(p, dtype=dtype))
    x = np.asarray(r_ref, dtype=dtype) + dx.reshape(n_atoms, 3)
    v = vel.reshape(n_atoms, 3)
    return x.astype(dtype, copy=False), v.astype(dtype, copy=False)


def _covariance_eigenbasis(
    x_mw: np.ndarray,
    window: np.ndarray,
    precision: FloatDTypeLike = "float64",
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate the mode matrix ``W`` from a windowed mass-weighted covariance.

    Implements the trajectory-based reference of the paper: form the windowed
    covariance estimate Ĉ ≈ ⟨x xᵀ⟩ of the mass-weighted displacement
    ``x = M^{1/2}(r - r0)`` and take ``W`` as its orthonormal eigenvectors,
    ordered by descending eigenvalue (dominant collective motion first).

    Returns ``(cov_eigenvalues_desc, W)`` where ``W`` columns are the modes.
    The covariance eigenvalues are returned only for diagnostics/ordering;
    frequencies are *not* taken from them (they carry a known bias) but from
    the projected-velocity spectra instead.
    """
    dtype = resolve_precision(precision)
    xw = (np.asarray(x_mw, dtype=dtype) * np.asarray(window, dtype=dtype)[:, None])
    n_frames = xw.shape[0]
    # Centre the windowed mass-weighted displacement before forming covariance.
    xw = xw - xw.mean(axis=0, keepdims=True)
    cov = (xw.T @ xw) / dtype.type(max(n_frames - 1, 1))
    cov = dtype.type(0.5) * (cov + cov.T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    return evals[order].astype(dtype, copy=False), evecs[:, order].astype(dtype, copy=False)


def _frequencies_from_velocity_spectra(
    v_mw: np.ndarray,
    W: np.ndarray,
    dt_fs: float,
    window: np.ndarray,
    precision: FloatDTypeLike = "float64",
    min_cm1: float = 1.0,
) -> np.ndarray:
    """Assign each mode an angular frequency (rad/fs) from its velocity spectrum.

    The mass-weighted velocity is projected onto each column of ``W`` to give
    the modal velocity time series ``v_ν(t) = w_νᵀ M^{1/2} ṙ``.  The frequency of
    mode ν is the location of the peak of the power spectrum ``|FFT(v_ν)|²``.
    A Hann (or supplied) window is applied before the transform; the DC bin is
    excluded so that drift does not masquerade as a zero-frequency peak.
    """
    dtype = resolve_precision(precision)
    vw = (np.asarray(v_mw, dtype=dtype) * np.asarray(window, dtype=dtype)[:, None])
    # Modal velocity time series: (n_frames, n_modes).
    v_modal = vw @ np.asarray(W, dtype=dtype)
    n_frames = v_modal.shape[0]
    spec = np.abs(np.fft.rfft(v_modal, axis=0)) ** 2
    freq_hz = np.fft.rfftfreq(n_frames, d=float(dt_fs) * 1.0e-15)
    freq_cm1 = freq_hz / C_CM_PER_S
    # Ignore the DC bin (and anything below min_cm1) when locating the peak.
    valid = freq_cm1 >= float(min_cm1)
    omega = np.zeros(v_modal.shape[1], dtype=dtype)
    if not np.any(valid):
        return omega
    idx_valid = np.where(valid)[0]
    spec_valid = spec[idx_valid, :]
    peak_local = np.argmax(spec_valid, axis=0)
    peak_rows = idx_valid[peak_local]
    # Parabolic (quadratic) sub-bin interpolation around each peak to avoid
    # quantising frequencies onto the coarse FFT grid. Fit log-magnitude of the
    # peak and its two neighbours; the vertex offset refines the bin location.
    df_cm1 = float(freq_cm1[1] - freq_cm1[0]) if len(freq_cm1) > 1 else 0.0
    peak_cm1 = freq_cm1[peak_rows].astype(dtype).copy()
    n_bins = spec.shape[0]
    for m in range(v_modal.shape[1]):
        k = int(peak_rows[m])
        if 0 < k < n_bins - 1 and df_cm1 > 0.0:
            y0, y1, y2 = spec[k - 1, m], spec[k, m], spec[k + 1, m]
            with np.errstate(divide="ignore", invalid="ignore"):
                l0 = np.log(y0 + 1e-30); l1 = np.log(y1 + 1e-30); l2 = np.log(y2 + 1e-30)
                denom = (l0 - 2.0 * l1 + l2)
                if abs(denom) > 1e-20:
                    delta = 0.5 * (l0 - l2) / denom  # in [-1, 1] bins
                    delta = float(np.clip(delta, -0.5, 0.5))
                    peak_cm1[m] = freq_cm1[k] + delta * df_cm1
    omega = np.asarray(cm1_to_radfs(peak_cm1), dtype=dtype)
    return omega.astype(dtype, copy=False)


def _effective_hessian_from_modes(
    W: np.ndarray,
    omega: np.ndarray,
    masses: np.ndarray,
    precision: FloatDTypeLike = "float64",
) -> np.ndarray:
    """Build the effective Cartesian Hessian K implied by (W, Ω).

    The paper defines the effective harmonic operator via
    ``M^{-1/2} H0 M^{-1/2} = W Ω² Wᵀ`` (Eq. 2).  Inverting the mass weighting,
    ``K = M^{1/2} W diag(Ω²) Wᵀ M^{1/2}`` in mass-weighted-frequency units; the
    eigenvalue stored internally is ``Ω²/EIG_TO_RADFS2`` so that the runtime
    source-term Taylor baseline ``F_model = F_ref - K dx`` is consistent with
    the same Ω used in the harmonic drift.  This is what keeps the kick–drift–
    kick split exact in the harmonic limit when the basis is trajectory-derived.
    """
    dtype = resolve_precision(precision)
    sqrt_M = np.repeat(np.sqrt(np.asarray(masses, dtype=dtype)), 3)
    eig = (np.asarray(omega, dtype=dtype) ** 2) / dtype.type(EIG_TO_RADFS2)
    H_mw = np.asarray(W, dtype=dtype) @ np.diag(eig).astype(dtype) @ np.asarray(W, dtype=dtype).T
    K = (sqrt_M[:, None] * H_mw) * sqrt_M[None, :]
    return (dtype.type(0.5) * (K + K.T)).astype(dtype, copy=False)


def _compute_source_term(
    q: np.ndarray,
    W: np.ndarray,
    masses: np.ndarray,
    r_ref: np.ndarray,
    K: np.ndarray,
    reference_force: np.ndarray,
    atoms: Atoms,
    band_mask: np.ndarray,
    precision: FloatDTypeLike = "float64",
) -> np.ndarray:
    dtype = resolve_precision(precision)
    n_atoms = len(masses)
    sqrt_M_inv = np.repeat(dtype.type(1.0) / np.sqrt(np.asarray(masses, dtype=dtype)), 3)
    dx = sqrt_M_inv * (W @ np.asarray(q, dtype=dtype))
    x_cart = np.asarray(r_ref, dtype=dtype) + dx.reshape(n_atoms, 3)
    atoms.set_positions(x_cart)

    F_total = np.asarray(atoms.get_forces(), dtype=dtype).reshape(-1)
    # Linear+quadratic Taylor reference force about r_ref:
    # F_ref_model(r) = F(r_ref) - K @ (r-r_ref).
    # This reduces to -K dx at a true minimum, but remains correct for
    # trajectory-centred finite-temperature references where F(r_ref) may be nonzero.
    F_ref = np.asarray(reference_force, dtype=dtype).reshape(-1)
    F_model = F_ref - (np.asarray(K, dtype=dtype) @ dx)
    F_residual = F_total - F_model
    S = W.T @ (sqrt_M_inv * F_residual)
    S = S * dtype.type(FORCE_CONV)
    S = np.where(band_mask, S, dtype.type(0.0))
    return S.astype(dtype, copy=False)


def _harmonic_step(
    q: np.ndarray,
    p: np.ndarray,
    omega: np.ndarray,
    dt: float,
    band_mask: np.ndarray,
    precision: FloatDTypeLike = "float64",
) -> Tuple[np.ndarray, np.ndarray]:
    dtype = resolve_precision(precision)
    q = np.asarray(q, dtype=dtype)
    p = np.asarray(p, dtype=dtype)
    omega = np.asarray(omega, dtype=dtype)
    eps = dtype.type(1.0e-12)
    mask_zero = np.abs(omega) < eps
    omega_safe = np.where(mask_zero, dtype.type(1.0), omega)
    wt = omega * dtype.type(dt)
    c = np.cos(wt).astype(dtype)
    s = np.sin(wt).astype(dtype)
    q_new = c * q + (s / omega_safe) * p
    p_new = -omega * s * q + c * p
    if np.any(mask_zero):
        q_new[mask_zero] = q[mask_zero] + p[mask_zero] * dtype.type(dt)
        p_new[mask_zero] = p[mask_zero]
    q_new = np.where(band_mask, q_new, q)
    p_new = np.where(band_mask, p_new, p)
    return q_new.astype(dtype, copy=False), p_new.astype(dtype, copy=False)


def _langevin_thermostat(
    p: np.ndarray,
    dt: float,
    gamma: float,
    temperature: float,
    band_mask: np.ndarray,
    dtype: np.dtype,
) -> np.ndarray:
    kT_eV = dtype.type(units.kB * temperature)
    kT_modal = kT_eV * dtype.type(EV_TO_AMU_A2_FS2)
    friction = dtype.type(np.exp(-gamma * float(dt)))
    noise_std = np.sqrt(kT_modal * (dtype.type(1.0) - friction**2)).astype(dtype)
    xi = np.random.standard_normal(len(p)).astype(dtype)
    p_new = friction * np.asarray(p, dtype=dtype) + noise_std * xi
    return np.where(band_mask, p_new, p).astype(dtype, copy=False)


def _create_band_mask(
    omega: np.ndarray,
    band_min_cm1: float,
    band_max_cm1: float,
    n_rigid_modes: int = 6,
    eig: Optional[np.ndarray] = None,
    eig_tol: float = 1.0e-6,
) -> np.ndarray:
    """Select the active vibrational modes within a wavenumber band.

    Rigid-body and imaginary modes must be excluded by their *eigenvalue*, not
    by array index.  ``omega`` is built with ``sqrt(maximum(eig, 0))``, so every
    non-positive-curvature direction collapses to ``omega == 0`` and would
    otherwise be propagated as a free drift that still receives the full
    residual-force kick -- an unbounded, self-amplifying mode.  When ``eig`` is
    supplied we therefore (a) drop any mode whose mass-weighted-Hessian
    eigenvalue is not safely positive and (b) drop the ``n_rigid_modes`` modes
    of smallest ``|eig|`` (the true translations/rotations near zero).
    """
    freq_cm1 = np.asarray(radfs_to_cm1(omega))
    mask = (freq_cm1 >= band_min_cm1) & (freq_cm1 <= band_max_cm1)
    if eig is not None:
        eig = np.asarray(eig)
        # (a) exclude imaginary / non-positive-curvature modes outright.
        mask &= eig > eig_tol
        # (b) exclude the genuine rigid modes: smallest |eigenvalue|.
        if n_rigid_modes > 0:
            rigid_idx = np.argsort(np.abs(eig))[: min(n_rigid_modes, len(eig))]
            mask[rigid_idx] = False
    else:
        # Backwards-compatible fallback: assume an index-sorted spectrum with
        # the rigid modes first.  Only correct at a clean minimum.
        mask[: min(n_rigid_modes, len(mask))] = False
    return mask.astype(bool)


def run_fimd_from_trajectory(
    trajectory_file: str,
    reference_file: str,
    calculator: Any,
    band: Tuple[float, float],
    nsteps: int = 1000,
    timestep: float = 1.0,
    temperature: Optional[float] = None,
    output_trajectory: Optional[str] = None,
    traj_dt: float = 1.0,
    precision: FloatDTypeLike = "float64",
    velocity_unit: str = "angstrom/ps",
) -> FIMDynamics:
    """Backward-compatible convenience wrapper for file-based workflows."""
    trajectory = load_trajectory(
        trajectory_file,
        dt_fs=traj_dt,
        precision=precision,
        velocity_unit=velocity_unit,
    )
    reference = read(reference_file)
    basis = FIMDBasis.from_trajectory(
        trajectory=trajectory,
        reference=reference,
        calculator=calculator,
        band=band,
        timestep_reference=traj_dt,
        precision=precision,
    )
    atoms = trajectory[-1].copy()
    atoms.calc = calculator
    dyn = FIMDynamics(
        atoms,
        basis=basis,
        timestep_fs=timestep,
        temperature=temperature,
        trajectory=output_trajectory,
        precision=precision,
    )
    dyn.run(nsteps)
    return dyn