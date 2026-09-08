#!/usr/bin/env python3
"""
compute_force_spring_constants.py

Build a FIMDBasis for liquid water from the Hungarian-permuted coordinate and
force trajectories, using force-displacement correlations to assign ω per mode.

WHY NOT VELOCITY FFT?
The permuted trajectory is not smooth: each frame's molecule labelling may
differ from the previous by O(10) swaps, giving velocity discontinuities that
contaminate the velocity power spectrum.  Forces, by contrast, are purely
per-frame quantities — permuting them applies the same bijection, giving
physically meaningful modal forces with no continuity requirement.

ALGORITHM
---------
1. r_ref  = ⟨r_permuted⟩          trajectory-centred reference geometry
2. W      = eigenvectors of windowed mass-weighted covariance of δr  (PCA)
3. ω_k²   = -⟨τ_k q_k⟩ / ⟨q_k²⟩   force-displacement estimator

       q_k(t) = W[:,k] · M^{+1/2} δr(t)    modal displacement  [sqrt(amu)·Å]
       τ_k(t) = W[:,k] · M^{-1/2} F(t)     modal force         [eV/(Å·sqrt(amu))]

4. K      = M^{1/2} W diag(ω²/EIG) Wᵀ M^{1/2}   effective Cartesian Hessian
5. F_ref  = ⟨F_permuted⟩           mean force ≈ F(r_ref) for small fluctuations
6. Save   FIMDBasis-compatible npz loadable with FIMDBasis.load()

UNITS NOTE
----------
AMBER force trajectories are in kcal/mol/Å.
Internal FIMD units are eV for energy, Å for length, amu for mass, fs for time.
Conversion:  1 kcal/mol = 1/23.0609 eV  (KCAL_TO_EV below)
ω_k² estimator output is in rad²/fs² after multiplying by EV_TO_AMU_A2_FS2.

NEGATIVE ω²
-----------
Modes where ⟨τ_k q_k⟩ > 0 (force and displacement in phase) are non-restoring
and physically meaningless for a vibration basis.  They arise from:
  - Insufficient sampling of low-frequency collective modes
  - Modes at the edge of the PCA spectrum with poor signal-to-noise
Negative ω² modes are zeroed and excluded from the band mask.

Usage:
    python compute_force_spring_constants.py \\
        --coords w240_permute.nc \\
        --forces w240_permute_frc.nc \\
        --band 100 2000 \\
        --out  w240_fimd_basis.npz

Output: w240_fimd_basis.npz — load with FIMDBasis.load()
"""

import argparse
import sys
import numpy as np
from scipy.io import netcdf_file

# ── Physical constants (matching fimd/core.py exactly) ────────────────────────

KCAL_TO_EV         = 1.0 / 23.0609          # AMBER force → eV/Å
EV_TO_AMU_A2_FS2   = 9.648533212331001e-3   # 1 eV = X amu·Å²/fs²
CM1_TO_RADFS       = 2.0 * np.pi * 2.99792458e10 * 1.0e-15
RADFS_TO_CM1       = 1.0 / CM1_TO_RADFS

# ── TIP4P/2005 system layout ───────────────────────────────────────────────────

ATOMS_PER_MOL   = 4
MASSES_PER_MOL  = [15.999, 1.008, 1.008, 1.0e-3]   # O, H, H, EP(virtual)
SYMBOLS_PER_MOL = ['O', 'H', 'H', 'X']


# ── PBC helper ─────────────────────────────────────────────────────────────────

def mic(dr: np.ndarray, box: np.ndarray) -> np.ndarray:
    return dr - box * np.round(dr / box)


# ── NetCDF loaders ─────────────────────────────────────────────────────────────

def load_nc_coords(path: str, stride: int = 1) -> tuple:
    """Load all coordinate frames from an AMBER NetCDF trajectory.

    Returns (coords [n_out, n_atoms, 3], box [3], times [n_out]).
    Wraps coordinates to [0, box) on load.
    """
    with netcdf_file(path, 'r', mmap=False) as nc:
        raw    = nc.variables['coordinates'][::stride].astype(np.float64)
        box    = nc.variables['cell_lengths'][0].astype(np.float64)
        times  = nc.variables['time'][::stride].astype(np.float64)
    print(f"  {path}: {raw.shape[0]} frames, {raw.shape[1]} atoms")
    return raw, box, times


def load_nc_forces(path: str, stride: int = 1) -> np.ndarray:
    """Load all force frames from an AMBER NetCDF force trajectory (kcal/mol/Å)."""
    with netcdf_file(path, 'r', mmap=False) as nc:
        raw = nc.variables['forces'][::stride].astype(np.float64)
    print(f"  {path}: {raw.shape[0]} frames")
    return raw


# ── Mass vector helpers ────────────────────────────────────────────────────────

def build_mass_vectors(n_mol: int) -> tuple:
    """Return (masses [n_atoms], sqrt_M [3N], inv_sqrt_M [3N])."""
    masses  = np.array(MASSES_PER_MOL * n_mol, dtype=np.float64)  # (n_atoms,)
    sqrt_M      = np.repeat(np.sqrt(masses), 3)                    # (3N,)
    inv_sqrt_M  = np.repeat(1.0 / np.sqrt(masses), 3)             # (3N,)
    return masses, sqrt_M, inv_sqrt_M


# ── Step 1: Reference geometry ────────────────────────────────────────────────

def compute_reference(coords: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    r_ref = mean position, PBC-corrected relative to first frame.

    Consistent with FIMDBasis.from_trajectory_covariance which sets
    r0 = traj_mean after Kabsch alignment.
    """
    dr   = mic(coords - coords[0:1], box)       # (n_frames, n_atoms, 3)
    r_ref = coords[0] + dr.mean(axis=0)          # (n_atoms, 3)
    return r_ref


# ── Step 2: Covariance PCA → W ────────────────────────────────────────────────

def covariance_pca(coords: np.ndarray, r_ref: np.ndarray,
                   sqrt_M: np.ndarray, box: np.ndarray,
                   window: str = 'hann') -> tuple:
    """
    Mass-weighted covariance eigenbasis.

    Applies a window function (Hann by default) to reduce edge effects,
    matching FIMDBasis.from_trajectory_covariance step [1/3].

    Returns (cov_evals [n_modes] descending, W [n_modes, n_modes]).
    W[:, k] is mode k as a unit vector in mass-weighted Cartesian space.
    """
    n_frames, n_atoms, _ = coords.shape
    n_modes = 3 * n_atoms

    # Mass-weighted displacements
    dr   = mic(coords - r_ref[None], box)          # (n_frames, n_atoms, 3)
    x_mw = sqrt_M * dr.reshape(n_frames, n_modes)  # (n_frames, n_modes)

    # Window
    if window.lower() in ('hann', 'hanning'):
        win = np.hanning(n_frames)
    elif window.lower() == 'hamming':
        win = np.hamming(n_frames)
    elif window.lower() in ('none', 'rect'):
        win = np.ones(n_frames)
    else:
        raise ValueError(f"Unknown window '{window}'; use hann/hamming/none.")

    xw = x_mw * win[:, None]
    xw -= xw.mean(axis=0)                          # centre windowed data

    print(f"  Building covariance ({n_modes}×{n_modes}) ...")
    C = xw.T @ xw / max(n_frames - 1, 1)
    C = 0.5 * (C + C.T)

    print(f"  Diagonalising ...")
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]                # descending variance
    return evals[order], evecs[:, order], x_mw     # also return raw x_mw


# ── Step 3: Force-displacement ω estimator ────────────────────────────────────

def force_displacement_omega(
    x_mw:       np.ndarray,   # (n_frames, n_modes) unwindowed, M^{1/2} δr
    forces_kcal: np.ndarray,  # (n_frames, n_atoms, 3) AMBER forces kcal/mol/Å
    inv_sqrt_M:  np.ndarray,  # (n_modes,)
    W:           np.ndarray,  # (n_modes, n_modes) mode matrix
) -> tuple:
    """
    Compute per-mode angular frequencies via force-displacement correlation.

        ω_k² = -⟨τ_k q_k⟩ / ⟨q_k²⟩  ×  EV_TO_AMU_A2_FS2

    The force-displacement estimator measures the actual harmonic curvature
    of the potential at finite temperature, unlike:
      • covariance-based ω  → PMF curvature (too low for anharmonic V)
      • velocity FFT ω      → contaminated by permutation discontinuities

    Returns (omega [n_modes] rad/fs, omega_sq_raw [n_modes] eV/(amu·Å²)).
    """
    n_frames, n_modes = x_mw.shape

    # Convert forces: kcal/mol/Å → eV/Å, flatten, mass-weight
    f_ev  = forces_kcal.reshape(n_frames, n_modes) * KCAL_TO_EV  # eV/Å
    f_mw  = inv_sqrt_M * f_ev                                     # eV/(Å·√amu)

    print(f"  Projecting {n_frames} frames onto {n_modes} modes ...")

    # Modal coordinates and forces: (n_frames, n_modes)
    # q[:,k] = W[:,k] · x_mw[t]  — broadcast via matrix multiply
    q_modal = x_mw  @ W     # (n_frames, n_modes)
    τ_modal = f_mw  @ W     # (n_frames, n_modes)

    mean_tq = (τ_modal * q_modal).mean(axis=0)   # ⟨τ_k q_k⟩
    mean_q2 = (q_modal ** 2).mean(axis=0)        # ⟨q_k²⟩

    # ω_k² in eV/(amu·Å²); convert to rad²/fs²
    # Use a minimum variance threshold: modes with ⟨q²⟩ below this have
    # too little displacement signal for a reliable force-displacement ratio.
    # Threshold: 1e-6 amu·Å² ≈ RMSD of 1e-3 Å for a 1-amu mode — well below
    # any physical vibration but above pure numerical noise.
    q2_threshold = 1.0e-6
    sampled      = mean_q2 > q2_threshold
    safe_q2      = np.where(sampled, mean_q2, 1.0)
    omega_sq_raw = np.where(sampled, -mean_tq / safe_q2, 0.0)
    omega_sq     = omega_sq_raw * EV_TO_AMU_A2_FS2    # rad²/fs²

    # Hard cap at 5000 cm⁻¹ (well above OH stretch ~3700 cm⁻¹) to suppress
    # noise amplification in near-zero-variance modes.
    omega_max    = 5000.0 * CM1_TO_RADFS
    omega_sq     = np.minimum(omega_sq, omega_max ** 2)

    omega = np.sqrt(np.maximum(omega_sq, 0.0))

    return omega, omega_sq_raw, mean_q2


# ── Step 4: Effective Hessian K from (W, ω) ──────────────────────────────────

def effective_hessian(W: np.ndarray, omega: np.ndarray,
                      sqrt_M: np.ndarray) -> np.ndarray:
    """
    K = M^{1/2} W diag(ω²/EIG_TO_RADFS2) Wᵀ M^{1/2}   in eV/Å²

    Replicates fimd.core._effective_hessian_from_modes so that the FIMD
    runtime source-term kick (F_residual = F_actual - F_ref + K @ dx) uses
    the same ω as the harmonic drift — keeping the kick-drift-kick split
    exact in the harmonic limit.
    """
    eig = omega**2 / EV_TO_AMU_A2_FS2      # eigenvalues in eV/(amu·Å²)
    H_mw = W @ (eig[:, None] * W.T)        # = W diag(eig) Wᵀ,  (n_modes, n_modes)
    K = (sqrt_M[:, None] * H_mw) * sqrt_M[None, :]
    return 0.5 * (K + K.T)


# ── Step 5: Reference force F(r_ref) ─────────────────────────────────────────

def mean_force(forces_kcal: np.ndarray) -> np.ndarray:
    """
    Mean permuted force ≈ F(r_ref) in eV/Å (flattened).

    For a trajectory centred at r_ref = ⟨r⟩, the mean force ≈ F(r_ref) to
    first order in the fluctuation amplitude — the same approximation used
    by FIMDBasis.from_trajectory_covariance.
    """
    return forces_kcal.reshape(forces_kcal.shape[0], -1).mean(axis=0) * KCAL_TO_EV


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--coords',  default='w240_permute.nc',
                    help='Permuted coordinate trajectory (AMBER NetCDF)')
    ap.add_argument('--forces',  default='w240_permute_frc.nc',
                    help='Permuted force trajectory (AMBER NetCDF, kcal/mol/Å)')
    ap.add_argument('--band',    nargs=2, type=float, default=[100.0, 2000.0],
                    metavar=('LO', 'HI'),
                    help='Frequency band in cm⁻¹ (default: 100 2000)')
    ap.add_argument('--out',     default='w240_fimd_basis.npz',
                    help='Output basis file (FIMDBasis.load()-compatible)')
    ap.add_argument('--stride',  type=int, default=1,
                    help='Use every Nth frame (default 1 = all)')
    ap.add_argument('--window',  default='hann',
                    choices=['hann', 'hamming', 'none'],
                    help='Window function for covariance PCA (default: hann)')
    args = ap.parse_args()

    band_lo, band_hi = args.band
    print(f"\n{'='*64}")
    print(f"  Force-displacement spring constants for FIMD")
    print(f"  Band: {band_lo:.0f}–{band_hi:.0f} cm⁻¹")
    print(f"{'='*64}\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading trajectories ...")
    coords, box, times = load_nc_coords(args.coords, stride=args.stride)
    forces             = load_nc_forces(args.forces, stride=args.stride)

    n_frames, n_atoms, _ = coords.shape
    n_mol   = n_atoms // ATOMS_PER_MOL
    n_modes = 3 * n_atoms

    # Trim to same length
    n_frames = min(n_frames, forces.shape[0])
    coords   = coords[:n_frames]
    forces   = forces[:n_frames]
    times    = times[:n_frames]

    print(f"\n  Frames   : {n_frames}  (stride={args.stride})")
    print(f"  Atoms    : {n_atoms}  ({n_mol} molecules × {ATOMS_PER_MOL} atoms)")
    print(f"  Modes    : {n_modes}")
    print(f"  Box      : {box[0]:.3f} × {box[1]:.3f} × {box[2]:.3f} Å")

    # Zero EP forces explicitly (AMBER redistributes them, but numerical noise
    # on a massless site would be amplified by inv_sqrt_M[EP] ≈ 31.6)
    forces[:, 3::ATOMS_PER_MOL, :] = 0.0

    # Build mass vectors
    masses, sqrt_M, inv_sqrt_M = build_mass_vectors(n_mol)
    symbols = SYMBOLS_PER_MOL * n_mol

    # ── Step 1: Reference geometry ────────────────────────────────────────────
    print("\n[1/5] Computing reference geometry r_ref = ⟨r⟩ ...")
    r_ref = compute_reference(coords, box)
    mean_dr = mic(coords - r_ref[None], box)
    rms_dr  = np.sqrt((mean_dr**2).mean())
    print(f"  RMS displacement from r_ref : {rms_dr:.4f} Å")

    # ── Step 2: Covariance PCA ────────────────────────────────────────────────
    print(f"\n[2/5] Covariance PCA (window={args.window}) ...")
    cov_evals, W, x_mw = covariance_pca(
        coords, r_ref, sqrt_M, box, window=args.window
    )
    print(f"  Top-10 covariance eigenvalues (amu·Å², descending variance):")
    print(f"    {cov_evals[:10].round(4)}")

    # ── Step 3: Force-displacement ω ──────────────────────────────────────────
    print(f"\n[3/5] Force-displacement ω estimator ...")
    omega, omega_sq_raw, mean_q2 = force_displacement_omega(
        x_mw, forces, inv_sqrt_M, W
    )
    freq_cm1 = omega * RADFS_TO_CM1

    n_physical  = int(np.sum(omega_sq_raw > 0))
    n_negative  = int(np.sum(omega_sq_raw < 0))
    n_zero      = int(np.sum(omega_sq_raw == 0))
    print(f"\n  Mode frequency statistics:")
    print(f"    Physical (ω² > 0) : {n_physical}")
    print(f"    Non-restoring     : {n_negative}  (zeroed)")
    print(f"    Unsampled/zero    : {n_zero}")
    if n_physical > 0:
        pos_mask = omega > 0
        print(f"    Frequency range   : {freq_cm1[pos_mask].min():.1f} – "
              f"{freq_cm1[pos_mask].max():.1f} cm⁻¹")

    # Histogram of physical modes across bands of interest
    edges   = [0, 50, 100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
    counts, _ = np.histogram(freq_cm1[pos_mask], bins=edges)
    print(f"\n  Frequency histogram (physical modes):")
    for lo, hi, c in zip(edges[:-1], edges[1:], counts):
        bar = '█' * min(c // 10, 40)
        print(f"    {lo:5d}–{hi:5d} cm⁻¹ : {c:5d}  {bar}")

    # ── Step 4: Band selection ────────────────────────────────────────────────
    print(f"\n[4/5] Selecting band {band_lo:.0f}–{band_hi:.0f} cm⁻¹ ...")
    eps_radfs = 1.0 * CM1_TO_RADFS           # exclude anything below 1 cm⁻¹
    band_mask = (
        (freq_cm1 >= band_lo) &
        (freq_cm1 <= band_hi) &
        (omega    > eps_radfs)
    )
    n_active = int(band_mask.sum())
    print(f"  Active modes : {n_active}")
    if n_active > 0:
        af = freq_cm1[band_mask]
        print(f"  Active range : {af.min():.1f} – {af.max():.1f} cm⁻¹")
        print(f"  Median ω     : {np.median(af):.1f} cm⁻¹")

    if n_active == 0:
        print("  WARNING: No modes in band — check trajectory quality or widen band.")

    # ── Step 5: Build and save FIMDBasis ──────────────────────────────────────
    print(f"\n[5/5] Building K, F_ref and saving {args.out} ...")

    K     = effective_hessian(W, omega, sqrt_M)
    F_ref = mean_force(forces)                    # (n_modes,) eV/Å

    # Initial conditions: zero (will be set at FIMD startup from current atoms)
    q0 = np.zeros(n_modes, dtype=np.float64)
    p0 = np.zeros(n_modes, dtype=np.float64)

    # FFT diagnostics: empty (not computed via velocity route)
    fft_frequencies = np.array([], dtype=np.float64)
    fft_amplitudes  = np.array([], dtype=np.complex128)

    dt_fs = float(times[1] - times[0]) if n_frames > 1 else 2.0   # fs

    metadata = {
        'basis_source'         : 'force_displacement',
        'window'               : args.window,
        'n_frames'             : n_frames,
        'stride'               : args.stride,
        'remove_rigid_motion'  : False,   # PBC bulk: no rigid-body removal
        'n_negative_omega_sq'  : n_negative,
        'coords_file'          : args.coords,
        'forces_file'          : args.forces,
        'force_unit'           : 'kcal/mol/Å',
        'kcal_to_ev'           : KCAL_TO_EV,
        'reference_force_correction': True,
    }

    np.savez_compressed(
        args.out,
        reference_positions = r_ref,
        masses              = masses,
        symbols             = np.array(symbols, dtype=object),
        W                   = W,
        omega               = omega,
        K                   = K,
        reference_force     = F_ref,
        reference_energy    = np.array([], dtype=float),   # not available
        band                = np.array([band_lo, band_hi]),
        band_mask           = band_mask,
        q0                  = q0,
        p0                  = p0,
        fft_frequencies     = fft_frequencies,
        fft_amplitudes      = fft_amplitudes,
        timestep_reference  = np.array(dt_fs),
        precision           = np.array('float64', dtype=object),
        n_rigid_modes       = np.array(0, dtype=int),
        metadata            = np.array(metadata, dtype=object),
    )

    print(f"\n{'='*64}")
    print(f"  Saved : {args.out}")
    print(f"  Active modes : {n_active} / {n_modes}")
    if n_active > 0:
        print(f"  Band         : {freq_cm1[band_mask].min():.1f} – "
              f"{freq_cm1[band_mask].max():.1f} cm⁻¹")
    print(f"  Load with    : FIMDBasis.load('{args.out}')")
    print(f"{'='*64}")

    # Extra diagnostic: show ω² < 0 rate as quality indicator
    neg_rate = 100.0 * n_negative / n_modes
    if neg_rate > 20:
        print(f"\n  WARNING: {neg_rate:.0f}% of modes have ω² < 0.")
        print(f"  Likely cause: insufficient frames for low-frequency modes.")
        print(f"  Suggested fix: increase trajectory length or use --stride 1.")
    elif neg_rate > 5:
        print(f"\n  NOTE: {neg_rate:.0f}% negative ω² modes (excluded from band).")
        print(f"  These are typically near-zero soft modes with poor SNR.")
    print()


if __name__ == '__main__':
    main()
