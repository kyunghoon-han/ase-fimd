#!/usr/bin/env python3
"""
run_fimd_so3lr.py – FIMD using a pre-computed SO3LR Hessian basis.

Mirrors run_fimd_hessian.py (AMBER/sander) but uses the SO3LR ASE calculator
for force evaluations, and reads the SO3LR Hessian npz format.

The SO3LR Hessian npz (hessian_basis_so3lr.npz) contains:
    W          – (3N, 3N) float32  mass-weighted eigenvectors
    eigvals    – (3N,)    float64  kcal/(mol·amu·Å²), sorted by |ω|
    ref_pos    – (3N,)    float64  Å, flattened
    masses     – (N,)     float64  amu
    box        – (3,)     float64  Å
    n_mol      – int32
    lr_cutoff  – float64  Å  (SO3LR long-range cutoff used for the Hessian)

Usage:
    python3 run_fimd_so3lr.py
    python3 run_fimd_so3lr.py --band-lo 50 --band-hi 900 --out-dir fimd_so3lr_50_900
    python3 run_fimd_so3lr.py --fimd-steps 400000   # 200 ps production
"""

import gc
import os
import sys
import argparse

# Cap CPU threads — SO3LR uses the GPU; BLAS threads only waste CPU.
for _var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
             'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import numpy as np
from ase import Atoms, units

from fimd.core import (
    FIMDBasis,
    FIMDynamics,
    _effective_hessian_from_modes,
    CM1_TO_RADFS,
    EIG_TO_RADFS2,
    set_velocities_ang_per_fs,
    complex_dtype_for,
)

# ── Physical constants (match compute_hessian_so3lr.py) ───────────────────────
_KCAL_TO_RAD2_FS2 = 4.184e-4               # kcal/(mol·amu·Å²) → (rad/fs)²
_C_INV_FS         = 2.99792458e10 * 1e-15  # cm⁻¹·fs

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_NPZ     = 'hessian_basis_so3lr.npz'
DEFAULT_OUT_DIR = 'fimd_so3lr'
TEMP            = 300.0    # K
BAND_LO         = 0.0      # cm⁻¹
BAND_HI         = 4000.0   # cm⁻¹
FIMD_DT_FS      = 0.5      # fs
FIMD_STEPS      = 20_000   # steps (10 ps)
EQUIL_STEPS     = 1_000
SAVE_INTERVAL   = 10


# ── Helpers ───────────────────────────────────────────────────────────────────

def _symbols_from_masses(masses):
    """Infer element symbols from atomic masses (O > 10 amu, else H)."""
    return ['O' if m > 10.0 else 'H' for m in masses]


def _make_so3lr_calc(lr_cutoff, dtype='float32'):
    # The so3lr ASE calculator defaults to float32, which heats NVE water by ~1.5 eV/ps at 0.25 fs even
    # under plain velocity Verlet; float64 conserves (drift_controls, 2026-10-13).
    from so3lr.ase_utils import make_ase_calculator
    if dtype == 'float64':
        import jax
        jax.config.update('jax_enable_x64', True)
    return make_ase_calculator(lr_cutoff=float(lr_cutoff), dtype=np.dtype(dtype).type)


# ── Basis construction ────────────────────────────────────────────────────────

def build_hessian_fimd_basis(npz_path, calc, temp_K, band_lo, band_hi,
                              allow_imaginary, precision, verbose):
    """
    Load hessian_basis_so3lr.npz and build a FIMDBasis for FIMDynamics.

    Closely mirrors the AMBER version in run_fimd_hessian.py; differences:
      - Reads SO3LR npz format (no prmtop key).
      - Derives atom symbols from masses (TIP3P: O H H, no EP virtual site).
      - Uses SO3LR (passed-in calc) for the reference force evaluation.
    """
    dtype  = np.dtype(precision)
    cdtype = complex_dtype_for(dtype)

    if verbose:
        print(f"\nLoading SO3LR Hessian basis: {npz_path}")
    data    = np.load(npz_path, allow_pickle=True)
    W       = np.asarray(data['W'],       dtype=dtype)    # (3N, 3N)
    eigvals = np.asarray(data['eigvals'], dtype=dtype)    # kcal/(mol·amu·Å²)
    ref_pos = np.asarray(data['ref_pos'], dtype=dtype)    # (3N,)  Å
    masses  = np.asarray(data['masses'],  dtype=dtype)    # (N,)   amu
    box     = np.asarray(data['box'],     dtype=dtype)    # (3,)   Å
    n_mol   = int(data['n_mol'])

    n_atoms = len(masses)
    n_modes = 3 * n_atoms
    symbols = _symbols_from_masses(masses)

    if verbose:
        print(f"  {n_atoms} atoms ({n_mol} molecules, TIP3P O H H), {n_modes} DOF")
        n_imag   = int((eigvals < -1e-6).sum())
        n_near0  = int((np.abs(eigvals) <= 1e-6).sum())
        n_stable = n_modes - n_imag - n_near0
        print(f"  stable: {n_stable}  near-zero: {n_near0}  imaginary: {n_imag}")

    # ── Angular frequencies ────────────────────────────────────────────────────
    omega_rad = np.sqrt(
        np.maximum(eigvals * _KCAL_TO_RAD2_FS2, dtype.type(0.0))
    ).astype(dtype)

    # ── Reference Atoms object ────────────────────────────────────────────────
    ref_atoms = Atoms(
        symbols   = symbols,
        positions = ref_pos.reshape(-1, 3),
    )
    ref_atoms.set_masses(masses)
    ref_atoms.cell = np.diag(box)
    ref_atoms.pbc  = True

    # ── Reference force: one SO3LR single-point at ref_pos ───────────────────
    if verbose:
        print("  Evaluating reference force (one SO3LR call) ...")
    ref_atoms.calc   = calc
    reference_force  = np.asarray(ref_atoms.get_forces(), dtype=dtype).ravel()
    reference_energy = float(ref_atoms.get_potential_energy())
    if verbose:
        f0_rms = float(np.sqrt(np.mean(reference_force**2)))
        print(f"  reference_force RMS: {f0_rms:.4e} eV/Å   "
              f"energy: {reference_energy:.6f} eV")

    # ── Effective Cartesian Hessian consistent with (W, Ω) ───────────────────
    K = _effective_hessian_from_modes(W, omega_rad, masses, precision=dtype)

    # ── Band mask ─────────────────────────────────────────────────────────────
    freq_cm1  = omega_rad / (2.0 * np.pi * _C_INV_FS)
    eps       = dtype.type(CM1_TO_RADFS)
    band_mask = (freq_cm1 >= dtype.type(band_lo)) & (freq_cm1 <= dtype.type(band_hi))
    if allow_imaginary:
        band_mask |= (omega_rad < eps)
    else:
        band_mask &= (omega_rad >= eps)
    n_active = int(band_mask.sum())

    if verbose:
        n_zero_in_band = int((omega_rad[band_mask] < eps).sum())
        af_pos = freq_cm1[band_mask & (omega_rad >= eps)]
        hi_str = f"{af_pos.max():.1f} cm⁻¹" if len(af_pos) else "n/a"
        print(f"  Band {band_lo:.0f}–{band_hi:.0f} cm⁻¹: {n_active} active modes  "
              f"({n_zero_in_band} zero-ω, {len(af_pos)} positive up to {hi_str})")

    # ── Initial conditions ────────────────────────────────────────────────────
    q0 = np.zeros(n_modes, dtype=dtype)
    p0 = np.zeros(n_modes, dtype=dtype)
    if n_active > 0:
        kT_modal = dtype.type(units.kB * float(temp_K)) * dtype.type(EIG_TO_RADFS2)
        rng = np.random.default_rng()
        p0[band_mask] = rng.normal(
            scale=float(np.sqrt(kT_modal)), size=n_active
        ).astype(dtype)

    fft_freq = np.array([], dtype=dtype)
    fft_amp  = np.array([], dtype=cdtype)

    return FIMDBasis(
        reference_positions = ref_pos.reshape(-1, 3).astype(dtype),
        masses              = masses,
        symbols             = symbols,
        W                   = W,
        omega               = omega_rad,
        K                   = K,
        reference_force     = reference_force,
        reference_energy    = reference_energy,
        band                = (float(band_lo), float(band_hi)),
        band_mask           = band_mask,
        q0                  = q0,
        p0                  = p0,
        fft_frequencies     = fft_freq,
        fft_amplitudes      = fft_amp,
        n_atoms             = n_atoms,
        n_modes             = n_modes,
        n_active_modes      = n_active,
        timestep_reference  = FIMD_DT_FS,
        precision           = precision,
        n_rigid_modes       = 0,
        metadata            = {
            'basis_source': 'so3lr_hessian',
            'npz'         : os.path.abspath(npz_path),
            'band_lo'     : band_lo,
            'band_hi'     : band_hi,
            'temp_K'      : temp_K,
        },
    )


# ── In-band constrained minimisation ──────────────────────────────────────────

def minimise_in_band(basis, calc, box, fmax=0.01, max_steps=3000,
                     max_step_A=0.1, max_disp_A=None, omega_floor_cm1=50.0,
                     verbose=True):
    """
    Relax the reference geometry within the active band only, so that the
    in-band reference force vanishes, then re-anchor the basis at r*.

    FIMD subtracts F(r0) in the residual kick, so at a non-stationary r0
    (e.g. an INM snapshot) the band dynamics feel a constant tilt −P_B F(r0).
    At r* with P_B F(r*) = 0 that tilt is gone: a one-mode basis oscillates
    about the constrained minimum, and a full basis reduces to plain MD.
    W, Ω and K are kept from r0 (they only define the drift/kick split).

    FIRE in Hessian-preconditioned modal coordinates y = q_B·sqrt(max(k, k_floor)),
    k = ω²/EIG_TO_RADFS2, so every mode above omega_floor_cm1 has curvature ≈ 1
    (soft and imaginary modes are left unscaled at the floor).  Out-of-band
    modes stay at r0.
    fmax       : convergence on the max per-atom in-band Cartesian force (eV/Å)
    max_step_A : cap on the max per-atom displacement per FIRE step (Å)
    max_disp_A : optional cap on the max per-atom total displacement from r0 (Å)
    """
    dtype  = np.dtype(basis.precision)
    mask   = np.asarray(basis.band_mask, dtype=bool)
    WB     = np.asarray(basis.W, dtype=dtype)[:, mask]
    sm     = np.repeat(np.sqrt(np.asarray(basis.masses, dtype=dtype)), 3)
    r0     = np.asarray(basis.reference_positions, dtype=dtype).reshape(-1)
    n_at   = len(basis.masses)
    k_q    = np.asarray(basis.omega, dtype=dtype)[mask] ** 2 / dtype.type(EIG_TO_RADFS2)
    k_flr  = (omega_floor_cm1 * CM1_TO_RADFS) ** 2 / EIG_TO_RADFS2
    sc     = 1.0 / np.sqrt(np.maximum(k_q, k_flr))            # q = sc * y

    atoms = Atoms(symbols=basis.symbols, positions=r0.reshape(-1, 3))
    atoms.set_masses(basis.masses)
    atoms.cell = np.diag(box)
    atoms.pbc  = True
    atoms.calc = calc

    def _eval(q):
        x = r0 + (WB @ q) / sm
        atoms.set_positions(x.reshape(-1, 3))
        F  = np.asarray(atoms.get_forces(), dtype=dtype).reshape(-1)
        fq = WB.T @ (F / sm)                                  # modal force
        FB = (sm * (WB @ fq)).reshape(-1, 3)                  # in-band Cartesian force
        return x, float(atoms.get_potential_energy()), fq, float(np.linalg.norm(FB, axis=1).max())

    # FIRE (Bitzek et al. 2006) with unit masses in preconditioned y-space
    q  = np.zeros(mask.sum(), dtype=dtype)
    v  = np.zeros_like(q)
    dt, dt_max, alpha, n_pos = 0.1, 1.0, 0.1, 0
    x, E, fq, fB = _eval(q)
    fy = sc * fq
    E_start, fB_start = E, fB
    if verbose:
        print(f"\n  In-band minimisation: {mask.sum()} modes, fmax target {fmax} eV/Å")
        print(f"    step     0  E {E:14.4f} eV  max|F_B| {fB:.4e} eV/Å")
    step, capped = 0, False
    for step in range(1, max_steps + 1):
        if fB < fmax:
            break
        P = float(fy @ v)
        if P > 0:
            vn, fn = np.linalg.norm(v), np.linalg.norm(fy)
            v = (1 - alpha) * v + alpha * vn * fy / max(fn, 1e-30)
            n_pos += 1
            if n_pos > 5:
                dt, alpha = min(dt * 1.1, dt_max), alpha * 0.99
        else:
            v[:] = 0.0; dt *= 0.5; alpha, n_pos = 0.1, 0
        v  = v + dt * fy
        dq = sc * (dt * v)
        dxs = np.linalg.norm(((WB @ dq) / sm).reshape(-1, 3), axis=1).max()
        if dxs > max_step_A:
            dq *= max_step_A / dxs
        q_new = q + dq
        if max_disp_A is not None:
            dtot = np.linalg.norm(((WB @ q_new) / sm).reshape(-1, 3), axis=1).max()
            if dtot > max_disp_A:
                q_new = q + dq * 0.5; v[:] = 0.0; dt *= 0.5; capped = True
        q = q_new
        x, E, fq, fB = _eval(q)
        fy = sc * fq
        if verbose and (step % 50 == 0):
            print(f"    step {step:5d}  E {E:14.4f} eV  max|F_B| {fB:.4e} eV/Å  dt {dt:.2f}")

    d = (x - r0).reshape(-1, 3)
    dn = np.linalg.norm(d, axis=1)
    F_star = np.asarray(atoms.get_forces(), dtype=dtype).reshape(-1)
    X = x.reshape(-1, 3); L = np.asarray(box, dtype=dtype)
    d_oh = np.r_[X[1::3] - X[0::3], X[2::3] - X[0::3]]
    oh = np.linalg.norm(d_oh - L * np.round(d_oh / L), axis=1)   # minimum image
    if verbose:
        print(f"  In-band minimisation {'converged' if fB < fmax else 'NOT converged'} "
              f"after {step} steps{' (displacement cap hit)' if capped else ''}")
        print(f"    E(r0) {E_start:.4f} → E(r*) {E:.4f} eV   (ΔE {E - E_start:+.4f})")
        print(f"    max|F_B|: {fB_start:.4e} → {fB:.4e} eV/Å   "
              f"full-force RMS at r*: {float(np.sqrt(np.mean(F_star**2))):.4e} eV/Å")
        print(f"    |r* − r0| per atom: rms {float(np.sqrt(np.mean(dn**2))):.4f}  "
              f"max {float(dn.max()):.4f} Å   O–H at r*: {oh.min():.3f}–{oh.max():.3f} Å")

    basis.reference_positions = x.reshape(-1, 3).astype(dtype)
    basis.reference_force     = F_star
    basis.reference_energy    = E
    basis.metadata.update({
        'min_in_band': True, 'min_fmax_band': fB, 'min_steps': step,
        'min_dE': E - E_start, 'min_rms_disp_A': float(np.sqrt(np.mean(dn**2))),
        'min_max_disp_A': float(dn.max()),
    })
    return basis


def minimise_cartesian(basis, calc, box, fmax=0.005, max_steps=5000, verbose=True):
    """Unconstrained Cartesian relaxation of the reference (all atoms, ASE FIRE).

    Intended for the full band, where the in-band minimiser stalls because its
    preconditioner (the Hessian at r0) goes stale.  W and omega are kept from r0;
    the reference is re-anchored at the relaxed r* with F(r*) as reference force.
    """
    from ase.optimize import FIRE
    dtype = np.dtype(basis.precision)
    atoms = Atoms(symbols=basis.symbols,
                  positions=np.asarray(basis.reference_positions, dtype=np.float64))
    atoms.set_masses(basis.masses)
    atoms.cell = np.diag(box)
    atoms.pbc = True
    atoms.calc = calc
    E0 = float(atoms.get_potential_energy())
    F0max = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
    opt = FIRE(atoms, maxstep=0.1, logfile='-' if verbose else None)
    converged = opt.run(fmax=fmax, steps=max_steps)
    F = np.asarray(atoms.get_forces(), dtype=dtype)
    E = float(atoms.get_potential_energy())
    fmax_end = float(np.linalg.norm(F, axis=1).max())
    if verbose:
        print(f"  Cartesian minimisation {'converged' if converged else 'NOT converged'} "
              f"after {opt.nsteps} steps")
        print(f"    E {E0:.4f} → {E:.4f} eV   max|F| {F0max:.4e} → {fmax_end:.4e} eV/Å   "
              f"RMS {float(np.sqrt(np.mean(F**2))):.4e} eV/Å")
    basis.reference_positions = atoms.get_positions().astype(dtype)
    basis.reference_force     = F.reshape(-1)
    basis.reference_energy    = E
    basis.metadata.update({'cart_min': True, 'cart_min_steps': int(opt.nsteps),
                           'cart_min_fmax_end': fmax_end})
    return basis


# ── FIMD logger ───────────────────────────────────────────────────────────────

def _make_fimd_logger(log_path, dyn, atoms, n_active_modes):
    lf  = open(log_path, 'w')
    hdr = "%-10s %12s %12s %12s %8s %9s\n" % (
        "Time[ps]", "Etot[eV]", "Epot[eV]", "Ekin[eV]", "T[K]", "T_FIMD[K]")
    lf.write(hdr)
    lf.flush()

    def _log():
        t      = dyn.get_time() / (1000.0 * units.fs)
        epot   = atoms.get_potential_energy()
        ekin   = atoms.get_kinetic_energy()
        t_ase  = atoms.get_temperature()
        t_fimd = 2.0 * ekin / (n_active_modes * units.kB)
        lf.write("%-10.4f %12.2f %12.2f %12.2f %8.1f %9.1f\n" % (
            t, epot + ekin, epot, ekin, t_ase, t_fimd))
        lf.flush()

    return _log, lf


# ── Production runner ─────────────────────────────────────────────────────────

def run_fimd_so3lr(
    npz_path        = DEFAULT_NPZ,
    out_dir         = DEFAULT_OUT_DIR,
    fimd_steps      = FIMD_STEPS,
    equil_steps     = EQUIL_STEPS,
    friction        = 0.1,
    temp_K          = TEMP,
    band_lo         = BAND_LO,
    band_hi         = BAND_HI,
    precision       = 'float64',
    allow_imaginary = False,
    timestep_fs     = FIMD_DT_FS,
    minimise_in_band_fmax = None,
    min_max_disp_A  = None,
    cart_minimise_fmax = None,
    cart_min_steps  = 5000,
    zero_reference_force = False,
    calc_dtype      = 'float32',
    save_interval   = SAVE_INTERVAL,
    verbose         = True,
):
    if not os.path.isfile(npz_path):
        sys.exit(f"ERROR: Hessian basis not found: {npz_path}")

    os.makedirs(out_dir, exist_ok=True)

    data      = np.load(npz_path, allow_pickle=True)
    lr_cutoff = float(data.get('lr_cutoff', 12.0))
    box       = np.asarray(data['box'], dtype=np.float64)

    calc = _make_so3lr_calc(lr_cutoff, calc_dtype)
    if verbose:
        print(f"  Calculator: SO3LR (lr_cutoff={lr_cutoff} Å, {calc_dtype})")

    basis = build_hessian_fimd_basis(
        npz_path        = npz_path,
        calc            = calc,
        temp_K          = temp_K,
        band_lo         = band_lo,
        band_hi         = band_hi,
        allow_imaginary = allow_imaginary,
        precision       = precision,
        verbose         = verbose,
    )

    if minimise_in_band_fmax is not None:
        r0_orig = np.asarray(basis.reference_positions).copy()
        F0_orig = np.asarray(basis.reference_force).copy()
        basis = minimise_in_band(basis, calc, box, fmax=minimise_in_band_fmax,
                                 max_disp_A=min_max_disp_A, verbose=verbose)
        np.savez(os.path.join(out_dir, 'min_reference.npz'),
                 r0=r0_orig, F0=F0_orig,
                 r_star=np.asarray(basis.reference_positions),
                 F_star=np.asarray(basis.reference_force),
                 E_star=basis.reference_energy, band_mask=basis.band_mask,
                 **{k: v for k, v in basis.metadata.items() if k.startswith('min_')})

    if cart_minimise_fmax is not None:
        r0_orig = np.asarray(basis.reference_positions).copy()
        basis = minimise_cartesian(basis, calc, box, fmax=cart_minimise_fmax,
                                   max_steps=cart_min_steps, verbose=verbose)
        np.savez(os.path.join(out_dir, 'cart_min_reference.npz'), r0=r0_orig,
                 r_star=np.asarray(basis.reference_positions),
                 F_star=np.asarray(basis.reference_force), E_star=basis.reference_energy)

    if zero_reference_force:
        # Use the true force: with F_ref = 0 the kick is F + K dx, so the total force
        # on the atoms is F(r) exactly (no frozen reference field).
        basis.reference_force = np.zeros_like(np.asarray(basis.reference_force))
        if verbose:
            print("  Reference force set to zero (true-force dynamics)")

    dtype  = np.dtype(precision)
    atoms  = Atoms(
        symbols   = basis.symbols,
        positions = basis.reference_positions,
    )
    atoms.set_masses(basis.masses)
    atoms.cell = np.diag(box)
    atoms.pbc  = True
    atoms.calc = calc

    inv_sqrt_m = np.repeat(dtype.type(1.0) / np.sqrt(basis.masses), 3)
    vel_flat   = inv_sqrt_m * (basis.W @ basis.p0)
    set_velocities_ang_per_fs(atoms, vel_flat.reshape(-1, 3), precision)

    # ── Langevin equilibration ────────────────────────────────────────────────
    equil_ps = equil_steps * timestep_fs * 1e-3
    prod_ps  = fimd_steps  * timestep_fs * 1e-3
    if verbose:
        print(f"\n{'='*60}")
        print(f"  FIMD SO3LR — Langevin equilibration")
        print(f"{'='*60}")
        print(f"  Steps : {equil_steps:,} × {timestep_fs} fs = {equil_ps:.2f} ps")
        print(f"  T     : {temp_K} K   friction = {friction} fs⁻¹")
        print(f"  Band  : {band_lo:.0f}–{band_hi:.0f} cm⁻¹   active: {basis.n_active_modes}")

    dyn_equil = FIMDynamics(
        atoms,
        basis           = basis,
        timestep_fs     = timestep_fs,
        loginterval     = save_interval,
        use_basis_init  = True,
        allow_imaginary = allow_imaginary,
        temperature     = temp_K,
        friction        = friction,
    )
    equil_log_fn, equil_lf = _make_fimd_logger(
        os.path.join(out_dir, 'fimd_equil.log'), dyn_equil, atoms, basis.n_active_modes)
    dyn_equil.attach(equil_log_fn, interval=save_interval)
    dyn_equil.run(equil_steps)
    equil_lf.close()
    del dyn_equil, equil_log_fn, equil_lf
    gc.collect()

    # ── NVE production ────────────────────────────────────────────────────────
    if verbose:
        print(f"\n{'='*60}")
        print(f"  FIMD SO3LR — NVE production")
        print(f"{'='*60}")
        print(f"  Steps : {fimd_steps:,} × {timestep_fs} fs = {prod_ps:.1f} ps")
        print(f"  Band  : {band_lo:.0f}–{band_hi:.0f} cm⁻¹  allow_imaginary={allow_imaginary}")
        print(f"  Output: {out_dir}/")

    traj_path = os.path.join(out_dir, 'fimd_trajectory.traj')
    log_path  = os.path.join(out_dir, 'fimd.log')

    dyn = FIMDynamics(
        atoms,
        basis           = basis,
        timestep_fs     = timestep_fs,
        trajectory      = traj_path,
        loginterval     = save_interval,
        use_basis_init  = False,
        allow_imaginary = allow_imaginary,
        temperature     = None,    # NVE
    )
    prod_log_fn, prod_lf = _make_fimd_logger(log_path, dyn, atoms, basis.n_active_modes)
    dyn.attach(prod_log_fn, interval=save_interval)

    band_energies = []
    def _record():
        band_energies.append(float(dyn.get_band_energy()))
        gc.collect()

    dyn.attach(_record, interval=save_interval)
    dyn.run(fimd_steps)
    prod_lf.close()

    be = np.array(band_energies)
    np.savez(
        os.path.join(out_dir, 'fimd_results.npz'),
        band_energies_eV = be,
        dt_fs            = timestep_fs,
        save_interval    = save_interval,
        band             = np.array(basis.band),
        n_active_modes   = basis.n_active_modes,
    )

    if verbose:
        print(f"\n  Done → {out_dir}/fimd_results.npz")
        print(f"  Band energy: {be.mean():.6f} ± {be.std():.6f} eV  "
              f"(drift {abs(be[-1]-be[0]):.4e} eV over {prod_ps:.1f} ps)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--npz',       default=DEFAULT_NPZ,
                    help=f'SO3LR Hessian basis npz (default: {DEFAULT_NPZ})')
    ap.add_argument('--out-dir',   default=DEFAULT_OUT_DIR,
                    help=f'Output directory (default: {DEFAULT_OUT_DIR})')
    ap.add_argument('--fimd-steps', type=int, default=FIMD_STEPS,
                    help=f'NVE steps (default: {FIMD_STEPS} = {FIMD_STEPS*FIMD_DT_FS*1e-3:.0f} ps; '
                         'use 400000 for 200 ps)')
    ap.add_argument('--equil-steps', type=int, default=EQUIL_STEPS,
                    help=f'Langevin equilibration steps (default: {EQUIL_STEPS})')
    ap.add_argument('--timestep',  type=float, default=FIMD_DT_FS,
                    help=f'FIMD timestep fs (default: {FIMD_DT_FS})')
    ap.add_argument('--friction',  type=float, default=0.1,
                    help='Langevin friction fs⁻¹ (default: 0.1)')
    ap.add_argument('--temp',      type=float, default=TEMP,
                    help=f'Temperature K (default: {TEMP})')
    ap.add_argument('--band-lo',   type=float, default=BAND_LO,
                    help=f'Lower band edge cm⁻¹ (default: {BAND_LO:.0f})')
    ap.add_argument('--band-hi',   type=float, default=BAND_HI,
                    help=f'Upper band edge cm⁻¹ (default: {BAND_HI:.0f})')
    ap.add_argument('--precision', default='float64',
                    choices=['float32', 'float64'])
    ap.add_argument('--allow-imaginary', action=argparse.BooleanOptionalAction,
                    default=False,
                    help='Propagate imaginary modes as free drift (default: False)')
    ap.add_argument('--minimise-in-band', type=float, default=None, metavar='FMAX',
                    help='Relax the reference within the active band to max in-band '
                         'force FMAX eV/Å before FIMD (default: off)')
    ap.add_argument('--min-max-disp', type=float, default=None, metavar='A',
                    help='Cap per-atom displacement from r0 during in-band minimisation (Å)')
    ap.add_argument('--cart-minimise', type=float, default=None, metavar='FMAX',
                    help='Unconstrained Cartesian (FIRE) relaxation of the reference to '
                         'max force FMAX eV/Å before FIMD (default: off)')
    ap.add_argument('--cart-min-steps', type=int, default=5000,
                    help='Max steps for --cart-minimise (default: 5000)')
    ap.add_argument('--zero-reference-force', action='store_true',
                    help='Set the reference force to zero (true-force dynamics)')
    ap.add_argument('--calc-dtype', default='float32', choices=['float32', 'float64'],
                    help='SO3LR calculator precision (default float32 = so3lr default; float64 conserves energy)')
    ap.add_argument('--save-interval', type=int, default=SAVE_INTERVAL,
                    help=f'Steps between log/trajectory frames (default: {SAVE_INTERVAL})')
    args = ap.parse_args()

    run_fimd_so3lr(
        npz_path        = args.npz,
        out_dir         = args.out_dir,
        fimd_steps      = args.fimd_steps,
        equil_steps     = args.equil_steps,
        friction        = args.friction,
        temp_K          = args.temp,
        band_lo         = args.band_lo,
        band_hi         = args.band_hi,
        precision       = args.precision,
        allow_imaginary = args.allow_imaginary,
        timestep_fs     = args.timestep,
        minimise_in_band_fmax = args.minimise_in_band,
        min_max_disp_A  = args.min_max_disp,
        cart_minimise_fmax = args.cart_minimise,
        cart_min_steps  = args.cart_min_steps,
        zero_reference_force = args.zero_reference_force,
        calc_dtype      = args.calc_dtype,
        save_interval   = args.save_interval,
    )


if __name__ == '__main__':
    main()
