#!/usr/bin/env python3
"""Per-mode Ornstein-Uhlenbeck thermostatting for band-limited FIMD.
 
WHAT THIS SCRIPT DOES
---------------------
Runs an FIMD simulation in which a *selectable subset* of the active band modes
is coupled to a heat bath at temperature T via an Ornstein-Uhlenbeck (stochastic
velocity-rescale) thermostat, while every other mode in the band evolves freely.
Setting --modes to a single index thermostats exactly one mode; omitting it
thermostats the whole active band.
 
It does NOT reimplement any FIMD physics. It runs in two stages:
 
  [A] BASIS -- call the package's own run_fimd_from_xyz, which performs the full
      validated pipeline: minimise the geometry, thermalise, run a reference MD,
      select the active normal modes by FFT of the projected trajectory, and
      build the FIMD basis. The pipeline returns a FIMDynamics object; we take
      its live, in-memory basis (result.basis).
 
      IMPORTANT: we use the in-memory basis object directly and never save/reload
      it. FIMDBasis.save() does not persist the mode matrix W or the effective
      Hessian K, so a reloaded basis has malformed W/K and the dynamics diverge
      from the first step. Reusing the live object keeps W/K intact.
 
  [B] THERMOSTATTED RUN -- attach OUFIMDynamics (a thin subclass of the package's
      FIMDynamics) to that basis and run the NVT stage. OUFIMDynamics does not
      copy or modify the integrator: it WRAPS the parent's validated step(),
      applying a half-strength OU kick to the modal band momentum before the step
      and another half after (a symmetric split), with the parent's own Langevin
      thermostat disabled so the OU thermostat is the only one acting. The OU
      target variance is matched to core.py's convention
      (kT_modal = k_B * T * EV_TO_AMU_A2_FS2), so coupled modes equipartition to
      kT/2. Only modes selected by --modes are kicked; the rest pass through
      untouched. core.py itself is not modified.
 
Because the dynamics ARE the package's dynamics on the package's own basis, if
the package's NVE conserves the band energy (it does), so does this run in the
absence of the thermostat; the thermostat is the only addition.
 
OUTPUT (in --out-dir)
---------------------
  nvt.traj, nvt.xyz          thermostatted trajectory (.xyz is clean 4-column,
                             VMD-safe), sampled every 10 steps
  nvt.log                    FIMD log
  nvt_modal_energies.npz     per-mode kinetic energies, the thermostat mask, and
                             the band mask
  _pkg_reference/            the package pipeline's own outputs + basis
 
SANITY CHECKS on the first run
------------------------------
  * The basis stage should report active modes starting at your band floor with
    NO imaginary modes; imaginary modes mean the reference is not a true minimum
    (tighten the minimisation / fix the structure).
  * Coupled modes should average to kT/2 (printed ratio near 1.0); free modes
    should keep a physical per-mode energy (~1e-2 eV), not blow up. If free modes
    explode, lower --dt and/or raise the --band floor off the near-free modes.
 
USAGE
-----
  JAX_ENABLE_X64=1 python run_ou_thermostat.py \
      --xyz molecule.xyz --calculator so3lr --band 100 1000 \
      --modes 7 --T 300 --tau 25 --out-dir ou_run
 
  --band LO HI   active window (cm^-1); keep LO off zero (e.g. 100) -- near-free
                 modes below ~100 cm^-1 destabilise the propagation.
  --modes k ...  active-mode indices (into the frequency-sorted active list) to
                 thermostat; omit to thermostat the whole active band.
  --T T          temperature (K).   --tau FS  OU relaxation time (fs; smaller =
                 stronger coupling).   --dt FS  FIMD timestep.   --nvt-steps N.
 
  The real run_fimd_from_xyz signature is printed at startup; extra pipeline
  arguments it accepts can be forwarded.
"""
from __future__ import annotations
import argparse, inspect, os, sys

os.environ.setdefault("JAX_ENABLE_X64", "1")

import warnings, logging
warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
logging.disable(logging.WARNING)
for _n in ("MLFF", "root", "absl", "jax", "jax._src", "so3lr", "mlff", "glp"):
    _lg = logging.getLogger(_n); _lg.setLevel(logging.ERROR); _lg.propagate = False

import numpy as np


def resolve_kwargs(func, desired):
    """Map desired intent onto the real parameters of `func`.

    `desired` is a dict of concept -> (candidate_param_names, value). For each
    concept we pick the first candidate that appears in func's signature. Returns
    (kwargs, missing) where missing lists concepts with no matching parameter.
    """
    params = set(inspect.signature(func).parameters)
    kwargs, missing = {}, []
    for concept, (candidates, value) in desired.items():
        hit = next((c for c in candidates if c in params), None)
        if hit is not None:
            kwargs[hit] = value
        else:
            missing.append((concept, candidates))
    return kwargs, missing, params


def build_desired(a, band, fimd_temperature):
    """Concept -> (candidate parameter names, value)."""
    d = {
        "xyz":        (["xyz_file", "input", "input_xyz", "xyz", "atoms_file"], a.xyz),
        "calculator": (["calculator", "calc"], a.calculator),
        "band":       (["band"], tuple(band)),
        "out_dir":    (["output_dir", "output", "out_dir", "outdir"], None),  # set per-run
        "thermalise_steps":  (["thermalise_steps", "thermalize_steps", "equilibration_steps"], a.thermalise_steps),
        "thermalise_T":      (["thermalise_temperature_K", "thermalize_temperature_K",
                               "thermalise_temperature", "equilibration_temperature_K"], a.T),
        "ref_md_steps":      (["reference_md_steps", "ref_md_steps", "md_steps"], a.ref_md_steps),
        "ref_md_dt":         (["reference_md_dt_fs", "ref_md_dt_fs", "md_dt_fs", "md_dt"], a.dt),
        "fimd_steps":        (["fimd_steps", "n_steps", "steps"], a.fimd_steps),
        "fimd_dt":           (["fimd_dt_fs", "fimd_dt", "dt_fs", "timestep_fs"], a.dt),
        "save_interval":     (["save_interval", "loginterval", "interval"], a.save_interval),
        "max_disp":          (["max_initial_displacement", "max_disp"], a.max_disp),
        "precision":         (["precision"], "float64"),
    }
    # band-limited separate: some versions take band_min/band_max instead of band
    d_bandpair = {
        "band_min": (["band_min"], band[0]),
        "band_max": (["band_max"], band[1]),
    }
    # fimd temperature only for NVT
    if fimd_temperature is not None:
        d["fimd_T"] = (["fimd_temperature_K", "fimd_temperature", "temperature_K", "temperature"],
                       fimd_temperature)
    return d, d_bandpair


def call_fimd(run_func, a, band, out_dir, fimd_temperature):
    desired, bandpair = build_desired(a, band, fimd_temperature)
    kwargs, missing, params = resolve_kwargs(run_func, desired)
    # fix out_dir value
    for name in ("output_dir", "output", "out_dir", "outdir"):
        if name in params:
            kwargs[name] = out_dir; break
    # if 'band' wasn't found, try band_min/band_max
    if not any(k in kwargs for k in ("band",)):
        bp, bp_missing, _ = resolve_kwargs(run_func, bandpair)
        kwargs.update(bp)
    # drop 'band'/'out_dir' entries that were only concepts, not real params
    critical = [c for (c, _) in missing if c in ("xyz", "calculator")]
    if critical:
        print("FATAL: run_fimd_from_xyz signature is missing critical params for:",
              critical)
        print("real signature:", inspect.signature(run_func))
        sys.exit(1)
    if missing:
        noncrit = [c for (c, _) in missing if c not in ("xyz", "calculator")]
        if noncrit:
            print(f"  note: no matching parameter for {noncrit}; the package default "
                  "will be used for those.")
    print(f"  calling run_fimd_from_xyz({', '.join(f'{k}={v!r}' for k,v in kwargs.items())})")
    return run_func(**kwargs)


def get_active_frequencies(result, band):
    """Pull active-mode frequencies (cm^-1) from the result/basis for enumeration."""
    C = 2.99792458e10
    # try the result object
    for attr in ("basis_file", "basis"):
        pass
    # try loading a saved basis npz next to the result
    bf = getattr(result, "basis_file", None)
    if bf and os.path.exists(bf):
        d = np.load(bf)
        if "omega" in d.files:
            freq = np.asarray(d["omega"]) / ((2 * np.pi * C) * 1e-15)
        elif "frequencies_cm1" in d.files:
            freq = np.asarray(d["frequencies_cm1"])
        else:
            return None, None
        mask = d["band_mask"] if "band_mask" in d.files else np.ones(len(freq), bool)
        return freq, mask
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xyz", required=True)
    ap.add_argument("--calculator", default="so3lr")
    ap.add_argument("--band", nargs=2, type=float, default=[0.0, 1000.0])
    ap.add_argument("--window", type=float, default=15.0,
                    help="half-width (cm^-1) of each single-mode band window")
    ap.add_argument("--T", type=float, default=300.0)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--thermalise-steps", type=int, default=4000)
    ap.add_argument("--ref-md-steps", type=int, default=2000)
    ap.add_argument("--fimd-steps", type=int, default=20000)
    ap.add_argument("--save-interval", type=int, default=1)
    ap.add_argument("--max-disp", type=float, default=0.15,
                    help="max_initial_displacement (Angstrom) -- REQUIRED for low bands")
    ap.add_argument("--modes", nargs="*", type=int, default=None)
    ap.add_argument("--nve-only", action="store_true", help="run NVE test only (no thermostat)")
    ap.add_argument("--out-dir", default="per_mode_study")
    a = ap.parse_args()

    try:
        from fimd import run_fimd_from_xyz
    except Exception as e:
        print("Could not import run_fimd_from_xyz from fimd:", e)
        print("Check the package is installed and the function name; "
              "if it lives elsewhere (e.g. fimd.pipeline), adjust the import.")
        sys.exit(1)

    os.makedirs(a.out_dir, exist_ok=True)
    print(f"[0] outputs -> {a.out_dir}/   (JAX_ENABLE_X64={os.environ.get('JAX_ENABLE_X64')})")
    print(f"    run_fimd_from_xyz signature: {inspect.signature(run_fimd_from_xyz)}")

    # ---- enumerate active modes: one full-band NVE run to get the basis ----
    print("[1] full-band reference run to enumerate active modes ...")
    full_dir = os.path.join(a.out_dir, "_fullband_reference")
    res = call_fimd(run_fimd_from_xyz, a, tuple(a.band), full_dir, fimd_temperature=None)
    freq, mask = get_active_frequencies(res, a.band)
    if freq is None:
        print("  could not read basis frequencies from the result; "
              "cannot enumerate modes. Inspect the result object:", res)
        sys.exit(1)
    active = np.where(mask)[0]; active = active[np.argsort(freq[active])]
    fnu = freq[active]
    print(f"    {len(active)} active modes in {fnu.min():.0f}-{fnu.max():.0f} cm^-1")

    if a.nve_only:
        print("nve-only requested; done (see _fullband_reference/).")
        return

    # ---- per-mode NVT runs via narrow windows, package does the physics ----
    sel = a.modes if a.modes is not None else range(len(active))
    print(f"[2] {len(list(sel))} per-mode NVT runs (window +/-{a.window} cm^-1) ...")
    for k in sel:
        f0 = float(fnu[k]); lo = max(0.0, f0 - a.window); hi = f0 + a.window
        mdir = os.path.join(a.out_dir, f"mode_{k:03d}_{f0:.0f}cm")
        print(f"    mode {k}: {f0:.1f} cm^-1  (band {lo:.0f}-{hi:.0f})")
        call_fimd(run_fimd_from_xyz, a, (lo, hi), mdir, fimd_temperature=a.T)

    print(f"\ndone. per-mode results in {a.out_dir}/mode_*/")


if __name__ == "__main__":
    main()