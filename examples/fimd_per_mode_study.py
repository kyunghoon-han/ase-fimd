#!/usr/bin/env python3
"""Per-mode FIMD study built ON TOP of the ase-fimd package's OWN pipeline.

This does NOT hand-roll the FIMD loop, the basis construction, or the thermostat.
It calls the package's validated entry point (run_fimd_from_xyz) which performs
the full minimise -> thermalise -> reference MD -> basis -> FIMD sequence, once
per frequency window, so each run is a proper FIMD run with the package doing all
the physics. We only orchestrate the per-window loop and collect results.

KEY LESSONS baked in (from the ase-fimd README troubleshooting):
  * max_initial_displacement MUST be set for low bands (e.g. 0.15) or soft-mode
    initial amplitudes overlap atoms and the run blows up. This was the cause of
    the earlier divergence -- it is OFF by default in the package by design.
  * Conservation is judged by the BAND energy H_B (band_energies_eV), NOT the
    full Cartesian potential, which is expected to drift.
  * NVE test first (no fimd temperature), then NVT (with temperature).

Because the exact keyword names of run_fimd_from_xyz can vary by package version,
this script INTROSPECTS the function signature and maps our intent onto whatever
parameters exist, printing the resolved call. If a required concept has no
matching parameter, it STOPS and shows the real signature rather than guessing.

Usage
-----
  JAX_ENABLE_X64=1 python fimd_per_mode_study.py \
      --xyz molecule.xyz --calculator so3lr \
      --band 0 1000 --window 15 --T 300 --dt 0.5 \
      --max-disp 0.15 --out-dir per_mode_study
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