#!/usr/bin/env python3
"""Diagnose a 'frozen' or unstable FIMD run from its output directory.

Reads <output_dir>/fimd_results.npz (and optionally re-evaluates forces with a
calculator) and reports, in order, the signatures that distinguish the likely
causes of an FIMD run that "vibrates then stops":

  * non-finite band energy           -> numerical blow-up (reduce --fimd-dt)
  * a frequency far above the band    -> bad reference / mode mis-assignment
  * energy draining then flat motion  -> soft-mode trap / lost residual force
  * positions literally frozen        -> calculator returning zero/stale forces

Usage:
    python diagnose_freeze.py out_mace
    python diagnose_freeze.py out_mace --calculator mace_mp   # re-eval forces
"""
from __future__ import annotations

import argparse
import os
import numpy as np


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("output_dir", help="FIMD output directory containing fimd_results.npz")
    p.add_argument("--calculator", default=None,
                   help="optional: re-evaluate forces on the final frame with this calculator")
    p.add_argument("--calculator-kwargs", default=None)
    args = p.parse_args(argv)

    npz = os.path.join(args.output_dir, "fimd_results.npz")
    d = np.load(npz, allow_pickle=True)
    t = d["times_fs"]; HB = d["band_energies_eV"]; E = d["energies_eV"]
    pos = d["positions"]; vel = d.get("velocities_ang_per_fs")
    print(f"== {npz} ==")
    print(f"precision: {str(d['precision'])}   frames: {len(t)}   band: {tuple(d['band'])}")

    # 1) Non-finite band energy?
    finite = np.all(np.isfinite(HB))
    print(f"\n[1] H_band finite throughout: {finite}")
    if not finite:
        bad = int(np.argmax(~np.isfinite(HB)))
        print(f"    -> first non-finite at frame {bad} (t={t[bad]:.1f} fs): NUMERICAL BLOW-UP.")
        print("       Fix: reduce --fimd-dt (try halving), check the reference is a true minimum.")
    else:
        span = float(HB.max() - HB.min()); drift = float(abs(HB[-1] - HB[0]))
        print(f"    H_band span={span:.6f} eV  drift={drift:.6f} eV")

    # 2) Frequency outliers vs band
    f = np.asarray(d["active_frequencies_cm1"], dtype=float)
    band_hi = float(d["band"][1])
    print(f"\n[2] active modes: {len(f)}  freq range: {f.min():.1f}-{f.max():.1f} cm^-1")
    if f.size and f.max() > 1.5 * band_hi:
        print(f"    -> a mode at {f.max():.1f} cm^-1 sits well above the band edge {band_hi:.0f}.")
        print("       Likely a bad reference (minimisation not converged on this PES) or")
        print("       mode mis-assignment. Fix: tighten --minimise-fmax, lengthen reference MD.")
    if f.size and (f < 1.0).any():
        print(f"    -> {int((f<1).sum())} near-zero-frequency modes are active (rigid/imaginary leak).")

    # 3) Energy redistribution then freeze
    print("\n[3] motion over time:")
    dpos = np.sqrt(((pos[1:] - pos[:-1]) ** 2).sum(-1)).max(1)  # max atomic step per frame
    frozen = np.where(dpos < 1e-6)[0]
    if frozen.size:
        fr = int(frozen[0])
        print(f"    -> positions FREEZE at frame {fr} (t={t[fr]:.1f} fs); "
              f"step before={dpos[max(fr-1,0)]:.2e} A, after={dpos[fr]:.2e} A")
        print("       Calculator likely returned zero/stale forces (graph cutoff?) or the")
        print("       band drained into a non-moving mode. Check forces with --calculator.")
    else:
        print(f"    positions keep moving (min per-frame step {dpos.min():.2e} A); no hard freeze.")

    # 4) Kinetic energy collapse (band 'stops' vibrating)
    if vel is not None:
        ke = 0.5 * (np.asarray(d["masses"])[None, :, None] * vel ** 2).sum((1, 2))
        print(f"\n[4] kinetic energy: start={ke[0]:.4f}  min={ke.min():.4f}  end={ke[-1]:.4f} eV")
        if ke[-1] < 0.1 * ke[0]:
            print("    -> KE collapsed to <10% of initial: the band lost its energy (trap/leak).")

    # 5) Optional: re-evaluate forces on first and last frame
    if args.calculator:
        from ase import Atoms
        import sys
        sys.path.insert(0, os.getcwd())
        from fimd.calculators import get_calculator, parse_calculator_kwargs
        kw = parse_calculator_kwargs(args.calculator_kwargs)
        symbols = [str(s) for s in d["symbols"]]
        print(f"\n[5] re-evaluating forces with '{args.calculator}':")
        for idx, tag in [(0, "first"), (-1, "last")]:
            a = Atoms(symbols=symbols, positions=pos[idx]); a.calc = get_calculator(args.calculator, **kw)
            try:
                F = a.get_forces(); fmax = np.abs(F).max()
                nzero = int(np.sum(np.all(np.abs(F) < 1e-10, axis=1)))
                print(f"    {tag} frame: max|F|={fmax:.4f} eV/A, atoms with ~zero force: {nzero}/{len(a)}")
            except Exception as e:
                print(f"    {tag} frame: calculator FAILED -> {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())