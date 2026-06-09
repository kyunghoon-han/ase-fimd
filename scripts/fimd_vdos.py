#!/usr/bin/env python3
"""Compute and compare the vibrational density of states (VDOS).

The VDOS is the Fourier transform of the mass-weighted velocity autocorrelation
function (VACF). This script reads velocities from either

  * an FIMD results file   (``fimd_results.npz``  -> ``velocities_ang_per_fs``), or
  * a reference-MD trajectory (``reference_md.xyz`` extended-XYZ with momenta),

computes the VACF and its spectrum, and (optionally) overlays the two so you can
see how the band-limited FIMD spectrum compares with the conventional reference.

Usage
-----
    # single file
    python fimd_vdos.py out/fimd_results.npz
    python fimd_vdos.py out/reference_md.xyz

    # overlay reference MD vs FIMD, save a figure
    python fimd_vdos.py out/reference_md.xyz out/fimd_results.npz \
        --labels "Reference MD" "FIMD 0-500" --fmax 2000 --save vdos.png

    # write the raw VDOS curves to .npz instead of/in addition to plotting
    python fimd_vdos.py out/fimd_results.npz --dump vdos_data.npz
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Tuple

import numpy as np

# Speed of light in cm/s, for converting FFT frequencies (Hz) to wavenumbers.
C_CM_PER_S = 2.99792458e10


# --------------------------------------------------------------------------- #
# Loading velocities (Angstrom / fs) and timestep (fs) from either file type
# --------------------------------------------------------------------------- #
def load_velocities(path: str) -> Tuple[np.ndarray, np.ndarray, float, str]:
    """Return (velocities [n_frames, n_atoms, 3] in A/fs, masses [n_atoms] amu,
    dt_fs, label) for an FIMD .npz or a reference-MD .xyz."""
    if path.endswith(".npz"):
        d = np.load(path, allow_pickle=True)
        if "velocities_ang_per_fs" not in d:
            raise KeyError(
                f"{path} has no 'velocities_ang_per_fs'. Re-run FIMD with a recent "
                "version (velocities are logged automatically)."
            )
        vel = np.asarray(d["velocities_ang_per_fs"], dtype=float)
        masses = np.asarray(d["masses"], dtype=float)
        dt_fs = float(d["fimd_dt_fs"])
        return vel, masses, dt_fs, os.path.basename(path)

    # Otherwise assume an ASE-readable trajectory (extended-XYZ, .traj, ...).
    from ase.io import read
    from ase import units

    frames = read(path, index=":")
    if len(frames) < 4:
        raise ValueError(f"{path}: need >= 4 frames for a meaningful VACF.")
    masses = np.asarray(frames[0].get_masses(), dtype=float)
    # ASE get_velocities() returns Angstrom / (ASE time unit); convert to A/fs.
    vel = np.asarray([a.get_velocities() for a in frames], dtype=float) * units.fs
    # Frame spacing: prefer the stored fimd_dt_fs, else assume 1 fs.
    dt_fs = float(frames[0].info.get("fimd_dt_fs", 1.0))
    if np.allclose(vel, 0.0):
        raise ValueError(
            f"{path}: velocities are all zero (trajectory stored positions only). "
            "Use a file that carries velocities/momenta."
        )
    return vel, masses, dt_fs, os.path.basename(path)


# --------------------------------------------------------------------------- #
# VACF and VDOS
# --------------------------------------------------------------------------- #
def mass_weighted_vacf(vel: np.ndarray, masses: np.ndarray,
                       max_lag: Optional[int] = None) -> np.ndarray:
    """Normalised mass-weighted velocity autocorrelation function.

    vel: (n_frames, n_atoms, 3) in A/fs; masses: (n_atoms,) in amu.
    Uses an FFT (Wiener-Khinchin) estimate of the autocorrelation, averaged over
    all mass-weighted Cartesian components.
    """
    n_frames = vel.shape[0]
    if max_lag is None:
        max_lag = n_frames // 2
    # Mass-weight and flatten to (n_frames, 3 n_atoms).
    vw = (vel * np.sqrt(masses)[None, :, None]).reshape(n_frames, -1)
    vw = vw - vw.mean(axis=0, keepdims=True)

    # Autocorrelation of each column via zero-padded FFT, then average columns.
    nfft = 1
    while nfft < 2 * n_frames:
        nfft *= 2
    F = np.fft.rfft(vw, n=nfft, axis=0)
    acf_full = np.fft.irfft(np.abs(F) ** 2, n=nfft, axis=0)[:n_frames]
    # Unbiased normalisation by the number of overlapping samples per lag.
    counts = (n_frames - np.arange(n_frames)).astype(float)
    acf = acf_full.sum(axis=1) / counts
    acf = acf[:max_lag]
    if acf[0] != 0:
        acf = acf / acf[0]
    return acf


def vacf_to_vdos(vacf: np.ndarray, dt_fs: float,
                 window: str = "hann") -> Tuple[np.ndarray, np.ndarray]:
    """Fourier transform the (one-sided) VACF into a VDOS.

    Returns (frequencies_cm^-1, vdos). A window is applied to the one-sided VACF
    and it is mirrored to make an even function so the transform is real.
    """
    n = len(vacf)
    if window == "hann":
        w = np.hanning(2 * n)[n:]
    elif window in ("none", None):
        w = np.ones(n)
    else:
        raise ValueError("window must be 'hann' or 'none'.")
    vacf_w = vacf * w
    # Even mirror -> real, non-negative spectrum.
    full = np.concatenate([vacf_w[::-1], vacf_w[1:]])
    spec = np.abs(np.fft.rfft(full))
    freq_hz = np.fft.rfftfreq(full.size, d=dt_fs * 1.0e-15)
    freq_cm1 = freq_hz / C_CM_PER_S
    return freq_cm1, spec


def compute_vdos(path: str, max_lag: Optional[int] = None,
                 window: str = "hann") -> Tuple[np.ndarray, np.ndarray, str]:
    vel, masses, dt_fs, label = load_velocities(path)
    vacf = mass_weighted_vacf(vel, masses, max_lag=max_lag)
    freq, vdos = vacf_to_vdos(vacf, dt_fs, window=window)
    return freq, vdos, label


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Compute / compare VDOS from FIMD or reference MD.")
    p.add_argument("inputs", nargs="+",
                   help="one or more files: fimd_results.npz or reference_md.xyz")
    p.add_argument("--labels", nargs="*", default=None, help="legend labels (one per input)")
    p.add_argument("--fmax", type=float, default=4000.0, help="max wavenumber to plot (cm^-1)")
    p.add_argument("--max-lag", type=int, default=None, help="VACF length in frames (default n_frames/2)")
    p.add_argument("--window", choices=["hann", "none"], default="hann")
    p.add_argument("--normalise", action="store_true", help="scale each VDOS to unit peak for shape comparison")
    p.add_argument("--save", default=None, help="save figure to this path instead of showing")
    p.add_argument("--dump", default=None, help="also write the VDOS curves to this .npz")
    args = p.parse_args(argv)

    labels = args.labels or [None] * len(args.inputs)
    if len(labels) != len(args.inputs):
        p.error("number of --labels must match number of inputs")

    curves = []
    for path, lab in zip(args.inputs, labels):
        freq, vdos, auto_label = compute_vdos(path, max_lag=args.max_lag, window=args.window)
        sel = freq <= args.fmax
        freq, vdos = freq[sel], vdos[sel]
        if args.normalise and vdos.max() > 0:
            vdos = vdos / vdos.max()
        curves.append((freq, vdos, lab or auto_label))
        # Report the dominant peaks as a quick textual summary.
        if len(vdos) > 3:
            order = np.argsort(vdos)[-5:][::-1]
            peaks = np.sort(freq[order])
            print(f"{lab or auto_label}: top VDOS peaks (cm^-1): "
                  + ", ".join(f"{x:.1f}" for x in peaks))

    if args.dump:
        out = {}
        for i, (freq, vdos, lab) in enumerate(curves):
            out[f"freq_cm1_{i}"] = freq
            out[f"vdos_{i}"] = vdos
            out[f"label_{i}"] = np.asarray(lab, dtype=object)
        np.savez_compressed(args.dump, **out)
        print(f"Wrote VDOS curves to {args.dump}")

    try:
        import matplotlib
        if args.save:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; use --dump to save the raw curves instead.")
        return 0

    fig, ax = plt.subplots(figsize=(7, 4))
    for freq, vdos, lab in curves:
        ax.plot(freq, vdos, label=lab, lw=1.2)
    ax.set_xlabel("Frequency (cm$^{-1}$)")
    ax.set_ylabel("VDOS (norm.)" if args.normalise else "VDOS (arb. units)")
    ax.set_xlim(0, args.fmax)
    ax.legend()
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Saved figure to {args.save}")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())