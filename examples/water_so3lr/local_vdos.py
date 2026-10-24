"""FIMD VDOS from extracted velocities (vel_f32.npy, Å/ps; vel_meta.npz), for local runs without ASE.
Same periodogram, normalisation and fimd_vdos.npz layout as remote_vdos.py (2026-10-13).
usage: python local_vdos.py <fimd_dir> [...]"""
import os, sys
import numpy as np
from scipy.constants import k as kB, atomic_mass as amu
C_LIGHT = 2.99792458e10
TEMP_K = 300.0

for d in sys.argv[1:]:
    r = np.load(os.path.join(d, 'fimd_results.npz'))
    dt_ps = float(r['dt_fs']) * int(r['save_interval']) / 1000.0
    vel = np.load(os.path.join(d, 'vel_f32.npy'), mmap_mode='r')
    masses = np.load(os.path.join(d, 'vel_meta.npz'))['masses']
    n_frames, n_atoms, _ = vel.shape
    dt_SI = dt_ps * 1e-12
    psd = np.zeros(n_frames // 2 + 1)
    for j in range(0, n_atoms, 256):
        V = np.fft.rfft(np.asarray(vel[:, j:j+256], dtype=np.float64), axis=0)
        psd += (2.0 * dt_SI * np.abs(V)**2 * amu * 1e4 * masses[None, j:j+256, None]).sum(axis=(1, 2)) / n_frames
    psd[0] /= 2; psd[-1] /= 2
    D_Hz = psd / (kB * TEMP_K)
    freq_Hz = np.arange(len(psd)) / (n_frames * dt_SI)
    np.savez(os.path.join(d, 'fimd_vdos.npz'), freq_icm=freq_Hz / C_LIGHT, dos_icm=D_Hz * C_LIGHT,
             freq_thz=freq_Hz * 1e-12, dos_thz=D_Hz * 1e12, n_atoms=np.int64(n_atoms))
    v2O = np.mean(np.asarray(vel[:, 0::3], dtype=np.float64) ** 2)
    v2H = np.mean(np.asarray(vel[:, 1::3], dtype=np.float64) ** 2)
    f = freq_Hz / C_LIGHT
    print(d, n_frames, 'T_O=%.1f T_H=%.1f K' % (masses[0]*amu*v2O*1e4/kB, masses[1]*amu*v2H*1e4/kB),
          ' integral %.1f (3N=%d)' % ((D_Hz * C_LIGHT).sum() * (f[1] - f[0]), 3 * n_atoms))
