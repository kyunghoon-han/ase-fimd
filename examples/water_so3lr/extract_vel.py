"""Extract FIMD production velocities (Å/ps, float32) + masses for local VDOS (2026-10-13).
Same conversion as remote_vdos.py (ASE velocity x fs*1000).  usage: extract_vel.py <fimd_dir>"""
import os, sys, numpy as np
from ase.io import Trajectory
from ase import units
d = sys.argv[1]; tr = Trajectory(os.path.join(d, 'fimd_trajectory.traj'))
n = len(tr); a0 = tr[0]; conv = units.fs * 1000.0
v = np.lib.format.open_memmap(os.path.join(d, 'vel_f32.npy'), mode='w+', dtype=np.float32, shape=(n, len(a0), 3))
for i, a in enumerate(tr): v[i] = a.get_velocities() * conv
v.flush(); np.savez(os.path.join(d, 'vel_meta.npz'), masses=a0.get_masses(), symbols=np.array(a0.get_chemical_symbols()))
print(d, v.shape)
