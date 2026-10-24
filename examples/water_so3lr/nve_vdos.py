"""MD VDOS of the true-NVE reference w300so3_nve (2026-10-13), by direct periodogram:
same convention as remote_vdos.py (FIMD) and md_species_vdos.py, total and O / H partials.
Step 1 streams w300so3_nve/w300so3_nve.xyz (extxyz, species pos momenta, ASE units) into
w300so3_nve/w300so3_nve_vel.npy (float32, Å/ps).  Step 2 writes w300so3_nve/w300so3_nve_vdos_pg.npz.
Checks: frame count and spacing against the YAML (1 fs/frame), equipartition per species,
and the integral of the DOS against 3N.  usage: python nve_vdos.py [dt_fs=1.0]"""
import os, sys, numpy as np
from ase import units
from scipy.constants import k as kB, atomic_mass as amu
D = os.path.expanduser('~/waterso3_300/w300so3_nve')
XYZ, VEL, OUT = f'{D}/w300so3_nve.xyz', f'{D}/w300so3_nve_vel.npy', f'{D}/w300so3_nve_vdos_pg.npz'
DT_FS = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
C_LIGHT = 2.99792458e10; T = 300.0; CONV = units.fs * 1000.0      # ASE velocity -> Å/ps
MASS = {'O': 15.999, 'H': 1.008}

with open(XYZ, 'rb') as f:
    n_atoms = int(f.readline())
    f.readline(); sp = [f.readline().split()[0].decode() for _ in range(n_atoms)]
masses = np.array([MASS[s] for s in sp])
assert sp[0::3] == ['O'] * (n_atoms // 3) and all(s == 'H' for i, s in enumerate(sp) if i % 3), 'expected OHH order'

if not os.path.exists(VEL):
    # pass 1: count frames
    n_lines = 0
    with open(XYZ, 'rb') as f:
        for buf in iter(lambda: f.read(1 << 26), b''):
            n_lines += buf.count(b'\n')
    n_frames = n_lines // (n_atoms + 2)
    assert n_frames * (n_atoms + 2) == n_lines, (n_lines, n_atoms)
    print('frames', n_frames, 'atoms', n_atoms, flush=True)
    vel = np.lib.format.open_memmap(VEL + '.part', mode='w+', dtype=np.float32, shape=(n_frames, n_atoms, 3))
    inv_m = (CONV / masses)[:, None]
    with open(XYZ, 'rb') as f:
        for t in range(n_frames):
            f.readline(); f.readline()
            block = b''.join(f.readline() for _ in range(n_atoms))
            a = np.fromstring(block.replace(b'O', b' ').replace(b'H', b' '), sep=' ').reshape(n_atoms, 6)[:, 3:6]
            vel[t] = a * inv_m
            if t % 20000 == 0: print('  frame', t, flush=True)
    vel.flush(); del vel; os.rename(VEL + '.part', VEL)

vel = np.load(VEL, mmap_mode='r'); n_frames = vel.shape[0]; DT_SI = DT_FS * 1e-15
kT_m = kB * T / (masses * amu) * 1e-4                                # <v^2> per component, (Å/ps)^2
sub = np.asarray(vel[::1000], dtype=np.float64)
for s, sl in (('O', slice(0, None, 3)), ('H', np.arange(n_atoms) % 3 != 0)):
    print(f'equipartition {s}: <v^2>/(kT/m) = {np.mean(sub[:, sl]**2) / np.mean(kT_m[sl]):.4f}')
psd = {'O': np.zeros(n_frames // 2 + 1), 'H': np.zeros(n_frames // 2 + 1)}
for j in range(0, n_atoms, 96):
    V = np.fft.rfft(np.asarray(vel[:, j:j + 96], dtype=np.float64), axis=0)
    p = 2.0 * DT_SI * np.abs(V)**2 * amu * 1e4 * masses[None, j:j + 96, None] / n_frames
    idx = np.arange(j, min(j + 96, n_atoms))
    psd['O'] += p[:, idx % 3 == 0].sum(axis=(1, 2)); psd['H'] += p[:, idx % 3 != 0].sum(axis=(1, 2))
for s in psd: psd[s][0] /= 2; psd[s][-1] /= 2
fi = np.arange(n_frames // 2 + 1) / (n_frames * DT_SI) / C_LIGHT; df = fi[1] - fi[0]
dO = psd['O'] / (kB * T) * C_LIGHT; dH = psd['H'] / (kB * T) * C_LIGHT
print(f'frames {n_frames} ({n_frames * DT_FS / 1000:.1f} ps); integrals O {dO.sum()*df:.1f} H {dH.sum()*df:.1f} '
      f'total {(dO + dH).sum()*df:.1f}, expect {n_atoms} / {2*n_atoms} / {3*n_atoms} (x T_actual/300)')
# Block spectra for the MD uncertainty (2026-10-13): NB contiguous blocks, same periodogram, plus the
# kinetic temperature of each block (D is normalised by kT at 300 K, so it scales with T_kin/300).
NB = 4; L = n_frames // NB
fb = np.arange(L // 2 + 1) / (L * DT_SI) / C_LIGHT
blocks = np.zeros((NB, L // 2 + 1)); Tb = np.zeros(NB)
for k in range(NB):
    pb = np.zeros(L // 2 + 1); ke = 0.0
    for j in range(0, n_atoms, 96):
        v = np.asarray(vel[k * L:(k + 1) * L, j:j + 96], dtype=np.float64)
        V = np.fft.rfft(v, axis=0)
        pb += (2.0 * DT_SI * np.abs(V)**2 * amu * 1e4 * masses[None, j:j + 96, None] / L).sum(axis=(1, 2))
        ke += (masses[None, j:j + 96, None] * v**2).sum() / L
    pb[0] /= 2; pb[-1] /= 2
    blocks[k] = pb / (kB * T) * C_LIGHT
    Tb[k] = ke * amu * 1e4 / (3 * n_atoms * kB)
    print(f'block {k}: T_kin {Tb[k]:.1f} K, integral {blocks[k].sum() * (fb[1] - fb[0]):.1f}', flush=True)
np.savez(OUT, freq_icm=fi, dos_icm=dO + dH, dos_O_icm=dO, dos_H_icm=dH, n_atoms=np.int64(n_atoms), dt_fs=DT_FS,
         freq_blocks_icm=fb, dos_blocks_icm=blocks, T_kin_blocks=Tb, T_kin=Tb.mean())
print('wrote', OUT)
