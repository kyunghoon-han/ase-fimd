#!/usr/bin/env python3
"""Protocol-matched plain-MD control for the full-band FIMD water runs (2026-10-19).

Same start as FIMD rep n (Hessian reference positions + basis.p0 velocities), same 300 K Langevin
equilibration (friction 0.1 fs^-1), SO3LR in float64, then plain ASE velocity Verlet NVE.  Before
production the mean kinetic temperature is steered to T_target (the FIMD rep's production mean) by
NVE windows, after each of which the energy 3N k_B (T_target - <T>_window) is added by rescaling the
velocities.  Output mimics run_fimd_so3lr.py (fimd_trajectory.traj, fimd.log, fimd_results.npz with
n_active_modes = 3N) so extract_vel.py / local_vdos.py work unchanged.

usage: python matched_vv.py --npz HESS --t-target K --out-dir D [--dt 0.25] [--equil-ps 5]
       [--window-ps 2] [--n-windows 5] [--settle-ps 1] [--prod-ps 20] [--save-interval 10]
"""
import os, sys, argparse, time, json
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '8')
import numpy as np
from ase import Atoms, units
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_fimd_so3lr import build_hessian_fimd_basis, _make_so3lr_calc, _make_fimd_logger
from fimd.core import set_velocities_ang_per_fs

ap = argparse.ArgumentParser()
ap.add_argument('--npz', required=True)
ap.add_argument('--t-target', type=float, required=True)
ap.add_argument('--out-dir', required=True)
ap.add_argument('--dt', type=float, default=0.25)
ap.add_argument('--equil-ps', type=float, default=5.0)
ap.add_argument('--friction', type=float, default=0.1, help='Langevin friction, fs^-1')
ap.add_argument('--window-ps', type=float, default=2.0)
ap.add_argument('--n-windows', type=int, default=5)
ap.add_argument('--settle-ps', type=float, default=1.0)
ap.add_argument('--prod-ps', type=float, default=20.0)
ap.add_argument('--save-interval', type=int, default=10)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
steps = lambda ps: int(round(ps * 1000.0 / a.dt))

data = np.load(a.npz)
calc = _make_so3lr_calc(float(data.get('lr_cutoff', 12.0)), 'float64')
basis = build_hessian_fimd_basis(a.npz, calc, 300.0, 0.0, 6000.0, allow_imaginary=True,
                                 precision='float64', verbose=True)

# start state exactly as run_fimd_so3lr.py builds it
atoms = Atoms(symbols=basis.symbols, positions=basis.reference_positions)
atoms.set_masses(basis.masses)
atoms.cell = np.diag(np.asarray(data['box'], float)); atoms.pbc = True
atoms.calc = calc
inv_sqrt_m = np.repeat(1.0 / np.sqrt(basis.masses), 3)
set_velocities_ang_per_fs(atoms, (inv_sqrt_m * (basis.W @ basis.p0)).reshape(-1, 3), 'float64')
n_dof = 3 * len(atoms)
t_wall = time.time()

# 1. Langevin equilibration at 300 K
dyn = Langevin(atoms, timestep=a.dt * units.fs, temperature_K=300.0,
               friction=a.friction / units.fs, fixcm=False)
log_fn, lf = _make_fimd_logger(os.path.join(a.out_dir, 'fimd_equil.log'), dyn, atoms, n_dof)
dyn.attach(log_fn, interval=a.save_interval)
dyn.run(steps(a.equil_ps)); lf.close()
print(f'equil done: T {atoms.get_temperature():.1f} K, {time.time() - t_wall:.0f} s', flush=True)

# 2. temperature steering in NVE
dyn = VelocityVerlet(atoms, timestep=a.dt * units.fs)
history = []
for k in range(a.n_windows):
    Ts = []
    dyn.attach(lambda: Ts.append(atoms.get_temperature()), interval=1)
    dyn.run(steps(a.window_ps))
    dyn.observers.clear()
    t_w = float(np.mean(Ts))
    dE = n_dof * units.kB * (a.t_target - t_w)
    ke = atoms.get_kinetic_energy()
    lam = np.sqrt((ke + dE) / ke)
    atoms.set_velocities(atoms.get_velocities() * lam)
    history.append(dict(window=k, T_mean=t_w, dE_eV=dE, scale=lam))
    print(f'window {k}: <T> {t_w:.2f} K (target {a.t_target:.2f}), added {dE:+.3f} eV, '
          f'v scale {lam:.5f}', flush=True)

# 3. settle (unrescaled NVE, not recorded)
dyn.run(steps(a.settle_ps))

# 4. production
dyn = VelocityVerlet(atoms, timestep=a.dt * units.fs,
                     trajectory=os.path.join(a.out_dir, 'fimd_trajectory.traj'),
                     loginterval=a.save_interval)
log_fn, lf = _make_fimd_logger(os.path.join(a.out_dir, 'fimd.log'), dyn, atoms, n_dof)
dyn.attach(log_fn, interval=a.save_interval)
Ts = []
dyn.attach(lambda: Ts.append(atoms.get_temperature()), interval=1)
dyn.run(steps(a.prod_ps)); lf.close()

np.savez(os.path.join(a.out_dir, 'fimd_results.npz'), dt_fs=a.dt, save_interval=a.save_interval,
         band=np.array([0.0, 6000.0]), n_active_modes=n_dof, t_target=a.t_target,
         t_prod_mean=float(np.mean(Ts)))
json.dump(dict(vars(a), steering=history, T_prod_mean=float(np.mean(Ts))),
          open(os.path.join(a.out_dir, 'matched_vv.json'), 'w'), indent=1)
d = np.loadtxt(os.path.join(a.out_dir, 'fimd.log'), skiprows=1)
print(f'DONE: production <T> {np.mean(Ts):.2f} K (target {a.t_target:.2f}), '
      f'drift {np.polyfit(d[:, 0], d[:, 1], 1)[0]:+.4f} eV/ps, wall {time.time() - t_wall:.0f} s')
