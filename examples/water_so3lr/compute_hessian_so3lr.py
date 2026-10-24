#!/usr/bin/env python3
"""
compute_hessian_so3lr.py – Finite-difference Hessian vibrational basis using SO3LR.

Uses one worker *process* per GPU (ProcessPoolExecutor + spawn).  Each worker
sets CUDA_VISIBLE_DEVICES before JAX initialises, so n_workers=4 on a 4-GPU
node fills the entire Hessian ~4× faster than a single GPU.

On restart the reference geometry is loaded from the checkpoint file so that
the accumulated columns remain consistent across job restarts.

    H_ij = −(F_i(r + δ e_j) − F_i(r − δ e_j)) / (2δ)  [eV/Å² → kcal/(mol·Å²)]
    D_ij = H_ij / sqrt(m_i · m_j)                        [kcal/(mol·amu·Å²)]

Usage:
    python3 compute_hessian_so3lr.py --n-workers 4
    python3 compute_hessian_so3lr.py --traj w300so3_prod.xyz --n-workers 4
"""

import os
import multiprocessing as mp

# Cap CPU-side threads.  JAX will use the GPU for heavy compute.
os.environ.setdefault('OMP_NUM_THREADS',      '2')
os.environ.setdefault('MKL_NUM_THREADS',      '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ['XLA_FLAGS'] = (
    os.environ.get('XLA_FLAGS', '') + ' --xla_cpu_multi_thread_eigen=false'
).strip()

import argparse
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from ase.io import read, write
from ase.optimize import LBFGS

_EV_TO_KCAL_MOL   = 23.0609
_C_INV_FS         = 2.99792458e10 * 1e-15
_KCAL_TO_RAD2_FS2 = 4.184e-4


# ── Per-worker globals (set once in initialiser, reused across tasks) ─────────

_W_ref_pos = None
_W_cell    = None
_W_numbers = None
_W_delta   = None
_W_cutoff  = None


def _worker_init(gpu_queue, ref_pos, cell, numbers, delta, cutoff):
    """
    Subprocess initialiser: claim a GPU from the queue, pin this process to it,
    then stash the shared geometry so each _task call doesn't re-pickle it.
    CUDA_VISIBLE_DEVICES must be set before any JAX / SO3LR import.
    """
    global _W_ref_pos, _W_cell, _W_numbers, _W_delta, _W_cutoff
    gpu_id = gpu_queue.get()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    _W_ref_pos = ref_pos
    _W_cell    = cell
    _W_numbers = numbers
    _W_delta   = delta
    _W_cutoff  = cutoff
    print(f'  Worker PID {os.getpid()} → GPU {gpu_id}', flush=True)


def _task(j):
    """Compute one Hessian column using the worker's pre-assigned GPU."""
    from ase import Atoms
    from so3lr.ase_utils import make_ase_calculator

    atom_idx = j // 3
    coord    = j  % 3
    calc     = make_ase_calculator(lr_cutoff=_W_cutoff)
    diff     = np.zeros(len(_W_ref_pos) * 3, dtype=np.float64)

    for sign in (+1, -1):
        pos = _W_ref_pos.copy()
        pos[atom_idx, coord] += sign * _W_delta
        atoms = Atoms(numbers=_W_numbers, positions=pos,
                      cell=_W_cell, pbc=True)
        atoms.calc = calc
        diff += sign * atoms.get_forces().ravel()

    col = -diff / (2.0 * _W_delta) * _EV_TO_KCAL_MOL
    return j, col


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--traj',      default='w300so3_prod.xyz')
    ap.add_argument('--out',       default='hessian_basis_so3lr.npz')
    ap.add_argument('--min-xyz',   default='w300so3_min.xyz')
    ap.add_argument('--fmax',      type=float, default=0.05)
    ap.add_argument('--delta',     type=float, default=0.01)
    ap.add_argument('--lr-cutoff', type=float, default=12.0)
    ap.add_argument('--n-workers', type=int,   default=4,
                    help='Worker processes / GPUs (default: 4)')
    ap.add_argument('--checkpoint', default=None,
                    help='Checkpoint path (default: <out>.ckpt.npz)')
    ap.add_argument('--skip-min',  action='store_true',
                    help='Skip minimisation (load structure from --min-xyz)')
    ap.add_argument('--frame-idx', type=int, default=-1,
                    help='Trajectory frame index to use as reference geometry '
                         '(default: -1, i.e. last frame); only used when '
                         '--skip-min is set and --min-xyz does not exist')
    args = ap.parse_args()

    t0        = time.time()
    ckpt_path = args.checkpoint or (args.out[:-4] + '.ckpt.npz')

    # ── 1. Reference geometry ─────────────────────────────────────────────────
    # Priority: checkpoint ref_pos > w300so3_min.xyz > raw MD frame
    # Loading from the checkpoint ensures that restarted runs accumulate columns
    # computed from exactly the same geometry.

    ckpt_data    = None
    ckpt_ref_pos = None

    if os.path.isfile(ckpt_path):
        try:
            ckpt_data    = np.load(ckpt_path)
            ckpt_ref_pos = np.asarray(ckpt_data['ref_pos'])   # (n_dof,) flat
            print(f'\nCheckpoint found: {ckpt_path}')
        except Exception as exc:
            print(f'WARNING: could not read checkpoint ({exc}); starting fresh')
            ckpt_data = ckpt_ref_pos = None

    if ckpt_ref_pos is not None:
        # Reconstruct Atoms from checkpoint — no minimisation needed.
        from ase import Atoms
        masses_ck  = np.asarray(ckpt_data['masses'])
        numbers_ck = np.asarray(ckpt_data['numbers'], dtype=int)
        box_ck     = np.asarray(ckpt_data['box'])
        atoms = Atoms(numbers   = numbers_ck,
                      positions = ckpt_ref_pos.reshape(-1, 3),
                      cell      = np.diag(box_ck),
                      pbc       = True)
        atoms.set_masses(masses_ck)
        print('  Using geometry from checkpoint (skipping minimisation).')

    else:
        from so3lr.ase_utils import make_ase_calculator

        if args.skip_min and os.path.isfile(args.min_xyz):
            print(f'\nReading minimised structure: {args.min_xyz}')
            atoms = read(args.min_xyz)
            atoms.pbc = True
        else:
            print(f'\nReading frame {args.frame_idx} from: {args.traj}')
            atoms = read(args.traj, index=args.frame_idx)
            atoms.pbc = True

        if not args.skip_min:
            atoms.calc = make_ase_calculator(lr_cutoff=args.lr_cutoff)
            print(f'Minimising (fmax < {args.fmax} eV/Å) ...')
            opt = LBFGS(atoms, logfile='w300so3_min.log')
            opt.run(fmax=args.fmax, steps=2000)
            fmax_val = float(np.abs(atoms.get_forces()).max())
            print(f'  fmax = {fmax_val:.4f} eV/Å')
            write(args.min_xyz, atoms)
            print(f'  Structure → {args.min_xyz}')

    n_atoms  = len(atoms)
    n_dof    = n_atoms * 3
    n_mol    = n_atoms // 3
    numbers  = atoms.get_atomic_numbers()
    masses   = atoms.get_masses()
    ref_pos  = atoms.get_positions().copy()     # (n_atoms, 3)
    cell_arr = np.array(atoms.get_cell())       # (3, 3) numpy — picklable
    box      = atoms.cell.lengths()
    print(f'  {n_atoms} atoms ({n_mol} mol) → {n_dof} DOF   cell {box}')

    # ── 2. Checkpoint ─────────────────────────────────────────────────────────
    H         = np.zeros((n_dof, n_dof), dtype=np.float64)
    completed = set()

    if ckpt_data is not None:
        try:
            H[:]      = ckpt_data['H']
            completed = set(int(j) for j in ckpt_data['completed'])
            print(f'  Resuming: {len(completed)}/{n_dof} columns already done')
        except Exception as exc:
            print(f'  WARNING: checkpoint H unreadable ({exc}); starting fresh')
            H[:]      = 0.0
            completed = set()

    def _save_checkpoint():
        """Atomic checkpoint write (must be called with lock held)."""
        tmp_base = ckpt_path[:-4] + '.tmp'
        np.savez(tmp_base,
                 H         = H,
                 completed = np.array(sorted(completed), dtype=np.int32),
                 ref_pos   = ref_pos.ravel().astype(np.float64),
                 masses    = masses.astype(np.float64),
                 numbers   = numbers.astype(np.int32),
                 box       = box.astype(np.float64))
        os.replace(tmp_base + '.npz', ckpt_path)

    # ── 3. Distribute GPUs to worker processes ────────────────────────────────
    # Read SLURM-assigned GPU IDs (e.g. "0,1,2,3") and pre-fill the queue so
    # each worker claims a unique device before JAX initialises.
    slurm_gpus = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if slurm_gpus:
        gpu_ids = [int(x) for x in slurm_gpus.split(',')]
    else:
        gpu_ids = list(range(args.n_workers))

    ctx       = mp.get_context('spawn')
    gpu_queue = ctx.Queue()
    for i in range(args.n_workers):
        gpu_queue.put(gpu_ids[i % len(gpu_ids)])

    # ── 4. Hessian loop ───────────────────────────────────────────────────────
    print(f'\nComputing Hessian: {n_dof} columns × 2 = {2*n_dof} SO3LR calls')
    print(f'  n_workers={args.n_workers}  GPUs={gpu_ids[:args.n_workers]}'
          f'  delta={args.delta} Å  lr_cutoff={args.lr_cutoff} Å')

    initial_done = len(completed)
    done         = [initial_done]
    lock         = threading.Lock()
    report_every = max(1, n_dof // 20)

    t_hess  = time.time()
    success = False
    try:
        with ProcessPoolExecutor(
                max_workers = args.n_workers,
                mp_context  = ctx,
                initializer = _worker_init,
                initargs    = (gpu_queue, ref_pos, cell_arr,
                               numbers, args.delta, args.lr_cutoff)) as pool:

            futures = {pool.submit(_task, j): j
                       for j in range(n_dof) if j not in completed}

            for fut in as_completed(futures):
                j, col = fut.result()
                H[:, j] = col
                with lock:
                    completed.add(j)
                    done[0] += 1
                    if done[0] % report_every == 0:
                        elapsed  = time.time() - t_hess
                        new_done = done[0] - initial_done
                        rate     = new_done / elapsed if elapsed > 0 else 0
                        eta      = (n_dof - done[0]) / rate if rate > 0 else 0
                        print(f'  {done[0]:5d}/{n_dof}  ({100*done[0]/n_dof:.0f}%)'
                              f'  {elapsed/60:.1f} min'
                              f'  {rate*60:.0f} cols/hr'
                              f'  ETA {eta/60:.0f} min', flush=True)
                        _save_checkpoint()
        success = True

    except Exception as e:
        print(f'\n*** Hessian failed: {e}')
        raise
    finally:
        if not success and completed:
            with lock:
                _save_checkpoint()
            print(f'  Emergency checkpoint: {len(completed)}/{n_dof} cols → {ckpt_path}')

    print(f'  Hessian done in {(time.time()-t_hess)/60:.1f} min')

    # ── 5. Symmetrise, mass-weight, diagonalise ───────────────────────────────
    print('Symmetrising and diagonalising ...')
    H = (H + H.T) * 0.5

    sqrt_m    = np.sqrt(np.repeat(masses, 3))
    D         = H / np.outer(sqrt_m, sqrt_m)
    eigvals, W = np.linalg.eigh(D)

    omega_cm1 = (np.sign(eigvals)
                 * np.sqrt(np.abs(eigvals) * _KCAL_TO_RAD2_FS2)
                 / (2.0 * np.pi * _C_INV_FS))

    order     = np.argsort(np.abs(omega_cm1))
    eigvals   = eigvals[order]
    omega_cm1 = omega_cm1[order]
    W         = W[:, order]

    n_imag   = int((omega_cm1 < -1.0).sum())
    n_zero   = int((np.abs(omega_cm1) <=  1.0).sum())
    n_stable = n_dof - n_imag - n_zero
    in_b     = (omega_cm1 > 50) & (omega_cm1 < 5000)
    print(f'  Stable: {n_stable}   zero/acoustic: {n_zero}   imaginary: {n_imag}')
    if in_b.any():
        print(f'  ω (50–5000 cm⁻¹): {omega_cm1[in_b].min():.0f} – {omega_cm1[in_b].max():.0f} cm⁻¹')

    # ── 6. Save ───────────────────────────────────────────────────────────────
    np.savez(args.out,
             W           = W.astype(np.float32),
             omega_cm1   = omega_cm1.astype(np.float32),
             eigvals     = eigvals.astype(np.float64),
             ref_pos     = ref_pos.ravel().astype(np.float64),
             masses      = masses.astype(np.float64),
             box         = box.astype(np.float64),
             n_mol       = np.int32(n_mol),
             delta       = np.float64(args.delta),
             lr_cutoff   = np.float64(args.lr_cutoff),
             source_traj = np.bytes_(args.traj))
    print(f'\nSaved → {args.out}  ({W.nbytes/1e6:.0f} MB)')

    if os.path.isfile(ckpt_path):
        os.remove(ckpt_path)

    print(f'Total wall time: {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
