#!/usr/bin/env python3
"""
hungarian_permute.py  –  Optimal molecule-to-lattice assignment for FIMD bases.

For each frame of a liquid MD trajectory, N molecule oxygens are assigned
bijectively to the N sites of a reference lattice (e.g. ice Ih) using the
Hungarian algorithm (linear_sum_assignment), minimising the total assignment
cost over all permutations:

    perm* = argmin_perm  Σ_j  |r_O^liq_{perm(j)} − r_O^ref_j|^p_mic

with p=4 by default (fourth-power cost penalises distant assignments more
strongly than squared distance, producing more local bijections).

WHY THIS MATTERS FOR FIMD
-------------------------
In a liquid, molecules diffuse: naively labelled trajectories have diffusive
character in the low-PCA modes, contaminating the vibrational basis.
Re-labelling each frame so that molecule j always refers to the liquid molecule
currently sitting closest to reference site j absorbs all inter-site hopping
into the label.  Residual displacements are then purely cage-rattling and
intramolecular vibrations — the liquid analogue of crystal phonons.

Both output files (permuted coordinates and permuted forces) use identical
per-frame molecule assignments, so the force-displacement estimator in
compute_force_spring_constants.py can be applied directly without any
continuity requirement on the trajectory.

WHOLE-MOLECULE PLACEMENT
------------------------
Hydrogen and virtual-site atoms are placed by applying the minimum-image
displacement of their oxygen as a rigid-body molecular translation, then
restoring the liquid-frame intramolecular vectors (also MIC-corrected).
This prevents split-imaging artefacts regardless of box boundary proximity.

Usage:
    python hungarian_permute.py \\
        --traj  coords.nc    --frc  forces.nc \\
        --ice   ref.rst7 \\
        --out   permuted_coords.nc  --out-frc permuted_forces.nc

    # coordinates only (no forces):
    python hungarian_permute.py --traj coords.nc --frc "" --out permuted_coords.nc

Input files:
    --traj   AMBER NetCDF coordinate trajectory
    --frc    AMBER NetCDF force trajectory (kcal/mol/Å); set to "" to skip
    --ice    AMBER NetCDF rst7 restart used as the mapping target lattice

TIP4P/2005 layout assumed by default (ATOMS_PER_MOL=4, O at index 0 per mol).
Adjust ATOMS_PER_MOL at the top of the script for other force fields.
"""

import argparse
import time
import numpy as np
from scipy.io import netcdf_file
from scipy.optimize import linear_sum_assignment

ATOMS_PER_MOL = 4   # O, H, H, EP for TIP4P/2005 (O is index 0 per molecule)


# ── PBC helper ────────────────────────────────────────────────────────────────

def mic(dr: np.ndarray, box: np.ndarray) -> np.ndarray:
    return dr - box * np.round(dr / box)


# ── Core mapping ──────────────────────────────────────────────────────────────

def make_cost(liq_oxy: np.ndarray, ice_oxy: np.ndarray,
              box: np.ndarray, power: int = 2) -> np.ndarray:
    """
    (N, N) cost matrix with PBC minimum image.
    power=2: squared distance (default).
    power=4: fourth power — strongly penalises distant pairs, making the
             assignment more local (each molecule pulled toward its
             nearest available site rather than a globally-cheap but
             locally-distant one).
    """
    dr = liq_oxy[:, None, :] - ice_oxy[None, :, :]   # (N, N, 3)
    dr = mic(dr, box)
    d2 = (dr ** 2).sum(axis=2)                         # (N, N)
    return d2 if power == 2 else d2 ** (power // 2)


def hungarian_perm(liq_oxy, ice_oxy, box, power: int = 2):
    """
    Return mol_perm such that liq_mol mol_perm[j] maps to ice site j.
    Minimises Σ_j cost(|liq_O_{mol_perm[j]} - ice_O_j|_mic) over all bijections.

    Each frame is mapped independently — a clean, well-defined bijection with
    no inter-frame coupling.  Swaps between consecutive frames are informative
    but irrelevant to PCA quality.
    """
    C = make_cost(liq_oxy, ice_oxy, box, power=power)
    row, col = linear_sum_assignment(C)
    perm = np.empty(len(ice_oxy), dtype=np.int32)
    perm[col] = row
    # Always report raw squared cost for comparability across power settings
    C_sq = make_cost(liq_oxy, ice_oxy, box, power=2)
    return perm, C_sq[row, col].sum()


def mol_to_atom_perm(mol_perm: np.ndarray) -> np.ndarray:
    """Expand molecule-level permutation to atom-level permutation."""
    A = ATOMS_PER_MOL
    return (mol_perm[:, None] * A + np.arange(A)[None, :]).ravel()


def permute_frame(coords: np.ndarray, mol_perm: np.ndarray,
                  ice_pos: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    Permute molecules to ice sites, keeping each molecule whole.

    Correct approach:
      1. For each assigned liquid molecule, compute the O→ice-O displacement
         using MIC — this is the molecular translation.
      2. Apply that same translation to every atom in the molecule.
      3. Place H/EP relative to the new O using their liquid intramolecular
         vectors (also MIC-corrected to handle any pre-existing split imaging
         in the input trajectory).

    This guarantees |new_H - new_O| = |liq_H - liq_O| ≈ 0.96 Å — no splits.
    """
    n_mol = len(mol_perm)
    A     = ATOMS_PER_MOL

    # Gather all atoms of each permuted molecule: shape (n_mol, A, 3)
    atom_perm   = mol_to_atom_perm(mol_perm)
    perm_coords = coords[atom_perm].reshape(n_mol, A, 3)

    # Ice site positions reshaped: (n_mol, A, 3)
    ice_mol = ice_pos.reshape(n_mol, A, 3)

    # Liquid O positions for each assigned molecule: (n_mol, 3)
    liq_O = perm_coords[:, 0, :]
    ice_O = ice_mol[:, 0, :]

    # Molecular translation: MIC displacement of liquid O from ice O
    t_O = mic(liq_O - ice_O, box)                  # (n_mol, 3)
    new_O = ice_O + t_O                             # (n_mol, 3)

    # Intramolecular vectors in the liquid frame, MIC-corrected to fix any
    # pre-existing split imaging in the input trajectory
    intra = mic(perm_coords - liq_O[:, None, :], box)   # (n_mol, A, 3)

    # Final positions: translate whole molecule so O lands near ice_O,
    # then attach H/EP at their liquid-frame intramolecular offsets
    mapped = new_O[:, None, :] + intra             # (n_mol, A, 3)

    return mapped.reshape(-1, 3).astype(np.float32)


def permute_forces(forces: np.ndarray, mol_perm: np.ndarray) -> np.ndarray:
    """
    Apply the same molecule permutation to a force frame.

    Forces are Cartesian vectors — no MIC or whole-molecule placement needed.
    Simply relabel atoms according to the molecule assignment:
      new_force[atom_perm[a]] = forces[a]
    which is equivalent to gathering forces at the permuted atom indices.
    """
    atom_perm = mol_to_atom_perm(mol_perm)
    return forces[atom_perm].astype(np.float32)


# ── NetCDF I/O ────────────────────────────────────────────────────────────────

def open_coord_writer(path: str, n_atoms: int, source_nc) -> tuple:
    """
    Open an AMBER NetCDF coordinate file for per-frame streaming writes.

    Returns (nc, vars_dict) where vars_dict has keys
    'time', 'coordinates', 'cell_lengths', 'cell_angles'.
    Call nc.close() when done.

    Uses an unlimited record dimension ('frame', size=0) so each
    frame is written directly to disk as it is produced — no full
    trajectory needs to be held in memory.
    """
    nc = netcdf_file(path, 'w', version=2)

    for attr in ('Conventions', 'ConventionVersion', 'title',
                 'application', 'program', 'programVersion'):
        val = getattr(source_nc, attr, None)
        if val is not None:
            setattr(nc, attr, val)
    nc.title = b'Hungarian-mapped water trajectory (test_hungarian.py)'

    nc.createDimension('frame',        0)   # 0 = unlimited record dimension
    nc.createDimension('atom',         n_atoms)
    nc.createDimension('spatial',      3)
    nc.createDimension('cell_spatial', 3)
    nc.createDimension('cell_angular', 3)
    nc.createDimension('label',        5)

    for vname in ('spatial', 'cell_spatial', 'cell_angular'):
        if vname in source_nc.variables:
            sv  = source_nc.variables[vname]
            nv  = nc.createVariable(vname, sv.typecode(), sv.dimensions)
            nv[:] = sv[:]

    tv             = nc.createVariable('time', 'f', ('frame',))
    tv.units       = b'picosecond'
    cv             = nc.createVariable('coordinates', 'f', ('frame', 'atom', 'spatial'))
    cv.units       = b'angstrom'
    clv            = nc.createVariable('cell_lengths', 'd', ('frame', 'cell_spatial'))
    clv.units      = b'angstrom'
    cav            = nc.createVariable('cell_angles', 'd', ('frame', 'cell_angular'))
    cav.units      = b'degree'

    return nc, {'time': tv, 'coordinates': cv,
                'cell_lengths': clv, 'cell_angles': cav}


def open_frc_writer(path: str, n_atoms: int, source_nc) -> tuple:
    """
    Open an AMBER NetCDF force file for per-frame streaming writes.

    Returns (nc, forces_var).  Call nc.close() when done.
    """
    nc = netcdf_file(path, 'w', version=2)

    for attr in ('Conventions', 'ConventionVersion', 'title',
                 'application', 'program', 'programVersion'):
        val = getattr(source_nc, attr, None)
        if val is not None:
            setattr(nc, attr, val)
    nc.title = b'Hungarian-mapped water forces (test_hungarian.py)'

    nc.createDimension('frame',   0)
    nc.createDimension('atom',    n_atoms)
    nc.createDimension('spatial', 3)

    tv       = nc.createVariable('time',   'f', ('frame',))
    tv.units = b'picosecond'
    fv       = nc.createVariable('forces', 'f', ('frame', 'atom', 'spatial'))
    fv.units = b'kilocalorie/mole/angstrom'

    return nc, {'time': tv, 'forces': fv}




# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--traj',       default='w240_prod.nc',
                    help='Input water production trajectory (AMBER NetCDF)')
    ap.add_argument('--frc',        default='w240_prod_frc.nc',
                    help='Input force trajectory (AMBER NetCDF); set to "" to skip')
    ap.add_argument('--ice',        default='i240_min2.rst7',
                    help='Ice Ih reference restart (NetCDF AMBER rst7)')
    ap.add_argument('--out',        default='w240_permute.nc',
                    help='Output permuted coordinate trajectory')
    ap.add_argument('--out-frc',    default='w240_permute_frc.nc',
                    help='Output permuted force trajectory')
    ap.add_argument('--stride',      type=int, default=1,
                    help='Use every N-th frame (default 1 = all frames)')
    ap.add_argument('--max-frames',  type=int, default=None,
                    help='Stop after this many OUTPUT frames')
    ap.add_argument('--cost-power',  type=int, default=4, choices=[2, 4],
                    help='Cost function exponent: 2=squared distance (default), '
                         '4=fourth power (more local, penalises distant assignments harder)')
    args = ap.parse_args()

    # ── Load ice reference ────────────────────────────────────────────────────
    print(f"\nLoading ice reference : {args.ice}")
    with netcdf_file(args.ice, 'r', mmap=False) as nc:
        i_coords = nc.variables['coordinates'][:].reshape(-1, 3).astype(float)
        i_box    = nc.variables['cell_lengths'][:].ravel()[:3].astype(float)
        i_ca     = nc.variables['cell_angles'][:].ravel()[:3].astype(float)

    n_atoms = len(i_coords)
    n_mol   = n_atoms // ATOMS_PER_MOL
    print(f"  Ice: {n_mol} molecules, box {i_box[0]:.3f}×{i_box[1]:.3f}×{i_box[2]:.3f} Å")

    # ── Load trajectory header ────────────────────────────────────────────────
    print(f"\nOpening trajectory    : {args.traj}")
    src_nc  = netcdf_file(args.traj, 'r', mmap=False)
    traj_co = src_nc.variables['coordinates']
    traj_cl = src_nc.variables['cell_lengths']
    traj_ca = src_nc.variables['cell_angles']
    traj_t  = src_nc.variables['time']

    n_frames_total = traj_co.shape[0]
    frame_indices  = range(0, n_frames_total, args.stride)
    if args.max_frames:
        frame_indices = list(frame_indices)[:args.max_frames]
    n_out = len(list(frame_indices))

    # Water box from first frame
    w_box = traj_cl[0].astype(float)
    print(f"  Water: {n_frames_total} frames, {n_atoms} atoms")
    print(f"         box {w_box[0]:.3f}×{w_box[1]:.3f}×{w_box[2]:.3f} Å")
    print(f"  Processing {n_out} frames (stride={args.stride})")

    # ── Open force trajectory (optional) ─────────────────────────────────────
    do_frc  = bool(args.frc)
    frc_nc  = None
    traj_fr = None
    if do_frc:
        print(f"Opening force traj    : {args.frc}")
        frc_nc  = netcdf_file(args.frc, 'r', mmap=False)
        traj_fr = frc_nc.variables['forces']
        n_frc_frames = traj_fr.shape[0]
        if n_frc_frames < n_frames_total:
            print(f"  WARNING: force file has {n_frc_frames} frames, "
                  f"coord file has {n_frames_total} — will cap to force file length")
            frame_indices = [fi for fi in frame_indices if fi < n_frc_frames]
            n_out = len(frame_indices)
        print(f"  Forces: {n_frc_frames} frames")

    # ── Scale ice O positions to water box ───────────────────────────────────
    # Linear scaling preserves inter-site topology without wrapping artefacts.
    scale    = w_box / i_box
    i_coords_scaled = i_coords * scale                   # all atoms scaled
    ice_oxy  = (i_coords_scaled[::ATOMS_PER_MOL]) % w_box   # O positions in water box

    ice_pos_full = i_coords_scaled % w_box               # all atoms, for permute_frame

    print(f"\n  Ice→water scale: x={scale[0]:.4f}  y={scale[1]:.4f}  z={scale[2]:.4f}")
    print(f"  Cost function  : |r|^{args.cost_power}")

    # ── Open streaming output writers ─────────────────────────────────────────
    # Frames are written to disk as they are produced: no full-trajectory
    # arrays are held in memory.
    print(f"\nOpening output files for streaming writes ...")
    coord_nc, coord_vars = open_coord_writer(args.out, n_atoms, src_nc)
    frc_nc_out = frc_vars = None
    if do_frc and args.out_frc:
        frc_nc_out, frc_vars = open_frc_writer(args.out_frc, n_atoms, frc_nc)

    # ── Process frames ────────────────────────────────────────────────────────
    rmsd_list      = []
    cost_list      = []
    swap_list      = []
    perm_prev      = None
    t_start        = time.perf_counter()
    report_every   = max(1, n_out // 20)    # print ~20 progress lines

    print(f"\n{'─'*60}")
    print(f"  {'frame':>8}  {'elapsed':>8}  {'RMSD(Å)':>9}  {'swaps':>6}  {'ETA':>8}")
    print(f"{'─'*60}")

    for out_idx, fi in enumerate(frame_indices):
        coords = traj_co[fi].astype(float)          # (n_atoms, 3)
        liq_oxy = (coords[::ATOMS_PER_MOL]) % w_box

        perm, total_cost = hungarian_perm(liq_oxy, ice_oxy, w_box,
                                          power=args.cost_power)

        # Diagnostics
        mapped_oxy   = liq_oxy[perm]
        delta_oxy    = mic(mapped_oxy - ice_oxy, w_box)
        rmsd         = np.sqrt((delta_oxy**2).sum(axis=1).mean())
        rmsd_list.append(rmsd)
        cost_list.append(total_cost)

        if perm_prev is not None:
            swap_list.append(int(np.sum(perm != perm_prev)))
        perm_prev = perm.copy()

        # Write permuted frame directly to disk
        t_val = float(traj_t[fi])
        coord_vars['coordinates'][out_idx]  = permute_frame(coords, perm, ice_pos_full, w_box)
        coord_vars['cell_lengths'][out_idx] = traj_cl[fi].astype(float)
        coord_vars['cell_angles'][out_idx]  = traj_ca[fi].astype(float)
        coord_vars['time'][out_idx]         = t_val

        if do_frc:
            frc_vars['forces'][out_idx] = permute_forces(traj_fr[fi].astype(float), perm)
            frc_vars['time'][out_idx]   = t_val

        # Progress
        if (out_idx + 1) % report_every == 0 or out_idx == n_out - 1:
            elapsed = time.perf_counter() - t_start
            rate    = (out_idx + 1) / elapsed
            eta     = (n_out - out_idx - 1) / rate if rate > 0 else 0
            swaps_here = swap_list[-1] if swap_list else 0
            print(f"  {fi:>8d}  {elapsed:>7.1f}s  {rmsd:>9.4f}  "
                  f"{swaps_here:>6d}  {eta:>7.1f}s")

    src_nc.close()
    if frc_nc is not None:
        frc_nc.close()

    coord_nc.close()
    print(f"  Written : {args.out}  ({n_out} frames, {n_atoms} atoms)")
    if do_frc and args.out_frc:
        frc_nc_out.close()
        print(f"  Written : {args.out_frc}  ({n_out} frames, {n_atoms} atoms)")

    # ── Sanity check: compare Hungarian to naive nearest-neighbour ────────────
    # If the assignment is working correctly, Hungarian should agree with NN
    # for most molecules (the globally optimal bijection should assign each
    # molecule to approximately its nearest available site).
    print(f"\n  Sanity check on first frame (Hungarian vs nearest-neighbour)...")
    with netcdf_file(args.traj, 'r', mmap=False) as snc:
        f0     = snc.variables['coordinates'][0].astype(float)
    liq_oxy_0 = f0[::ATOMS_PER_MOL] % w_box
    perm_h, _ = hungarian_perm(liq_oxy_0, ice_oxy, w_box, power=args.cost_power)

    # Naive NN: for each ice site j, find the closest liquid O (greedy, not bijective)
    C0     = make_cost(liq_oxy_0, ice_oxy, w_box, power=2)   # always use d² for reporting
    nn_liq = np.argmin(C0, axis=0)    # nn_liq[j] = liquid mol closest to ice site j

    agree       = np.sum(perm_h == nn_liq)
    n_unique_nn = len(np.unique(nn_liq))   # NN is not bijective; may have collisions
    print(f"  Hungarian vs NN agreement : {agree}/{n_mol} molecules ({100*agree/n_mol:.1f} %)")
    print(f"  NN unique assignments     : {n_unique_nn}/{n_mol} "
          f"({'bijective' if n_unique_nn == n_mol else 'has collisions — Hungarian needed'})")

    # Per-molecule distance: Hungarian assignment vs its NN distance
    h_dists = np.sqrt(C0[perm_h, np.arange(n_mol)])   # dist of Hungarian assignment
    nn_dists= np.sqrt(C0[nn_liq,  np.arange(n_mol)])  # dist of naive NN
    print(f"  Mean O–site dist (Hungarian) : {h_dists.mean():.4f} Å")
    print(f"  Mean O–site dist (NN)        : {nn_dists.mean():.4f} Å")
    print(f"  Max O–site dist  (Hungarian) : {h_dists.max():.4f} Å")

    # ── Summary statistics ────────────────────────────────────────────────────
    rmsd_arr  = np.array(rmsd_list)
    cost_arr  = np.array(cost_list)
    swap_arr  = np.array(swap_list) if swap_list else np.array([0])

    print(f"\n{'='*60}")
    print(f"  Summary over {n_out} frames (stride={args.stride})")
    print(f"{'='*60}")
    print(f"  O–site RMSD   : {rmsd_arr.mean():.4f} ± {rmsd_arr.std():.4f} Å")
    print(f"                  min={rmsd_arr.min():.4f}  max={rmsd_arr.max():.4f}")
    print(f"  Optimal cost  : {cost_arr.mean():.2f} ± {cost_arr.std():.2f} Å²  (per frame)")
    print(f"  Swaps/frame   : {swap_arr.mean():.2f} ± {swap_arr.std():.2f}")
    print(f"  (a 'swap' = one molecule changes ice-site assignment vs previous frame)")
    print(f"  (swaps count how often the globally optimal assignment differs "
          f"between consecutive frames — informative, not a quality concern for PCA)")
    if do_frc and args.out_frc:
        print(f"\n  Permuted forces  : {args.out_frc}")
        print(f"  (same per-frame molecule assignment applied to force vectors)")
    print(f"\n  To visualise: vmd {args.out} -parm7 water240.parm7")
    print(f"  Compare with:  vmd w240_prod.nc   -parm7 water240.parm7")


if __name__ == '__main__':
    main()
