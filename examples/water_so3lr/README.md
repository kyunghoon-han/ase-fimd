# Liquid water with SO3LR: full-band FIMD

These are the driver scripts behind the water VDOS results in the FIMD paper: 300 K liquid water with the SO3LR
potential, 4 replicates. Each replicate starts from a finite-difference Hessian of an unminimised thermal
snapshot, and the full band (0–6000 cm⁻¹, imaginary modes included) is propagated with true forces.

They are research scripts, kept as they were run. You need `so3lr` installed and `fimd` importable
(`pip install -e .` from the repo root, or `PYTHONPATH=<repo>`).

## Pipeline

| step | script | notes |
|---|---|---|
| 1 | `prepare_xyz.py` | TIP4P/2005 PDB → extxyz (drops the virtual site) |
| 2 | `compute_hessian_so3lr.py` | FD Hessian (δ = 0.01 Å, lr_cutoff 12 Å), one GPU per worker; writes `hessian_basis_so3lr_repN.npz` |
| 3 | `run_fimd_so3lr.py` | Langevin equilibration, then FIMD NVE production |
| 4 | `extract_vel.py` | trajectory → `vel_f32.npy` (Å/ps) + `vel_meta.npz` |
| 5 | `local_vdos.py` | periodogram VDOS → `fimd_vdos.npz` (needs numpy/scipy only) |
| – | `matched_vv.py` | plain velocity-Verlet control with the same start, equilibration and temperature as an FIMD replicate |
| – | `nve_vdos.py` | VDOS of a separate long NVE MD reference (only used here for the D(0) check) |

Replicate Hessians (a snapshot from the MD production trajectory, no minimisation):

```bash
python compute_hessian_so3lr.py --traj w300so3_prod.xyz --out hessian_basis_so3lr_rep1.npz \
    --frame-idx 500 --skip-min --delta 0.01 --lr-cutoff 12.0 --n-workers 4
# rep2/3/4: --frame-idx 1500 / 1999 / 1000
```

Production run for one replicate: 5 ps Langevin, then 20 ps NVE, with frames every 2.5 fs.

```bash
python run_fimd_so3lr.py --npz hessian_basis_so3lr_rep1.npz --out-dir fimd_rep1 \
    --band-lo 0 --band-hi 6000 --allow-imaginary --zero-reference-force \
    --calc-dtype float64 --timestep 0.03125 \
    --equil-steps 160000 --fimd-steps 640000 --save-interval 80 --friction 0.1 --temp 300
python extract_vel.py fimd_rep1 && python local_vdos.py fimd_rep1
```

Matched MD control (T_target = the FIMD replicate's mean production temperature):

```bash
python matched_vv.py --npz hessian_basis_so3lr_rep1.npz --t-target 299.92 --out-dir vv_rep1
```

## Options added for this system

- `FIMDynamics(..., allow_imaginary=True)` / `--allow-imaginary`: the Hessian of a thermal liquid snapshot has
  about 1100 imaginary modes. With this option, zero-ω modes are made active and propagated as free drift instead
  of raising an error. The low-frequency VDOS peak needs them: about 175 cm⁻¹ with them frozen, versus 64 in MD.
- `--zero-reference-force`: sets F_ref = 0, so the dynamics uses the true force. With a thermal (non-stationary)
  reference, a non-zero F_ref acts as a constant tilt that tears the system apart.
- `--minimise-in-band FMAX`: Hessian-preconditioned FIRE over the active modes only, after which the basis
  is re-anchored at the minimum. Use it for truncated bands. The full band does not need it when the true force is used.
- `--calc-dtype float64`: **use this.** The SO3LR ASE calculator defaults to float32, and in NVE that heats the
  system by about 1 eV/ps (+15–30 K over 20 ps for 2304 atoms). Plain velocity Verlet heats just as much in float32 and
  conserves energy in float64, so this comes from the force precision, not from FIMD.
- `--save-interval`: frame stride, so that small timesteps don't produce huge trajectories.

## Timestep

At full band, the kick–rotate–kick split loses energy. The loss has a fixed sign, and as a function of Ω_max·dt it
sits on a plateau until Ω_max·dt ≈ 0.05; below that it falls off roughly as dt². For water (Ω_max ≈ 3700 cm⁻¹,
the O–H stretch) this gives:

| dt (fs) | 0.25 | 0.125 | 0.0625 | 0.03125 |
|---|---|---|---|---|
| ΔE after 0.5 ps (eV, 2304 atoms) | −0.91 | −0.62 | −0.19 | −0.05 |

At 0.25 fs the libration band is blue-shifted (558 vs 547 cm⁻¹) and TS(0–1300) comes out about 0.2 kJ/mol low. At
0.03125 fs, FIMD matches the protocol-matched velocity-Verlet control: TS 13.20 vs 13.21 kJ/mol, libration peak 547
in both, and the per-replicate difference is 0.00 ± 0.11. With Ω = 0 (plain VV through the same code) energy is
conserved. Truncated bands tolerate much larger steps: at 0–2000 cm⁻¹ and dt 0.25 fs the drift was about 0.15 eV/ps.

## VDOS normalisation

`local_vdos.py` divides by k_B·300 K. When you compare runs whose temperatures differ by a few K, rescale each
curve so that ∫D = number of active modes (or 3N). Otherwise the temperature offset shows up as an apparent
spectral difference.
