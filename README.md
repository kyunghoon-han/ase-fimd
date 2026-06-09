<div align="center">
  <img src="docs/logo.png" alt="ase-fimd logo" width="180"/>
  <h1>ase-fimd</h1>
  <p><strong>Fourier-Integrator Molecular Dynamics for ASE</strong></p>
  <p><em>Symplectic, band-limited molecular dynamics in the vibrational/modal domain.</em></p>

  <p>
    <a href="https://github.com/kyunghoon-han/ase-fimd/actions/workflows/tests.yml"><img src="https://github.com/kyunghoon-han/ase-fimd/actions/workflows/tests.yml/badge.svg" alt="tests"/></a>
    <a href="https://www.python.org"><img src="https://img.shields.io/badge/python-3.10%E2%80%933.13-blue.svg" alt="Python"/></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"/></a>
    <a href="https://ase-lib.org"><img src="https://img.shields.io/badge/built%20on-ASE-orange.svg" alt="Built on ASE"/></a>
  </p>
</div>

---

## What is FIMD?

**Fourier-Integrator Molecular Dynamics (FIMD)** propagates selected vibrational
motion of a molecular system *stably and reversibly* in the frequency domain.
Instead of integrating in Cartesian time and filtering frequencies afterwards,
FIMD makes **band selection part of the integrator**: the harmonic component of
the dynamics is advanced as an *exact phase rotation* in a modal basis, and only
a chosen frequency window `[nu_min, nu_max]` is propagated. Everything outside
the band is held fixed.

This lets you:

- **Isolate a vibrational band** (e.g. the amide I/II region of a peptide, or the
  thermodynamically important low-frequency collective modes) and propagate *only*
  that band, suppressing out-of-band response.
- **Take larger time steps** for low-frequency dynamics than a conventional
  integrator allows, because the stiff high-frequency modes are not propagated.
- **Diagnose mode coupling and anharmonicity** through a physically transparent
  band Hamiltonian that is conserved to machine precision in the harmonic limit.

The method is built on a symmetric kick-drift-kick (Strang) splitting of the band
Hamiltonian into an exactly solvable quadratic reference flow and a residual-force
kick, giving a **second-order symplectic** update.

> This package is an open-source ASE implementation of the FIMD method.
> The conserved quantity is the **band Hamiltonian**
> `H_B = 1/2 sum_{nu in B}(pi_nu^2 + omega_nu^2 q_nu^2) + dV(r_B)`,
> **not** the full Cartesian potential energy &mdash; see [Outputs](#outputs).

---

## Table of Contents

- [What is FIMD?](#what-is-fimd)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
  - [Optional calculator backends](#optional-calculator-backends)
- [Quick start](#quick-start)
- [Calculators](#calculators)
- [Command reference](#command-reference)
  - [`fimd run`](#fimd-run)
  - [`fimd run-bands`](#fimd-run-bands)
  - [`fimd calculators`](#fimd-calculators)
- [Outputs](#outputs)
  - [`fimd_results.npz` keys](#fimd_resultsnpz-keys)
- [Analysis scripts](#analysis-scripts)
  - [Vibrational density of states (VDOS)](#vibrational-density-of-states-vdos)
  - [Diagnosing a problematic run](#diagnosing-a-problematic-run)
- [Choosing a proper test system](#choosing-a-proper-test-system)
- [Python API](#python-api)
- [Tips \& troubleshooting](#tips--troubleshooting)
- [Development \& testing](#development--testing)
- [Citation](#citation)
- [License](#license)

---

## Installation

Requires **Python >= 3.10** and **ASE >= 3.22**.

```bash
git clone https://github.com/kyunghoon-han/ase-fimd.git
cd ase-fimd
pip install -e .
```

This installs the `fimd` command-line tool and the importable `fimd` package.

### Optional calculator backends

The core install runs on ASE's built-in calculators (EMT, Lennard-Jones, Morse,
EAM, TIP3P). Heavier quantum-chemistry and machine-learned potentials are
optional extras:

```bash
pip install -e ".[tblite]"   # xTB semi-empirical QM (tblite)
pip install -e ".[mace]"     # MACE-MP machine-learned force field
pip install -e ".[xtb]"      # the xtb-python ASE calculator
pip install -e ".[so3lr]"    # SO3LR JAX force field  (see note below)
pip install -e ".[dev]"      # build, pytest, ruff, twine (for contributors)
```

> **SO3LR note.** SO3LR requires **Python >= 3.12** and a specific JAX version.
> Install it in a dedicated 3.12+ environment:
> ```bash
> conda create -n fimd312 python=3.12 && conda activate fimd312
> pip install "jax==0.5.3"
> pip install -e ".[so3lr]"
> ```
> SO3LR is JAX-based and **only numerically correct in float64**; the calculator
> factory enables `jax_enable_x64` automatically and defaults to `dtype=float64`.

---

## Quick start

```bash
# 1. A minimal run on a metallic cluster with EMT (ships with ASE)
fimd run --input cluster.xyz --calculator emt \
    --band-min 0 --band-max 200 --output out

# 2. A low-frequency band of a peptide with xTB, thermalised at 300 K
fimd run --input peptide.xyz --calculator tblite \
    --band-min 0 --band-max 500 --output out_xtb \
    --thermalise-steps 2000 --thermalise-temperature 300 \
    --reference-md-steps 2000 --reference-md-dt 0.5 \
    --fimd-steps 1000 --fimd-dt 0.5 --save-interval 1 \
    --max-initial-displacement 0.15

# 3. Build the modal basis from an existing conventional-MD trajectory
fimd run --input peptide.xyz --calculator mace_mp \
    --trajectory conventional_md.traj --basis-source covariance \
    --band-min 0 --band-max 200 --output out_cov

# 4. The full multi-band protocol (minimise -> thermalise -> equilibrate -> FIMD)
fimd run-bands --input peptide.xyz --calculator mace_mp \
    --bands "0:200,200:600,600:1200,1450:1750" --output out_bands

# 5. List available calculator shortcuts
fimd calculators
```

---

## Calculators

Pass a calculator with `--calculator/-c`. Built-in shortcuts:

| Name | Backend | Physically valid for | Extra |
|------|---------|----------------------|-------|
| `emt` | ASE EMT | Metal clusters (Al, Cu, Ag, Au, Ni, Pd, Pt) | &mdash; |
| `lj` | ASE Lennard-Jones | Noble-gas / van-der-Waals clusters | &mdash; |
| `morse` | ASE Morse | Diatomics / simple bonded clusters | &mdash; |
| `eam` | ASE EAM | Metals (`--calculator-kwargs '{"potential":"Cu.eam.alloy"}'`) | &mdash; |
| `tip3p` | ASE TIP3P | Rigid water | &mdash; |
| `tblite` / `xtb` | tblite (GFN2-xTB) | General molecules | `[tblite]` |
| `xtb-ase` | xtb-python | General molecules | `[xtb]` |
| `mace_mp` / `mace` | MACE-MP | General molecules & materials | `[mace]` |
| `so3lr` | SO3LR (JAX) | General molecules (float64) | `[so3lr]` |
| `orca` | ASE ORCA | General molecules (needs ORCA) | &mdash; |
| `module:object` | custom | Any ASE calculator factory/class | &mdash; |

Calculator keyword arguments are passed as JSON or `key=value` pairs:

```bash
--calculator-kwargs '{"method":"GFN2-xTB"}'
--calculator-kwargs 'epsilon=0.0103,sigma=3.4,rc=10.0'
```

> **Use the right calculator for your system.** Pair potentials (`lj`, `morse`)
> and metallic `emt` will *run* on any structure and even conserve energy, but the
> dynamics are only physically meaningful on appropriate systems. Running `emt` on
> an organic molecule, for instance, produces a conserved-but-meaningless
> trajectory. See [Choosing a proper test system](#choosing-a-proper-test-system).

---

## Command reference

### `fimd run`

Run a single band-limited FIMD trajectory. The standard pipeline is:

**minimise -> (optional) thermalise (NVT) -> reference MD (NVE) -> build modal basis -> FIMD**

```bash
fimd run --input STRUCTURE --band-min LOW --band-max HIGH [options]
```

**Required**

| Flag | Description |
|------|-------------|
| `--input`, `-i` | Input structure (any ASE-readable format, e.g. `.xyz`). |
| `--band-min` | Lower band edge, in cm^-1. |
| `--band-max` | Upper band edge, in cm^-1. |

**Calculator & I/O**

| Flag | Default | Description |
|------|---------|-------------|
| `--calculator`, `-c` | `emt` | Calculator name or `module:object` factory. |
| `--calculator-kwargs` | &mdash; | JSON or `key=value,...` calculator arguments. |
| `--output`, `-o` | `fimd_output` | Output directory. |
| `--precision` | `float64` | `float32` / `float64`. Drives the integrator **and** (for MACE) the model dtype. |
| `--velocity-unit` | `angstrom/ps` | Units for velocities read from an external trajectory: `angstrom/ps`, `angstrom/fs`, or `ase`. |
| `--quiet` | off | Suppress progress output. |
| `--seed` | &mdash; | RNG seed for reproducibility. |

**Reference structure & basis**

| Flag | Default | Description |
|------|---------|-------------|
| `--reference` | &mdash; | Use this structure as the reference instead of minimising the input. |
| `--trajectory` | &mdash; | External trajectory for band identification (skips the internal reference MD). |
| `--basis-source` | `hessian` | `hessian` (numerical Hessian) or `covariance` (mode basis from the external trajectory). **Only used with `--trajectory`.** |
| `--hessian-step` | `0.001` | Finite-difference displacement for the numerical Hessian (A). |
| `--no-minimise` | off | Use the input structure directly as the reference (no minimisation). |
| `--minimise-fmax` | `0.01` | Force convergence for minimisation (eV/A). |
| `--minimise-steps` | `1000` | Max minimisation steps. |
| `--no-remove-rigid-motion` | off | Skip Kabsch alignment of the reference trajectory before modal projection. |

**Thermalisation (NVT Langevin, optional)**

| Flag | Default | Description |
|------|---------|-------------|
| `--thermalise-steps` | `0` | NVT Langevin equilibration steps **before** reference MD. `0` skips it. |
| `--thermalise-temperature` | `300.0` | Target temperature (K). |
| `--thermalise-friction` | `0.01` | Langevin friction (1/fs). |

**Reference MD (for band identification)**

| Flag | Default | Description |
|------|---------|-------------|
| `--reference-md-steps` | `1000` | Reference MD steps. Set `0` to build the basis from the reference only. |
| `--reference-md-dt` | `1.0` | Reference MD time step (fs). |
| `--reference-temperature` | `300.0` | Initial Maxwell-Boltzmann temperature (K). |
| `--reference-ensemble` | `nve` | `nve` or `nvt`. |
| `--reference-save-interval` | `1` | Save every N steps. |

**FIMD production**

| Flag | Default | Description |
|------|---------|-------------|
| `--fimd-steps` | `1000` | Number of FIMD steps. |
| `--fimd-dt` | `1.0` | FIMD time step (fs). |
| `--fimd-temperature` | &mdash; | If set, applies a band thermostat (NVT). Omit for NVE. |
| `--fimd-friction` | `0.01` | Band Langevin friction (1/fs), used only with `--fimd-temperature`. |
| `--save-interval` | `10` | Log every N steps. **Set to `1` to store every time step.** |

**Initial conditions & trajectory output**

| Flag | Default | Description |
|------|---------|-------------|
| `--max-initial-displacement` | &mdash; | Cap (A) on the reconstructed initial band displacement. **Recommended for low-frequency bands** (try `0.1-0.2`). |
| `--initial-temperature` | &mdash; | Replace trajectory modal momenta with Maxwell-Boltzmann momenta at this temperature (K). |
| `--trajectory-format` | `xyz` | `xyz` (plain, VMD-safe), `extxyz` (with velocities, for ASE/OVITO), `traj` (ASE binary), `both`, or `none`. |
| `--keep-intermediate-traj` | off | Keep `minimise.traj` / `reference_md.traj` instead of deleting them. |

> **Why `--max-initial-displacement` matters.** For a low-frequency band, projecting
> the last reference frame onto soft modes can produce a large initial amplitude
> that places atoms on top of each other and blows up the run. The cap is **off by
> default** (silently altering initial conditions would undermine reproducibility),
> so set it explicitly when running narrow low-frequency bands.

---

### `fimd run-bands`

Run the **full multi-stage protocol** once and reuse the equilibrated reference
for several independent FIMD bands:

**minimise -> thermalise (NVT) -> equilibrate (NVT) -> FIMD per band**

```bash
fimd run-bands --input STRUCTURE --bands "0:200,200:600,600:1200" [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input`, `-i` | *(required)* | Input structure. |
| `--calculator`, `-c` | `mace_mp` | Calculator name or factory. |
| `--calculator-kwargs` | &mdash; | JSON / `key=value` calculator args. |
| `--bands` | `0:200,200:600,600:1200,1450:1750` | Comma-separated bands in cm^-1 (`lo:hi` or `lo-hi`). |
| `--output`, `-o` | `fimd_200ps_multiband` | Output directory. |
| `--precision` | `float64` | `float32` / `float64`. |
| `--minimise-fmax` | `0.02` | Minimisation force tolerance (eV/A). |
| `--minimise-steps` | `2000` | Max minimisation steps. |
| `--conventional-md-dt` | `0.5` | Conventional MD time step (fs). |
| `--temperature` | `300.0` | Thermalisation/equilibration temperature (K). |
| `--thermalisation-ps` | `20.0` | Thermalisation length (ps). |
| `--equilibration-ps` | `20.0` | Equilibration length (ps). |
| `--thermalisation-ensemble` | `nvt` | `nve` / `nvt`. |
| `--equilibration-ensemble` | `nvt` | `nve` / `nvt`. |
| `--friction` | `0.01` | Conventional MD Langevin friction (1/fs). |
| `--thermalisation-save-interval` | `200` | Save interval during thermalisation. |
| `--equilibration-save-interval` | `20` | Save interval during equilibration. |
| `--production-ps` | `200.0` | FIMD production length per band (ps). |
| `--fimd-dt` | `0.5` | FIMD time step (fs). |
| `--fimd-temperature` | &mdash; | Optional FIMD NVT temperature (omit for NVE). |
| `--fimd-friction` | `0.01` | FIMD band Langevin friction (1/fs). |
| `--fimd-save-interval` | `200` | FIMD save interval. |
| `--hessian-step` | `0.001` | Hessian finite-difference step (A). |
| `--max-initial-displacement` | &mdash; | Cap on initial band displacement (A). |
| `--seed` | `12345` | RNG seed. |
| `--quiet` | off | Suppress progress output. |

---

### `fimd calculators`

List the available calculator shortcuts and their descriptions:

```bash
fimd calculators
```

---

## Outputs

A `fimd run` writes the following into the output directory:

| File | Contents |
|------|----------|
| `fimd_results.npz` | Primary results archive (see keys below). |
| `fimd_trajectory.xyz` | Full FIMD trajectory (plain XYZ by default; VMD-safe). |
| `fimd_final.xyz` | Final structure. |
| `fimd_basis.npz` | Serialised modal basis (`W`, `omega`, `K`, band mask, reference energy, ...). |
| `fimd.log` | ASE MD log (per-step energies & temperature). |
| `reference.xyz`, `reference_md.{xyz,log}` | Reference structure and reference-MD trajectory. |
| `thermalise.log` | NVT thermalisation log (only if `--thermalise-steps > 0`). |

### `fimd_results.npz` keys

| Key | Description |
|-----|-------------|
| `times_fs` | Time of each logged frame (fs). |
| `energies_eV` | Full Cartesian potential energy `V(r)` &mdash; **not** the conserved quantity. |
| `band_energies_eV` | **Band Hamiltonian `H_B`** &mdash; the conserved quantity in NVE. |
| `positions` | `(n_frames, n_atoms, 3)` positions (A). |
| `velocities_ang_per_fs` | `(n_frames, n_atoms, 3)` velocities (A/fs) &mdash; use these for VACF / VDOS. |
| `velocity_unit` | String tag: `"angstrom/fs"`. |
| `symbols`, `masses` | Chemical symbols and atomic masses (amu). |
| `band` | `(nu_min, nu_max)` in cm^-1. |
| `fimd_dt_fs` | FIMD time step (fs). |
| `active_mode_index`, `active_frequencies_cm1` | Active band modes and their frequencies. |
| `final_mode_kinetic_eV`, `final_mode_potential_eV`, `final_mode_harmonic_eV` | Per-mode energy breakdown at the final frame. |
| `final_residual_dV_eV` | Residual potential `dV(r_B)` at the final frame. |

> **Energy conservation check.** Judge conservation by `band_energies_eV`
> (`H_B`), *not* `energies_eV`. The full Cartesian potential moves with
> out-of-band content even when the band dynamics are perfectly conserved; only
> `H_B` is the integrator's invariant.

---

## Analysis scripts

Two helper scripts live in [`scripts/`](scripts/).

### Vibrational density of states (VDOS)

Compute the VDOS (FFT of the mass-weighted velocity autocorrelation) from FIMD
results or a reference-MD trajectory, and overlay them:

```bash
# single spectrum
python scripts/fimd_vdos.py out/fimd_results.npz --fmax 2000

# overlay reference MD vs band-limited FIMD, normalised for shape
python scripts/fimd_vdos.py out/reference_md.xyz out/fimd_results.npz \
    --labels "Reference MD" "FIMD 0-500" --fmax 2000 --normalise --save vdos.png

# dump the raw curves to .npz for your own plotting
python scripts/fimd_vdos.py out/fimd_results.npz --dump vdos.npz
```

The script auto-detects the file type, reads velocities from either source, and
prints the dominant peak positions. VDOS resolution scales with trajectory
length (`df ~ 1/(N*dt)`) &mdash; use long runs with `--save-interval 1` for sharp peaks.

### Diagnosing a problematic run

If a run looks unstable or "frozen", this probe reports finite-ness, frequency
outliers, position freezes, and (optionally) re-evaluates forces:

```bash
python scripts/diagnose_freeze.py out --calculator emt
```

---

## Choosing a proper test system

The native ASE potentials are only physically meaningful on specific systems.
The `fimd.reference_systems` module builds appropriate, pre-relaxed structures:

```python
from fimd.reference_systems import emt_cluster, lj_cluster, morse_diatomic
from ase.io import write

write("cu13.xyz", emt_cluster())          # Cu icosahedron  -> emt
ar, lj_kw = lj_cluster()                  # Ar icosahedron  -> lj  (lj_kw = matching params)
write("ar13.xyz", ar)
write("h2.xyz", morse_diatomic())         # H2 at r0        -> morse
```

| Calculator | Proper system | Builder |
|------------|---------------|---------|
| `emt` | Metallic cluster | `emt_cluster()` |
| `lj` | Noble-gas cluster | `lj_cluster()` |
| `morse` | Diatomic / small cluster | `morse_diatomic()`, `morse_cluster()` |

---

## Python API

Everything the CLI does is available programmatically:

```python
from fimd import run_fimd_from_xyz

result = run_fimd_from_xyz(
    xyz_file="peptide.xyz",
    calculator="tblite",
    calculator_kwargs={"method": "GFN2-xTB"},
    band=(0.0, 500.0),
    output_dir="out",
    thermalise_steps=2000,
    thermalise_temperature_K=300.0,
    reference_md_steps=2000,
    reference_md_dt_fs=0.5,
    fimd_steps=1000,
    fimd_dt_fs=0.5,
    save_interval=1,
    max_initial_displacement=0.15,
    precision="float64",
)

print(result.results_file)     # path to fimd_results.npz
print(result.basis_file)       # path to fimd_basis.npz
```

Lower-level building blocks are exported too:

```python
from fimd import FIMDBasis, FIMDynamics

# Hessian-based basis ...
basis = FIMDBasis.from_trajectory(trajectory, reference, calculator, band=(0, 500))
# ... or a trajectory-derived (covariance) basis
basis = FIMDBasis.from_trajectory_covariance(trajectory, reference, calculator, band=(0, 200))

dyn = FIMDynamics(atoms, basis=basis, timestep_fs=0.5)
dyn.run(1000)
print(dyn.get_band_energy())            # conserved band Hamiltonian H_B
print(dyn.get_modal_energy_breakdown()) # per-mode kinetic/potential/harmonic + dV
```

---

## Tips & troubleshooting

- **"Energy isn't conserved."** Check `band_energies_eV` (`H_B`), not
  `energies_eV`. The full potential energy is expected to drift; `H_B` should not.
- **Run blows up immediately on a low band.** Set `--max-initial-displacement`
  (e.g. `0.15`). Soft-mode initial amplitudes can otherwise overlap atoms.
- **VMD shows the molecule "freezing" mid-trajectory.** Use the default
  `--trajectory-format xyz` (plain 4-column). Extended-XYZ with velocity columns
  can desync VMD's parser; use `extxyz` only with ASE/OVITO.
- **MACE seems noisy in low-frequency bands.** Keep `--precision float64`. MACE's
  float32 mode is faster but its force noise lands in exactly the soft modes FIMD
  targets.
- **Conserved but unphysical dynamics.** You're probably using a calculator on an
  inappropriate system (e.g. `emt` on a molecule). See
  [Choosing a proper test system](#choosing-a-proper-test-system).

---

## Development & testing

```bash
pip install -e ".[dev]"
pytest -q          # run the test suite
ruff check fimd    # lint
```

Continuous integration runs the suite and linter across Python 3.10-3.13 on every
push and pull request (see [`.github/workflows/tests.yml`](.github/workflows/tests.yml)).
The core suite runs entirely on ASE's built-in calculators; tests for `tblite`,
`mace`, and `so3lr` are skipped automatically when those backends are not
installed.

---

## Citation

If you use FIMD in your work, please cite the method paper:

> K. Han, A. Tkatchenko, and J. T. Berryman,
> *Symplectic and Thermodynamically Consistent Molecular Dynamics in the
> Frequency Domain* (Fourier-Integrator Molecular Dynamics, FIMD).
> Manuscript under review at *Physical Review Letters* (2026).

Full citation details will be updated here once the paper is published.

---

## License

Released under the [MIT License](LICENSE).
