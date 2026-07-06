#!/usr/bin/env python3
"""Per-mode Ornstein-Uhlenbeck thermostatting for band-limited FIMD.
 
WHAT IT DOES
------------
FIMD propagates a chosen band of vibrational modes as decoupled harmonic
oscillators in mass-weighted normal coordinates, with an anharmonic residual-
force kick coupling them. This script adds a thermostat that couples a SELECTED
SUBSET of those band modes to a heat bath at temperature T, while the remaining
band modes evolve without a bath. Pick one mode with --modes to thermostat a
single mode; omit --modes to thermostat the whole active band.
 
 
HOW THE THERMOSTAT IS APPLIED (per band mode)
---------------------------------------------
In FIMD's mass-weighted normal coordinates each active mode nu is a UNIT-MASS
oscillator with modal momentum p_nu. Its kinetic energy is 0.5 * p_nu^2 (in the
code's modal units), so the canonical (Boltzmann) target at temperature T is a
Gaussian momentum with
 
    <p_nu^2> = kT_modal ,   kT_modal = k_B * T * EV_TO_AMU_A2_FS2 ,
 
i.e. equipartition <0.5 p_nu^2> = kT/2. Note this target is the SAME for every
mode -- no reduced mass and no frequency dependence -- because the mass-weighting
has already made every modal oscillator unit-mass.
 
The thermostat is an Ornstein-Uhlenbeck (stochastic velocity-rescale) update
applied per mode:
 
    p_nu  <-  sqrt(1 - a) * p_nu  +  sqrt(a) * xi_nu * sqrt(kT_modal)
 
where xi_nu ~ N(0,1) is a fresh Gaussian draw and a in (0,1] is the coupling
strength for that application. `a` is set from a relaxation time tau by
a = 1 - exp(-dt/tau): small a (large tau) is a gentle bath, a -> 1 redraws the
momentum entirely each step. The update relaxes p_nu toward the canonical
distribution while preserving detailed balance.
 
BAND SELECTION. The update is applied only where a boolean mask is True:
 
    mask = band_mask  AND  (modes chosen by --modes)
 
so only the selected, active modes are kicked; all other modes (inactive, or
active-but-unselected) have their momentum passed through unchanged and keep
evolving under the bare FIMD dynamics. This is what makes it *per-mode*: one
True entry thermostats one mode while the rest of the band moves freely.
 
SYMMETRIC PLACEMENT. To keep the integrator second-order and time-symmetric, the
OU update is split around the conservative FIMD step: a HALF-strength kick is
applied to p before FIMD's kick-drift-kick, and another HALF-strength kick after
it (Strang splitting). The half strength is a_half = 1 - sqrt(1 - a). This is
done by WRAPPING the package's FIMDynamics.step(): the parent step (the exact,
validated harmonic drift + residual-force kick) runs untouched between the two
OU half-kicks, and FIMD's own built-in thermostat is left off so the OU update
is the only thermostat acting. The FIMD package (core.py) is not modified.
 
 
HOW THE RUN IS SET UP
---------------------
The FIMD basis (active-mode selection, mode matrix, reference geometry and force)
is produced by the package's own run_fimd_from_xyz, which minimises the geometry,
thermalises, runs a reference MD, and selects the active modes from the FFT of
the projected trajectory. The script uses the returned in-memory basis object and
attaches the thermostatted integrator to it, so the dynamics are exactly the
package's dynamics plus the per-mode OU thermostat.
 
 
OUTPUT (in --out-dir)
---------------------
  nvt.traj, nvt.xyz          thermostatted trajectory (.xyz is clean 4-column,
                             VMD-safe), sampled every 10 steps
  nvt.log                    FIMD log
  nvt_modal_energies.npz     per-mode kinetic energies, the thermostat mask,
                             and the band mask
  _pkg_reference/            the package pipeline's own outputs + basis
 
Sanity: coupled modes should average to kT/2 (printed ratio ~1.0); free
(unselected) modes should keep a physical per-mode energy (~1e-2 eV). Keep the
--band floor off zero (e.g. 100 cm^-1): near-free modes below ~100 cm^-1
destabilise the propagation.
 
 
USAGE
-----
  JAX_ENABLE_X64=1 python run_ou_thermostat.py \
      --xyz molecule.xyz --calculator so3lr --band 100 1000 \
      --modes 7 --T 300 --tau 25 --out-dir ou_run
 
  --band LO HI   active window (cm^-1); keep LO off zero.
  --modes k ...  active-mode indices (frequency-sorted) to thermostat; omit for
                 the whole active band.
  --T T          temperature (K).   --tau FS  OU relaxation time (fs).
  --dt FS        FIMD timestep.      --nvt-steps N  run length.
"""
from __future__ import annotations
import argparse, inspect, os, sys

os.environ.setdefault("JAX_ENABLE_X64", "1")
import warnings, logging
warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
logging.disable(logging.WARNING)
for _n in ("MLFF", "root", "absl", "jax", "jax._src", "so3lr", "mlff", "glp"):
    _lg = logging.getLogger(_n); _lg.setLevel(logging.ERROR); _lg.propagate = False

import numpy as np
from ase import units
from ase.io import read, Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

from fimd import run_fimd_from_xyz
from fimd.core import FIMDBasis, FIMDynamics, EV_TO_AMU_A2_FS2, radfs_to_cm1


# --------------------------------------------------------------------------- #
# OU thermostat helpers (matched to core.py units: self.p target variance is
# kT_modal = kB*T*EV_TO_AMU_A2_FS2, same as the package's own Langevin).
# --------------------------------------------------------------------------- #
def modal_kT(T):
    return units.kB * T * EV_TO_AMU_A2_FS2

def coupling_from_tau(dt_fs, tau_fs):
    return float(1.0 - np.exp(-dt_fs / tau_fs))

def half_coupling(alpha):
    return float(1.0 - np.sqrt(1.0 - alpha))


class OUFIMDynamics(FIMDynamics):
    """FIMDynamics + masked OU thermostat, WRAPPING the parent step (not copying).

    NVE (ou_temperature None) => exactly super().step().
    NVT => OU half-kick on self.p, super().step(), OU half-kick -- with the
    parent's own thermostat disabled (parent temperature=None).
    """
    def configure_ou(self, temperature=None, thermostat_modes=None, tau=25.0, seed=0):
        self.ou_temperature = temperature
        if thermostat_modes is None:
            self.thermostat_mask = self.band_mask.copy()
        else:
            m                                          = np.zeros_like(self.band_mask)
            m[np.asarray(thermostat_modes, dtype=int)] = True
            self.thermostat_mask                       = m & self.band_mask
        self.ou_tau     = float(tau)
        self.rng        = np.random.default_rng(seed)
        return self

    def _ou(self, p, alpha):
        kT_modal = modal_kT(float(self.ou_temperature))
        xi       = self.rng.standard_normal(len(p)).astype(self.dtype)
        sigma    = np.sqrt(self.dtype.type(kT_modal)).astype(self.dtype)
        a        = self.dtype.type(alpha)
        p_new    = np.sqrt(self.dtype.type(1.0) - a) * p + np.sqrt(a) * (sigma * xi)
        return np.where(self.thermostat_mask, p_new, p).astype(self.dtype, copy=False)

    def step(self, forces=None):
        if getattr(self, "ou_temperature", None) is None:
            return super().step(forces=forces)
        dt_fs  = float(self.dt / units.fs)
        ha     = half_coupling(coupling_from_tau(dt_fs, self.ou_tau))
        self.p = self._ou(self.p, ha)
        out    = super().step(forces=forces)
        self.p = self._ou(self.p, ha)
        return out


# --------------------------------------------------------------------------- #
def make_calc(kind):
    if kind == "so3lr":
        from so3lr import So3lrCalculator
        return So3lrCalculator(calculate_stress=False, lr_cutoff=1000.0, dtype=np.float64)
    if kind == "mace":
        from mace.calculators import mace_off
        return mace_off(model="small", device="cpu", default_dtype="float64")
    raise SystemExit(f"unknown calculator {kind}")


def call_pipeline(xyz, calculator, band, out_dir, extra):
    """Call run_fimd_from_xyz, mapping our intent onto its real signature."""
    params = set(inspect.signature(run_fimd_from_xyz).parameters)
    cand   = {
        "xyz": (["xyz_file", "input", "input_xyz", "xyz"], xyz),
        "calc": (["calculator", "calc"], calculator),
        "band": (["band"], tuple(band)),
        "out": (["output_dir", "output", "out_dir"], out_dir),
        "temperature": (["temperature", "fimd_temperature", "fimd_temperature_K"], None),  # NVE
        # we only need the BASIS from stage A, so keep the pipeline's own FIMD run
        # short if the arg exists (the basis is built before any FIMD steps).
        "nsteps": (["nsteps", "fimd_steps", "n_steps", "steps"], 1),
    }
    kwargs = {}
    for _c, (names, val) in cand.items():
        hit = next((n for n in names if n in params), None)
        if hit:
            kwargs[hit] = val
    for k, v in extra.items():
        if k in params:
            kwargs[k] = v
    print(f"    run_fimd_from_xyz({', '.join(f'{k}={v!r}' for k,v in kwargs.items())})")
    return run_fimd_from_xyz(**kwargs)


def load_basis_from_result(result, out_dir):
    """Get the IN-MEMORY basis from the pipeline result.

    CRITICAL: do NOT reload from a saved .npz -- FIMDBasis.save() does not persist
    W or K (the mode matrix and effective Hessian), so a reloaded basis has
    malformed W/K and the dynamics diverge immediately. The pipeline returns a
    FIMDynamics object that holds the live basis with intact W/K; use that.
    """
    # run_fimd_from_xyz returns a FIMDynamics (or an object exposing .basis)
    if hasattr(result, "basis") and result.basis is not None:
        return result.basis
    # some versions may return the basis directly
    if isinstance(result, FIMDBasis):
        return result
    raise RuntimeError(
        "Could not get the in-memory basis from run_fimd_from_xyz's result. "
        "It returned: " + repr(type(result)) + ". Do NOT fall back to loading "
        "basis.npz -- save() drops W/K and the reloaded basis diverges.")


def clean_xyz(md, atoms, path, interval=10):
    sym = atoms.get_chemical_symbols(); fh = open(path, "w")
    def w():
        pos = atoms.get_positions(); fh.write(f"{len(sym)}\n\n")
        for s, (x, y, z) in zip(sym, pos):
            fh.write(f"{s} {x:.6f} {y:.6f} {z:.6f}\n")
        fh.flush()
    md.attach(w, interval=interval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xyz", required=True)
    ap.add_argument("--calculator", default="so3lr")
    ap.add_argument("--band", nargs=2, type=float, default=[100.0, 1000.0])
    ap.add_argument("--modes", nargs="*", type=int, default=None)
    ap.add_argument("--T", type=float, default=300.0)
    ap.add_argument("--tau", type=float, default=25.0)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--nvt-steps", type=int, default=20000)
    ap.add_argument("--out-dir", default="ou_run")
    a, unknown = ap.parse_known_args()

    os.makedirs(a.out_dir, exist_ok=True)
    print(f"[0] outputs -> {a.out_dir}/")
    print(f"    run_fimd_from_xyz signature: {inspect.signature(run_fimd_from_xyz)}")

    # [A] package builds a KNOWN-GOOD basis via its own validated pipeline (NVE)
    print("[A] package pipeline -> basis (minimise/thermalise/MD/FFT/basis) ...")
    ref_dir = os.path.join(a.out_dir, "_pkg_reference")
    result  = call_pipeline(a.xyz, make_calc(a.calculator), tuple(a.band), ref_dir, extra={})
    basis   = load_basis_from_result(result, ref_dir)
    freq    = np.asarray([radfs_to_cm1(w) for w in basis.omega])
    active  = np.where(basis.band_mask)[0]; active = active[np.argsort(freq[active])]
    print(f"    basis: {len(active)} active modes, "
          f"{freq[active].min():.0f}-{freq[active].max():.0f} cm^-1")

    # [B] NVT with OU thermostat on the package basis
    tag = "wholeband" if a.modes is None else "modes_" + "_".join(map(str, a.modes))
    print(f"[B] NVT OU thermostat ({tag}, {a.nvt_steps} steps) ...")
    atoms = read(a.xyz); 
    if a.calculator == "so3lr": atoms.info["charge"] = 0.0
    atoms.calc = make_calc(a.calculator)
    MaxwellBoltzmannDistribution(atoms, temperature_K=a.T); Stationary(atoms)

    md = OUFIMDynamics(atoms, basis, timestep_fs=a.dt, temperature=None,
                       logfile=os.path.join(a.out_dir, "nvt.log"))
    # translate --modes (indices into sorted active list) to global mode indices
    global_modes = None if a.modes is None else [int(active[k]) for k in a.modes]
    md.configure_ou(temperature=a.T, thermostat_modes=global_modes, tau=a.tau, seed=0)
    tr = Trajectory(os.path.join(a.out_dir, "nvt.traj"), "w", atoms); md.attach(tr.write, interval=10)
    clean_xyz(md, atoms, os.path.join(a.out_dir, "nvt.xyz"))
    acc = np.zeros_like(md.get_modal_energies()); n = 0
    def samp():
        nonlocal acc, n
        acc += md.get_modal_energies(); n += 1
    md.attach(samp, interval=4)
    md.run(a.nvt_steps)
    ke = acc / max(n, 1)

    kT_half = 0.5 * units.kB * a.T
    mm = md.thermostat_mask
    np.savez(os.path.join(a.out_dir, "nvt_modal_energies.npz"),
             frequencies_cm1=freq, ke=ke, thermostat_mask=mm, band_mask=basis.band_mask)
    print("\nNVT result:")
    print(f"  target kT/2 = {kT_half:.4e} eV")
    if mm.any():
        print(f"  coupled modes mean KE = {ke[mm].mean():.4e}  ratio {ke[mm].mean()/kT_half:.3f}")
    free = basis.band_mask & ~mm
    if free.any():
        print(f"  free (evolving) modes mean KE = {ke[free].mean():.4e}")
    print(f"\nsaved: nvt.traj, nvt.xyz, nvt_modal_energies.npz (basis in {ref_dir}/)")


if __name__ == "__main__":
    main()