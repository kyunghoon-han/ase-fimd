"""Command-line interface for the modular FIMD package."""

from __future__ import annotations

import argparse

from .calculators import available_calculators, parse_calculator_kwargs
from .protocols import run_multiband_fimd_from_xyz
from .workflow import run_fimd_from_xyz


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fimd",
        description="Run Fourier-Integrator Molecular Dynamics from ASE-readable structures.",
    )
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="run FIMD from an XYZ/ASE-readable structure")
    p_run.add_argument("--input", "-i", required=True, help="input structure, e.g. molecule.xyz")
    p_run.add_argument("--calculator", "-c", default="emt", help="calculator name or module:object factory")
    p_run.add_argument("--calculator-kwargs", default=None, help="JSON or key=value,key=value kwargs for calculator")
    p_run.add_argument("--reference", default=None, help="optional minimised/reference structure")
    p_run.add_argument("--trajectory", default=None, help="optional external trajectory for band identification")
    p_run.add_argument("--band-min", type=float, required=True, help="lower band edge in cm^-1")
    p_run.add_argument("--band-max", type=float, required=True, help="upper band edge in cm^-1")
    p_run.add_argument("--output", "-o", default="fimd_output", help="output directory")
    p_run.add_argument("--precision", choices=["float32", "float64", "float", "double"], default="float64")
    p_run.add_argument("--velocity-unit", default="angstrom/ps", choices=["angstrom/ps", "angstrom/fs", "ase"])

    p_run.add_argument("--no-minimise", action="store_true", help="use input structure as reference if --reference is absent")
    p_run.add_argument("--minimise-fmax", type=float, default=0.01)
    p_run.add_argument("--minimise-steps", type=int, default=1000)
    p_run.add_argument("--thermalise-steps", type=int, default=0, help="NVT Langevin equilibration steps before reference MD (0 = skip)")
    p_run.add_argument("--thermalise-temperature", type=float, default=300.0, help="target temperature for NVT thermalisation in K")
    p_run.add_argument("--thermalise-friction", type=float, default=0.01, help="Langevin friction for thermalisation in 1/fs")
    p_run.add_argument("--hessian-step", type=float, default=0.001)
    p_run.add_argument("--basis-source", choices=["hessian", "covariance"], default="hessian", help="how to build W and omega when an external --trajectory is given")
    p_run.add_argument("--no-remove-rigid-motion", action="store_true", help="do not Kabsch-align the reference trajectory before modal projection")
    p_run.add_argument("--max-initial-displacement", type=float, default=None, help="optional cap on reconstructed initial band displacement in Å")
    p_run.add_argument("--initial-temperature", type=float, default=None, help="replace trajectory modal momenta with Maxwell-Boltzmann momenta at this K")
    p_run.add_argument("--trajectory-format", choices=["xyz", "extxyz", "traj", "both", "none"], default="xyz", help="FIMD trajectory output: 'xyz' plain (VMD-safe), 'extxyz' with velocities, 'traj' ASE binary, 'both', or 'none'")
    p_run.add_argument("--keep-intermediate-traj", action="store_true", help="keep minimise.traj / reference_md.traj instead of deleting them")

    p_run.add_argument("--reference-md-steps", type=int, default=1000)
    p_run.add_argument("--reference-md-dt", type=float, default=1.0)
    p_run.add_argument("--reference-temperature", type=float, default=300.0)
    p_run.add_argument("--reference-ensemble", choices=["nve", "nvt"], default="nve")
    p_run.add_argument("--reference-save-interval", type=int, default=1)

    p_run.add_argument("--fimd-steps", type=int, default=1000)
    p_run.add_argument("--fimd-dt", type=float, default=1.0)
    p_run.add_argument("--fimd-temperature", type=float, default=None)
    p_run.add_argument("--fimd-friction", type=float, default=0.01)
    p_run.add_argument("--save-interval", type=int, default=10)
    p_run.add_argument("--seed", type=int, default=None)
    p_run.add_argument("--quiet", action="store_true")


    p_bands = sub.add_parser("run-bands", help="minimise, equilibrate, then run several FIMD bands")
    p_bands.add_argument("--input", "-i", required=True, help="input structure, e.g. molecule.xyz")
    p_bands.add_argument("--calculator", "-c", default="mace_mp", help="calculator name or module:object factory")
    p_bands.add_argument("--calculator-kwargs", default=None, help="JSON or key=value,key=value kwargs for calculator")
    p_bands.add_argument("--bands", default="0:200,200:600,600:1200,1450:1750", help="comma-separated bands, e.g. '0:200,200:600,600:1200,1450:1750'")
    p_bands.add_argument("--output", "-o", default="fimd_200ps_multiband", help="output directory")
    p_bands.add_argument("--precision", choices=["float32", "float64", "float", "double"], default="float64")

    p_bands.add_argument("--minimise-fmax", type=float, default=0.02)
    p_bands.add_argument("--minimise-steps", type=int, default=2000)
    p_bands.add_argument("--conventional-md-dt", type=float, default=0.5, help="ordinary ASE MD timestep in fs")
    p_bands.add_argument("--temperature", type=float, default=300.0, help="thermalisation/equilibration temperature in K")
    p_bands.add_argument("--thermalisation-ps", type=float, default=20.0)
    p_bands.add_argument("--equilibration-ps", type=float, default=20.0)
    p_bands.add_argument("--thermalisation-ensemble", choices=["nve", "nvt"], default="nvt")
    p_bands.add_argument("--equilibration-ensemble", choices=["nve", "nvt"], default="nvt")
    p_bands.add_argument("--friction", type=float, default=0.01, help="ordinary MD Langevin friction in 1/fs")
    p_bands.add_argument("--thermalisation-save-interval", type=int, default=200)
    p_bands.add_argument("--equilibration-save-interval", type=int, default=20)

    p_bands.add_argument("--production-ps", type=float, default=200.0, help="FIMD production length per band in ps")
    p_bands.add_argument("--fimd-dt", type=float, default=0.5, help="FIMD timestep in fs")
    p_bands.add_argument("--fimd-temperature", type=float, default=None, help="optional FIMD NVT temperature; omit for NVE")
    p_bands.add_argument("--fimd-friction", type=float, default=0.01)
    p_bands.add_argument("--fimd-save-interval", type=int, default=200)
    p_bands.add_argument("--hessian-step", type=float, default=0.001)
    p_bands.add_argument("--max-initial-displacement", type=float, default=None, help="optional cap on reconstructed initial band displacement in Å")
    p_bands.add_argument("--seed", type=int, default=12345)
    p_bands.add_argument("--quiet", action="store_true")

    sub.add_parser("calculators", help="list calculator shortcuts")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "calculators":
        for name, desc in available_calculators().items():
            print(f"{name:14s} {desc}")
        return 0


    if args.command == "run-bands":
        kwargs = parse_calculator_kwargs(args.calculator_kwargs)
        result = run_multiband_fimd_from_xyz(
            xyz_file=args.input,
            calculator=args.calculator,
            calculator_kwargs=kwargs,
            bands=args.bands,
            output_dir=args.output,
            minimise_fmax=args.minimise_fmax,
            minimise_steps=args.minimise_steps,
            conventional_md_dt_fs=args.conventional_md_dt,
            temperature_K=args.temperature,
            thermalisation_ps=args.thermalisation_ps,
            equilibration_ps=args.equilibration_ps,
            thermalisation_ensemble=args.thermalisation_ensemble,
            equilibration_ensemble=args.equilibration_ensemble,
            friction_per_fs=args.friction,
            thermalisation_save_interval=args.thermalisation_save_interval,
            equilibration_save_interval=args.equilibration_save_interval,
            production_ps=args.production_ps,
            fimd_dt_fs=args.fimd_dt,
            fimd_temperature_K=args.fimd_temperature,
            fimd_friction_per_fs=args.fimd_friction,
            fimd_save_interval=args.fimd_save_interval,
            hessian_step=args.hessian_step,
            max_initial_displacement=args.max_initial_displacement,
            precision=args.precision,
            seed=args.seed,
            verbose=not args.quiet,
        )
        print("Multi-band FIMD protocol complete.")
        print(f"  Reference: {result.prepared_reference.reference_file}")
        print(f"  Equilibration trajectory: {result.prepared_reference.equilibration_traj}")
        print(f"  Metadata: {result.metadata_file}")
        for slug, band_result in result.band_results.items():
            print(f"  {slug}: {band_result.results_file}")
        return 0

    if args.command in {None, "run"}:
        if args.command is None:
            parser.print_help()
            return 2
        kwargs = parse_calculator_kwargs(args.calculator_kwargs)
        result = run_fimd_from_xyz(
            xyz_file=args.input,
            calculator=args.calculator,
            calculator_kwargs=kwargs,
            band=(args.band_min, args.band_max),
            output_dir=args.output,
            reference_file=args.reference,
            trajectory_file=args.trajectory,
            minimise=not args.no_minimise,
            minimise_fmax=args.minimise_fmax,
            minimise_steps=args.minimise_steps,
            thermalise_steps=args.thermalise_steps,
            thermalise_temperature_K=args.thermalise_temperature,
            thermalise_friction_per_fs=args.thermalise_friction,
            reference_md_steps=args.reference_md_steps,
            reference_md_dt_fs=args.reference_md_dt,
            reference_temperature_K=args.reference_temperature,
            reference_ensemble=args.reference_ensemble,
            reference_save_interval=args.reference_save_interval,
            fimd_steps=args.fimd_steps,
            fimd_dt_fs=args.fimd_dt,
            fimd_temperature_K=args.fimd_temperature,
            fimd_friction_per_fs=args.fimd_friction,
            save_interval=args.save_interval,
            hessian_step=args.hessian_step,
            basis_source=args.basis_source,
            remove_rigid_motion=not args.no_remove_rigid_motion,
            max_initial_displacement=args.max_initial_displacement,
            initial_temperature_K=args.initial_temperature,
            precision=args.precision,
            velocity_unit=args.velocity_unit,
            trajectory_format=args.trajectory_format,
            keep_intermediate_traj=args.keep_intermediate_traj,
            seed=args.seed,
            verbose=not args.quiet,
        )
        print("FIMD run complete.")
        print(f"  Results:    {result.results_file}")
        print(f"  Basis:      {result.basis_file}")
        print(f"  Trajectory: {result.trajectory_file}")
        print(f"  Final XYZ:  {result.final_xyz}")
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())