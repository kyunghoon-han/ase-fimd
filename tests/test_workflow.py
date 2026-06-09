"""End-to-end workflow tests on EMT: output files, formats, and logged data.

These exercise run_fimd_from_xyz the way the CLI does, on a cheap EMT cluster so
they finish quickly on CI.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase.cluster import Icosahedron
from ase.io import read, write

from fimd.workflow import run_fimd_from_xyz


@pytest.fixture
def cu_xyz(tmp_path):
    atoms = Icosahedron("Cu", noshells=2)  # 13 atoms, EMT-appropriate
    path = tmp_path / "cu13.xyz"
    write(str(path), atoms)
    return str(path)


def _run(cu_xyz, outdir, **overrides):
    kw = dict(
        xyz_file=cu_xyz,
        calculator="emt",
        band=(0.0, 200.0),
        output_dir=str(outdir),
        minimise=True,
        reference_md_steps=300,
        reference_md_dt_fs=2.0,
        reference_temperature_K=150.0,
        fimd_steps=60,
        fimd_dt_fs=1.0,
        save_interval=1,
        max_initial_displacement=0.2,
        verbose=False,
    )
    kw.update(overrides)
    return run_fimd_from_xyz(**kw)


def test_workflow_runs_and_writes_results(cu_xyz, tmp_path):
    out = tmp_path / "out"
    result = _run(cu_xyz, out)
    assert result.results_file.endswith("fimd_results.npz")
    assert (out / "fimd_results.npz").exists()
    assert (out / "fimd_basis.npz").exists()

    d = np.load(out / "fimd_results.npz", allow_pickle=True)
    # 60 steps + initial frame, with save_interval 1.
    assert d["positions"].shape[0] == 61
    assert np.all(np.isfinite(d["energies_eV"]))
    assert np.all(np.isfinite(d["band_energies_eV"]))


def test_velocities_saved_for_vacf(cu_xyz, tmp_path):
    out = tmp_path / "out"
    _run(cu_xyz, out)
    d = np.load(out / "fimd_results.npz", allow_pickle=True)
    assert "velocities_ang_per_fs" in d
    v = d["velocities_ang_per_fs"]
    assert v.shape == d["positions"].shape
    assert str(d["velocity_unit"]) == "angstrom/fs"
    assert np.all(np.isfinite(v))


def test_default_xyz_is_vmd_safe_four_columns(cu_xyz, tmp_path):
    """Default trajectory format must be plain 4-column XYZ (element + xyz),
    which VMD reads reliably. Regression: extended-XYZ extra columns froze VMD."""
    out = tmp_path / "out"
    _run(cu_xyz, out, trajectory_format="xyz")
    traj = out / "fimd_trajectory.xyz"
    assert traj.exists()
    lines = traj.read_text().splitlines()
    natoms = int(lines[0])
    # Third line is the first atom record; must be exactly 4 whitespace fields.
    first_atom = lines[2].split()
    assert len(first_atom) == 4, f"expected element+xyz, got {len(first_atom)} columns"
    # And it must still load back as a full trajectory.
    frames = read(str(traj), index=":")
    assert len(frames) == 61
    assert len(frames[0]) == natoms


def test_extxyz_format_carries_velocities(cu_xyz, tmp_path):
    out = tmp_path / "out"
    _run(cu_xyz, out, trajectory_format="extxyz")
    traj = out / "fimd_trajectory.xyz"
    lines = traj.read_text().splitlines()
    # Extended XYZ data line has more than 4 columns (pos + masses + momenta).
    assert len(lines[2].split()) > 4


def test_trajectory_format_none(cu_xyz, tmp_path):
    out = tmp_path / "out"
    _run(cu_xyz, out, trajectory_format="none")
    assert not (out / "fimd_trajectory.xyz").exists()
    assert not (out / "fimd_trajectory.traj").exists()


def test_intermediate_traj_cleaned_by_default(cu_xyz, tmp_path):
    out = tmp_path / "out"
    _run(cu_xyz, out)
    # minimise.traj / reference_md.traj should be cleaned unless kept.
    assert not (out / "minimise.traj").exists()
    assert not (out / "reference_md.traj").exists()


def test_band_energy_conservation_end_to_end(cu_xyz, tmp_path):
    out = tmp_path / "out"
    _run(cu_xyz, out, fimd_steps=200)
    d = np.load(out / "fimd_results.npz", allow_pickle=True)
    HB = d["band_energies_eV"]
    assert np.all(np.isfinite(HB))
    # EMT is smooth; band energy should be reasonably conserved.
    assert (HB.max() - HB.min()) < 1.0
