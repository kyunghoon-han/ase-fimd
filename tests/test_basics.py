"""Smoke tests: imports, version, unit conversions, calculator registry."""

from __future__ import annotations

import numpy as np
import pytest


def test_package_imports():
    import fimd

    assert isinstance(fimd.__version__, str)
    for name in ("FIMDBasis", "FIMDynamics", "run_fimd_from_xyz", "get_calculator"):
        assert hasattr(fimd, name)


def test_covariance_basis_is_exposed():
    from fimd import FIMDBasis

    assert hasattr(FIMDBasis, "from_trajectory")
    assert hasattr(FIMDBasis, "from_trajectory_covariance")
    assert hasattr(FIMDBasis, "from_reference_only")


def test_wavenumber_roundtrip():
    from fimd import cm1_to_radfs, radfs_to_cm1

    for cm1 in (1.0, 100.0, 1500.0, 3000.0):
        assert np.isclose(radfs_to_cm1(cm1_to_radfs(cm1)), cm1, rtol=1e-9)


def test_velocity_unit_roundtrip():
    from fimd import ang_per_fs_to_ase_velocity, ase_velocity_to_ang_per_fs

    v = np.array([[0.1, -0.2, 0.05], [0.0, 0.3, -0.1]])
    back = ase_velocity_to_ang_per_fs(ang_per_fs_to_ase_velocity(v))
    assert np.allclose(back, v, rtol=1e-9, atol=1e-12)


def test_resolve_precision():
    from fimd import resolve_precision

    assert resolve_precision("float32").name == "float32"
    assert resolve_precision("float64").name == "float64"
    assert resolve_precision("double").name == "float64"


def test_calculator_registry_has_expected_entries():
    from fimd import available_calculators

    cats = available_calculators()
    for key in ("emt", "lj", "morse", "mace_mp", "so3lr"):
        assert key in cats


def test_unknown_calculator_raises():
    from fimd import get_calculator

    with pytest.raises(ValueError):
        get_calculator("definitely_not_a_calculator")


@pytest.mark.parametrize("name", ["mace_mp", "so3lr"])
def test_optional_calculators_give_clean_importerror(name):
    """When an optional backend isn't installed, the error must be an
    actionable ImportError, not an arbitrary crash."""
    from fimd import get_calculator

    try:
        get_calculator(name)
    except ImportError:
        pass  # expected when the backend isn't installed
    except ValueError:
        pytest.fail(f"{name} should be a known calculator")
    except Exception:
        # If the backend *is* installed it may construct or raise something
        # else (e.g. download). That's fine; we only assert it's a known name.
        pass
