"""Calculator factory for modular FIMD workflows."""

from __future__ import annotations

import importlib
import json
from typing import Any, Dict, Mapping, Optional


def parse_calculator_kwargs(value: Optional[str]) -> Dict[str, Any]:
    """Parse calculator keyword arguments from JSON or comma-separated key=value."""
    if value is None or str(value).strip() == "":
        return {}
    text = str(value).strip()
    if text.startswith("{"):
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("Calculator kwargs JSON must decode to an object/dict.")
        return parsed

    out: Dict[str, Any] = {}
    for item in text.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError("Use JSON or comma-separated key=value calculator kwargs.")
        key, raw = item.split("=", 1)
        key = key.strip()
        raw = raw.strip()
        try:
            out[key] = json.loads(raw)
        except Exception:
            out[key] = raw
    return out


def get_calculator(name: Any, **kwargs: Any) -> Any:
    """
    Construct an ASE calculator by name.

    Built-ins currently supported:
      - ``emt``
      - ``lj`` / ``lennardjones``
      - ``morse``
      - ``eam``
      - ``tip3p``
      - ``tblite`` / ``xtb`` via ``tblite.ase.TBLite``
      - ``xtb-ase`` via ``xtb.ase.calculator.XTB``
      - ``mace_mp`` / ``mace``
      - ``orca``

    For custom calculators, pass ``module:object``. The object can be a class
    or a factory function returning an ASE calculator.
    """
    if not isinstance(name, str):
        return name
    key = name.strip()
    lower = key.lower().replace("_", "-")
    # Optional precision hint threaded from the workflow; consumed only by
    # calculators that support a dtype (currently MACE). Pop it here so it is
    # never forwarded to calculators that would reject an unknown kwarg.
    _precision_hint = kwargs.pop("_precision_hint", None)

    if ":" in key:
        module_name, object_name = key.split(":", 1)
        module = importlib.import_module(module_name)
        obj = getattr(module, object_name)
        return obj(**kwargs)

    if lower == "emt":
        from ase.calculators.emt import EMT
        return EMT(**kwargs)

    if lower in {"lj", "lennardjones", "lennard-jones"}:
        from ase.calculators.lj import LennardJones
        return LennardJones(**kwargs)

    if lower == "morse":
        from ase.calculators.morse import MorsePotential
        return MorsePotential(**kwargs)

    if lower == "eam":
        from ase.calculators.eam import EAM
        return EAM(**kwargs)

    if lower == "tip3p":
        from ase.calculators.tip3p import TIP3P
        return TIP3P(**kwargs)

    if lower in {"tblite", "xtb", "gfn2-xtb", "gfn1-xtb", "ipea1-xtb"}:
        try:
            from tblite.ase import TBLite
        except ImportError as exc:
            raise ImportError(
                "The 'tblite' calculator requires tblite-python. Install with "
                "conda install -c conda-forge tblite-python or pip install tblite."
            ) from exc
        method = kwargs.pop("method", None)
        if method is None:
            method = "GFN2-xTB" if lower in {"tblite", "xtb", "gfn2-xtb"} else lower.upper()
        # tblite prints SCF cycle tables to stdout by default; silence unless
        # the user explicitly asks for output via verbosity=...
        kwargs.setdefault("verbosity", 0)
        return TBLite(method=method, **kwargs)

    if lower in {"xtb-ase", "ase-xtb"}:
        try:
            from xtb.ase.calculator import XTB
        except ImportError as exc:
            raise ImportError("The 'xtb-ase' calculator requires the xtb Python package.") from exc
        return XTB(**kwargs)

    if lower in {"mace", "mace-mp", "mace_mp"}:
        try:
            from mace.calculators import mace_mp
        except ImportError as exc:
            raise ImportError("The 'mace_mp' calculator requires mace-torch.") from exc
        # MACE's model precision should follow the run precision: float32 is
        # faster (MACE's MD default) but its float32 force noise lands in the
        # low-frequency band FIMD targets, so float64 is preferable for
        # band-limited work. We honour an explicit default_dtype in
        # calculator_kwargs; otherwise use the precision hint threaded from the
        # workflow (which defaults to float64). _precision_hint is popped so it
        # never reaches mace_mp itself.
        hint = _precision_hint
        if "default_dtype" not in kwargs:
            kwargs["default_dtype"] = "float32" if str(hint) == "float32" else "float64"
        return mace_mp(**kwargs)

    if lower in {"so3lr", "so3lr-ase", "solar"}:
        # SO3LR is JAX-based and only numerically correct in float64. JAX
        # defaults to float32 and must be switched to x64 *before* it
        # initializes, so we enable it at import time here as well as passing
        # dtype=float64 to the calculator.
        import os as _os
        _os.environ.setdefault("JAX_ENABLE_X64", "1")
        try:
            import jax as _jax
            _jax.config.update("jax_enable_x64", True)
        except Exception:
            pass
        try:
            import numpy as _np
            from so3lr import So3lrCalculator
        except ImportError as exc:
            raise ImportError(
                "The 'so3lr' calculator requires the so3lr package and JAX, which "
                "need Python >= 3.12. In a 3.12+ environment install with:\n"
                "  pip install 'jax==0.5.3'\n"
                "  pip install git+https://github.com/general-molecular-simulations/so3lr"
            ) from exc
        # Sensible defaults: gas-phase long-range cutoff and float64 precision,
        # both overridable via calculator_kwargs.
        kwargs.setdefault("lr_cutoff", 1000.0)
        kwargs.setdefault("calculate_stress", False)
        kwargs.setdefault("dtype", _np.float64)
        return So3lrCalculator(**kwargs)

    if lower == "orca":
        from ase.calculators.orca import ORCA
        return ORCA(**kwargs)

    raise ValueError(
        f"Unknown calculator {name!r}. Use a built-in name or a custom 'module:object' factory."
    )


def available_calculators() -> Mapping[str, str]:
    return {
        "emt": "ASE EMT toy metallic potential",
        "lj": "ASE Lennard-Jones potential",
        "morse": "ASE Morse potential",
        "eam": "ASE EAM potential; pass potential=...",
        "tip3p": "ASE TIP3P water calculator",
        "tblite": "tblite xTB calculator",
        "xtb-ase": "xtb.ase.calculator.XTB calculator",
        "mace_mp": "MACE-MP calculator",
        "so3lr": "SO3LR JAX ML force field (float64; needs so3lr + jax)",
        "orca": "ASE ORCA calculator",
        "module:object": "custom calculator factory or class",
    }