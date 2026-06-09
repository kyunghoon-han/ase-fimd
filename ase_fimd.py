#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backward-compatible shim for the modular ``fimd`` package.

Existing scripts that used

    from ase_fimd import FIMDBasis, FIMDynamics, load_trajectory

can continue to work, while new code should prefer

    from fimd import FIMDBasis, FIMDynamics, run_fimd_from_xyz
"""

from fimd.core import *  # noqa: F401,F403
