#!/usr/bin/env python3
"""
prepare_xyz.py  –  Convert AMBER TIP4P/2005 PDB → extended XYZ for SO3LR.

Reads ../water.pdb (or --pdb <path>), strips the virtual EP/MW site,
and writes an extxyz with Lattice from the CRYST1 record.

SO3LR only supports physical atoms (H, O); the TIP4P/2005 extra-point
(atom name EP or MW) is discarded.

Output: water300_init.xyz  (or --out <path>)
"""

import argparse
import os

# Element map: PDB atom name → element symbol
_NAME_TO_ELEM = {
    'OW':  'O',
    'O':   'O',
    'HW1': 'H',
    'HW2': 'H',
    'HW3': 'H',
    'H1':  'H',
    'H2':  'H',
}
_SKIP_NAMES = {'EP', 'MW', 'LP'}   # virtual sites


def parse_pdb(pdb_path):
    box = None
    atoms = []  # list of (elem, x, y, z)

    with open(pdb_path) as fh:
        for line in fh:
            rec = line[:6].strip()

            if rec == 'CRYST1':
                a = float(line[6:15])
                b = float(line[15:24])
                c = float(line[24:33])
                box = (a, b, c)
                continue

            if rec in ('ATOM', 'HETATM'):
                name = line[12:16].strip()
                if name in _SKIP_NAMES:
                    continue
                elem = _NAME_TO_ELEM.get(name)
                if elem is None:
                    # fall back to the element column (cols 76-77) if present
                    elem_col = line[76:78].strip() if len(line) > 76 else ''
                    elem = elem_col if elem_col in ('O', 'H') else None
                if elem is None:
                    print(f"  Warning: skipping unrecognised atom name '{name}'")
                    continue
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                atoms.append((elem, x, y, z))

    return box, atoms


def write_extxyz(out_path, box, atoms):
    a, b, c = box
    lattice = f'{a:.6f} 0.0 0.0 0.0 {b:.6f} 0.0 0.0 0.0 {c:.6f}'
    with open(out_path, 'w') as fh:
        fh.write(f'{len(atoms)}\n')
        fh.write(
            f'Lattice="{lattice}" '
            f'Properties=species:S:1:pos:R:3 '
            f'pbc="T T T"\n'
        )
        for elem, x, y, z in atoms:
            fh.write(f'{elem:2s}  {x:14.8f}  {y:14.8f}  {z:14.8f}\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pdb', default=os.path.join('..', 'water.pdb'),
                    help='Input AMBER PDB (default: ../water.pdb)')
    ap.add_argument('--out', default='water300_init.xyz',
                    help='Output extxyz (default: water300_init.xyz)')
    args = ap.parse_args()

    print(f'Reading: {args.pdb}')
    box, atoms = parse_pdb(args.pdb)

    if box is None:
        raise RuntimeError('No CRYST1 record found in PDB – cannot determine cell.')

    n_O = sum(1 for e, *_ in atoms if e == 'O')
    n_H = sum(1 for e, *_ in atoms if e == 'H')
    print(f'  Box:  {box[0]:.4f} x {box[1]:.4f} x {box[2]:.4f} Å')
    print(f'  Atoms: {n_O} O + {n_H} H = {len(atoms)} total  ({n_O} water molecules)')

    write_extxyz(args.out, box, atoms)
    print(f'Written: {args.out}')


if __name__ == '__main__':
    main()
