#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 01 (single model): deltas of ONE GCM on the input IFS/ERA5 grid.

For several models + ENSMEAN use step01_prepare_ensemble.py (called by
run_PGW_WRF_ML.sh). This script is kept for one-off use.

Usage:
  python3 step01_prepare_deltas.py <ref_grib> <path_deltas> <out_dir>
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pgw_ml.prepare import read_grid, check_rotated_grid, prepare_model   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref_grib")
    ap.add_argument("path_deltas")
    ap.add_argument("out_dir")
    ap.add_argument("--margin", type=float, default=3.0,
                    help="Margin (degrees) of the GCM cropping box")
    args = ap.parse_args()

    print(f">> Reference grid: {args.ref_grib}")
    gi = read_grid(args.ref_grib)
    print(f"  {gi['gridType']} {gi['Ni']}x{gi['Nj']} = {gi['numberOfPoints']} points")
    if gi["gridType"] == "rotated_ll":
        check_rotated_grid(gi)
    out_path = os.path.join(args.out_dir, "deltas_on_grid.nc")
    info = prepare_model(gi, args.path_deltas, out_path, args.margin)
    print(f">> Saved {out_path}  ({info})")


if __name__ == "__main__":
    main()
