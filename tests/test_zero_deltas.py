#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Round-trip test: with ALL deltas set to zero, the PGW output must equal the
input up to GRIB repacking noise (one packing unit).

Checks the internal consistency of the package (q -> RH -> q, ps adjustment,
vertical interpolation, surface-file coupling, GRIB writing).

Usage:
  python3 tests/test_zero_deltas.py <model_level_grib> <deltas_on_grid.nc> <tmp_dir> [<surface_grib>]
"""
import os
import sys
from datetime import datetime

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import eccodes as ec                                          # noqa: E402
from pgw_ml.apply import PGWOptions, apply_pgw_to_file        # noqa: E402
from pgw_ml.deltas import Deltas                              # noqa: E402
from pgw_ml.surface import apply_pgw_to_sfc_file, save_state  # noqa: E402

src, deltas_nc, tmp = sys.argv[1:4]
sfc = sys.argv[4] if len(sys.argv) > 4 else None
os.makedirs(tmp, exist_ok=True)
ds = xr.open_dataset(deltas_nc).load()
for v in ds.data_vars:
    ds[v][:] = 0.0
zero_nc = os.path.join(tmp, "deltas_zero.nc")
ds.to_netcdf(zero_nc)

opt = PGWOptions(month="auto", perturb_wind=True)
dl = Deltas(zero_nc)
pairs = [(src, os.path.join(tmp, "zero_ml.grb"))]
diag = apply_pgw_to_file(src, pairs[0][1], dl, opt)
st = diag.pop("_state")
print("diagnostics:", {k: diag[k] for k in ("ps_iterations", "dps_mean", "dT_lowest_mean")})
if sfc:
    sdir = os.path.join(tmp, "_state")
    save_state(sdir, datetime.fromisoformat(diag["valid"]), st)
    pairs.append((sfc, os.path.join(tmp, "zero_sfc.grb")))
    dg = apply_pgw_to_sfc_file(sfc, pairs[1][1], dl, opt, sdir)
    print(f"surface: {dg['n_times']} times, {len(dg['coupled'])} coupled")

worst = {}
for a, b in pairs:
    with open(a, "rb") as fa, open(b, "rb") as fb:
        while True:
            ga, gb = ec.codes_grib_new_from_file(fa), ec.codes_grib_new_from_file(fb)
            if ga is None:
                break
            sn = ec.codes_get(ga, "shortName")
            va, vb = ec.codes_get_values(ga), ec.codes_get_values(gb)
            unit = (2.0 ** ec.codes_get(gb, "binaryScaleFactor")) * \
                   (10.0 ** -ec.codes_get(gb, "decimalScaleFactor")) + 1e-12
            rel = np.abs(va - vb).max() / unit
            worst[sn] = max(worst.get(sn, 0.0), rel)
            ec.codes_release(ga)
            ec.codes_release(gb)

ok = True
for sn, r in sorted(worst.items()):
    flag = "OK" if r <= 1.0 else "FAIL"
    ok &= r <= 1.0
    print(f"  {sn:6s} max|Δ| = {r:.3f} packing units  [{flag}]")
print("TEST PASSED" if ok else "TEST FAILED")
sys.exit(0 if ok else 1)
