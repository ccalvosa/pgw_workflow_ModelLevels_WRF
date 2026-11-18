"""
Prepared deltas (output of step01) and temporal selection.

Format of 'deltas_on_grid.nc':
  3D  (month, plev, point): ta [K], hur [%], zg [m], ua, va [m s-1]
  2D  (month, point)      : tas, ts [K], hurs [%], ps [Pa]
  ua/va are already ROTATED to the target-grid basis when the GRIB winds are
  grid-relative (attribute 'winds_rotated_to_grid').
  'point' follows the same order as the values of the GRIB messages.
"""

from datetime import datetime

import numpy as np
import xarray as xr

VARS_3D = ("ta", "hur", "zg", "ua", "va")
VARS_2D = ("tas", "ts", "hurs", "ps")


def _mid_month_dates(year, ds):
    """Central date of each month (from the original GCM time axis)."""
    days = ds["mid_day"].values if "mid_day" in ds else np.full(12, 16)
    return [datetime(year, m, int(d)) for m, d in zip(range(1, 13), days)]


def time_weights(valid_dt, month_spec, ds):
    """
    Weights {month_index: weight} used to combine the monthly deltas.

    month_spec = 1..12  -> that month is used as is (as in the pressure-level
                           workflows: cdo seltimestep).
    month_spec = 'auto' -> linear interpolation between the two months whose
                           central dates bracket the validity time of the file
                           (avoids jumps at month boundaries in long runs).
    """
    if str(month_spec).lower() != "auto":
        m = int(month_spec)
        if not 1 <= m <= 12:
            raise ValueError(f"Month out of range: {m}")
        return {m - 1: 1.0}

    y = valid_dt.year
    centers = (_mid_month_dates(y - 1, ds)[-1:] + _mid_month_dates(y, ds)
               + _mid_month_dates(y + 1, ds)[:1])
    idx = [11] + list(range(12)) + [0]
    for k in range(len(centers) - 1):
        if centers[k] <= valid_dt <= centers[k + 1]:
            w = (valid_dt - centers[k]) / (centers[k + 1] - centers[k])
            return {idx[k]: 1.0 - w, idx[k + 1]: w}
    raise RuntimeError("Could not place the date between months")


class Deltas:
    """Access to the prepared deltas for a given validity time.

    The file is NOT loaded entirely: each call to at() reads only the (at
    most two) months needed, so several members (models + ENSMEAN) can share
    the step02 worker processes without blowing up memory.
    """

    def __init__(self, path):
        self.path = path
        self.ds = xr.open_dataset(path)
        self.plev = self.ds["plev"].values.astype(float)
        self.winds_rotated = bool(self.ds.attrs.get("winds_rotated_to_grid", 0))
        self._annual = {}

    def at(self, valid_dt, month_spec):
        w = time_weights(valid_dt, month_spec, self.ds)
        out = {}
        for v in VARS_3D + VARS_2D:
            if v in self.ds:
                da = self.ds[v]
                out[v] = sum(wi * da.isel(month=i).values.astype(float)
                             for i, wi in w.items())
        out["_weights"] = w
        return out

    def annual_mean(self, var):
        if var not in self._annual:
            self._annual[var] = self.ds[var].values.astype(float).mean(axis=0)
        return self._annual[var]

    def close(self):
        self.ds.close()
