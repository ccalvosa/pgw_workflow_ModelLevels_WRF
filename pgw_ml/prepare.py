"""
Preparation of the deltas of ONE CMIP6 model on the input IFS/ERA5 grid.

  1. Reads the monthly deltas <var>_delta.nc (12 months, global lat-lon, on
     the native GCM grid).
  2. Fills NaNs (below the GCM orography or empty upper levels) with
     constant vertical extrapolation from the nearest valid level.
  3. Interpolates bilinearly to the target grid points, with the grid of
     each variable (some models store 3D and 2D variables on staggered
     grids, e.g., ACCESS-ESM1-5).
  4. Rotates ua/va to the grid basis if the GRIB winds are grid-relative.
  5. Writes <out_path> (dims: month, plev, point).
"""

import os

import numpy as np
import xarray as xr

from .gribio import IFSFile
from .grids import GCMToPoints, geo_to_rot, wind_rotation_matrix

VARS_3D = ["ta", "hur", "zg", "ua", "va"]
VARS_2D = ["tas", "ts", "hurs", "ps"]
REQUIRED = {"ta", "hur", "zg", "tas", "ts", "hurs"}


def read_grid(ref_grib):
    f = IFSFile(ref_grib)
    try:
        return f.grid_info()
    finally:
        f.close()


def check_rotated_grid(gi, log=print):
    """Geographical lat/lon rotated back must form a regular grid."""
    rlon, rlat = geo_to_rot(gi["lon"], gi["lat"], gi["lat_sp"], gi["lon_sp"])
    rlon = rlon.reshape(gi["Nj"], gi["Ni"])
    rlat = rlat.reshape(gi["Nj"], gi["Ni"])
    err_lon = np.abs(rlon - rlon[0:1, :]).max()      # rlon constant along columns
    err_lat = np.abs(rlat - rlat[:, 0:1]).max()      # rlat constant along rows
    log(f"  Rotated grid: rlon [{rlon.min():.3f}, {rlon.max():.3f}], "
        f"rlat [{rlat.min():.3f}, {rlat.max():.3f}]  "
        f"(regularity error: {err_lon:.2e}° / {err_lat:.2e}°)")
    if max(err_lon, err_lat) > 1.0e-3:
        raise RuntimeError("eccodes lat/lon are inconsistent with the rotated pole")


def fill_vertical_nans(arr, plev):
    """Constant downward (and upward) extrapolation of NaNs in (t, plev, lat, lon)."""
    if not np.isnan(arr).any():
        return arr, 0
    n = int(np.isnan(arr).sum())
    da = xr.DataArray(arr, dims=("t", "p", "y", "x"))
    order_up = plev[0] > plev[-1]     # True: index 0 = surface
    da = da.bfill("p") if order_up else da.ffill("p")
    da = da.ffill("p") if order_up else da.bfill("p")
    return da.values, n


def available_vars(path_deltas):
    return {v for v in VARS_3D + VARS_2D
            if os.path.isfile(os.path.join(path_deltas, f"{v}_delta.nc"))}


def prepare_model(gi, path_deltas, out_path, margin=3.0, log=print, tag=""):
    """Regrid the deltas of one model to the grid 'gi' (from read_grid)."""
    lat, lon = gi["lat"], gi["lon"]
    rotate = gi["gridType"] == "rotated_ll" and gi["uv_relative_to_grid"] == 1
    pre = f"  [{tag}] " if tag else "  "

    regrids = {}          # one interpolator per GCM grid (variables can be staggered)
    data_vars = {}
    plev = None
    mid_day = None
    for var in VARS_3D + VARS_2D:
        path = os.path.join(path_deltas, f"{var}_delta.nc")
        if not os.path.isfile(path):
            if var in REQUIRED:
                raise FileNotFoundError(path)
            continue
        with xr.open_dataset(path) as ds:
            da = ds[var]
            if da.sizes["time"] != 12:
                raise ValueError(f"{var}: 12 months expected, found {da.sizes['time']}")
            if mid_day is None:
                mid_day = ds["time"].dt.day.values
            glat, glon = ds["lat"].values, ds["lon"].values
            key = (glat.size, glon.size, round(float(glat[0]), 6), round(float(glat[-1]), 6),
                   round(float(glon[0]), 6), round(float(glon[-1]), 6))
            if key not in regrids:
                regrids[key] = GCMToPoints(glat, glon, lat, lon, margin=margin)
            regrid = regrids[key]
            arr = da.values.astype(float)
            units = da.attrs.get("units", "")
            if var in VARS_3D:
                p = ds["plev"].values.astype(float)
                if plev is None:
                    plev = p
                elif not np.allclose(plev, p):
                    raise ValueError(f"{var}: pressure levels differ from those of ta")
                arr, nfill = fill_vertical_nans(arr, p)
                if nfill:
                    log(f"{pre}{var}: {nfill} NaNs filled (constant vertical extrapolation)")
                out = regrid(arr)                          # (12, plev, npts)
                dims = ("month", "plev", "point")
            else:
                out = regrid(arr)                          # (12, npts)
                dims = ("month", "point")
        if np.isnan(out).any():
            raise ValueError(f"{var}: NaNs remain after regridding")
        data_vars[var] = (dims, out.astype("float32"), {"units": units})

    if rotate and "ua" in data_vars and "va" in data_vars:
        cos_a, sin_a = wind_rotation_matrix(lon, lat, gi["lat_sp"], gi["lon_sp"])
        ua = data_vars["ua"][1].astype(float)
        va = data_vars["va"][1].astype(float)
        ug = cos_a * ua - sin_a * va
        vg = sin_a * ua + cos_a * va
        data_vars["ua"] = (data_vars["ua"][0], ug.astype("float32"), data_vars["ua"][2])
        data_vars["va"] = (data_vars["va"][0], vg.astype("float32"), data_vars["va"][2])

    ds_out = xr.Dataset(
        data_vars,
        coords={"month": np.arange(1, 13), "plev": plev,
                "point": np.arange(lat.size),
                "lat": ("point", lat), "lon": ("point", lon),
                "mid_day": ("month", mid_day)},
        attrs={"source_deltas": os.path.abspath(path_deltas),
               "Ni": gi["Ni"], "Nj": gi["Nj"],
               "winds_rotated_to_grid": int(rotate),
               "history": f"pgw_ml.prepare ({os.path.dirname(os.path.abspath(__file__))})"},
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp = out_path + ".tmp"
    ds_out.to_netcdf(tmp)
    os.replace(tmp, out_path)
    return {"vars": sorted(data_vars), "nplev": int(plev.size),
            "winds_rotated": int(rotate)}
