"""
Reading/writing of the IFS/ERA5 GRIB files used by WPS/WRF as initial and
boundary conditions.

Model-level files (one per time, e.g., ECMWF_PL_20171007_00.grb):
  hybrid (GRIB2)      : t, q, u, v  x 137 levels; lnsp and z on level 1
  regular_ll grid, Earth-relative winds.
Surface file (all times in one file, e.g., ECMWF_SFC_SPs.grb, GRIB1):
  surface             : sp, msl, 2t, 2d, 10u, 10v, skt, sst*, ci*, lsm, sd, rsn, z
  depthBelowLandLayer : stl1-4, swvl1-4
  (*) with bitmap: missing values over land.

Messages are identified by paramId (independent of local definition tables).
On writing, the ORDER and METADATA of all messages are kept and only the
values of the modified fields are replaced.
"""

import fnmatch
import os
from datetime import datetime

import eccodes as ec
import numpy as np

PARAM = {
    "z": 129, "t": 130, "u": 131, "v": 132, "q": 133, "lnsp": 152,
    "sst": 34, "skt": 235, "lsm": 172,
    "stl1": 139, "stl2": 170, "stl3": 183, "stl4": 236,
    "sp": 134, "msl": 151, "2t": 167, "2d": 168, "10u": 165, "10v": 166,
}

# Mid-layer depth (m) of the IFS soil layers: 0-7, 7-28, 28-100, 100-289 cm
SOIL_DEPTH = {"stl1": 0.035, "stl2": 0.175, "stl3": 0.64, "stl4": 1.945}


class IFSFile:
    """IFS GRIB file loaded in memory (eccodes handles)."""

    def __init__(self, path):
        self.path = path
        self.handles = []
        with open(path, "rb") as f:
            while True:
                gid = ec.codes_grib_new_from_file(f)
                if gid is None:
                    break
                self.handles.append(gid)
        if not self.handles:
            raise ValueError(f"{path}: contains no GRIB messages")
        self.index = {}                     # (paramId, typeOfLevel, level) -> i
        for i, gid in enumerate(self.handles):
            key = (ec.codes_get(gid, "paramId"), ec.codes_get(gid, "typeOfLevel"),
                   ec.codes_get(gid, "level"))
            self.index[key] = i
        self.new_values = {}                # i -> array

    # ── geometry ─────────────────────────────────────────────────────────────
    def _ref_handle(self):
        return self.handles[self.index[(PARAM["t"], "hybrid", 1)]]

    def grid_info(self):
        gid = self._ref_handle()
        info = {k: ec.codes_get(gid, k) for k in
                ("gridType", "Ni", "Nj", "numberOfPoints")}
        if info["gridType"] == "rotated_ll":
            info["lat_sp"] = ec.codes_get(gid, "latitudeOfSouthernPoleInDegrees")
            info["lon_sp"] = ec.codes_get(gid, "longitudeOfSouthernPoleInDegrees")
            info["angle"] = ec.codes_get(gid, "angleOfRotationInDegrees")
        info["lat"] = ec.codes_get_array(self.handles[0], "latitudes")
        info["lon"] = ec.codes_get_array(self.handles[0], "longitudes")
        info["uv_relative_to_grid"] = ec.codes_get(
            self.handles[self.index[(PARAM["u"], "hybrid", 1)]], "uvRelativeToGrid")
        return info

    def pv(self):
        return ec.codes_get_array(self._ref_handle(), "pv")

    def valid_datetime(self):
        gid = self._ref_handle()
        return (ec.codes_get(gid, "validityDate"), ec.codes_get(gid, "validityTime"))

    # ── reading ──────────────────────────────────────────────────────────────
    def has(self, name, type_of_level, level):
        return (PARAM[name], type_of_level, level) in self.index

    def get2d(self, name, type_of_level="surface", level=0):
        """Returns (values, missing_mask)."""
        i = self.index[(PARAM[name], type_of_level, level)]
        gid = self.handles[i]
        vals = ec.codes_get_values(gid).astype(float)
        if ec.codes_get(gid, "bitmapPresent"):
            miss = vals == ec.codes_get(gid, "missingValue")
        else:
            miss = np.zeros(vals.shape, dtype=bool)
        return vals, miss

    def hybrid_levels(self, name):
        return sorted(lev for (p, tl, lev) in self.index
                      if p == PARAM[name] and tl == "hybrid")

    def get3d(self, name):
        """Field on hybrid levels, shape (nlev, npts), level 1 = model top."""
        levs = self.hybrid_levels(name)
        if levs != list(range(1, len(levs) + 1)):
            raise ValueError(f"{name}: non-consecutive hybrid levels")
        return np.stack([self.get2d(name, "hybrid", l)[0] for l in levs])

    # ── writing ──────────────────────────────────────────────────────────────
    def set2d(self, name, values, type_of_level="surface", level=0, missing=None):
        i = self.index[(PARAM[name], type_of_level, level)]
        vals = np.asarray(values, dtype=float)          # no copy: callers do not reuse it
        if missing is not None and missing.any():
            vals = vals.copy()
            vals[missing] = ec.codes_get(self.handles[i], "missingValue")
        if not np.all(np.isfinite(vals)):
            raise ValueError(f"{name} level {level}: non-finite values")
        self.new_values[i] = vals

    def set3d(self, name, values):
        for k, lev in enumerate(self.hybrid_levels(name)):
            self.set2d(name, values[k], "hybrid", lev)

    def write(self, path_out):
        with open(path_out, "wb") as f:
            for i, gid in enumerate(self.handles):
                if i in self.new_values:
                    ec.codes_set_values(gid, self.new_values[i])
                ec.codes_write(gid, f)

    def close(self):
        for gid in self.handles:
            ec.codes_release(gid)
        self.handles = []


def find_files(root, pattern):
    """Files under 'root' (recursive) whose name matches 'pattern'."""
    out = []
    for dirpath, _, files in os.walk(root):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                out.append(os.path.join(dirpath, name))
    return sorted(out)


def read_valid_datetime(path):
    """Validity time (datetime), reading only the first message of the file."""
    with open(path, "rb") as f:
        gid = ec.codes_grib_new_from_file(f)
        if gid is None:
            raise ValueError(f"{path}: contains no GRIB messages")
        try:
            d, t = ec.codes_get(gid, "validityDate"), ec.codes_get(gid, "validityTime")
        finally:
            ec.codes_release(gid)
    return datetime.strptime(f"{d}{t:04d}", "%Y%m%d%H%M")


class GribTimeFile:
    """
    GRIB file with several validity times (e.g., the ERA5/IFS surface file
    used by WPS, ECMWF_SFC*.grb). Messages are indexed by
    (validity datetime, paramId, typeOfLevel, level). On writing, the ORDER
    and METADATA of all messages are kept and only the values of the
    modified messages are replaced.
    """

    def __init__(self, path):
        self.path = path
        self.handles = []
        self.index = {}
        with open(path, "rb") as f:
            while True:
                gid = ec.codes_grib_new_from_file(f)
                if gid is None:
                    break
                i = len(self.handles)
                self.handles.append(gid)
                vt = datetime.strptime(f"{ec.codes_get(gid, 'validityDate')}"
                                       f"{ec.codes_get(gid, 'validityTime'):04d}", "%Y%m%d%H%M")
                key = (vt, ec.codes_get(gid, "paramId"), ec.codes_get(gid, "typeOfLevel"),
                       ec.codes_get(gid, "level"))
                self.index[key] = i
        if not self.handles:
            raise ValueError(f"{path}: contains no GRIB messages")
        self.new_values = {}

    def times(self):
        return sorted({k[0] for k in self.index})

    def has(self, vt, name, type_of_level="surface", level=0):
        return (vt, PARAM[name], type_of_level, level) in self.index

    def levels(self, vt, name, type_of_level):
        return sorted(k[3] for k in self.index
                      if k[0] == vt and k[1] == PARAM[name] and k[2] == type_of_level)

    def get2d(self, vt, name, type_of_level="surface", level=0):
        """Returns (values, missing_mask)."""
        gid = self.handles[self.index[(vt, PARAM[name], type_of_level, level)]]
        vals = ec.codes_get_values(gid).astype(float)
        if ec.codes_get(gid, "bitmapPresent"):
            miss = vals == ec.codes_get(gid, "missingValue")
        else:
            miss = np.zeros(vals.shape, dtype=bool)
        return vals, miss

    def set2d(self, vt, name, values, type_of_level="surface", level=0, missing=None):
        i = self.index[(vt, PARAM[name], type_of_level, level)]
        vals = np.asarray(values, dtype=float).copy()
        if missing is not None and missing.any():
            vals[missing] = ec.codes_get(self.handles[i], "missingValue")
        if not np.all(np.isfinite(vals)):
            raise ValueError(f"{name} {vt}: non-finite values")
        self.new_values[i] = vals

    def write(self, path_out):
        with open(path_out, "wb") as f:
            for i, gid in enumerate(self.handles):
                if i in self.new_values:
                    ec.codes_set_values(gid, self.new_values[i])
                ec.codes_write(gid, f)

    def close(self):
        for gid in self.handles:
            ec.codes_release(gid)
        self.handles = []


def read_time_subset(path, vt):
    """{(paramId, typeOfLevel, level): (values, missing)} of the messages of
    'path' valid at 'vt' (only those messages are decoded)."""
    out = {}
    with open(path, "rb") as f:
        while True:
            gid = ec.codes_grib_new_from_file(f)
            if gid is None:
                break
            try:
                t = datetime.strptime(f"{ec.codes_get(gid, 'validityDate')}"
                                      f"{ec.codes_get(gid, 'validityTime'):04d}", "%Y%m%d%H%M")
                if t != vt:
                    continue
                vals = ec.codes_get_values(gid).astype(float)
                miss = (vals == ec.codes_get(gid, "missingValue")) if ec.codes_get(gid, "bitmapPresent") \
                    else np.zeros(vals.shape, dtype=bool)
                out[(ec.codes_get(gid, "paramId"), ec.codes_get(gid, "typeOfLevel"),
                     ec.codes_get(gid, "level"))] = (vals, miss)
            finally:
                ec.codes_release(gid)
    return out


def pack_unit(gid):
    """Quantization step of a message: 2^E · 10^-D."""
    e = ec.codes_get(gid, "binaryScaleFactor")
    d = ec.codes_get(gid, "decimalScaleFactor")
    return (2.0 ** e) * (10.0 ** -d)
