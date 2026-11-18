"""
CMIP6 multi-model ensemble: scenarios, model discovery, variable checks,
selection (list / all / automatic) and ensemble mean (ENSMEAN).

Delta layout (root CMIP_ROOT, e.g., /ec/res4/hpcperm/ecme2143/CMIP6_deltas):
    <root>/Future_ssp245/<MODEL>/<var>_delta.nc
    <root>/Future_ssp585/<MODEL>/<var>_delta.nc
    <root>/Attribution/ssp245/<YEAR>/<MODEL>/<var>_delta.nc   (per event year)
Scenario = relative path under the root ("Future_ssp585", "Attribution/ssp245";
"Attribution_ssp245" is also accepted). If its directory contains year
subdirectories (YYYY), the event year is used (the year of the validity
times of the files, or the one set by the user).
Backward compatibility: if the root directly contains models (or ta_delta.nc)
and no scenario is given, the root is the only scenario.

ENSMEAN: equal-weight mean of the deltas of the selected models ALREADY
interpolated to the input grid (all share the same plev). The intermodel
standard deviation is also stored (<var>_spread).

Automatic selection of N models. Each model (and scenario) is described by
three features, averaged over the domain box on the NATIVE GCM grid (the
~40 models need not be regridded) and over the months of the files:
    dT_trop : ΔT averaged over 850-300 hPa     [K]
    dtas    : Δtas                              [K]
    dRH     : ΔRH averaged over 850-500 hPa    [%]
With several scenarios and common models, the feature vector of each model
concatenates those of all scenarios. Features are standardized (z-score) and:
    spread         : (default) starts from the model closest to the
                     multi-model mean and greedily adds the model whose
                     minimum distance to the already selected ones is
                     largest (farthest-point sampling): the N members span
                     the spread of the full ensemble.
    representative : the N models closest to the multi-model mean.
"""

import json
import os

import numpy as np
import xarray as xr

from .deltas import time_weights
from .prepare import REQUIRED, available_vars

FEATURES = ("dT_trop", "dtas", "dRH")


# ──────────────────────────────────────────────────────────────────────────────
#  Discovery and selection by name
# ──────────────────────────────────────────────────────────────────────────────
def discover(cmip_root, suffix=""):
    """{name: path} of the models available in the directory of a scenario."""
    cmip_root = os.path.abspath(cmip_root)
    if os.path.isfile(os.path.join(cmip_root, "ta_delta.nc")):
        name = os.path.basename(cmip_root.rstrip("/"))
        if suffix and name.endswith(suffix):
            name = name[: -len(suffix)]
        return {name: cmip_root}
    out = {}
    for d in sorted(os.listdir(cmip_root)):
        p = os.path.join(cmip_root, d)
        if not os.path.isdir(p) or d.startswith((".", "_")):
            continue
        if suffix:
            if not d.endswith(suffix):
                continue
            d = d[: -len(suffix)]
        if d.upper().startswith("ENSMEAN"):
            continue
        out[d] = p
    return out


def _year_dirs(path):
    return sorted(d for d in os.listdir(path)
                  if d.isdigit() and len(d) == 4 and os.path.isdir(os.path.join(path, d)))


def list_scenarios(root):
    """Scenarios available under the root (relative paths)."""
    out = []
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if not os.path.isdir(p) or d.startswith((".", "_")):
            continue
        subs = [s for s in sorted(os.listdir(p)) if os.path.isdir(os.path.join(p, s))
                and not s.startswith((".", "_"))]
        if _year_dirs(p) or any(os.path.isfile(os.path.join(p, s, "ta_delta.nc")) for s in subs):
            out.append(d)
        else:
            out += [f"{d}/{s}" for s in subs
                    if _year_dirs(os.path.join(p, s)) or
                    any(os.path.isfile(os.path.join(p, s, m, "ta_delta.nc"))
                        for m in os.listdir(os.path.join(p, s)))]
    return out


def resolve_scenario(root, name, year, nearest=False):
    """
    (tag, model_directory, year|None) of a scenario.
    'year': int or None. If the scenario has year subdirectories and the year
    is not available, an error is raised, unless nearest=True (automatic
    year), in which case the closest available year is used.
    """
    cands = [name, name.replace("_", "/", 1)]
    for c in cands:
        p = os.path.join(root, c)
        if os.path.isdir(p):
            break
    else:
        raise FileNotFoundError(f"Scenario '{name}' does not exist in {root}. "
                                f"Available: {' '.join(list_scenarios(root))}")
    tag = c.strip("/").replace("/", "_")
    years = _year_dirs(p)
    if years:
        if year is None:
            raise ValueError(f"{name}: an event year is required ({years[0]}..{years[-1]})")
        if str(year) not in years:
            if not nearest:
                raise ValueError(f"{name}: no deltas for {year} (available {years[0]}..{years[-1]})")
            closest = min(years, key=lambda y: abs(int(y) - int(year)))
            print(f"  [WARNING] {name}: no deltas for {year}; the closest year {closest} is used "
                  f"(set ATTRIB_YEAR to choose another)")
            year = int(closest)
        return tag, os.path.join(p, str(year)), int(year)
    return tag, p, None


def required_vars(ps_mode="hydrostatic", perturb_wind=False):
    req = set(REQUIRED)
    if ps_mode == "gcm":
        req.add("ps")
    if perturb_wind:
        req |= {"ua", "va"}
    return req


def missing_vars(path, required):
    return sorted(required - available_vars(path))


def resolve_names(spec_list, available):
    """User-requested names -> names in 'available' (case-insensitive, with or
    without suffix). Returns (found, not_found)."""
    low = {k.lower(): k for k in available}
    ok, bad = [], []
    for s in spec_list:
        k = low.get(s.lower()) or low.get(s.lower().rsplit("_", 1)[0])
        if k is None:
            bad.append(s)
        elif k not in ok:
            ok.append(k)
    return ok, bad


# ──────────────────────────────────────────────────────────────────────────────
#  Domain features and automatic selection
# ──────────────────────────────────────────────────────────────────────────────
def mean_weights(valid_dts, month_spec, ds):
    """Monthly weights averaged over all validity times of the files."""
    acc = np.zeros(12)
    for dt in valid_dts:
        for i, w in time_weights(dt, month_spec, ds).items():
            acc[i] += w
    return acc / acc.sum()


def domain_features(model_dir, lat, lon, valid_dts, month_spec, margin=1.0):
    """Domain features of a model on its NATIVE grid (lat/lon box of the domain)."""
    lon_c = np.rad2deg(np.angle(np.mean(np.exp(1j * np.deg2rad(lon)))))
    lon_rel = (np.asarray(lon) - lon_c + 180.0) % 360.0 - 180.0
    box = (lat.min() - margin, lat.max() + margin,
           lon_rel.min() - margin, lon_rel.max() + margin)

    def wmean(var, pmin=None, pmax=None):
        with xr.open_dataset(os.path.join(model_dir, f"{var}_delta.nc")) as ds:
            glon = (ds["lon"].values - lon_c + 180.0) % 360.0 - 180.0
            jl = np.where((glon >= box[2]) & (glon <= box[3]))[0]
            il = np.where((ds["lat"].values >= box[0]) & (ds["lat"].values <= box[1]))[0]
            if jl.size == 0 or il.size == 0:          # very coarse GCM: nearest point
                jl = [int(np.argmin(np.abs(glon)))]
                il = [int(np.argmin(np.abs(ds["lat"].values - lat.mean())))]
            da = ds[var].isel(lat=il, lon=jl)
            if pmin is not None:
                da = da.sel(plev=(da["plev"] >= pmin) & (da["plev"] <= pmax))
            w = mean_weights(valid_dts, month_spec,
                             xr.Dataset({"mid_day": ("month", ds["time"].dt.day.values)}))
            a = np.tensordot(w, da.values.astype(float), axes=(0, 0))
            coslat = np.cos(np.deg2rad(da["lat"].values))[:, None]
            wa = np.broadcast_to(coslat, a.shape[-2:])
            a = a.reshape((-1,) + a.shape[-2:])
            ok = np.isfinite(a)
            return float((np.where(ok, a, 0) * wa).sum() / (ok * wa).sum())

    return {"dT_trop": wmean("ta", 30000, 85000), "dtas": wmean("tas"),
            "dRH": wmean("hur", 50000, 85000), "dT850": wmean("ta", 85000, 85000),
            "dts": wmean("ts")}


def select_auto(features, n, method="spread"):
    """features: {model: {feature: value}} -> (ordered list, info)."""
    names = sorted(features)
    keys = sorted(next(iter(features.values())))
    if n >= len(names):
        return names, {"note": f"N={n} >= candidates ({len(names)}): all are used"}
    X = np.array([[features[m][k] for k in keys] for m in names])
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X - X.mean(axis=0)) / sd
    d_mean = np.linalg.norm(Z, axis=1)
    if method == "representative":
        idx = list(np.argsort(d_mean)[:n])
    elif method == "spread":
        idx = [int(np.argmin(d_mean))]
        while len(idx) < n:
            dmin = np.min(np.linalg.norm(Z[:, None, :] - Z[None, idx, :], axis=2), axis=1)
            dmin[idx] = -1.0
            idx.append(int(np.argmax(dmin)))
    else:
        raise ValueError(f"Unknown selection method: {method}")
    sel = [names[i] for i in idx]
    info = {"method": method, "features": keys,
            "zscores": {m: [round(float(v), 3) for v in Z[i]] for i, m in enumerate(names)},
            "dist_to_mean": {m: round(float(d_mean[i]), 3) for i, m in enumerate(names)},
            "ens_mean_all": {k: round(float(v), 4) for k, v in zip(keys, X.mean(axis=0))},
            "ens_mean_selected": {k: round(float(v), 4) for k, v in
                                  zip(keys, X[idx].mean(axis=0))}}
    return sel, info


# ──────────────────────────────────────────────────────────────────────────────
#  ENSMEAN
# ──────────────────────────────────────────────────────────────────────────────
SPREAD_VARS = ("ta", "hur", "zg", "tas", "ts", "hurs")


def ensmean(member_paths, out_path):
    """Mean (and spread) of the prepared deltas of several models."""
    dss = [xr.open_dataset(p) for p in member_paths.values()]
    try:
        ref = dss[0]
        for name, ds in zip(member_paths, dss):
            if not np.allclose(ds["plev"].values, ref["plev"].values):
                raise ValueError(f"{name}: plev differ from those of {list(member_paths)[0]}")
            if ds.sizes["point"] != ref.sizes["point"]:
                raise ValueError(f"{name}: different number of points")
        common = set(ref.data_vars)
        for ds in dss[1:]:
            common &= set(ds.data_vars)
        data_vars = {}
        for v in sorted(common):
            stack = np.stack([ds[v].values.astype(float) for ds in dss])
            data_vars[v] = (ref[v].dims, stack.mean(axis=0).astype("float32"),
                            dict(ref[v].attrs))
            if v in SPREAD_VARS:
                data_vars[v + "_spread"] = (ref[v].dims, stack.std(axis=0).astype("float32"),
                                            {"units": ref[v].attrs.get("units", ""),
                                             "long_name": f"intermodel standard deviation of {v}"})
        rot = {int(ds.attrs.get("winds_rotated_to_grid", 0)) for ds in dss}
        attrs = dict(ref.attrs)
        attrs.update({"members": json.dumps(list(member_paths)),
                      "source_deltas": json.dumps({k: ds.attrs.get("source_deltas", "")
                                                   for k, ds in zip(member_paths, dss)}),
                      "winds_rotated_to_grid": int(rot == {1}),
                      "history": "pgw_ml.ensemble.ensmean (equal-weight mean)"})
        out = xr.Dataset(data_vars,
                         coords={c: ref[c] for c in ("month", "plev", "point", "lat",
                                                     "lon", "mid_day") if c in ref.coords},
                         attrs=attrs)
        tmp = out_path + ".tmp"
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        out.to_netcdf(tmp)
        os.replace(tmp, out_path)
        return sorted(common)
    finally:
        for ds in dss:
            ds.close()
