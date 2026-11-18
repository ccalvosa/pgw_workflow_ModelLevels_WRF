"""
PGW applied to the surface file read by WPS (ECMWF/ERA5 surface fields, all
validity times in one GRIB file, e.g., ECMWF_SFC_SPs.grb).

WPS uses these fields together with the model-level file:
  * calc_ecmwf_p computes the pressure and geopotential height of the model
    levels from PSFC (sp) and the hybrid coefficients, so sp must follow the
    adjusted surface pressure of the model levels;
  * metgrid/real use PMSL (msl), 2-m temperature and dewpoint, 10-m wind,
    skin, sea surface and soil temperatures.

For each validity time, the state of the lowest model level saved by the
model-level step (ps, T_L, Δu, Δv; see apply.py) is used, so that the surface
file stays consistent with the perturbed model levels:

  sp'  = sp · ps'/ps                    (relative change: sp and exp(lnsp) are
                                          not identical fields in the inputs)
  msl' = msl · ps'/ps · exp[ Φs/Rd · (1/T̄' − 1/T̄) ],   T̄ = T_L + 0.0065·zs/2
                                         (hydrostatic reduction to sea level)
  2t'  = 2t + (T_L' − T_L)              (ΔT of the lowest model level, which
                                          the surface anchor ties to Δtas)
  2d'  : RH2 = e_w(2d)/e_w(2t);  RH2' = bounded(RH2 + Δhurs);
         2d' = Td(RH2' · e_w(2t'))      (saturation over water, IFS Tetens)
  10u, 10v + Δu_L, Δv_L                 (only with perturb_wind)
  skt, sst, stl1-4                      (as in the model-level version)

Times without a model-level file (no saved state) keep sp and msl unchanged,
use Δtas for 2t, and are reported as "uncoupled" in the diagnostics.
"""

import os
from datetime import datetime

import numpy as np
import eccodes as ec

from .apply import PGWOptions, surface_ts_delta, soil_delta
from .constants import G, RD
from .gribio import GribTimeFile, SOIL_DEPTH
from .thermo import esat_water, dewpoint_from_e

LAPSE = 0.0065      # K m-1, standard lapse rate for the sea-level reduction


def state_path(state_dir, vt):
    return os.path.join(state_dir, f"{vt:%Y%m%d%H%M}.npz")


def save_state(state_dir, vt, state):
    os.makedirs(state_dir, exist_ok=True)
    arrs = {k: v for k, v in state.items() if v is not None}
    tmp = state_path(state_dir, vt) + ".tmp.npz"
    np.savez(tmp, **arrs)
    os.replace(tmp, state_path(state_dir, vt))


def load_state(state_dir, vt):
    p = state_path(state_dir, vt)
    if not os.path.isfile(p):
        return None
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def tbar(t_lowest, phis):
    """Mean temperature of the fictitious column between the surface and sea level."""
    return t_lowest + LAPSE * (phis / G) / 2.0


def msl_factor(st):
    """msl'/msl from the state of the lowest model level."""
    t0, t1 = tbar(st["tL0"], st["phis"]), tbar(st["tL1"], st["phis"])
    return (st["ps1"] / st["ps0"]) * np.exp(st["phis"] / RD * (1.0 / t1 - 1.0 / t0))


def dewpoint_perturbed(t2, d2, t2_new, dhurs_frac, rh_min_factor):
    """New 2-m dewpoint keeping RH2 + ΔRH (bounded as in 3D)."""
    rh2 = esat_water(d2) / esat_water(t2)
    rh2_new = np.clip(rh2 + dhurs_frac, rh_min_factor * rh2, np.maximum(rh2, 1.0))
    return dewpoint_from_e(rh2_new * esat_water(t2_new)), rh2, rh2_new


def apply_pgw_to_sfc_file(path_in, path_out, deltas, opt: PGWOptions, state_dir):
    f = GribTimeFile(path_in)
    try:
        return _apply_sfc(f, path_out, deltas, opt, state_dir)
    finally:
        f.close()


def _apply_sfc(f, path_out, deltas, opt, state_dir):
    gid = f.handles[0]
    nj, ni = ec.codes_get(gid, "Nj"), ec.codes_get(gid, "Ni")
    diag = {"file": f.path, "kind": "sfc", "times": [], "coupled": [], "uncoupled": []}
    first = None
    for vt in f.times():
        d = deltas.at(vt, opt.month)
        st = load_state(state_dir, vt)
        diag["times"].append(vt.isoformat())
        (diag["coupled"] if st is not None else diag["uncoupled"]).append(vt.isoformat())
        rec = {}

        # Surface and sea-level pressure
        if st is not None and opt.ps_mode != "none":
            ratio = st["ps1"] / st["ps0"]
            if f.has(vt, "sp"):
                sp, _ = f.get2d(vt, "sp")
                f.set2d(vt, "sp", sp * ratio)
                rec["dsp"] = float((sp * ratio - sp).mean())
            if f.has(vt, "msl") and opt.perturb_msl:
                msl, _ = f.get2d(vt, "msl")
                fac = msl_factor(st)
                f.set2d(vt, "msl", msl * fac)
                rec["dmsl"] = float((msl * fac - msl).mean())

        # 2-m temperature and dewpoint
        if st is not None:
            dt2 = st["tL1"] - st["tL0"]
        else:
            dt2 = d["tas"] if opt.perturb_t else 0.0
        if f.has(vt, "2t"):
            t2, _ = f.get2d(vt, "2t")
            t2n = t2 + (dt2 if opt.perturb_t else 0.0)
            if opt.perturb_t:
                f.set2d(vt, "2t", t2n)
                rec["d2t"] = float((t2n - t2).mean())
            if f.has(vt, "2d") and (opt.perturb_q or opt.perturb_t):
                d2, _ = f.get2d(vt, "2d")
                dh = d["hurs"] / 100.0 if opt.perturb_q else 0.0
                d2n, _, _ = dewpoint_perturbed(t2, d2, t2n, dh, opt.rh_min_factor)
                f.set2d(vt, "2d", d2n)
                rec["d2d"] = float((d2n - d2).mean())

        # 10-m wind
        if opt.perturb_wind and st is not None and "du" in st:
            for name, k in (("10u", "du"), ("10v", "dv")):
                if f.has(vt, name):
                    w, _ = f.get2d(vt, name)
                    f.set2d(vt, name, w + st[k])

        # Skin, sea surface and soil temperatures
        if f.has(vt, "lsm"):
            dts_comb, dts_sea, sea, _ = surface_ts_delta(d["ts"], f.get2d(vt, "lsm")[0],
                                                         nj, ni, opt)
            if opt.perturb_skt and f.has(vt, "skt"):
                skt, _ = f.get2d(vt, "skt")
                f.set2d(vt, "skt", skt + dts_comb)
            if opt.perturb_sst and f.has(vt, "sst"):
                sst, miss = f.get2d(vt, "sst")
                f.set2d(vt, "sst", np.where(miss, sst, sst + dts_comb), missing=miss)
                rec["dsst"] = float(dts_comb[~miss].mean()) if (~miss).any() else float("nan")
            if opt.perturb_soil:
                dts_ann = deltas.annual_mean("ts")
                for name, depth in SOIL_DEPTH.items():
                    for lev in f.levels(vt, name, "depthBelowLandLayer"):
                        s, _ = f.get2d(vt, name, "depthBelowLandLayer", lev)
                        f.set2d(vt, name, s + soil_delta(d["ts"], dts_ann, depth, opt),
                                "depthBelowLandLayer", lev)
        if first is None:
            first = rec
    f.write(path_out)
    diag.update({f"first_{k}": v for k, v in (first or {}).items()})
    diag["n_times"] = len(diag["times"])
    diag["n_uncoupled"] = len(diag["uncoupled"])
    return diag
