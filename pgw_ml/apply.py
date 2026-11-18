"""
PGW applied to one IFS file on model levels.

Algorithm (adaptation of step_03_apply_to_era.py of PGW4ERA5, Brogli et al.
2023, https://github.com/Potopoles/PGW4ERA5, with i_reinterp = 0):

 1. Full-level p with the original ps (p = a + b·ps).
 2. Original RH = f(q, T, p)   (IFS mixed water/ice saturation).
 3. Deltas ta, hur (and ua, va) interpolated in ln(p) to the ORIGINAL
    pressures of each level. For ta/hur a continuous surface correction
    toward tas/hurs is added over the lowest 'sfc_blend_depth' Pa
    (anchored at the IFS ps; see README).
 4. T_pgw = T + ΔT ;  RH_pgw = RH + ΔRH ;  u,v_pgw = u,v + Δu,v (rotated).
    RH_pgw is bounded to [rh_min_factor·RH, max(RH, 1)] and the change in q
    is tapered in ln(p) between q_taper_bottom and q_taper_top (q unchanged
    in the stratosphere).
 5. Iterative ps adjustment: ps_pgw is sought such that
        Φ_pgw(p_ref) - Φ_orig(p_ref) = g · Δzg(p_ref)
    recomputing q_pgw = f(RH_pgw, T_pgw, p(ps_pgw)) at each iteration.
        ps += -adj · ps / (Rd · T_pgw,lowest level) · error_Φ
 6. lnsp_pgw = ln(ps_pgw). The surface geopotential is unchanged.
 7. Surface: skt + Δts ("open-sea" Δts over the sea), sst + coast-cleaned
    Δts over the sea, stl1-4 + Δts damped with depth:
        Δst(z) = Δts_annual + exp(-z/2.8 m) · (Δts - Δts_annual)
"""

from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np

from .constants import G, RD
from .deltas import Deltas
from .gribio import IFSFile, PARAM, SOIL_DEPTH
from .sst import ocean_only_delta
from .thermo import q_to_rh, rh_to_q
from .vertical import (split_pv, full_level_pressure, geopotential_at_pressure,
                       interp_delta_to_model_levels)


@dataclass
class PGWOptions:
    month: str = "auto"              # 1..12 or 'auto' (temporal interpolation)
    perturb_t: bool = True
    perturb_q: bool = True           # keeping RH + ΔRH
    q_taper_bottom: float = 15000.0  # Pa: below (higher p) full change in q
    q_taper_top: float = 10000.0     # Pa: above (lower p) q is not modified
    rh_min_factor: float = 0.5       # RH_pgw >= rh_min_factor * RH_orig
    perturb_wind: bool = False
    ps_mode: str = "hydrostatic"     # hydrostatic | gcm | none
    p_ref: float = 30000.0           # Pa (PGW4ERA5: p_ref_inp = 30000)
    adj_factor: float = 0.95
    thresh_phi: float = 0.15         # m2 s-2
    max_iter: int = 20
    sfc_anchor: bool = True          # anchor ta/hur to tas/hurs at the surface
    sfc_blend_depth: float = 10000.0 # Pa: depth of the surface transition layer
    perturb_skt: bool = True
    perturb_sst: bool = True
    perturb_soil: bool = True
    perturb_msl: bool = True         # msl rescaled with the adjusted ps (surface file)
    soil_efold: float = 2.8          # m
    lsm_threshold: float = 0.5
    coast_buffer: int = 2
    k_neighbors: int = 8
    idw_power: float = 2.0
    sst_smooth_sigma: float = 1.0
    chunk: int = 20000               # columns per block (memory control)


def _interp_p_ref(delta_pl, plev, p_ref):
    """Delta at p_ref (ln p interpolation) for each column."""
    order = np.argsort(plev)
    lp = np.log(plev[order])
    d = delta_pl[order]
    k = int(np.clip(np.searchsorted(lp, np.log(p_ref)), 1, lp.size - 1))
    w = (np.log(p_ref) - lp[k - 1]) / (lp[k] - lp[k - 1])
    return d[k - 1] * (1 - w) + d[k] * w


def apply_pgw_to_file(path_in, path_out, deltas: Deltas, opt: PGWOptions, log=print):
    f = IFSFile(path_in)
    try:
        return _apply(f, path_out, deltas, opt, log)
    finally:
        f.close()


def _apply(f, path_out, deltas, opt, log):
    diag = {"file": f.path}
    vdate, vtime = f.valid_datetime()
    valid_dt = datetime.strptime(f"{vdate}{vtime:04d}", "%Y%m%d%H%M")
    d = deltas.at(valid_dt, opt.month)
    diag["valid"] = valid_dt.isoformat()
    diag["month_weights"] = {int(k) + 1: round(float(v), 4) for k, v in d["_weights"].items()}
    plev = deltas.plev

    # ── Original state ───────────────────────────────────────────────────────
    a, b = split_pv(f.pv())
    lnsp, _ = f.get2d("lnsp", "hybrid", 1)
    ps0 = np.exp(lnsp)
    phis, _ = f.get2d("z", "hybrid", 1)
    t0 = f.get3d("t")
    q0 = f.get3d("q")
    ptop = float(full_level_pressure(a, b, ps0[:1]).min())
    if ptop < plev.min():
        log(f"  [INFO] Model top ({ptop:.2f} Pa) above the highest GCM level "
            f"({plev.min():.0f} Pa): constant extrapolation of the delta.")

    # ── Column blocks (bounded memory for large domains) ─────────────────────
    npts = ps0.size
    t_pgw, q_pgw, ps = np.empty_like(t0), np.empty_like(q0), ps0.copy()
    its, errs = [0], [0.0]
    near = {"dT850": 0.0, "dT500": 0.0}
    for s0 in range(0, npts, opt.chunk):
        sl = slice(s0, min(s0 + opt.chunk, npts))
        dc = {k: v[..., sl] for k, v in d.items() if isinstance(v, np.ndarray)}
        t_c, q_c, ps_c, it, err = perturb_columns(a, b, ps0[sl], phis[sl], t0[:, sl],
                                                  q0[:, sl], dc, plev, opt)
        t_pgw[:, sl], q_pgw[:, sl], ps[sl] = t_c, q_c, ps_c
        its.append(it)
        errs.append(err)
        pf = full_level_pressure(a, b, ps0[sl])
        for key, pref in (("dT850", 85000.0), ("dT500", 50000.0)):
            near[key] += float(_sum_near(t_c - t0[:, sl], pf, pref))
    if opt.ps_mode == "hydrostatic":
        diag["ps_iterations"] = int(max(its))
        diag["phi_err_max"] = float(max(errs))
    elif opt.ps_mode == "gcm":
        diag["ps_iterations"] = 0

    # ── 3D output ────────────────────────────────────────────────────────────
    if opt.perturb_t:
        f.set3d("t", t_pgw)
    if opt.perturb_q:
        f.set3d("q", q_pgw)
    if opt.ps_mode != "none":
        f.set2d("lnsp", np.log(ps), "hybrid", 1)
    du = dv = None
    if opt.perturb_wind:
        if not deltas.winds_rotated and f.grid_info()["uv_relative_to_grid"]:
            raise RuntimeError("Grid-relative winds but the deltas are not rotated")
        for c, dname in (("u", "ua"), ("v", "va")):
            x = f.get3d(c)
            for s0 in range(0, npts, opt.chunk):
                sl = slice(s0, min(s0 + opt.chunk, npts))
                x[:, sl] += interp_delta_to_model_levels(d[dname][:, sl], plev,
                                                         full_level_pressure(a, b, ps0[sl]))
            f.set3d(c, x)
            dlow = x[-1] - f.get2d(c, "hybrid", f.hybrid_levels(c)[-1])[0]
            if c == "u":
                du = dlow
            else:
                dv = dlow
            del x
        diag["du_mean_lowest"] = float(du.mean())

    # ── Surface (only if the surface fields are in the same file) ────────────
    n_clean = 0
    dts_sea = sea = None
    if f.has("lsm", "surface", 0):
        gi = f.grid_info()
        dts_comb, dts_sea, sea, n_clean = surface_ts_delta(d["ts"], f.get2d("lsm")[0],
                                                           gi["Nj"], gi["Ni"], opt)
        if opt.perturb_skt and f.has("skt", "surface", 0):
            skt, _ = f.get2d("skt")
            f.set2d("skt", skt + dts_comb)
        if opt.perturb_sst and f.has("sst", "surface", 0):
            sst, miss = f.get2d("sst")
            f.set2d("sst", np.where(miss, sst, sst + dts_comb), missing=miss)
        if opt.perturb_soil:
            dts_ann = deltas.annual_mean("ts")
            for name, depth in SOIL_DEPTH.items():
                lev = _soil_level(f, name)
                if lev >= 0:
                    st, _ = f.get2d(name, "depthBelowLandLayer", lev)
                    f.set2d(name, st + soil_delta(d["ts"], dts_ann, depth, opt),
                            "depthBelowLandLayer", lev)

    f.write(path_out)

    # State of the lowest model level, used to perturb the surface file
    diag["_state"] = {"ps0": ps0, "ps1": ps, "tL0": t0[-1], "tL1": t_pgw[-1], "phis": phis,
                      "du": du, "dv": dv}

    # ── Diagnostics ──────────────────────────────────────────────────────────
    diag.update({
        "dps_mean": float((ps - ps0).mean()),
        "dps_min": float((ps - ps0).min()),
        "dps_max": float((ps - ps0).max()),
        "dT_lowest_mean": float((t_pgw - t0)[-1].mean()),
        "dT_850_mean": near["dT850"] / npts,
        "dT_500_mean": near["dT500"] / npts,
        "dq_lowest_rel_mean": float(((q_pgw - q0)[-1] / q0[-1]).mean()),
        "dts_sea_mean": float(dts_sea[sea].mean()) if sea is not None and sea.any() else float("nan"),
        "n_open_sea_points": n_clean,
    })
    return diag


def perturb_columns(a, b, ps0, phis, t0, q0, d, plev, opt):
    """
    PGW of a block of columns. Returns (T', q', ps', iterations, max |E|).
    d: deltas restricted to the block (3D: (nplev, n); 2D: (n,)).
    """
    pf0 = full_level_pressure(a, b, ps0)
    rh0 = q_to_rh(q0, t0, pf0)

    # Deltas on model levels (original pressures)
    anchor = opt.sfc_anchor
    bd = opt.sfc_blend_depth
    dta = interp_delta_to_model_levels(d["ta"], plev, pf0,
                                       d["tas"] if anchor else None, ps0, bd)
    t_pgw = t0 + dta if opt.perturb_t else t0
    del dta
    if opt.perturb_q:
        dhur = interp_delta_to_model_levels(d["hur"], plev, pf0,
                                            d["hurs"] if anchor else None, ps0, bd) / 100.0
        # Relative floor (avoids RH <= 0 in very dry areas) and ceiling at
        # saturation (or at the original supersaturation, which the IFS
        # allows with respect to ice).
        rh_pgw = np.clip(rh0 + dhur, opt.rh_min_factor * rh0, np.maximum(rh0, 1.0))
        del dhur
        # Vertical weight of the change in q: 1 in the troposphere, 0 in the
        # stratosphere (stratospheric vapor is not controlled by RH, and
        # keeping RH with the GCM ΔT produces unrealistic changes in q).
        w_q = np.clip((np.log(pf0) - np.log(opt.q_taper_top)) /
                      (np.log(opt.q_taper_bottom) - np.log(opt.q_taper_top)), 0.0, 1.0)
    else:
        rh_pgw = None
    del rh0

    def q_for(ps):
        if rh_pgw is None:
            return q0
        q_rh = rh_to_q(rh_pgw, t_pgw, full_level_pressure(a, b, ps))
        return q0 + w_q * (q_rh - q0)

    # Surface pressure
    ps = ps0.copy()
    it, err_max = 0, 0.0
    if opt.ps_mode == "gcm":
        ps = ps0 + d["ps"]
    elif opt.ps_mode == "hydrostatic":
        phi_ref0 = geopotential_at_pressure(a, b, ps0, t0, q0, phis, opt.p_ref)
        dphi_clim = G * _interp_p_ref(d["zg"], plev, opt.p_ref)
        err_max = np.inf
        while err_max > opt.thresh_phi:
            it += 1
            if it > opt.max_iter:
                raise RuntimeError(f"The ps adjustment does not converge ({err_max:.3f} m2/s2 "
                                   f"after {opt.max_iter} iterations)")
            phi = geopotential_at_pressure(a, b, ps, t_pgw, q_for(ps), phis, opt.p_ref)
            err = (phi - phi_ref0) - dphi_clim
            err_max = float(np.abs(err).max())
            if err_max > opt.thresh_phi:
                ps = ps - opt.adj_factor * ps / (RD * t_pgw[-1]) * err
    elif opt.ps_mode != "none":
        raise ValueError(f"Unknown ps_mode: {opt.ps_mode}")
    return t_pgw, q_for(ps), ps, it, err_max


def surface_ts_delta(dts, lsm, nj, ni, opt):
    """Δts combined: open-sea Δts over the sea and GCM Δts over land."""
    sea = lsm < opt.lsm_threshold
    dts_sea, n_clean = ocean_only_delta(
        dts.reshape(nj, ni), lsm.reshape(nj, ni), opt.lsm_threshold,
        opt.coast_buffer, opt.k_neighbors, opt.idw_power, opt.sst_smooth_sigma)
    dts_sea = dts_sea.ravel()
    return np.where(sea, dts_sea, dts), dts_sea, sea, n_clean


def soil_delta(dts, dts_ann, depth, opt):
    """Δts damped with depth (PGW4ERA5)."""
    return dts_ann + np.exp(-depth / opt.soil_efold) * (dts - dts_ann)


def _soil_level(f, name):
    """GRIB level of soil layer 'name' (-1 if not present in the file)."""
    levs = [lev for (p, tl, lev) in f.index if p == PARAM[name] and tl == "depthBelowLandLayer"]
    return levs[0] if levs else -1


def _sum_near(field, pf, p):
    """Sum over columns of the field on the model level closest to p."""
    k = np.abs(pf - p).argmin(axis=0)
    return field[k, np.arange(field.shape[1])].sum()


def options_dict(opt):
    return asdict(opt)
