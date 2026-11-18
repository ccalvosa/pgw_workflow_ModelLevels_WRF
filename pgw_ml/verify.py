"""
Quantitative verification that the deltas have been applied correctly.

For an (original, PGW) model-level pair, its surface counterpart (same
validity time in the surface file) and the deltas of its member, the
EXPECTED change of each field is computed and compared with the change
ACTUALLY written to the GRIB files. Columns are processed in blocks, so the
memory does not grow with the 3D size of the domain.

 A) Exact replica (whole domain): the expected change is recomputed with the
    package formulas and the final ps read from the PGW file. The error must
    stay within the GRIB packing noise, expressed in packing units
    (p.u. = quantization step of the message, 2^E · 10^-D). OK if <= 1 p.u.
      T, q, skt, sst, stl1-4, (u, v)
    For sp, msl, 2t and 2d, whose expected values depend on other packed
    fields (ln ps', T of the lowest level, 2t'), the tolerance adds the
    error propagated from those fields ("tol. ratio" <= 1).

 B) Physical and independent checks:
      * Δzg reached at p_ref versus the GCM target                    [m]
      * ΔRH reached versus the target ΔRH (troposphere, outside the
        bounded points) and RH2 reached versus RH2 + Δhurs           [%]
      * stratospheric q unchanged (above q_taper_top)
      * ΔT of sample columns with np.interp (independent interpolation)
      * SST mask preserved, q >= 0
"""

from datetime import datetime

import numpy as np
import eccodes as ec

from .apply import PGWOptions, surface_ts_delta, soil_delta
from .constants import G
from .gribio import IFSFile, PARAM, SOIL_DEPTH, read_time_subset
from .surface import msl_factor, dewpoint_perturbed
from .thermo import q_to_rh, rh_to_q, esat_water
from .vertical import (split_pv, full_level_pressure, geopotential_at_pressure,
                       interp_delta_to_model_levels)

# Tolerances of the checks
TOL_PACK = 1.0          # p.u.
TOL_DZ = 0.25           # m       error of Δzg at p_ref
TOL_DRH = 0.5           # %       RMSE of ΔRH at non-bounded points
TOL_STRAT_Q = 1.0       # p.u.    change of q in the stratosphere
TOL_INDEP_T = 1.0e-3    # K       independent ΔT replica


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _unit_of(gid):
    try:
        e = ec.codes_get(gid, "binaryScaleFactor")
        d = ec.codes_get(gid, "decimalScaleFactor")
        step = (2.0 ** e) * (10.0 ** -d)
        if np.isfinite(step) and step > 0:
            return step
    except Exception:
        pass
    bits = ec.codes_get(gid, "bitsPerValue")
    v = ec.codes_get_values(gid)
    rng = float(v.max() - v.min()) if v.size else 0.0
    return max(rng / (2.0 ** bits) if bits else 0.0, 1.0e-12)


def _pack_unit(f, name, type_of_level, level):
    return _unit_of(f.handles[f.index[(PARAM[name], type_of_level, level)]])


def _pack_units_3d(f, name):
    return np.array([_pack_unit(f, name, "hybrid", lev) for lev in f.hybrid_levels(name)])


class Acc:
    """Streaming statistics of an error field (bias, RMSE, max |e|, max p.u.)."""

    def __init__(self):
        self.n, self.s, self.s2, self.mx, self.mpu = 0, 0.0, 0.0, 0.0, None

    def add(self, e, unit=None, mask=None):
        if mask is not None:
            if unit is not None and np.ndim(unit) > 0:
                unit = np.broadcast_to(unit, e.shape)[mask]
            e = e[mask]
        if e.size == 0:
            return
        self.n += e.size
        self.s += float(e.sum())
        self.s2 += float((e ** 2).sum())
        self.mx = max(self.mx, float(np.abs(e).max()))
        if unit is not None:
            pu = float((np.abs(e) / unit).max())
            self.mpu = pu if self.mpu is None else max(self.mpu, pu)

    def result(self):
        if self.n == 0:
            out = {"bias": 0.0, "rmse": 0.0, "maxabs": 0.0, "n": 0}
        else:
            out = {"bias": self.s / self.n, "rmse": float(np.sqrt(self.s2 / self.n)),
                   "maxabs": self.mx, "n": self.n}
        out["max_pu"] = self.mpu if self.mpu is not None else 0.0
        return out


def _stats(e, unit=None, mask=None):
    a = Acc()
    a.add(e, unit, mask)
    r = a.result()
    if unit is None:
        r.pop("max_pu")
    return r


def _indep_dT(d_ta, plev, pf0, d_tas, ps0, depth):
    """Expected ΔT of the given columns with np.interp (independent of vertical.py)."""
    order = np.argsort(plev)
    lp = np.log(plev[order])
    out = np.empty(pf0.shape)
    for c in range(pf0.shape[1]):
        prof = d_ta[order, c]
        dstd = np.interp(np.log(pf0[:, c]), lp, prof)        # constant extrapolation
        if d_tas is not None:
            d_ps = np.interp(np.log(ps0[c]), lp, prof)
            w = np.clip(1.0 - (ps0[c] - pf0[:, c]) / depth, 0.0, 1.0)
            dstd = dstd + (d_tas[c] - d_ps) * w
        out[:, c] = dstd
    return out


class _SurfaceSource:
    """Uniform access to surface fields: from an IFSFile (fields in the
    model-level file) or from a time subset of the surface file."""

    def __init__(self, f=None, subset=None, units=None):
        self.f, self.sub, self.units = f, subset, units

    def has(self, name, tol="surface", lev=0):
        if self.f is not None:
            return self.f.has(name, tol, lev)
        return (PARAM[name], tol, lev) in self.sub

    def get(self, name, tol="surface", lev=0):
        if self.f is not None:
            return self.f.get2d(name, tol, lev)
        return self.sub[(PARAM[name], tol, lev)]

    def unit(self, name, tol="surface", lev=0):
        if self.f is not None:
            return _pack_unit(self.f, name, tol, lev)
        return self.units[(PARAM[name], tol, lev)]

    def levels(self, name, tol):
        keys = self.f.index if self.f is not None else self.sub
        return sorted(lev for (p, t, lev) in keys if p == PARAM[name] and t == tol)


def _subset_with_units(path, vt):
    sub = read_time_subset(path, vt)
    units = {}
    with open(path, "rb") as fh:
        while True:
            gid = ec.codes_grib_new_from_file(fh)
            if gid is None:
                break
            try:
                t = datetime.strptime(f"{ec.codes_get(gid, 'validityDate')}"
                                      f"{ec.codes_get(gid, 'validityTime'):04d}", "%Y%m%d%H%M")
                if t == vt:
                    units[(ec.codes_get(gid, "paramId"), ec.codes_get(gid, "typeOfLevel"),
                           ec.codes_get(gid, "level"))] = _unit_of(gid)
            finally:
                ec.codes_release(gid)
    return sub, units


# ──────────────────────────────────────────────────────────────────────────────
#  Main entry
# ──────────────────────────────────────────────────────────────────────────────
def verify_file(src, dst, deltas, opt: PGWOptions, sfc_pair=None, n_cols=300, seed=0):
    """Returns (metrics, extras). metrics: {name: {stats..., ok}}.
    sfc_pair: (original_surface_file, pgw_surface_file) or None."""
    fa, fb = IFSFile(src), IFSFile(dst)
    try:
        return _verify(fa, fb, deltas, opt, sfc_pair, n_cols, seed)
    finally:
        fa.close()
        fb.close()


def _verify(fa, fb, deltas, opt, sfc_pair, n_cols, seed):
    vdate, vtime = fa.valid_datetime()
    valid = datetime.strptime(f"{vdate}{vtime:04d}", "%Y%m%d%H%M")
    d = deltas.at(valid, opt.month)
    plev = deltas.plev
    gi = fa.grid_info()
    nj, ni = gi["Nj"], gi["Ni"]
    anchor, bd = opt.sfc_anchor, opt.sfc_blend_depth

    a, b = split_pv(fa.pv())
    ps0 = np.exp(fa.get2d("lnsp", "hybrid", 1)[0])
    ps1 = np.exp(fb.get2d("lnsp", "hybrid", 1)[0])
    u_lnsp = _pack_unit(fb, "lnsp", "hybrid", 1)
    phis = fa.get2d("z", "hybrid", 1)[0]
    t0, t1 = fa.get3d("t"), fb.get3d("t")
    q0, q1 = fa.get3d("q"), fb.get3d("q")
    uT = _pack_units_3d(fb, "t")[:, None]
    uq = _pack_units_3d(fb, "q")[:, None]
    nlev, npts = t0.shape
    rng = np.random.default_rng(seed)
    cols = np.sort(rng.choice(npts, size=min(n_cols, npts), replace=False))

    A = {k: Acc() for k in ("T", "Tind", "q", "dRH", "qstrat", "qnc", "dz")}
    n_trop = n_clip = 0
    prof = {k: np.zeros(nlev) for k in ("p", "dT", "dT_exp", "dRH", "dRH_tgt", "dq", "q0")}
    near = {k: 0.0 for k in ("dT850", "dT500", "dRH850", "dq850", "q0850")}
    map_dT_err = np.empty(npts)
    dz_err = np.full(npts, np.nan) if opt.ps_mode == "hydrostatic" else None
    sc_exp, sc_app, dT_cols, dRH_cols = [], [], [], []

    for s0 in range(0, npts, opt.chunk):
        sl = slice(s0, min(s0 + opt.chunk, npts))
        dc = {k: v[..., sl] for k, v in d.items() if isinstance(v, np.ndarray)}
        T0, T1, Q0, Q1 = t0[:, sl], t1[:, sl], q0[:, sl], q1[:, sl]
        P0, P1 = full_level_pressure(a, b, ps0[sl]), full_level_pressure(a, b, ps1[sl])
        dta = interp_delta_to_model_levels(dc["ta"], plev, P0,
                                           dc["tas"] if anchor else None, ps0[sl], bd)
        dT_exp = dta if opt.perturb_t else np.zeros_like(T0)
        dT = T1 - T0
        A["T"].add(dT - dT_exp, uT)
        map_dT_err[sl] = (dT - dT_exp)[-1]
        loc = cols[(cols >= sl.start) & (cols < sl.stop)] - sl.start
        if loc.size:
            if opt.perturb_t:
                ind = _indep_dT(dc["ta"][:, loc], plev, P0[:, loc],
                                dc["tas"][loc] if anchor else None, ps0[sl][loc], bd)
                A["Tind"].add(dT[:, loc] - ind)
            sc_exp.append(dT_exp[:, loc].ravel())
            sc_app.append(dT[:, loc].ravel())
            dT_cols.append(dT[:, loc].astype("float32"))

        rh0, rh1 = q_to_rh(Q0, T0, P0), q_to_rh(Q1, T1, P1)
        dhur = interp_delta_to_model_levels(dc["hur"], plev, P0,
                                            dc["hurs"] if anchor else None, ps0[sl], bd) / 100.0
        if opt.perturb_q:
            rh_tgt = rh0 + dhur
            rh_pgw = np.clip(rh_tgt, opt.rh_min_factor * rh0, np.maximum(rh0, 1.0))
            clipped = rh_pgw != rh_tgt
            w_q = np.clip((np.log(P0) - np.log(opt.q_taper_top)) /
                          (np.log(opt.q_taper_bottom) - np.log(opt.q_taper_top)), 0.0, 1.0)
            q_exp = Q0 + w_q * (rh_to_q(rh_pgw, T0 + dT_exp, P1) - Q0)
            A["q"].add(Q1 - q_exp, uq)
            trop = P0 >= opt.q_taper_bottom
            A["dRH"].add(100 * ((rh1 - rh0) - dhur), mask=trop & ~clipped)
            n_trop += int(trop.sum())
            n_clip += int((clipped & trop).sum())
            A["qstrat"].add(Q1 - Q0, uq, mask=P0 <= opt.q_taper_top)
        else:
            A["qnc"].add(Q1 - Q0, uq)
        if loc.size:
            dRH_cols.append((100 * (rh1 - rh0))[:, loc].astype("float32"))

        if opt.ps_mode == "hydrostatic":
            order = np.argsort(plev)
            lp = np.log(plev[order])
            k = int(np.clip(np.searchsorted(lp, np.log(opt.p_ref)), 1, lp.size - 1))
            w = (np.log(opt.p_ref) - lp[k - 1]) / (lp[k] - lp[k - 1])
            zg = dc["zg"][order]
            dzg_tgt = zg[k - 1] * (1 - w) + zg[k] * w
            phi0 = geopotential_at_pressure(a, b, ps0[sl], T0, Q0, phis[sl], opt.p_ref)
            phi1 = geopotential_at_pressure(a, b, ps1[sl], T1, Q1, phis[sl], opt.p_ref)
            e = (phi1 - phi0) / G - dzg_tgt
            dz_err[sl] = e
            A["dz"].add(e)

        # Relative change of q as a ratio of sums (robust to q ~ 0 or q < 0 in ERA5)
        dq = Q1 - Q0
        drh = 100 * (rh1 - rh0)
        for key, arr in (("p", P0 / 100.0), ("dT", dT), ("dT_exp", dT_exp), ("dRH", drh),
                         ("dRH_tgt", 100 * dhur), ("dq", dq), ("q0", Q0)):
            prof[key] += arr.sum(axis=1)
        for key, arr, p in (("dT850", dT, 85000), ("dT500", dT, 50000),
                            ("dRH850", drh, 85000), ("dq850", dq, 85000), ("q0850", Q0, 85000)):
            kk = np.abs(P0 - p).argmin(axis=0)
            near[key] += float(arr[kk, np.arange(arr.shape[1])].sum())

    M = {}
    M["T (replica)"] = A["T"].result()
    M["T (replica)"]["ok"] = M["T (replica)"]["max_pu"] <= TOL_PACK
    if opt.perturb_t:
        k = "ΔT columns (np.interp) [K]"
        M[k] = A["Tind"].result()
        M[k].pop("max_pu")
        M[k]["ok"] = M[k]["maxabs"] <= TOL_INDEP_T + 2 * uT.max()
    if opt.perturb_q:
        M["q (replica)"] = A["q"].result()
        M["q (replica)"]["ok"] = M["q (replica)"]["max_pu"] <= TOL_PACK
        k = "ΔRH troposphere [%]"
        M[k] = A["dRH"].result()
        M[k].pop("max_pu")
        M[k]["ok"] = M[k]["rmse"] <= TOL_DRH
        M[k]["frac_clipped"] = n_clip / max(n_trop, 1)
        k = "q stratosphere unchanged"
        M[k] = A["qstrat"].result()
        M[k]["ok"] = M[k]["max_pu"] <= TOL_STRAT_Q
    else:
        M["q (unchanged)"] = A["qnc"].result()
        M["q (unchanged)"]["ok"] = M["q (unchanged)"]["max_pu"] <= TOL_PACK
    # ERA5/IFS q contains a few slightly negative values; GRIB packing can round
    # tiny positive values to the (negative) reference value of the message.
    new_neg = (q1 < 0) & (q0 >= 0)
    beyond = new_neg & (q1 < -uq)
    M["q >= 0 (no new negatives > 1 p.u.)"] = {
        "min": float(q1.min()), "n_new_negative": int(new_neg.sum()),
        "n_beyond_1pu": int(beyond.sum()), "n_negative_in_original": int((q0 < 0).sum()),
        "ok": bool(not beyond.any())}
    if opt.ps_mode == "hydrostatic":
        k = f"Δzg({opt.p_ref / 100:.0f} hPa) [m]"
        M[k] = A["dz"].result()
        M[k].pop("max_pu")
        M[k]["ok"] = M[k]["maxabs"] <= TOL_DZ
    elif opt.ps_mode == "gcm":
        M["Δps vs GCM [Pa]"] = _stats((ps1 - ps0) - d["ps"])
        M["Δps vs GCM [Pa]"]["ok"] = M["Δps vs GCM [Pa]"]["maxabs"] <= 2.0
    else:
        M["ps unchanged [Pa]"] = _stats(ps1 - ps0)
        M["ps unchanged [Pa]"]["ok"] = M["ps unchanged [Pa]"]["maxabs"] <= 2.0

    # ── Wind ─────────────────────────────────────────────────────────────────
    dw_low = {}
    if opt.perturb_wind:
        for c, dv_ in (("u", "ua"), ("v", "va")):
            acc = Acc()
            uu = _pack_units_3d(fb, c)[:, None]
            x0, x1 = fa.get3d(c), fb.get3d(c)
            for s0 in range(0, npts, opt.chunk):
                sl = slice(s0, min(s0 + opt.chunk, npts))
                P0 = full_level_pressure(a, b, ps0[sl])
                exp = interp_delta_to_model_levels(d[dv_][:, sl], plev, P0)
                acc.add((x1[:, sl] - x0[:, sl]) - exp, uu)
            M[f"{c} (replica)"] = acc.result()
            M[f"{c} (replica)"]["ok"] = M[f"{c} (replica)"]["max_pu"] <= TOL_PACK
            dw_low[c] = x1[-1] - x0[-1]
            del x0, x1

    # ── Surface ──────────────────────────────────────────────────────────────
    if fa.has("lsm", "surface", 0):
        sa, sb = _SurfaceSource(f=fa), _SurfaceSource(f=fb)
    elif sfc_pair is not None:
        sa = _SurfaceSource(None, *_subset_with_units(sfc_pair[0], valid))
        sb = _SurfaceSource(None, *_subset_with_units(sfc_pair[1], valid))
    else:
        sa = sb = None
    dsst, sfc_means = None, {}
    if sa is not None and sa.has("lsm"):
        state = {"ps0": ps0, "ps1": ps1, "tL0": t0[-1], "tL1": t1[-1], "phis": phis}
        dsst, sfc_means = _verify_surface(M, sa, sb, d, deltas, opt, state, nj, ni,
                                          u_lnsp, float(uT[-1, 0]), dw_low)

    extras = {
        "valid": valid.isoformat(),
        "shape": (nj, ni),
        "p_hPa": prof["p"] / npts,
        "prof_dT": prof["dT"] / npts,
        "prof_dT_exp": prof["dT_exp"] / npts,
        "prof_dRH": prof["dRH"] / npts,
        "prof_dRH_tgt": prof["dRH_tgt"] / npts,
        "prof_dq_rel": 100 * prof["dq"] / np.where(prof["q0"] > 0, prof["q0"], np.nan),
        "dT_cols": np.concatenate(dT_cols, axis=1),
        "dRH_cols": np.concatenate(dRH_cols, axis=1),
        "map_dps": ps1 - ps0,
        "map_dT_low": t1[-1] - t0[-1],
        "map_dT_low_err": map_dT_err,
        "map_dsst": dsst,
        "map_dz_err": dz_err,
        "scatter_T": (np.concatenate(sc_exp), np.concatenate(sc_app)),
        "means": {"dps": float((ps1 - ps0).mean()),
                  "dT_low": float((t1 - t0)[-1].mean()),
                  "dT850": near["dT850"] / npts,
                  "dT500": near["dT500"] / npts,
                  "dRH850": near["dRH850"] / npts,
                  "dq_rel850": 100 * near["dq850"] / near["q0850"],
                  "dsst": float(np.nanmean(dsst)) if dsst is not None else float("nan"),
                  **sfc_means},
    }
    return M, extras


def _verify_surface(M, sa, sb, d, deltas, opt, st, nj, ni, u_lnsp, u_tlow, dw_low):
    """Surface checks (sp, msl, 2t, 2d, 10u/10v, skt, sst, stl1-4)."""
    means = {}
    lsm = sa.get("lsm")[0]
    dts_comb, _, _, _ = surface_ts_delta(d["ts"], lsm, nj, ni, opt)
    ratio = st["ps1"] / st["ps0"]

    def tol_check(name, err, tol):
        M[name] = _stats(err)
        M[name]["max_tol_ratio"] = float((np.abs(err) / tol).max())
        M[name]["ok"] = bool(M[name]["max_tol_ratio"] <= 1.0)

    # sp and msl: rounding + error propagated from the packed ln ps' (and T_L)
    if opt.ps_mode != "none":
        if sa.has("sp"):
            sp0, sp1 = sa.get("sp")[0], sb.get("sp")[0]
            tol_check("sp (replica) [Pa]", sp1 - sp0 * ratio,
                      0.5 * sb.unit("sp") + 0.5 * u_lnsp * np.abs(sp1) * 1.05 + 1e-9)
            means["dsp"] = float((sp1 - sp0).mean())
        if sa.has("msl") and opt.perturb_msl:
            m0, m1 = sa.get("msl")[0], sb.get("msl")[0]
            tol_check("msl (replica) [Pa]", m1 - m0 * msl_factor(st),
                      0.5 * sb.unit("msl")
                      + (0.5 * u_lnsp + 2.0 * u_tlow / 200.0) * np.abs(m1) * 1.05 + 1e-9)
            means["dmsl"] = float((m1 - m0).mean())

    # 2t and 2d
    if sa.has("2t"):
        t20, t21 = sa.get("2t")[0], sb.get("2t")[0]
        dt2 = (st["tL1"] - st["tL0"]) if opt.perturb_t else 0.0
        u2t = sb.unit("2t")
        tol_check("2t (replica) [K]", (t21 - t20) - dt2, 0.5 * u2t + u_tlow * 1.05 + 1e-9)
        means["d2t"] = float((t21 - t20).mean())
        if sa.has("2d"):
            d20, d21 = sa.get("2d")[0], sb.get("2d")[0]
            dh = d["hurs"] / 100.0 if opt.perturb_q else 0.0
            d2_exp, rh2, rh2_new = dewpoint_perturbed(t20, d20, t21, dh, opt.rh_min_factor)
            tol_check("2d (replica) [K]", d21 - d2_exp, 0.5 * sb.unit("2d") + 0.6 * u2t + 1e-9)
            rh2_reached = esat_water(d21) / esat_water(t21)
            bounded = rh2_new != rh2 + dh
            k = "RH2 reached vs RH2+Δhurs [%]"
            M[k] = _stats(100 * (rh2_reached - (rh2 + dh)), mask=~bounded)
            M[k]["ok"] = M[k]["rmse"] <= TOL_DRH
            M[k]["frac_clipped"] = float(np.mean(bounded))
            means["d2d"] = float((d21 - d20).mean())

    # 10-m wind
    for name, c in (("10u", "u"), ("10v", "v")):
        if opt.perturb_wind and c in dw_low and sa.has(name):
            w0, w1 = sa.get(name)[0], sb.get(name)[0]
            tol_check(f"{name} (replica) [m/s]", (w1 - w0) - dw_low[c],
                      0.5 * sb.unit(name) + 1e-3)

    # skt, sst, stl
    if sa.has("skt"):
        s0, s1 = sa.get("skt")[0], sb.get("skt")[0]
        M["skt (replica)"] = _stats((s1 - s0) - (dts_comb if opt.perturb_skt else 0.0), sb.unit("skt"))
        M["skt (replica)"]["ok"] = M["skt (replica)"]["max_pu"] <= TOL_PACK
    dsst = None
    if sa.has("sst"):
        (s0, m0), (s1, m1) = sa.get("sst"), sb.get("sst")
        M["SST mask preserved"] = {"ok": bool(np.array_equal(m0, m1))}
        M["sst (replica)"] = _stats((s1 - s0) - (dts_comb if opt.perturb_sst else 0.0),
                                    sb.unit("sst"), mask=~m0)
        M["sst (replica)"]["ok"] = M["sst (replica)"]["max_pu"] <= TOL_PACK
        dsst = np.where(m0, np.nan, s1 - s0)
    dts_ann = deltas.annual_mean("ts")
    for name, depth in SOIL_DEPTH.items():
        for lev in sa.levels(name, "depthBelowLandLayer"):
            s0 = sa.get(name, "depthBelowLandLayer", lev)[0]
            s1 = sb.get(name, "depthBelowLandLayer", lev)[0]
            exp = soil_delta(d["ts"], dts_ann, depth, opt) if opt.perturb_soil else 0.0
            k = f"{name} (replica)"
            M[k] = _stats((s1 - s0) - exp, sb.unit(name, "depthBelowLandLayer", lev))
            M[k]["ok"] = M[k]["max_pu"] <= TOL_PACK
    return dsst, means
