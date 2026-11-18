#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 04: verification that the deltas have been applied correctly, with
metrics, figures and an HTML report, for all scenarios and members
(models + ENSMEAN).

Checks (see pgw_ml/verify.py):
  * Per file and member: exact replica of T, q, skt, sst, stl1-4 (u, v)
    within the packing noise; Δzg(p_ref) reached versus the target; ΔRH
    reached versus the target; stratospheric q unchanged; ΔT of sample
    columns with an independent interpolation (np.interp); SST mask; q >= 0.
  * Prepared deltas: horizontal interpolation compared with xarray.interp on
    the native GCM deltas (sample points); ENSMEAN equal to the mean of the
    model deltas.
  * Ensemble: applied ΔT of the ENSMEAN versus the mean of the applied ΔT of
    the models (must agree: ΔT is linear in the delta). ΔRH and Δps are
    reported (nonlinear because of the RH bounds and the ps adjustment).

Output in <output_root>/verification/:
  report.html (self-contained), metrics.csv, summary.txt, figs/*.png

Usage:
  python3 step04_verify_deltas.py <input_root> <output_root> [--nproc N]
"""

import argparse
import base64
import csv
import html
import json
import os
import sys
from datetime import datetime
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pgw_ml.apply import PGWOptions                                     # noqa: E402
from pgw_ml.deltas import Deltas, time_weights                          # noqa: E402
from pgw_ml.verify import verify_file                                   # noqa: E402

TOL_ENS_T = 0.01        # K: ΔT(ENSMEAN) - mean(ΔT models)
TOL_REGRID = 0.02       # K: regridding versus xarray.interp
TOL_ENS_DELTA = 1e-4    # ENSMEAN deltas versus the mean


# ──────────────────────────────────────────────────────────────────────────────
#  Discovery of what has to be verified
# ──────────────────────────────────────────────────────────────────────────────
def load_plan(output_root):
    """{scenario: {"members": {m: {"deltas", "root", "opt", "files"}}, ...}}"""
    mj = os.path.join(output_root, "_deltas", "members.json")
    plan = {}
    if os.path.isfile(mj):
        info = json.load(open(mj))
        for tag, sc in info["scenarios"].items():
            plan[tag] = {"info": sc, "members": {}}
            for m, dpath in sc["members"].items():
                plan[tag]["members"][m] = {"deltas": dpath,
                                           "root": os.path.join(output_root, tag, m)}
    else:                                           # single member (old layout)
        plan[""] = {"info": {}, "members": {"": {
            "deltas": json.load(open(os.path.join(output_root, "pgw_options.json")))["deltas"],
            "root": output_root}}}
    for tag in plan:
        for m, mm in plan[tag]["members"].items():
            o = json.load(open(os.path.join(mm["root"], "pgw_options.json")))
            mm["opt"] = o["options"]
            mm["input_root"] = o["input_root"]
            diags = [json.loads(l) for l in open(os.path.join(mm["root"], "pgw_diagnostics.jsonl"))]
            sfc_by_time = {}
            mm["uncoupled"] = []
            for dg in diags:
                if dg["status"] == "ok" and dg.get("kind") == "sfc":
                    for t in dg["times"]:
                        sfc_by_time[t] = (dg["file"], dg["output"])
                    mm["uncoupled"] += [f"{os.path.basename(dg['file'])} {t}" for t in dg["uncoupled"]]
            mm["files"] = {os.path.relpath(dg["file"], o["input_root"]):
                           (dg["file"], dg["output"], sfc_by_time.get(dg.get("valid")))
                           for dg in diags if dg["status"] == "ok" and dg.get("kind") != "sfc"}
            mm["failed"] = [dg["file"] for dg in diags if dg["status"] != "ok"]
    return plan


# ──────────────────────────────────────────────────────────────────────────────
#  Work per (scenario, file): all members at once
# ──────────────────────────────────────────────────────────────────────────────
def _work(job):
    tag, rel, members = job          # members: {m: (src, dst, sfc_pair, deltas_path, opt_dict)}
    out = {"tag": tag, "rel": rel, "members": {}, "ens": None, "errors": []}
    dT, dRH, maps_low, dps = {}, {}, {}, {}
    for m, (src, dst, sfc_pair, dpath, optd) in members.items():
        try:
            opt = PGWOptions(**optd)
            dl = Deltas(dpath)
            M, ex = verify_file(src, dst, dl, opt, sfc_pair=sfc_pair)
            dl.close()
        except Exception as e:
            out["errors"].append(f"{tag}/{m}/{rel}: {e!r}")
            continue
        dT[m], dRH[m] = ex.pop("dT_cols"), ex.pop("dRH_cols")
        maps_low[m], dps[m] = ex["map_dT_low"], ex["map_dps"]
        out["members"][m] = {"metrics": M, "extras": ex}

    models = [m for m in dT if m != "ENSMEAN"]
    if "ENSMEAN" in dT and len(models) >= 2:
        # dT, dRH: sampled columns (300 x nlev); maps_low: lowest level, whole domain
        mT = np.mean([dT[m] for m in models], axis=0)
        mR = np.mean([dRH[m] for m in models], axis=0)
        eT = np.concatenate([(dT["ENSMEAN"] - mT).ravel(),
                             (maps_low["ENSMEAN"] - np.mean([maps_low[m] for m in models], axis=0))])
        eR = dRH["ENSMEAN"] - mR
        ep = dps["ENSMEAN"] - np.mean([dps[m] for m in models], axis=0)
        out["ens"] = {
            "ΔT(ENSMEAN) − mean ΔT of models [K]": {
                "bias": float(eT.mean()), "rmse": float(np.sqrt((eT ** 2).mean())),
                "maxabs": float(np.abs(eT).max()), "ok": bool(np.abs(eT).max() <= TOL_ENS_T)},
            "ΔRH(ENSMEAN) − mean ΔRH of models [%] (informative)": {
                "bias": float(eR.mean()), "rmse": float(np.sqrt((eR ** 2).mean())),
                "maxabs": float(np.abs(eR).max()), "ok": True},
            "Δps(ENSMEAN) − mean Δps of models [Pa] (informative)": {
                "bias": float(ep.mean()), "rmse": float(np.sqrt((ep ** 2).mean())),
                "maxabs": float(np.abs(ep).max()), "ok": True},
        }
        stack = np.stack([maps_low[m] for m in models])
        out["ens_maps"] = {"spread_dT_low": stack.std(axis=0),
                           "spread_dT_prof": np.stack([out["members"][m]["extras"]["prof_dT"]
                                                       for m in models]).std(axis=0)}
    return out


def _check_deltas(job):
    """Checks of the prepared deltas of a member (once per member)."""
    tag, m, dpath, member_paths, valid_dt, month = job
    import xarray as xr
    res = {}
    try:
        ds = xr.open_dataset(dpath)
        w = time_weights(valid_dt, month, ds)
        i = max(w, key=w.get)
        if m == "ENSMEAN":
            for v in ("ta", "hur", "zg", "tas", "ts", "hurs"):
                if v not in ds:
                    continue
                mean = np.mean([xr.open_dataset(p)[v].isel(month=i).values.astype(float)
                                for p in member_paths], axis=0)
                e = ds[v].isel(month=i).values.astype(float) - mean
                res[f"ENSMEAN {v} = mean of models"] = {
                    "maxabs": float(np.abs(e).max()), "ok": bool(np.abs(e).max() <= TOL_ENS_DELTA)}
        else:
            src = ds.attrs["source_deltas"]
            lat, lon = ds["lat"].values, ds["lon"].values
            rng = np.random.default_rng(1)
            pts = rng.choice(lat.size, size=min(300, lat.size), replace=False)
            lon_c = np.rad2deg(np.angle(np.mean(np.exp(1j * np.deg2rad(lon)))))
            for v, sel in (("tas", None), ("ta", 50000.0), ("ts", None)):
                nat = xr.open_dataset(os.path.join(src, f"{v}_delta.nc"))[v].isel(time=i)
                if sel is not None:
                    nat = nat.sel(plev=sel)
                nat = nat.assign_coords(lon=((nat["lon"] - lon_c + 180) % 360) - 180 + lon_c)
                nat = nat.sortby("lon").sortby("lat")
                ref = nat.interp(lat=xr.DataArray(lat[pts], dims="p"),
                                 lon=xr.DataArray(((lon[pts] - lon_c + 180) % 360) - 180 + lon_c,
                                                  dims="p")).values
                ours = ds[v].isel(month=i).values
                if sel is not None:
                    ours = ours[int(np.argmin(np.abs(ds["plev"].values - sel)))]
                ours = ours[pts]
                ok = np.isfinite(ref)
                e = ours[ok] - ref[ok]
                key = f"regrid {v}" + (f" {sel / 100:.0f} hPa" if sel else "") + " vs xarray.interp [K]"
                res[key] = {"maxabs": float(np.abs(e).max()) if e.size else 0.0,
                            "rmse": float(np.sqrt((e ** 2).mean())) if e.size else 0.0,
                            "n": int(ok.sum()),
                            "ok": bool(e.size == 0 or np.abs(e).max() <= TOL_REGRID)}
        ds.close()
    except Exception as e:
        res["error"] = {"msg": repr(e), "ok": False}
    return tag, m, res


# ──────────────────────────────────────────────────────────────────────────────
#  Figures
# ──────────────────────────────────────────────────────────────────────────────
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
    return plt


def _colors(names):
    import matplotlib
    models = [n for n in names if n != "ENSMEAN"]
    cmap = matplotlib.colormaps["tab20"].resampled(max(len(models), 1))
    c = {m: cmap(i) for i, m in enumerate(models)}
    c["ENSMEAN"] = "k"
    return c


def _savefig(fig, path, dpi=95):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def fig_profiles(tag, prof, colors, path):
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(14, 5.2), sharey=True)
    for m, p in prof.items():
        lw, z = (2.4, 10) if m == "ENSMEAN" else (1.0, 2)
        for i, k in enumerate(("dT", "dRH", "dq")):
            ax[i].plot(p[k], p["p"], color=colors[m], lw=lw, zorder=z, label=m)
        if m == "ENSMEAN":
            ax[0].plot(p["dT_exp"], p["p"], "--", color="tab:red", lw=1.4, zorder=11,
                       label="ENSMEAN expected")
            ax[1].plot(p["dRH_tgt"], p["p"], "--", color="tab:red", lw=1.4, zorder=11,
                       label="ENSMEAN target")
            if "spread" in p:
                ax[0].fill_betweenx(p["p"], p["dT"] - p["spread"], p["dT"] + p["spread"],
                                    color="0.6", alpha=0.35, zorder=1, label="±1σ models")
    for a, lab in zip(ax, ("ΔT [K]", "ΔRH [%]", "Δq/q [%]")):
        a.set_yscale("log")
        a.set_ylim(1050, 1)
        a.axvline(0, color="k", lw=0.5)
        a.set_xlabel(lab)
    ax[0].set_ylabel("mean level pressure [hPa]")
    ax[2].legend(fontsize=7, loc="lower left", ncol=1)
    fig.suptitle(f"{tag}: domain-mean profiles (mean over files)")
    return _savefig(fig, path)


def fig_bars(tag, means, colors, path):
    plt = _plt()
    keys = [("dT850", "ΔT 850 hPa [K]"), ("dT500", "ΔT 500 hPa [K]"),
            ("dRH850", "ΔRH 850 hPa [%]"), ("dq_rel850", "Δq/q 850 hPa (ratio of means) [%]"),
            ("dps", "Δps [Pa]"), ("dsst", "ΔSST sea [K]")]
    names = list(means)
    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    for a, (k, lab) in zip(ax.ravel(), keys):
        vals = [means[m][k] for m in names]
        a.bar(range(len(names)), vals, color=[colors[m] for m in names],
              edgecolor=["k" if m == "ENSMEAN" else "none" for m in names])
        mod = [means[m][k] for m in names if m != "ENSMEAN"]
        if mod:
            a.axhline(np.mean(mod), color="tab:red", lw=1, ls="--", label="mean of models")
        a.set_xticks(range(len(names)))
        a.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
        a.set_title(lab)
        a.axhline(0, color="k", lw=0.5)
    ax[0, 0].legend(fontsize=7)
    fig.suptitle(f"{tag}: domain means per member")
    fig.tight_layout()
    return _savefig(fig, path)


def _imshow(a, f2, title, plt, cmap="RdBu_r", sym=True):
    f2 = np.asarray(f2, dtype=float)
    lim = np.nanmax(np.abs(f2)) if sym else None
    if sym and (not np.isfinite(lim) or lim == 0):
        lim = 1.0
    im = a.imshow(f2, origin="upper", cmap=cmap,
                  vmin=-lim if sym else None, vmax=lim if sym else None)
    a.set_title(title, fontsize=9)
    a.set_xticks([])
    a.set_yticks([])
    plt.colorbar(im, ax=a, shrink=0.75)


def fig_maps_ens(tag, ex, spread, shape, path):
    plt = _plt()
    nj, ni = shape
    fig, ax = plt.subplots(1, 4, figsize=(17, 4))
    _imshow(ax[0], ex["map_dT_low"].reshape(nj, ni), "ENSMEAN ΔT level 137 [K]", plt)
    if spread is not None:
        _imshow(ax[1], spread.reshape(nj, ni), "intermodel σ of ΔT level 137 [K]", plt,
                cmap="viridis", sym=False)
    else:
        ax[1].axis("off")
    _imshow(ax[2], ex["map_dps"].reshape(nj, ni), "ENSMEAN Δps [Pa]", plt)
    if ex["map_dsst"] is not None:
        _imshow(ax[3], ex["map_dsst"].reshape(nj, ni), "ENSMEAN ΔSST [K]", plt)
    fig.suptitle(f"{tag}: maps (first file)")
    return _savefig(fig, path)


def fig_errors(tag, worst, colors, path):
    """Worst error per member: replica (p.u.) and physical checks, each in its
    own panel with its tolerance."""
    from pgw_ml import verify as V
    plt = _plt()
    names = list(worst)
    pu = sorted({k for m in names for k, v in worst[m].items() if "max_pu" in v})
    allk = sorted({k for m in names for k in worst[m]})
    phys = [(k, "rmse", V.TOL_DRH) for k in allk if k.startswith("ΔRH")] + \
           [(k, "maxabs", V.TOL_DZ) for k in allk if k.startswith("Δzg")] + \
           [(k, "maxabs", V.TOL_INDEP_T) for k in allk if k.startswith("ΔT col")] + \
           [(k, "maxabs", TOL_REGRID) for k in allk if k.startswith("regrid")]
    ncol = 1 + len(phys)
    fig, ax = plt.subplots(1, ncol, figsize=(6 + 2.6 * len(phys), 4.6),
                           gridspec_kw={"width_ratios": [3] + [1] * len(phys)})
    ax = np.atleast_1d(ax)
    x = np.arange(len(pu))
    wdt = 0.8 / max(len(names), 1)
    for i, m in enumerate(names):
        ax[0].bar(x + i * wdt, [worst[m].get(k, {}).get("max_pu", 0) for k in pu], wdt,
                  color=colors[m], label=m)
    ax[0].axhline(1.0, color="tab:red", ls="--", lw=1, label="tolerance")
    ax[0].set_xticks(x + 0.4 - wdt / 2)
    ax[0].set_xticklabels(pu, rotation=30, ha="right", fontsize=8)
    ax[0].set_ylabel("max |error| [GRIB p.u.]")
    ax[0].set_title("Exact replica (worst file)")
    ax[0].legend(fontsize=6, ncol=2)
    for a, (k, stat, tol) in zip(ax[1:], phys):
        vals = [worst[m].get(k, {}).get(stat, np.nan) for m in names]
        a.bar(range(len(names)), vals, color=[colors[m] for m in names])
        a.axhline(tol, color="tab:red", ls="--", lw=1)
        a.text(0.5, 0.92, f"max = {np.nanmax(vals):.2g}", transform=a.transAxes,
               ha="center", fontsize=8)
        a.set_xticks([])
        a.set_title(f"{k}\n({'RMSE' if stat == 'rmse' else 'max'}; tol. {tol:g})", fontsize=8)
    fig.suptitle(f"{tag}: errors in the application of the deltas")
    fig.tight_layout()
    return _savefig(fig, path)


def fig_scatter(tag, scat, colors, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    lo, hi = np.inf, -np.inf
    for m, (e, a) in scat.items():
        ax.scatter(e, a, s=2, alpha=0.4, color=colors[m], label=m, rasterized=True)
        lo, hi = min(lo, e.min(), a.min()), max(hi, e.max(), a.max())
    ax.plot([lo, hi], [lo, hi], "k-", lw=0.8)
    ax.set_xlabel("expected ΔT [K]")
    ax.set_ylabel("applied ΔT [K]")
    ax.set_title(f"{tag}: applied vs expected ΔT\n(300 columns × 137 levels, first file)",
                 fontsize=9)
    ax.legend(fontsize=6, markerscale=4)
    return _savefig(fig, path)


def fig_selection(tag, info, path):
    feats = info.get("features") or {}
    if len(feats) < 2:
        return None
    plt = _plt()
    sel = set(info.get("selected_models", []))
    names = sorted(feats)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for a, (kx, ky, lx, ly) in zip(ax, [("dT_trop", "dRH", "ΔT 850–300 hPa [K]", "ΔRH 850–500 hPa [%]"),
                                         ("dtas", "dT_trop", "Δtas [K]", "ΔT 850–300 hPa [K]")]):
        x = np.array([feats[m][kx] for m in names])
        y = np.array([feats[m][ky] for m in names])
        s = np.array([m in sel for m in names])
        a.scatter(x[~s], y[~s], c="0.7", s=25, label="candidates")
        a.scatter(x[s], y[s], c="tab:red", s=45, edgecolor="k", label="selected")
        a.scatter([x[s].mean()], [y[s].mean()], marker="*", s=200, c="tab:red",
                  edgecolor="k", label="ENSMEAN (selected)")
        a.scatter([x.mean()], [y.mean()], marker="*", s=200, c="0.4", edgecolor="k",
                  label="mean of all")
        for m, xi, yi in zip(names, x, y):
            a.annotate(m, (xi, yi), fontsize=6, alpha=0.8, xytext=(2, 2),
                       textcoords="offset points")
        a.set_xlabel(lx)
        a.set_ylabel(ly)
    ax[0].legend(fontsize=7)
    mode = info.get("selection", {}).get("mode", "")
    fig.suptitle(f"{tag}: domain features of the models (selection: {mode})")
    fig.tight_layout()
    return _savefig(fig, path)


def fig_member(tag, m, ex, path):
    plt = _plt()
    nj, ni = ex["shape"]
    fig = plt.figure(figsize=(17, 8))
    gs = fig.add_gridspec(2, 4)
    p = ex["p_hPa"]
    a = fig.add_subplot(gs[0, 0])
    a.plot(ex["prof_dT"], p, "-", label="applied")
    a.plot(ex["prof_dT_exp"], p, "--", label="expected")
    a.set_yscale("log"); a.set_ylim(1050, 1); a.set_xlabel("ΔT [K]"); a.legend(fontsize=7)
    a.set_ylabel("p [hPa]")
    a = fig.add_subplot(gs[0, 1])
    a.plot(ex["prof_dRH"], p, "-", label="reached")
    a.plot(ex["prof_dRH_tgt"], p, "--", label="target (unbounded)")
    a.set_yscale("log"); a.set_ylim(1050, 1); a.set_xlabel("ΔRH [%]"); a.legend(fontsize=7)
    a = fig.add_subplot(gs[0, 2])
    a.plot(ex["prof_dq_rel"], p)
    a.set_yscale("log"); a.set_ylim(1050, 1); a.set_xlabel("Δq/q [%]")
    a = fig.add_subplot(gs[0, 3])
    e, ap_ = ex["scatter_T"]
    a.scatter(e, ap_, s=2, alpha=0.4, rasterized=True)
    lo, hi = min(e.min(), ap_.min()), max(e.max(), ap_.max())
    a.plot([lo, hi], [lo, hi], "k-", lw=0.8)
    a.set_xlabel("expected ΔT [K]"); a.set_ylabel("applied ΔT [K]")
    _imshow(fig.add_subplot(gs[1, 0]), ex["map_dps"].reshape(nj, ni), "Δps [Pa]", plt)
    _imshow(fig.add_subplot(gs[1, 1]), ex["map_dT_low_err"].reshape(nj, ni),
            "error ΔT level 137 [K]", plt)
    if ex["map_dz_err"] is not None:
        _imshow(fig.add_subplot(gs[1, 2]), ex["map_dz_err"].reshape(nj, ni),
                "error Δzg(p_ref) [m]", plt)
    if ex["map_dsst"] is not None:
        _imshow(fig.add_subplot(gs[1, 3]), ex["map_dsst"].reshape(nj, ni), "ΔSST [K]", plt)
    fig.suptitle(f"{tag} / {m}  ({ex['valid']})")
    fig.tight_layout()
    return _savefig(fig, path, dpi=70)


def fig_scenarios(prof_by_tag, path):
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(14, 5.2), sharey=True)
    for tag, p in prof_by_tag.items():
        for i, k in enumerate(("dT", "dRH", "dq")):
            l, = ax[i].plot(p[k], p["p"], lw=2, label=tag)
            if k == "dT" and "spread" in p:
                ax[i].fill_betweenx(p["p"], p[k] - p["spread"], p[k] + p["spread"],
                                    color=l.get_color(), alpha=0.2)
    for a, lab in zip(ax, ("ΔT [K]", "ΔRH [%]", "Δq/q [%]")):
        a.set_yscale("log"); a.set_ylim(1050, 1); a.axvline(0, color="k", lw=0.5)
        a.set_xlabel(lab)
    ax[0].set_ylabel("mean level pressure [hPa]")
    ax[0].legend(fontsize=8)
    fig.suptitle("Scenario comparison (ENSMEAN, or the only member; band ±1σ models)")
    return _savefig(fig, path)


# ──────────────────────────────────────────────────────────────────────────────
#  Report
# ──────────────────────────────────────────────────────────────────────────────
CSS = """
:root{--bg:#fff;--fg:#1d232a;--mut:#5b6670;--line:#dde2e6;--card:#f6f8f9;--ok:#1a7f37;--bad:#c62828}
@media (prefers-color-scheme:dark){:root{--bg:#15191d;--fg:#e4e8eb;--mut:#9aa6b0;--line:#2d343a;--card:#1d2328;--ok:#4cc26b;--bad:#ff6b6b}}
body{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;margin:0 auto;max-width:1400px;padding:16px}
h1{font-size:22px;margin:.2em 0}h2{font-size:18px;border-bottom:1px solid var(--line);padding-bottom:4px;margin-top:2em}
h3{font-size:15px;margin-top:1.4em}.mut{color:var(--mut)}
table{border-collapse:collapse;margin:8px 0;font-size:12.5px;display:block;overflow-x:auto}
th,td{border:1px solid var(--line);padding:3px 8px;text-align:right;white-space:nowrap}
th{background:var(--card)}td:first-child,th:first-child{text-align:left}
.ok{color:var(--ok);font-weight:600}.bad{color:var(--bad);font-weight:700}
img{max-width:100%;background:#fff;border:1px solid var(--line);border-radius:4px;margin:6px 0}
.badge{display:inline-block;padding:3px 10px;border-radius:12px;font-weight:700;color:#fff}
.b-ok{background:var(--ok)}.b-bad{background:var(--bad)}
details{margin:6px 0}summary{cursor:pointer;font-weight:600}
code{background:var(--card);padding:1px 4px;border-radius:3px}
"""


def _img(path):
    if not path or not os.path.isfile(path):
        return ""
    b = base64.b64encode(open(path, "rb").read()).decode()
    return f'<img src="data:image/png;base64,{b}" alt="{html.escape(os.path.basename(path))}">'


def _okc(ok):
    return '<span class="ok">OK</span>' if ok else '<span class="bad">FAIL</span>'


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.3g}" if abs(v) < 1e4 else f"{v:.3e}"
    return html.escape(str(v))


def write_html(path, ctx):
    h = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
         f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
         f"<title>PGW Verification</title><style>{CSS}</style></head><body>"]
    badge = ('<span class="badge b-ok">VERIFICATION PASSED</span>' if ctx["n_bad"] == 0 else
             f'<span class="badge b-bad">{ctx["n_bad"]} FAILED CHECKS</span>')
    h.append(f"<h1>PGW delta verification · WRF (IFS/ERA5 model levels)</h1>{badge}")
    h.append(f"<p class='mut'>Generated {datetime.now():%Y-%m-%d %H:%M} · input "
             f"<code>{html.escape(ctx['input_root'])}</code> · output "
             f"<code>{html.escape(ctx['output_root'])}</code></p>")
    h.append("<h2>Summary</h2><table><tr><th>Scenario</th><th>Member</th><th>Files</th>"
             "<th>Failed checks</th><th>ΔT850 [K]</th><th>ΔT500 [K]</th><th>ΔRH850 [%]</th>"
             "<th>Δps [Pa]</th><th>ΔSST [K]</th></tr>")
    for (tag, m), r in ctx["summary"].items():
        h.append(f"<tr><td>{html.escape(tag)}</td><td>{html.escape(m) or '—'}</td><td>{r['n']}</td>"
                 f"<td>{'<span class=bad>' + str(r['bad']) + '</span>' if r['bad'] else '<span class=ok>0</span>'}</td>"
                 + "".join(f"<td>{r['means'][k]:+.2f}</td>" for k in
                           ("dT850", "dT500", "dRH850", "dps", "dsst")) + "</tr>")
    h.append("</table>")
    h.append("<details><summary>What is checked and tolerances</summary><ul>"
             "<li><b>Exact replica</b> (T, q, skt, sst, stl1-4, u, v): the expected change is "
             "recomputed with the package formulas and the final ps of the file; the maximum "
             "error must be ≤ 1 GRIB packing unit (p.u. = quantization step 2<sup>E</sup>·10<sup>−D</sup>; "
             "rounding gives 0.5).</li>"
             "<li><b>ΔT columns (np.interp)</b>: 300 columns with an independent vertical "
             "interpolation; ≤ 1e-3 K + packing.</li>"
             "<li><b>Δzg(p_ref)</b>: Φ<sub>pgw</sub>−Φ<sub>orig</sub> at p_ref versus g·Δzg of the GCM; "
             "max ≤ 0.25 m.</li>"
             "<li><b>ΔRH troposphere</b>: RH recomputed with the final T, q, p versus RH+ΔRH, "
             "outside the bounded points; RMSE ≤ 0.5 %.</li>"
             "<li><b>q stratosphere</b>: q does not change above q_taper_top (≤ 1 p.u.).</li>"
             "<li><b>regrid</b>: deltas on the grid versus <code>xarray.interp</code> on the "
             "native deltas (300 points); ≤ 0.02 K.</li>"
             "<li><b>ENSMEAN</b>: deltas = mean of the models (≤ 1e-4) and applied ΔT = mean "
             "of the applied ΔT (≤ 0.01 K).</li></ul></details>")
    if ctx.get("fig_scen"):
        h.append("<h2>Scenario comparison</h2>" + _img(ctx["fig_scen"]))
    for tag, sc in ctx["scen"].items():
        h.append(f"<h2>Scenario {html.escape(tag) or '(single)'}</h2>")
        inf = sc["info"]
        if inf:
            sel = inf.get("selection", {})
            h.append(f"<p>Delta directory: <code>{html.escape(str(inf.get('dir')))}</code>"
                     + (f" · event year {inf['year']}" if inf.get("year") else "")
                     + f"<br>Selection: <b>{html.escape(str(sel.get('mode')))}</b>"
                     + (f" ({html.escape(str(sel.get('method')))})" if sel.get("method") else "")
                     + f" · models: {html.escape(' '.join(inf.get('selected_models', [])))}</p>")
            if inf.get("excluded"):
                h.append("<details><summary>Excluded models ("
                         f"{len(inf['excluded'])})</summary><table><tr><th>Model</th><th>Reason</th></tr>"
                         + "".join(f"<tr><td>{html.escape(k)}</td><td style='text-align:left'>"
                                   f"{html.escape(v)}</td></tr>" for k, v in inf["excluded"].items())
                         + "</table></details>")
        for k in ("selection", "profiles", "bars", "maps", "errors", "scatter"):
            if sc["figs"].get(k):
                h.append(_img(sc["figs"][k]))
        if sc["ens"]:
            h.append("<h3>ENSMEAN consistency (worst file)</h3><table><tr><th>Check</th>"
                     "<th>bias</th><th>RMSE</th><th>max |e|</th><th></th></tr>")
            for k, v in sc["ens"].items():
                h.append(f"<tr><td>{html.escape(k)}</td><td>{_fmt(v.get('bias', ''))}</td>"
                         f"<td>{_fmt(v.get('rmse', ''))}</td><td>{_fmt(v['maxabs'])}</td>"
                         f"<td>{_okc(v['ok'])}</td></tr>")
            h.append("</table>")
        h.append("<h3>Metrics per member (worst file of each check)</h3>")
        for m, W in sc["worst"].items():
            nb = sum(not v.get("ok", True) for v in W.values())
            h.append(f"<details{' open' if nb else ''}><summary>{html.escape(m) or 'member'}: "
                     f"{_okc(nb == 0)} {nb} failed</summary>")
            h.append("<table><tr><th>Check</th><th>bias</th><th>RMSE</th><th>max |e|</th>"
                     "<th>max p.u.</th><th>other</th><th></th></tr>")
            for k, v in W.items():
                extra = {kk: vv for kk, vv in v.items()
                         if kk not in ("bias", "rmse", "maxabs", "max_pu", "ok", "n", "file")}
                h.append(f"<tr><td>{html.escape(k)}</td><td>{_fmt(v.get('bias', ''))}</td>"
                         f"<td>{_fmt(v.get('rmse', ''))}</td><td>{_fmt(v.get('maxabs', ''))}</td>"
                         f"<td>{_fmt(v.get('max_pu', ''))}</td>"
                         f"<td>{html.escape(', '.join(f'{a}={_fmt(b)}' for a, b in extra.items()))}</td>"
                         f"<td>{_okc(v.get('ok', True))}</td></tr>")
            h.append("</table>")
            if sc["figs"].get(("member", m)):
                h.append(_img(sc["figs"][("member", m)]))
            h.append("</details>")
    if ctx.get("warnings"):
        h.append("<h2>Warnings</h2><ul>" + "".join(f"<li>{html.escape(w)}</li>"
                                                    for w in ctx["warnings"]) + "</ul>")
    if ctx["errors"]:
        h.append("<h2>Errors</h2><ul>" + "".join(f"<li class='bad'>{html.escape(e)}</li>"
                                                  for e in ctx["errors"]) + "</ul>")
    h.append("</body></html>")
    open(path, "w").write("\n".join(h))


# ──────────────────────────────────────────────────────────────────────────────
def _worse(a, b):
    """Selects the 'worse' metric entry (for the per-member summary)."""
    if a is None:
        return b
    if a.get("ok", True) != b.get("ok", True):
        return a if not a.get("ok", True) else b
    for k in ("max_pu", "maxabs", "rmse"):
        if k in a and k in b:
            return a if a[k] >= b[k] else b
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_root")
    ap.add_argument("output_root")
    ap.add_argument("--nproc", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--chunk", type=int, default=0, help="columns per block (0: as in step02)")
    args = ap.parse_args()
    out_root = os.path.abspath(args.output_root)
    vdir = os.path.join(out_root, "verification")
    fdir = os.path.join(vdir, "figs")
    os.makedirs(fdir, exist_ok=True)

    plan = load_plan(out_root)
    jobs, djobs = [], []
    for tag, sc in plan.items():
        rels = sorted(set().union(*[set(mm["files"]) for mm in sc["members"].values()]))
        for rel in rels:
            members = {m: (*mm["files"][rel], mm["deltas"],
                           dict(mm["opt"], **({"chunk": args.chunk} if args.chunk else {})))
                       for m, mm in sc["members"].items() if rel in mm["files"]}
            jobs.append((tag, rel, members))
        first = next(iter(sc["members"].values()))
        if rels:
            from pgw_ml.gribio import read_valid_datetime
            vdt = read_valid_datetime(first["files"][rels[0]][0])
            mpaths = [mm["deltas"] for m, mm in sc["members"].items() if m != "ENSMEAN"]
            for m, mm in sc["members"].items():
                djobs.append((tag, m, mm["deltas"], mpaths, vdt, mm["opt"]["month"]))
    print(f">> Verifying {len(jobs)} (scenario, file) pairs with "
          f"{sum(len(j[2]) for j in jobs)} original/PGW pairs; {args.nproc} processes")

    rows, errors = [], []
    results = {tag: [] for tag in plan}
    dchecks = {tag: {} for tag in plan}
    with Pool(max(1, min(args.nproc, len(jobs) or 1))) as pool:
        for tag, m, res in pool.imap_unordered(_check_deltas, djobs):
            dchecks[tag][m] = res
        for r in pool.imap_unordered(_work, jobs):
            results[r["tag"]].append(r)
            errors += r["errors"]
            nb = sum(not v.get("ok", True) for mm in r["members"].values()
                     for v in mm["metrics"].values())
            print(f"  [{'OK ' if nb == 0 and not r['errors'] else 'BAD'}] {r['tag']}/{r['rel']}: "
                  f"{len(r['members'])} members, {nb} failed checks")

    ctx = {"input_root": args.input_root, "output_root": out_root, "summary": {},
           "scen": {}, "errors": errors, "n_bad": len(errors)}
    prof_ens = {}
    for tag, sc in plan.items():
        res = sorted(results[tag], key=lambda r: r["rel"])
        names = list(sc["members"])
        colors = _colors(names)
        worst = {m: {} for m in names}
        prof_acc, mean_acc, nfile = {}, {}, {m: 0 for m in names}
        ens_worst = {}
        spread_prof = []
        for r in res:
            for m, mm in r["members"].items():
                nfile[m] += 1
                for k, v in mm["metrics"].items():
                    v = dict(v, file=r["rel"])
                    worst[m][k] = _worse(worst[m].get(k), v)
                    rows.append({"scenario": tag, "member": m, "file": r["rel"], "check": k,
                                 **{c: v.get(c, "") for c in ("ok", "bias", "rmse", "maxabs",
                                                             "max_pu", "n")}})
                ex = mm["extras"]
                pa = prof_acc.setdefault(m, {k: [] for k in ("p", "dT", "dT_exp", "dRH",
                                                             "dRH_tgt", "dq")})
                for k, ek in (("p", "p_hPa"), ("dT", "prof_dT"), ("dT_exp", "prof_dT_exp"),
                              ("dRH", "prof_dRH"), ("dRH_tgt", "prof_dRH_tgt"), ("dq", "prof_dq_rel")):
                    pa[k].append(ex[ek])
                mean_acc.setdefault(m, []).append(ex["means"])
            if r.get("ens"):
                for k, v in r["ens"].items():
                    ens_worst[k] = _worse(ens_worst.get(k), v)
                    rows.append({"scenario": tag, "member": "ENSMEAN", "file": r["rel"], "check": k,
                                 **{c: v.get(c, "") for c in ("ok", "bias", "rmse", "maxabs")}})
                spread_prof.append(r["ens_maps"]["spread_dT_prof"])
        for m, res_d in dchecks[tag].items():
            for k, v in res_d.items():
                worst[m][k] = v
                rows.append({"scenario": tag, "member": m, "file": "(deltas)", "check": k,
                             **{c: v.get(c, "") for c in ("ok", "rmse", "maxabs", "n")}})
        prof = {m: {k: np.mean(v, axis=0) for k, v in pa.items()} for m, pa in prof_acc.items()}
        if spread_prof and "ENSMEAN" in prof:
            prof["ENSMEAN"]["spread"] = np.mean(spread_prof, axis=0)
        means = {m: {k: float(np.nanmean([d[k] for d in v])) for k in v[0]}
                 for m, v in mean_acc.items()}
        for m in names:
            if m not in means:
                continue
            bad = sum(not v.get("ok", True) for v in worst[m].values())
            if m == "ENSMEAN":
                bad += sum(not v["ok"] for v in ens_worst.values())
            ctx["n_bad"] += bad
            ctx["summary"][(tag, m)] = {"n": nfile[m], "bad": bad, "means": means[m]}

        safe = tag or "single"
        figs = {}
        try:
            figs["selection"] = fig_selection(tag, sc["info"], os.path.join(fdir, f"selection_{safe}.png"))
            if prof:
                figs["profiles"] = fig_profiles(tag, prof, colors, os.path.join(fdir, f"profiles_{safe}.png"))
                figs["bars"] = fig_bars(tag, means, colors, os.path.join(fdir, f"bars_{safe}.png"))
                figs["errors"] = fig_errors(tag, {m: worst[m] for m in means}, colors,
                                            os.path.join(fdir, f"errors_{safe}.png"))
            if res:
                r0 = res[0]
                key = "ENSMEAN" if "ENSMEAN" in r0["members"] else next(iter(r0["members"]), None)
                if key:
                    figs["maps"] = fig_maps_ens(tag, r0["members"][key]["extras"],
                                                r0.get("ens_maps", {}).get("spread_dT_low"),
                                                r0["members"][key]["extras"]["shape"],
                                                os.path.join(fdir, f"maps_{safe}.png"))
                    figs["scatter"] = fig_scatter(tag, {m: mm["extras"]["scatter_T"] for m, mm in
                                                        r0["members"].items()}, colors,
                                                  os.path.join(fdir, f"scatter_{safe}.png"))
                for m, mm in r0["members"].items():
                    figs[("member", m)] = fig_member(tag, m, mm["extras"], os.path.join(
                        fdir, f"member_{safe}_{m or 'single'}.png"))
        except ImportError:
            print("  (no matplotlib: no figures)")
        if prof:
            prof_ens[tag or "single"] = prof.get("ENSMEAN") or next(iter(prof.values()))
        ctx["scen"][tag] = {"info": sc["info"], "figs": figs, "worst": worst, "ens": ens_worst}
        for m, mm in sc["members"].items():
            if mm["failed"]:
                errors += [f"{tag}/{m}: step02 failed for {f}" for f in mm["failed"]]
                ctx["n_bad"] += len(mm["failed"])
            if mm.get("uncoupled"):
                ctx.setdefault("warnings", []).append(
                    f"{tag}/{m}: {len(mm['uncoupled'])} surface times without model-level file "
                    f"(sp, msl unchanged; not verified): {mm['uncoupled'][0]}"
                    + (f" ... {mm['uncoupled'][-1]}" if len(mm['uncoupled']) > 1 else ""))

    if len(prof_ens) > 1:
        try:
            ctx["fig_scen"] = fig_scenarios(prof_ens, os.path.join(fdir, "scenarios.png"))
        except ImportError:
            pass

    with open(os.path.join(vdir, "metrics.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["scenario", "member", "file", "check", "ok", "bias",
                                           "rmse", "maxabs", "max_pu", "n"])
        w.writeheader()
        w.writerows(rows)
    lines = [f"{'scenario':22s} {'member':20s} {'file':>4s} {'failed':>6s} {'ΔT850':>7s} "
             f"{'ΔT500':>7s} {'ΔRH850':>7s} {'Δps':>7s} {'ΔSST':>6s}"]
    for (tag, m), r in ctx["summary"].items():
        mm = r["means"]
        lines.append(f"{tag:22s} {m:20s} {r['n']:4d} {r['bad']:6d} {mm['dT850']:+7.2f} "
                     f"{mm['dT500']:+7.2f} {mm['dRH850']:+7.2f} {mm['dps']:+7.1f} {mm['dsst']:+6.2f}")
    for (tag, m), r in ctx["summary"].items():
        for k, v in ctx["scen"][tag]["worst"][m].items():
            if not v.get("ok", True):
                lines.append(f"  FAIL {tag}/{m}: {k}: " + ", ".join(
                    f"{a}={_fmt(b)}" for a, b in v.items() if a != "ok"))
        if m == "ENSMEAN":
            for k, v in ctx["scen"][tag]["ens"].items():
                if not v["ok"]:
                    lines.append(f"  FAIL {tag}/ENSMEAN: {k}: maxabs={_fmt(v['maxabs'])}")
    lines += [f"  WARNING {w}" for w in ctx.get("warnings", [])]
    lines += [f"  ERROR {e}" for e in errors]
    txt = "\n".join(lines)
    open(os.path.join(vdir, "summary.txt"), "w").write(txt + "\n")
    print(txt)
    write_html(os.path.join(vdir, "report.html"), ctx)
    print(f">> Report: {os.path.join(vdir, 'report.html')}")
    print(f">> Metrics: {os.path.join(vdir, 'metrics.csv')}")
    if ctx["n_bad"]:
        sys.exit(f"ERROR: {ctx['n_bad']} failed checks")
    print(">> Delta verification passed.")


if __name__ == "__main__":
    main()
