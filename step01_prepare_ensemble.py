#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 01: deltas of X CMIP6 models (+ ENSMEAN) for one or more scenarios on
the input IFS/ERA5 grid.

  1. Scenarios (--scenarios) under <cmip_root>, e.g.
       "Attribution_ssp245 Future_ssp585"   or   all
     Scenarios with year subdirectories (Attribution/ssp245/YYYY) use the
     event year: --attrib-year YYYY, or 'auto' = year of the validity times
     of the files.
     Without --scenarios, <cmip_root> is directly the model directory.
  2. Models (--models):
       "A B C"  explicit list (case-insensitive)
       all      all available and complete models
       auto     --n-models N selected automatically (--auto-method)
     Models without all the required variables, or that fail, are EXCLUDED
     with a warning and the workflow continues. With --common-models true
     the same models are used in all scenarios (intersection).
  3. Domain features on the native grid (for the selection and the report).
  4. Regridding to the input grid of the SELECTED models only (in
     parallel; existing files are reused if newer than the original deltas).
  5. ENSMEAN per scenario (if at least 2 models).
  6. <out_dir>/members.json and <out_dir>/members.txt ("SCENARIO/MEMBER PATH").

Usage:
  python3 step01_prepare_ensemble.py <input_root> <cmip_root> <out_dir> [options]
"""

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pgw_ml import ensemble as ens                                            # noqa: E402
from pgw_ml.gribio import find_files, read_valid_datetime                      # noqa: E402
from pgw_ml.parallel import robust_map                                         # noqa: E402
from pgw_ml.prepare import read_grid, check_rotated_grid, prepare_model       # noqa: E402

_GI = None


def _init(gi):
    global _GI
    _GI = gi


def _is_fresh(out_path, src_dir):
    if not os.path.isfile(out_path):
        return False
    t_out = os.path.getmtime(out_path)
    srcs = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.endswith("_delta.nc")]
    return all(os.path.getmtime(s) < t_out for s in srcs)


def _features(job):
    key, model_dir, valid, month = job
    try:
        return key, ens.domain_features(model_dir, _GI["lat"], _GI["lon"], valid, month), None
    except Exception as e:
        return key, None, repr(e)


def _regrid(job):
    key, src, out_path, margin, reuse = job
    if reuse and _is_fresh(out_path, src):
        return key, "reused", None
    logs = []
    try:
        prepare_model(_GI, src, out_path, margin, log=logs.append, tag="/".join(key))
        return key, "ok", logs
    except Exception as e:
        return key, "error", [repr(e)]


def str2bool(s):
    return str(s).lower() in ("1", "true", "yes", "y")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_root")
    ap.add_argument("cmip_root")
    ap.add_argument("out_dir")
    ap.add_argument("--scenarios", default="", help="'Attribution_ssp245 Future_ssp585' | all | ''")
    ap.add_argument("--attrib-year", default="auto", help="YYYY or auto (year of the files)")
    ap.add_argument("--models", default="all", help="'A B C' | all | auto")
    ap.add_argument("--n-models", type=int, default=5, help="N for --models auto")
    ap.add_argument("--auto-method", default="spread", choices=["spread", "representative"])
    ap.add_argument("--common-models", default="true")
    ap.add_argument("--suffix", default="", help="suffix of the model directories (e.g., _global)")
    ap.add_argument("--month", default="auto")
    ap.add_argument("--pattern", default="ECMWF_PL_*.grb", help="model-level files")
    ap.add_argument("--ps-mode", default="hydrostatic")
    ap.add_argument("--perturb-wind", default="false")
    ap.add_argument("--no-ensmean", action="store_true")
    ap.add_argument("--nproc", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--margin", type=float, default=3.0)
    ap.add_argument("--no-reuse", action="store_true")
    args = ap.parse_args()
    common = str2bool(args.common_models)
    root = os.path.abspath(args.cmip_root)

    # ── Files and grid ───────────────────────────────────────────────────────
    files = find_files(args.input_root, args.pattern)
    if not files:
        sys.exit(f"ERROR: no '{args.pattern}' files in {args.input_root}")
    valid = [read_valid_datetime(f) for f in files]
    print(f">> {len(files)} files; valid {min(valid):%Y-%m-%d %H} -> {max(valid):%Y-%m-%d %H}")
    gi = read_grid(files[0])
    print(f">> Grid ({files[0]}): {gi['gridType']} {gi['Ni']}x{gi['Nj']}; "
          f"uvRelativeToGrid={gi['uv_relative_to_grid']}")
    if gi["gridType"] == "rotated_ll":
        check_rotated_grid(gi)

    # ── Scenarios ────────────────────────────────────────────────────────────
    if args.attrib_year.lower() == "auto":
        cnt = collections.Counter(v.year for v in valid)
        year = cnt.most_common(1)[0][0]
        if len(cnt) > 1:
            print(f"  [WARNING] Files from several years {sorted(cnt)}: {year} is used for Attribution")
    else:
        year = int(args.attrib_year)
    spec = args.scenarios.replace(",", " ").split()
    if not spec:
        scen = {os.path.basename(root): (root, None)}
    else:
        if [s.lower() for s in spec] == ["all"]:
            spec = ens.list_scenarios(root)
        scen = {}
        for s in spec:
            tag, d, y = ens.resolve_scenario(root, s, year,
                                             nearest=args.attrib_year.lower() == "auto")
            scen[tag] = (d, y)
    print(">> Scenarios:")
    for tag, (d, y) in scen.items():
        print(f"     {tag:22s} {d}" + (f"   (event year {y})" if y else ""))

    # ── Candidates ───────────────────────────────────────────────────────────
    req = ens.required_vars(args.ps_mode, str2bool(args.perturb_wind))
    mode = args.models.strip()
    available, cands, excluded = {}, {}, {}
    for tag, (d, _) in scen.items():
        available[tag] = ens.discover(d, args.suffix)
        excluded[tag] = {}
        if mode.lower() in ("all", "auto"):
            c = list(available[tag])
        else:
            c, bad = ens.resolve_names(mode.replace(",", " ").split(), available[tag])
            for b in bad:
                excluded[tag][b] = "not found in the scenario"
        for m in list(c):
            miss = ens.missing_vars(available[tag][m], req)
            if miss:
                excluded[tag][m] = f"missing variables: {' '.join(miss)}"
                c.remove(m)
        cands[tag] = c
    if common and len(scen) > 1:
        inter = [m for m in cands[next(iter(scen))] if all(m in cands[t] for t in scen)]
        for tag in scen:
            for m in cands[tag]:
                if m not in inter:
                    lacking = [t for t in scen if m not in cands[t]]
                    excluded[tag][m] = f"not complete in {' '.join(lacking)} (common models)"
            cands[tag] = list(inter)
    for tag in scen:
        print(f">> {tag}: {len(available[tag])} models, {len(cands[tag])} candidates")
        for m, why in excluded[tag].items():
            print(f"     [EXCLUDED] {m}: {why}")
        if not cands[tag]:
            sys.exit(f"ERROR: {tag}: no valid model")

    nproc = max(1, args.nproc)
    # ── Domain features (native grid) ───────────────────────────────────
    jobs = [((tag, m), available[tag][m], valid, args.month)
            for tag in scen for m in cands[tag]]
    print(f">> Domain features of {len(jobs)} (scenario, model) pairs on the native grid ...")
    feats = {tag: {} for tag in scen}
    for (tag, m), f, err in robust_map(_features, jobs, nproc, _init, (gi,)):
        if err:
            excluded[tag][m] = f"failed to read deltas: {err}"
            print(f"     [EXCLUDED] {tag}/{m}: {err}")
        else:
            feats[tag][m] = f
    for tag in scen:
        cands[tag] = [m for m in cands[tag] if m in feats[tag]]
    if common and len(scen) > 1:
        inter = [m for m in cands[next(iter(scen))] if all(m in feats[t] for t in scen)]
        cands = {t: list(inter) for t in scen}
    for tag in scen:
        print(f"   {tag}:  {'model':20s} {'ΔT850-300':>10s} {'Δtas':>7s} {'ΔRH850-500':>11s} {'Δts':>7s}")
        for m in cands[tag]:
            f = feats[tag][m]
            print(f"   {'':{len(tag)}s}   {m:20s} {f['dT_trop']:+10.2f} {f['dtas']:+7.2f} "
                  f"{f['dRH']:+11.2f} {f['dts']:+7.2f}")

    # ── Selection ───────────────────────────────────────────────────────
    selected, sel_info = {}, {}
    if mode.lower() == "auto":
        if common:
            comb = {m: {f"{t}:{k}": feats[t][m][k] for t in scen for k in ens.FEATURES}
                    for m in cands[next(iter(scen))]}
            sel, info = ens.select_auto(comb, args.n_models, args.auto_method)
            for tag in scen:
                selected[tag], sel_info[tag] = list(sel), dict(info, mode="auto", common=True)
        else:
            for tag in scen:
                sub = {m: {k: feats[tag][m][k] for k in ens.FEATURES} for m in cands[tag]}
                selected[tag], info = ens.select_auto(sub, args.n_models, args.auto_method)
                sel_info[tag] = dict(info, mode="auto", common=False)
        for tag in scen:
            print(f">> {tag}: automatic selection ({args.auto_method}, N={args.n_models}): "
                  f"{' '.join(selected[tag])}")
            if "note" in sel_info[tag]:
                print(f"     {sel_info[tag]['note']}")
    else:
        for tag in scen:
            selected[tag] = list(cands[tag])
            sel_info[tag] = {"mode": mode}
            print(f">> {tag}: models ({len(selected[tag])}): {' '.join(selected[tag])}")

    # ── Regridding to the input grid ─────────────────────────────────
    on_grid = {tag: {m: os.path.join(args.out_dir, tag, m, "deltas_on_grid.nc")
                     for m in selected[tag]} for tag in scen}
    jobs = [((tag, m), available[tag][m], on_grid[tag][m], args.margin, not args.no_reuse)
            for tag in scen for m in selected[tag]]
    print(f">> Regridding {len(jobs)} (scenario, model) pairs to the grid ...")
    for (tag, m), status, logs in robust_map(_regrid, jobs, nproc, _init, (gi,)):
        if status == "error":
            excluded[tag][m] = f"regridding failed: {logs[0]}"
            selected[tag].remove(m)
            del on_grid[tag][m]
            print(f"     [EXCLUDED] {tag}/{m}: {logs[0]}")
        else:
            print(f"     [{'OK' if status == 'ok' else 'REUSED'}] {tag}/{m}")
            for line in logs or []:
                print("       ", line.strip())

    # ── ENSMEAN and output ───────────────────────────────────────────────────
    summary, lines = {}, []
    for tag, (d, y) in scen.items():
        members = {m: on_grid[tag][m] for m in selected[tag]}
        if not members:
            sys.exit(f"ERROR: {tag}: no model left")
        ensmean_path = None
        if len(members) >= 2 and not args.no_ensmean:
            ensmean_path = os.path.join(args.out_dir, tag, "ENSMEAN", "deltas_on_grid.nc")
            ens.ensmean(members, ensmean_path)
            print(f">> {tag}: ENSMEAN of {len(members)} models -> {ensmean_path}")
        elif len(members) < 2:
            print(f">> {tag}: only 1 model, no ENSMEAN")
        allm = dict(members)
        if ensmean_path:
            allm["ENSMEAN"] = ensmean_path
        for k, p in allm.items():
            lines.append(f"{tag}/{k} {p}")
        summary[tag] = {"dir": d, "year": y, "members": allm, "selected_models": selected[tag],
                        "candidates": cands[tag], "excluded": excluded[tag],
                        "features": feats[tag], "selection": sel_info[tag]}
    with open(os.path.join(args.out_dir, "members.json"), "w") as fh:
        json.dump({"scenarios": summary, "cmip_root": root, "month": args.month,
                   "common_models": common, "valid_dates": [v.isoformat() for v in valid]},
                  fh, indent=2)
    with open(os.path.join(args.out_dir, "members.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f">> {len(lines)} members in {os.path.join(args.out_dir, 'members.txt')}")


if __name__ == "__main__":
    main()
