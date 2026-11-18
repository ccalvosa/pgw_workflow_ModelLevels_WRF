#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 02: PGW applied to the IFS/ERA5 model-level and surface GRIB files read
by WPS (ungrib) for WRF.

Searches <input_root> recursively for
  * model-level files matching --pattern     (default 'ECMWF_PL_*.grb'),
  * surface files matching --sfc-pattern     (default 'ECMWF_SFC*.grb'),
and writes the perturbed files under <output_root> with the SAME directory
structure and the SAME file names, so that WPS can read them by changing the
input path only.

Two phases, both parallel over (member, file):
  A) model levels: T, q, lnsp (and u, v); the state of the lowest model level
     is saved to <member>/_state/YYYYMMDDHHMM.npz;
  B) surface files: sp, msl, 2t, 2d, (10u, 10v), skt, sst, stl1-4, using the
     saved state of each validity time (see pgw_ml/surface.py).

<deltas> can be:
  * a deltas_on_grid.nc  -> single member; output in <output_root>/...
  * a members.txt (from step01_prepare_ensemble.py, lines "NAME PATH")
    -> several members (models + ENSMEAN); output in <output_root>/<NAME>/...
    All (member, file) tasks share the same Pool.

Usage:
  python3 step02_apply_pgw.py <input_root> <output_root> <deltas> [options]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pgw_ml.apply import PGWOptions, apply_pgw_to_file, options_dict   # noqa: E402
from pgw_ml.deltas import Deltas                                        # noqa: E402
from pgw_ml.gribio import find_files                                    # noqa: E402
from pgw_ml.surface import apply_pgw_to_sfc_file, save_state            # noqa: E402

_DELTAS = {}
_OPT = None


def _init(opt):
    global _OPT
    _OPT = opt


def _get_deltas(path):
    """One Deltas object per member and process (lazy monthly reading)."""
    if path not in _DELTAS:
        _DELTAS[path] = Deltas(path)
    return _DELTAS[path]


def _work(job):
    member, deltas_path, src, dst, kind, state_dir = job
    t0 = time.time()
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".tmp"
    try:
        if kind == "ml":
            diag = apply_pgw_to_file(src, tmp, _get_deltas(deltas_path), _OPT, log=lambda *a: None)
            st = diag.pop("_state")
            save_state(state_dir, datetime.fromisoformat(diag["valid"]), st)
            diag["kind"] = "ml"
        else:
            diag = apply_pgw_to_sfc_file(src, tmp, _get_deltas(deltas_path), _OPT, state_dir)
        os.replace(tmp, dst)
        diag["seconds"] = round(time.time() - t0, 1)
        diag["output"] = dst
        diag["status"] = "ok"
    except Exception as e:                       # report and continue
        if os.path.exists(tmp):
            os.remove(tmp)
        diag = {"file": src, "status": "error", "error": repr(e)}
    diag["member"] = member
    return diag


def read_members(path):
    """{name: deltas_on_grid.nc} from members.txt, or a single member ''."""
    if path.endswith(".txt"):
        out = {}
        for line in open(path):
            if line.strip():
                name, p = line.split(None, 1)
                out[name] = p.strip()
        return out
    return {"": os.path.abspath(path)}


def str2bool(s):
    return str(s).lower() in ("1", "true", "yes", "y")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_root")
    ap.add_argument("output_root")
    ap.add_argument("deltas")
    ap.add_argument("--pattern", default="ECMWF_PL_*.grb", help="model-level files")
    ap.add_argument("--sfc-pattern", default="ECMWF_SFC*.grb", help="surface files")
    ap.add_argument("--nproc", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    d = PGWOptions()
    ap.add_argument("--month", default=d.month, help="1..12 or 'auto'")
    ap.add_argument("--perturb-t", type=str2bool, default=d.perturb_t)
    ap.add_argument("--perturb-q", type=str2bool, default=d.perturb_q)
    ap.add_argument("--q-taper-bottom", type=float, default=d.q_taper_bottom)
    ap.add_argument("--q-taper-top", type=float, default=d.q_taper_top)
    ap.add_argument("--rh-min-factor", type=float, default=d.rh_min_factor)
    ap.add_argument("--perturb-wind", type=str2bool, default=d.perturb_wind)
    ap.add_argument("--ps-mode", choices=["hydrostatic", "gcm", "none"], default=d.ps_mode)
    ap.add_argument("--p-ref", type=float, default=d.p_ref)
    ap.add_argument("--sfc-anchor", type=str2bool, default=d.sfc_anchor)
    ap.add_argument("--sfc-blend-depth", type=float, default=d.sfc_blend_depth)
    ap.add_argument("--perturb-skt", type=str2bool, default=d.perturb_skt)
    ap.add_argument("--perturb-sst", type=str2bool, default=d.perturb_sst)
    ap.add_argument("--perturb-soil", type=str2bool, default=d.perturb_soil)
    ap.add_argument("--perturb-msl", type=str2bool, default=d.perturb_msl)
    ap.add_argument("--chunk", type=int, default=d.chunk, help="columns per block")
    ap.add_argument("--coast-buffer", type=int, default=d.coast_buffer)
    ap.add_argument("--sst-smooth-sigma", type=float, default=d.sst_smooth_sigma)
    args = ap.parse_args()

    opt = PGWOptions(month=args.month, perturb_t=args.perturb_t, perturb_q=args.perturb_q,
                     q_taper_bottom=args.q_taper_bottom, q_taper_top=args.q_taper_top,
                     rh_min_factor=args.rh_min_factor,
                     perturb_wind=args.perturb_wind, ps_mode=args.ps_mode, p_ref=args.p_ref,
                     sfc_anchor=args.sfc_anchor, sfc_blend_depth=args.sfc_blend_depth,
                     perturb_skt=args.perturb_skt,
                     perturb_sst=args.perturb_sst, perturb_soil=args.perturb_soil,
                     perturb_msl=args.perturb_msl, chunk=args.chunk,
                     coast_buffer=args.coast_buffer, sst_smooth_sigma=args.sst_smooth_sigma)

    in_root = os.path.abspath(args.input_root)
    out_root = os.path.abspath(args.output_root)
    if in_root == out_root:
        sys.exit("ERROR: input_root and output_root must differ")
    files = find_files(in_root, args.pattern)
    if not files:
        sys.exit(f"ERROR: no '{args.pattern}' files in {in_root}")
    sfc_files = find_files(in_root, args.sfc_pattern)
    if not sfc_files:
        print(f"  [WARNING] no '{args.sfc_pattern}' surface files in {in_root}: "
              "only model levels are perturbed")
    members = read_members(args.deltas)
    mroot = {m: os.path.join(out_root, m) if m else out_root for m in members}
    sdir = {m: os.path.join(mroot[m], "_state") for m in members}
    jobs_ml = [(m, members[m], f, os.path.join(mroot[m], os.path.relpath(f, in_root)), "ml", sdir[m])
               for m in members for f in files]
    jobs_sfc = [(m, members[m], f, os.path.join(mroot[m], os.path.relpath(f, in_root)), "sfc", sdir[m])
                for m in members for f in sfc_files]

    print(f">> {len(files)} model-level + {len(sfc_files)} surface files x {len(members)} members "
          f"= {len(jobs_ml) + len(jobs_sfc)} tasks, {args.nproc} processes")
    if members.keys() != {""}:
        print(f">> Members: {' '.join(members)}")
    print(">> Options:", json.dumps(options_dict(opt)))
    fdiag = {}
    for m in members:
        os.makedirs(mroot[m], exist_ok=True)
        with open(os.path.join(mroot[m], "pgw_options.json"), "w") as fh:
            json.dump({"options": options_dict(opt), "input_root": in_root,
                       "member": m or None, "deltas": os.path.abspath(members[m])},
                      fh, indent=2)
        fdiag[m] = open(os.path.join(mroot[m], "pgw_diagnostics.jsonl"), "w")

    n_err = 0
    try:
        with Pool(args.nproc, initializer=_init, initargs=(opt,)) as pool:
            for phase, jobs in (("A: model levels", jobs_ml), ("B: surface", jobs_sfc)):
                if not jobs:
                    continue
                print(f">> Phase {phase} ({len(jobs)} tasks)")
                for diag in pool.imap_unordered(_work, jobs):
                    m = diag["member"]
                    fdiag[m].write(json.dumps(diag) + "\n")
                    fdiag[m].flush()
                    rel = os.path.join(m, os.path.relpath(diag["file"], in_root))
                    if diag["status"] == "ok" and diag.get("kind") == "sfc":
                        print(f"  [OK] {rel}  ({diag['seconds']} s)  times={diag['n_times']} "
                              f"(uncoupled={diag['n_uncoupled']})  first time: "
                              f"Δsp={diag.get('first_dsp', float('nan')):+.1f} Pa  "
                              f"Δmsl={diag.get('first_dmsl', float('nan')):+.1f} Pa  "
                              f"Δ2t={diag.get('first_d2t', float('nan')):+.2f} K  "
                              f"Δ2d={diag.get('first_d2d', float('nan')):+.2f} K  "
                              f"ΔSST={diag.get('first_dsst', float('nan')):+.2f} K")
                        if diag["n_uncoupled"]:
                            u = diag["uncoupled"]
                            print(f"     [WARNING] {len(u)} times without model-level file (sp, msl "
                                  f"unchanged; 2t with Δtas): {u[0]}" + (f" ... {u[-1]}" if len(u) > 1 else ""))
                    elif diag["status"] == "ok":
                        print(f"  [OK] {rel}  ({diag['seconds']} s)  months={diag['month_weights']}  "
                              f"Δps={diag['dps_mean']:+.1f} Pa [{diag['dps_min']:+.1f},{diag['dps_max']:+.1f}] "
                              f"it={diag.get('ps_iterations', '-')}  "
                              f"ΔT(level 137)={diag['dT_lowest_mean']:+.2f} K  "
                              f"ΔT850={diag['dT_850_mean']:+.2f}  ΔT500={diag['dT_500_mean']:+.2f}")
                    else:
                        n_err += 1
                        print(f"  [ERROR] {rel}: {diag['error']}")
    finally:
        for fh in fdiag.values():
            fh.close()
    print(f">> Diagnostics: <member>/pgw_diagnostics.jsonl under {out_root}")
    if n_err:
        sys.exit(f"ERROR: {n_err} tasks failed")


if __name__ == "__main__":
    main()
