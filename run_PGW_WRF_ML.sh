#!/bin/bash
# ==============================================================================
#  PGW WORKFLOW FOR WRF WITH IFS/ERA5 MODEL-LEVEL INPUT (137 hybrid levels)
#  CMIP6 multi-model (X models + ENSMEAN) and multi-scenario.
#
#  Perturbs the IFS/ERA5 GRIB files that WPS (ungrib + calc_ecmwf_p) reads
#  for WRF: model-level files (t, q, u, v, lnsp, z; one per time) and the
#  surface file (sp, msl, 2t, 2d, 10u, 10v, skt, sst, soil; all times), with
#  monthly CMIP6 climate deltas on pressure levels. For each selected scenario
#  it produces X+1 sets of files: one per model and the ensemble mean
#  (ENSMEAN) of the X models.
#
#  Steps:
#    STEP 01  step01_prepare_ensemble.py  Scenarios, model selection,
#                                         regridding and ENSMEAN.
#    STEP 02  step02_apply_pgw.py         A) model levels (T, q, lnsp, [u,v]);
#                                         B) surface file (sp, msl, 2t, 2d,
#                                         [10u,10v], skt, sst, stl1-4) consistent
#                                         with the adjusted model levels.
#    STEP 03  step03_check_pgw.py         Integrity of each member (metadata,
#                                         untouched fields bit-identical).
#    STEP 04  step04_verify_deltas.py     Verification that the deltas were
#                                         applied correctly: metrics, figures
#                                         and HTML report.
#
#  Usage:
#    sbatch run_PGW_WRF_ML.sh <path_input> <path_output> <cmip_root> <month>
#
#    path_input  : directory with the GRIB files for WPS (searched recursively
#                  for ML_PATTERN and SFC_PATTERN; e.g.,
#                  .../WRF_HAILCAST/input/Ophelia500m/conditions)
#    path_output : output root:
#                    <path_output>/<SCENARIO>/<MEMBER>/<same file names>
#                    <path_output>/_deltas/      prepared deltas
#                    <path_output>/verification/ report (report.html)
#    cmip_root   : root of the CMIP6 deltas (e.g., /ec/res4/hpcperm/ecme2143/CMIP6_deltas
#                  with Future_ssp245/ Future_ssp370/ Future_ssp585/ Attribution/ssp245/YYYY/).
#                  With SCENARIOS="" it is directly a model directory.
#    month       : 1..12 (fixed month, as cdo seltimestep) or 'auto'
#                  (linear interpolation between months from the validity time)
#
#  Example:
#    sbatch run_PGW_WRF_ML.sh /perm/spcc/WRF_HAILCAST/input/Ophelia500m/conditions /ec/res4/scratch/ecme2143/PGW_Ophelia500m /ec/res4/hpcperm/ecme2143/CMIP6_deltas auto
# ==============================================================================

#SBATCH --job-name=pgw_wrf_ml
#SBATCH --output=logs/pgw_wrf_ml_%j.out
#SBATCH --error=logs/pgw_wrf_ml_%j.out
#SBATCH --qos=nf
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -eo pipefail     # (no -u: the spcc bashrc uses undefined variables)

# ——————————————————————————————————————————————————————————————————————————————
#  PERTURBATION SETTINGS  (see README.md, "Perturbation options")
# ——————————————————————————————————————————————————————————————————————————————
PERTURB_T=true           # T + ΔT
PERTURB_Q=true           # q such that RH_pgw = RH + ΔRH  (with the new T and p)
PERTURB_WIND=false       # u,v + Δua,Δva (rotated to the grid)
PS_MODE=hydrostatic      # hydrostatic | gcm | none
P_REF=30000              # Pa, reference level of the hydrostatic adjustment
SFC_ANCHOR=true          # close ΔT/ΔRH at the surface with Δtas/Δhurs
SFC_BLEND_DEPTH=10000    # Pa, depth of the surface transition layer
Q_TAPER_BOTTOM=15000     # Pa, below: full change in q
Q_TAPER_TOP=10000        # Pa, above: q unchanged (stratosphere)
RH_MIN_FACTOR=0.5        # RH_pgw >= factor * RH_orig
PERTURB_SKT=true         # skt + Δts ("open-sea" Δts over the sea)
PERTURB_SST=true         # sst + coast-cleaned Δts (IDW)
PERTURB_SOIL=true        # stl1-4 + Δts damped with depth
PERTURB_MSL=true         # msl rescaled hydrostatically with the adjusted ps
COAST_BUFFER=2           # erosion of the sea mask (grid points)
SST_SMOOTH_SIGMA=1.0     # Gaussian smoothing of ΔSST (grid points)
ML_PATTERN='ECMWF_PL_*.grb'   # model-level files (one per time)
SFC_PATTERN='ECMWF_SFC*.grb'  # surface files (all times)
CHUNK=20000              # columns per block (memory control)
# Memory: about 1.5-2 GB per process for a 401x401x137 domain. Reduce NPROC
# (default SLURM_CPUS_PER_TASK) if the domain is larger.

# ——————————————————————————————————————————————————————————————————————————————
#  SCENARIOS AND MODELS  (see README.md, "Multi-model and scenarios")
# ——————————————————————————————————————————————————————————————————————————————
# Scenarios (paths under cmip_root; "_" and "/" are equivalent), separated by
# spaces, or "all". Available: Attribution_ssp245 Future_ssp245
# Future_ssp370 Future_ssp585.
SCENARIOS="Attribution_ssp245 Future_ssp585"
ATTRIB_YEAR=auto         # event year for Attribution (YYYY) or auto (year of the files)
# Models: explicit list ("EC-Earth3 MIROC6 MRI-ESM2-0"), "all" or "auto".
MODELS=auto
N_MODELS=5               # number of models with MODELS=auto
AUTO_METHOD=spread       # spread (spans the ensemble) | representative (closest to the mean)
COMMON_MODELS=true       # same models in all scenarios (intersection)
MAKE_ENSMEAN=true        # additional ENSMEAN member of the X models
MODEL_SUFFIX=""          # suffix of the model directories (e.g., _global in CMIP6_ATTRIB_ssp245)

# ——————————————————————————————————————————————————————————————————————————————
#  ARGUMENTS
# ——————————————————————————————————————————————————————————————————————————————
if [[ $# -ne 4 ]]; then
    sed -n '2,40p' "$0"
    exit 1
fi

# Argument paths are resolved BEFORE changing directory, so relative paths
# refer to the directory the script is called from (under sbatch the job
# starts in $SLURM_SUBMIT_DIR).
path_input=$(realpath "$1")
mkdir -p "$2"
path_output=$(realpath "$2")
cmip_root=$(realpath "$3")
month="$4"

# Package directory = physical location of this script. Under sbatch "$0" is
# a copy in the SLURM spool, so the real path is obtained from scontrol
# (Command field of the job).
script_path="$0"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    cmd=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
          | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -n 1)
    [[ -n "$cmd" ]] && script_path="$cmd"
fi
PKG_DIR=$(cd "$(dirname "$script_path")" && pwd)
if [[ ! -f "$PKG_DIR/pgw_ml/apply.py" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PKG_DIR="$SLURM_SUBMIT_DIR"
fi
if [[ ! -f "$PKG_DIR/pgw_ml/apply.py" ]]; then
    echo "ERROR: the pgw_ml package was not found next to the script ($PKG_DIR)"
    exit 1
fi
cd "$PKG_DIR"
source "$PKG_DIR/env.sh"

nproc="${NPROC:-${SLURM_CPUS_PER_TASK:-4}}"
work_deltas="$path_output/_deltas"
mkdir -p "$work_deltas"

echo "=================================================================="
echo "  PGW WRF (IFS/ERA5 model levels), multi-model   $(date)"
echo "  Input      : $path_input"
echo "  Output     : $path_output"
echo "  CMIP6      : $cmip_root"
echo "  Scenarios  : ${SCENARIOS:-(cmip_root)}   Attribution year: $ATTRIB_YEAR"
echo "  Models     : $MODELS$([[ $MODELS == auto ]] && echo " (N=$N_MODELS, $AUTO_METHOD)")   common: $COMMON_MODELS   ENSMEAN: $MAKE_ENSMEAN"
echo "  Month      : $month      Processes: $nproc"
echo "  Package    : $PKG_DIR"
echo "=================================================================="

# ——————————————————————————————————————————————————————————————————————————————
echo ""
echo ">> [STEP 01] Scenarios, models and deltas on the input grid ..."
ens_flag=""
[[ "$MAKE_ENSMEAN" == true ]] || ens_flag="--no-ensmean"
python3 -u step01_prepare_ensemble.py "$path_input" "$cmip_root" "$work_deltas" \
    --scenarios "$SCENARIOS" --attrib-year "$ATTRIB_YEAR" \
    --models "$MODELS" --n-models "$N_MODELS" --auto-method "$AUTO_METHOD" \
    --common-models "$COMMON_MODELS" --suffix "$MODEL_SUFFIX" \
    --month "$month" --pattern "$ML_PATTERN" --ps-mode "$PS_MODE" \
    --perturb-wind "$PERTURB_WIND" --nproc "$nproc" $ens_flag

# ——————————————————————————————————————————————————————————————————————————————
echo ""
echo ">> [STEP 02] Applying PGW to all members ..."
python3 -u step02_apply_pgw.py "$path_input" "$path_output" \
    "$work_deltas/members.txt" \
    --pattern "$ML_PATTERN" --sfc-pattern "$SFC_PATTERN" --nproc "$nproc" --month "$month" \
    --perturb-t "$PERTURB_T" --perturb-q "$PERTURB_Q" --perturb-wind "$PERTURB_WIND" \
    --ps-mode "$PS_MODE" --p-ref "$P_REF" \
    --sfc-anchor "$SFC_ANCHOR" --sfc-blend-depth "$SFC_BLEND_DEPTH" \
    --q-taper-bottom "$Q_TAPER_BOTTOM" --q-taper-top "$Q_TAPER_TOP" \
    --rh-min-factor "$RH_MIN_FACTOR" \
    --perturb-skt "$PERTURB_SKT" --perturb-sst "$PERTURB_SST" \
    --perturb-soil "$PERTURB_SOIL" --perturb-msl "$PERTURB_MSL" --chunk "$CHUNK" \
    --coast-buffer "$COAST_BUFFER" --sst-smooth-sigma "$SST_SMOOTH_SIGMA"

# ——————————————————————————————————————————————————————————————————————————————
#  STEP 03/04: do not abort at the first failure; rc gives the final status.
rc=0
echo ""
echo ">> [STEP 03] Integrity of the files of each member ..."
while read -r member _; do
    [[ -z "$member" ]] && continue
    echo "  -- $member"
    python3 -u step03_check_pgw.py "$path_input" "$path_output/$member" --plots 0 < /dev/null || rc=1
done < "$work_deltas/members.txt"

echo ""
echo ">> [STEP 04] Verification of the applied deltas (metrics, figures, HTML) ..."
python3 -u step04_verify_deltas.py "$path_input" "$path_output" --nproc "$nproc" --chunk "$CHUNK" || rc=1

echo ""
echo "=================================================================="
echo "  Workflow finished: $(date)   (status: $([[ $rc == 0 ]] && echo OK || echo 'WITH PROBLEMS'))"
echo "  PGW files : $path_output/<SCENARIO>/<MEMBER>/"
echo "  Report    : $path_output/verification/report.html"
echo "=================================================================="
exit $rc
