#!/bin/bash
# ==============================================================================
#  env.sh: software environment of the model-level PGW workflow.
#
#  IMPORTANT: the HARMONIE IFS files pack q with CCSDS (grid_ccsds). Several
#  ecmwf-toolbox/python combinations segfault when decoding them. The tested
#  combination is that of user spcc (ecmwf-toolbox/2026.04.0.0 + HARMONIE
#  ECCODES_DEFINITION_PATH) with python3/3.11.10-01 (eccodes-python, xarray,
#  scipy, numpy, matplotlib).
#
#  Usage:  source env.sh
# ==============================================================================

source /home/spcc/bashrc_spcc > /dev/null 2>&1
module load python3/3.11.10-01 > /dev/null 2>&1

# Silence the eccodes-python version warning
export PYTHONWARNINGS="ignore::UserWarning"
