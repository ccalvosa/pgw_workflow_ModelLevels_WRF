"""
pgw_ml: pseudo-global warming (PGW) on IFS/ERA5 hybrid model levels for the
initial and boundary conditions of WRF (GRIB files read by WPS).

Modules
-------
constants : physical constants (IFS values).
thermo    : saturation vapor pressure (IFS mixed water/ice and over water),
            RH <-> q, dewpoint, Tv.
vertical  : hybrid pressures, hydrostatic geopotential, ln(p) interpolation of
            the deltas with a surface anchor.
grids     : rotated-pole transform, GCM -> input-grid regridding, wind rotation
            (only needed for rotated grids).
gribio    : reading/writing of the IFS GRIB files (one time per file, or
            several times per file for the surface file).
prepare   : deltas of one CMIP6 model -> input grid.
ensemble  : scenarios, model discovery and selection, ensemble mean.
deltas    : access to the prepared deltas and temporal selection.
sst       : coastal cleaning of the SST delta (IDW from the open sea).
apply     : PGW of a model-level file (column blocks; includes the ps adjustment).
surface   : PGW of the WPS surface file, consistent with the model levels.
verify    : verification metrics for one model-level file and its surface time.
"""
