"""
Vertical utilities for IFS hybrid levels.

Conventions
-----------
* 3D arrays have shape (nlev, npts): level 0 = model top, level -1 = lowest
  model level (IFS order, level 1 ... 137).
* Half levels: p_{k+1/2} = a_k + b_k * ps,   k = 0..nlev.
* Full levels: p_k = (p_{k-1/2} + p_{k+1/2}) / 2.
"""

import numpy as np

from .constants import RD
from .thermo import virtual_temperature


# ──────────────────────────────────────────────────────────────────────────────
#  Hybrid pressures
# ──────────────────────────────────────────────────────────────────────────────
def split_pv(pv):
    """Split the GRIB 'pv' array into half-level coefficients (a, b)."""
    pv = np.asarray(pv, dtype=float)
    nh = pv.size // 2
    return pv[:nh], pv[nh:]


def half_level_pressure(a, b, ps):
    """Half-level pressure, shape (nlev+1, npts)."""
    return a[:, None] + b[:, None] * ps[None, :]


def full_level_pressure(a, b, ps):
    """Full-level pressure, shape (nlev, npts)."""
    ph = half_level_pressure(a, b, ps)
    return 0.5 * (ph[:-1] + ph[1:])


# ──────────────────────────────────────────────────────────────────────────────
#  Geopotential
# ──────────────────────────────────────────────────────────────────────────────
def geopotential_at_pressure(a, b, ps, t, q, phis, p_ref):
    """
    Geopotential [m2 s-2] on the isobaric surface p_ref, integrating the
    hydrostatic equation upward from the surface over the half levels:

        Φ_{k-1/2} = Φ_{k+1/2} + Rd · Tv_k · ln(p_{k+1/2} / p_{k-1/2})

    and interpolating linearly in ln(p) between the two half levels that
    bracket p_ref. The integration stops at p_ref, so the model top (p = 0)
    is not involved.
    """
    ph = half_level_pressure(a, b, ps)
    ph = np.maximum(ph, 1.0e-2)                # avoids ln(0) at the model top
    lnph = np.log(ph)
    tv = virtual_temperature(t, q)

    nlev, npts = t.shape
    phi_h = np.empty((nlev + 1, npts))
    phi_h[-1] = phis
    dphi = RD * tv * (lnph[1:] - lnph[:-1])    # (nlev, npts)
    # cumulative sum from the bottom
    phi_h[:-1] = phis[None, :] + np.cumsum(dphi[::-1], axis=0)[::-1]

    # index of the half level immediately below p_ref
    kb = np.sum(ph < p_ref, axis=0)            # ph[kb-1] < p_ref <= ph[kb]
    if np.any(kb == 0) or np.any(kb > nlev):
        raise ValueError(f"p_ref={p_ref} Pa outside the model pressure range.")
    cols = np.arange(npts)
    lp_up, lp_dn = lnph[kb - 1, cols], lnph[kb, cols]
    f_up, f_dn = phi_h[kb - 1, cols], phi_h[kb, cols]
    w = (np.log(p_ref) - lp_dn) / (lp_up - lp_dn)
    return f_dn + w * (f_up - f_dn)


# ──────────────────────────────────────────────────────────────────────────────
#  Vertical interpolation of the deltas
# ──────────────────────────────────────────────────────────────────────────────
def interp_delta_to_model_levels(delta_pl, plev, p_target, delta_sfc=None, ps=None,
                                 blend_depth=10000.0):
    """
    Interpolate a delta defined on GCM pressure levels to the (column-varying)
    pressures of the model levels.

    Parameters
    ----------
    delta_pl    : (nplev, npts)  delta on pressure levels
    plev        : (nplev,)       pressure levels [Pa] (any order)
    p_target    : (nlev, npts)   target pressures [Pa]
    delta_sfc   : (npts,) or None surface delta (tas, hurs)
    ps          : (npts,) or None column surface pressure [Pa]
    blend_depth : Pa             depth of the surface transition layer

    Behavior
    --------
    1. Linear interpolation in ln(p) between the GCM levels, with constant
       extrapolation above the highest and below the lowest level -> d_std(p).
    2. With a surface anchor (delta_sfc), a correction is added that equals
       (delta_sfc - d_std(ps)) at the surface and decays linearly to 0 at
       'blend_depth' Pa above it:

           d(p) = d_std(p) + [delta_sfc - d_std(ps)] · max(0, 1 - (ps - p)/blend_depth)

       The lowest model level therefore receives ~delta_sfc (tas, hurs), the
       "below-ground" GCM values over orography are corrected, and the
       result is CONTINUOUS in ps and p (no jumps when ps crosses a plev).
    """
    order = np.argsort(plev)                    # ascending: top -> surface
    lp = np.log(np.asarray(plev, dtype=float)[order])
    d = np.asarray(delta_pl, dtype=float)[order]

    def _interp(lpt):
        """lpt: (n, npts) -> (n, npts)"""
        nplev = lp.size
        cols = np.broadcast_to(np.arange(d.shape[1]), lpt.shape)
        k = np.clip(np.searchsorted(lp, lpt), 1, nplev - 1)
        w = np.clip((lpt - lp[k - 1]) / (lp[k] - lp[k - 1]), 0.0, 1.0)
        return d[k - 1, cols] * (1.0 - w) + d[k, cols] * w

    out = _interp(np.log(p_target))
    if delta_sfc is None:
        return out
    if ps is None:
        raise ValueError("ps is required when delta_sfc is given.")
    d_at_ps = _interp(np.log(ps)[None, :])[0]
    w_sfc = np.clip(1.0 - (ps[None, :] - p_target) / blend_depth, 0.0, 1.0)
    return out + (delta_sfc - d_at_ps)[None, :] * w_sfc
