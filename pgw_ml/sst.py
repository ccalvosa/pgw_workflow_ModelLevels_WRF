"""
Coastal cleaning of the surface temperature delta over the sea.

The GCM 'ts' delta (~1° grid) mixes land signal (which usually warms more)
with sea signal near the coast. For SST (and skt over the sea) an
"open-sea only" delta is used:

  1) sea mask from the IFS land-sea mask (lsm < threshold), eroded by
     'coast_buffer' points to keep only reliable open-sea points;
  2) the remaining points are filled by inverse-distance weighting (IDW;
     k neighbors, power p) from those open-sea points;
  3) optional Gaussian smoothing.

Same logic as add_forcing_gfs_2D_SST in the PGW-WRF workflow.
"""

import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter
from scipy.spatial import cKDTree


def ocean_only_delta(delta2d, lsm2d, lsm_threshold=0.5, coast_buffer=2,
                     k_neighbors=8, idw_power=2.0, smooth_sigma=1.0):
    """delta2d, lsm2d: (Nj, Ni). Returns the cleaned delta (Nj, Ni)."""
    ocean = lsm2d < lsm_threshold
    clean = binary_erosion(ocean, iterations=int(coast_buffer)) if coast_buffer else ocean
    if clean.sum() < k_neighbors:
        # Domain with very little sea: return the original delta
        return delta2d.copy(), int(clean.sum())
    jj, ii = np.indices(delta2d.shape)
    tree = cKDTree(np.column_stack([jj[clean], ii[clean]]))
    dist, idx = tree.query(np.column_stack([jj.ravel(), ii.ravel()]), k=k_neighbors)
    w = 1.0 / (dist ** idw_power + 1.0e-9)
    w /= w.sum(axis=1, keepdims=True)
    out = (delta2d[clean][idx] * w).sum(axis=1).reshape(delta2d.shape)
    out[clean] = delta2d[clean]
    if smooth_sigma and smooth_sigma > 0:
        out = gaussian_filter(out, sigma=smooth_sigma, mode="nearest")
    return out, int(clean.sum())
