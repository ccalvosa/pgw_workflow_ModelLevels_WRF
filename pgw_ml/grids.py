"""
Grids: rotated-pole transform, GCM -> input-grid regridding and rotation of
wind vectors.

On a regular lat-lon grid (ERA5/IFS for WPS) the winds are Earth-relative
and no rotation is needed. On a ROTATED lat-lon grid with grid-relative winds
(uvRelativeToGrid = 1), the Earth-relative GCM ua/va deltas are rotated
before being added.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator


# ──────────────────────────────────────────────────────────────────────────────
#  Rotated pole (GRIB convention: rotated SOUTH pole at (lat_sp, lon_sp), angle 0)
# ──────────────────────────────────────────────────────────────────────────────
def geo_to_rot(lon, lat, lat_sp, lon_sp):
    """Geographical -> rotated coordinates (degrees)."""
    theta = np.deg2rad(90.0 + lat_sp)
    lon_r = np.deg2rad(np.asarray(lon) - lon_sp)
    lat_r = np.deg2rad(np.asarray(lat))
    x = np.cos(lat_r) * np.cos(lon_r)
    y = np.cos(lat_r) * np.sin(lon_r)
    z = np.sin(lat_r)
    xr_ = np.cos(theta) * x + np.sin(theta) * z
    zr_ = -np.sin(theta) * x + np.cos(theta) * z
    return (np.rad2deg(np.arctan2(y, xr_)),
            np.rad2deg(np.arcsin(np.clip(zr_, -1.0, 1.0))))


def wind_rotation_matrix(lon, lat, lat_sp, lon_sp, eps=1.0e-3):
    """
    Components of the geographical EAST unit vector expressed in the local
    basis of the rotated grid: (cos α, sin α). Computed numerically (finite
    differences of geo_to_rot), which avoids sign errors.

        u_grid =  cos α · u_geo - sin α · v_geo
        v_grid =  sin α · u_geo + cos α · v_geo
    """
    rlon0, rlat0 = geo_to_rot(lon, lat, lat_sp, lon_sp)
    rlon1, rlat1 = geo_to_rot(np.asarray(lon) + eps, lat, lat_sp, lon_sp)
    dx = ((rlon1 - rlon0 + 180.0) % 360.0 - 180.0) * np.cos(np.deg2rad(rlat0))
    dy = rlat1 - rlat0
    norm = np.hypot(dx, dy)
    return dx / norm, dy / norm


def rotate_wind_to_grid(u_geo, v_geo, cos_a, sin_a):
    """Rotate geographical (u, v) to the rotated-grid basis."""
    return cos_a * u_geo - sin_a * v_geo, sin_a * u_geo + cos_a * v_geo


# ──────────────────────────────────────────────────────────────────────────────
#  Horizontal regridding GCM (regular global lat-lon) -> target points
# ──────────────────────────────────────────────────────────────────────────────
class GCMToPoints:
    """
    Bilinear interpolation of fields on a regular lat-lon grid (GCM) to an
    arbitrary list of geographical (lat, lon) points. A box around the
    domain (plus a margin) is cropped first, and longitude periodicity is
    handled. With GCM resolutions of ~0.7-2.5° and ~0.1-0.25° for the input grid,
    bilinear interpolation is sufficient (the delta is a smooth field).
    """

    def __init__(self, gcm_lat, gcm_lon, tgt_lat, tgt_lon, margin=3.0):
        gcm_lat = np.asarray(gcm_lat, dtype=float)
        gcm_lon = np.asarray(gcm_lon, dtype=float) % 360.0
        self.lat_flip = gcm_lat[0] > gcm_lat[-1]
        lat_sorted = gcm_lat[::-1] if self.lat_flip else gcm_lat

        tlon = np.asarray(tgt_lon, dtype=float)
        tlat = np.asarray(tgt_lat, dtype=float)

        # Target longitudes centered on the domain (avoids cuts at 0°)
        lon_c = np.rad2deg(np.angle(np.mean(np.exp(1j * np.deg2rad(tlon)))))
        tlon_rel = (tlon - lon_c + 180.0) % 360.0 - 180.0
        glon_rel = (gcm_lon - lon_c + 180.0) % 360.0 - 180.0
        jlon = np.where((glon_rel >= tlon_rel.min() - margin) &
                        (glon_rel <= tlon_rel.max() + margin))[0]
        jlon = jlon[np.argsort(glon_rel[jlon])]
        ilat = np.where((lat_sorted >= tlat.min() - margin) &
                        (lat_sorted <= tlat.max() + margin))[0]
        if jlon.size < 2 or ilat.size < 2:
            raise ValueError("The domain box contains no GCM points.")

        self.ilat = ilat
        self.jlon = jlon
        self.sub_lat = lat_sorted[ilat]
        self.sub_lon = glon_rel[jlon]
        self.points = np.column_stack([tlat, tlon_rel])

    def _subset(self, field2d):
        f = np.asarray(field2d)
        if self.lat_flip:
            f = f[..., ::-1, :]
        return f[..., self.ilat, :][..., self.jlon]

    def __call__(self, field):
        """field: (..., nlat, nlon) -> (..., npts)."""
        sub = self._subset(field)
        lead = sub.shape[:-2]
        flat = sub.reshape((-1,) + sub.shape[-2:])
        out = np.empty((flat.shape[0], self.points.shape[0]))
        for i in range(flat.shape[0]):
            itp = RegularGridInterpolator((self.sub_lat, self.sub_lon), flat[i],
                                          method="linear", bounds_error=True)
            out[i] = itp(self.points)
        return out.reshape(lead + (self.points.shape[0],))
