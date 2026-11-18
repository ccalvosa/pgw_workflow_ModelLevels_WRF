"""
Moisture thermodynamics consistent with the IFS.

Relative humidity is computed with respect to the IFS "mixed" saturation:
over water for T >= T0, over ice for T <= TICE, and a quadratic blend in
between. The key property for PGW is that the q -> RH -> q conversion is
exactly invertible: with zero deltas, q_pgw == q_original.
"""

import numpy as np

from .constants import (EPS, T0, TICE, ES_A1, ES_A3_WATER, ES_A4_WATER,
                        ES_A3_ICE, ES_A4_ICE)


def esat_mixed(t):
    """Saturation vapor pressure [Pa] (IFS mixed water/ice)."""
    es_w = ES_A1 * np.exp(ES_A3_WATER * (t - T0) / (t - ES_A4_WATER))
    es_i = ES_A1 * np.exp(ES_A3_ICE * (t - T0) / (t - ES_A4_ICE))
    alpha = np.clip((t - TICE) / (T0 - TICE), 0.0, 1.0) ** 2
    return alpha * es_w + (1.0 - alpha) * es_i


def q_to_e(q, p):
    """Specific humidity [kg/kg] -> vapor pressure [Pa]."""
    return q * p / (EPS + (1.0 - EPS) * q)


def e_to_q(e, p):
    """Vapor pressure [Pa] -> specific humidity [kg/kg]."""
    return EPS * e / (p - (1.0 - EPS) * e)


def q_to_rh(q, t, p):
    """Relative humidity (fraction, 0-1) from q, T, p."""
    return q_to_e(q, p) / esat_mixed(t)


def rh_to_q(rh, t, p):
    """Specific humidity from RH (fraction), T, p."""
    return e_to_q(rh * esat_mixed(t), p)


def virtual_temperature(t, q):
    """Virtual temperature [K]."""
    return t * (1.0 + (1.0 / EPS - 1.0) * q)


def esat_water(t):
    """Saturation vapor pressure over water [Pa] (IFS Tetens), used for 2d."""
    return ES_A1 * np.exp(ES_A3_WATER * (t - T0) / (t - ES_A4_WATER))


def dewpoint_from_e(e):
    """Dewpoint [K] from vapor pressure [Pa] (inverse of esat_water)."""
    x = np.log(np.maximum(e, 1.0e-6) / ES_A1)
    return (ES_A3_WATER * T0 - ES_A4_WATER * x) / (ES_A3_WATER - x)
