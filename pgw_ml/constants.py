"""Physical constants (IFS values)."""

RD = 287.0597      # gas constant of dry air          [J kg-1 K-1]
RV = 461.5250      # gas constant of water vapor      [J kg-1 K-1]
G = 9.80665        # gravitational acceleration       [m s-2]
EPS = RD / RV      # Rd/Rv ratio (~0.622)             [1]

# Saturation vapor pressure (Tetens, IFS documentation, Part IV, eq. 7.5)
T0 = 273.16        # triple point                     [K]
TICE = 250.16      # below: saturation over ice       [K]
ES_A1 = 611.21     # [Pa]
ES_A3_WATER = 17.502
ES_A4_WATER = 32.19
ES_A3_ICE = 22.587
ES_A4_ICE = -0.7
