"""Small pieces of asteroid physics used for explanations and baselines."""

from __future__ import annotations

import numpy as np

# NASA's potentially hazardous asteroid (PHA) definition has two parts:
#   absolute magnitude H <= 22.0   (roughly: bigger than ~140 m)
#   minimum orbit intersection distance (MOID) <= 0.05 AU
# The dataset has H but not MOID, which is the single biggest limit on any
# model trained on it.
PHA_MAX_ABSOLUTE_MAGNITUDE = 22.0
PHA_MAX_MOID_AU = 0.05
AU_KM = 149_597_870.7

# Albedo range NeoWs uses to turn H into a diameter range.
ALBEDO_BRIGHT = 0.25  # gives the minimum diameter
ALBEDO_DARK = 0.05  # gives the maximum diameter


def diameter_km(absolute_magnitude, albedo: float):
    """Standard H to diameter conversion: D = 1329 / sqrt(albedo) * 10^(-H/5)."""
    h = np.asarray(absolute_magnitude, dtype=float)
    return 1329.0 / np.sqrt(albedo) * np.power(10.0, -h / 5.0)


def diameter_range_km(absolute_magnitude) -> tuple:
    """(min, max) diameter in km, matching NeoWs' est_diameter_min / max."""
    return (
        diameter_km(absolute_magnitude, ALBEDO_BRIGHT),
        diameter_km(absolute_magnitude, ALBEDO_DARK),
    )


def brightness_rule(absolute_magnitude):
    """The H half of NASA's PHA rule. Used as the baseline every model must beat."""
    return np.asarray(absolute_magnitude, dtype=float) <= PHA_MAX_ABSOLUTE_MAGNITUDE
