"""
Stage 13 -- Multipath Channel Realism

Adds a small number of reflected paths to the existing clean single-path
(LOS-only) near-field channel used throughout every prior stage,
controlled via a Rician K-factor (ratio of LOS power to total scattered
power) rather than fully re-randomized reflections each run -- per the
Extension Plan v2's request for a CONTROLLED sweep.

Design: for a given "environment" (a fixed set of reflector locations
and random complex gains, generated once per trial), the K-factor sweep
changes ONLY the LOS-vs-scattered power split:

    h_total = sqrt(K/(K+1)) * h_LOS + sqrt(1/(K+1)) * h_scattered

where h_LOS and each reflector's channel are individually power-
normalized (sum|h|^2 = N*F, this project's standard convention since
Stage 3), and h_scattered is the random-phase-weighted sum of the
reflectors' channels, rescaled so its own expected power is also N*F.//
K -> infinity (or the K_DB=None sentinel, 0 reflectors) reduces EXACTLY
to the existing single-path near_field_channel already used everywhere
-- this is Stage 13's regression gate, same discipline as Stage 8/12.

Reflector geometry/phases are generated once per trial (via a supplied
RNG) and REUSED across the entire K sweep for that trial -- isolating
K's effect from environment-to-environment variation. Multiple trials
(different seeds) are then averaged in validate_stage13.py for a
statistically robust trend, since a single random reflector
configuration could show K-dependence that's really just luck of the
draw on reflector placement/phase.
"""

from __future__ import annotations

import numpy as np

from arraymodel import ArrayGeometry
from channel import near_field_channel


def _normalize_channel(channel: np.ndarray) -> np.ndarray:
    """Standard project convention: rescale to sum|h|^2 = N*F."""
    N, F = channel.shape
    power = np.sum(np.abs(channel) ** 2)
    return channel * np.sqrt(N * F / power)


def generate_reflector_config(
    rng: np.random.Generator,
    n_reflectors: int,
    r_frac_range: tuple,
    theta_range_deg: tuple,
) -> dict:
    """Generate one fixed reflector environment: n_reflectors locations
    (as fractions of the array's own Rayleigh distance, matching this
    project's normalization convention) and one random phase per
    reflector. Reused across an entire K sweep by
    multipath_near_field_channel below.

    Args:
        rng: numpy random Generator (caller controls the seed).
        n_reflectors: number of reflected paths.
        r_frac_range: (min, max) reflector distance, as a fraction of
            the array's Rayleigh distance.
        theta_range_deg: (min, max) reflector angle, degrees.

    Returns:
        dict with 'r_fracs', 'thetas_deg', 'phases' (each length
        n_reflectors).
    """
    return {
        "r_fracs": rng.uniform(r_frac_range[0], r_frac_range[1], size=n_reflectors),
        "thetas_deg": rng.uniform(theta_range_deg[0], theta_range_deg[1], size=n_reflectors),
        "phases": rng.uniform(0, 2 * np.pi, size=n_reflectors),
    }


def multipath_near_field_channel(
    array: ArrayGeometry,
    r_los: float,
    theta_los: float,
    freqs: np.ndarray,
    k_db: float,
    reflector_config: dict,
) -> np.ndarray:
    """Build the multipath channel: direct LOS path + n_reflectors
    scattered paths, combined per a Rician K-factor.

    Args:
        array: ArrayGeometry.
        r_los, theta_los: the direct/LOS path's user location.
        freqs: (F,) frequencies, Hz.
        k_db: Rician K-factor in dB (LOS power / total scattered
            power). If None, returns the pure-LOS channel unchanged
            (0 reflectors) -- the exact regression case matching every
            prior stage's single-path channel.
        reflector_config: dict from generate_reflector_config, reused
            unchanged across an entire K sweep so only K varies.

    Returns:
        (N, F) complex channel, power-normalized to sum|h|^2 = N*F.
    """
    h_los = _normalize_channel(near_field_channel(array, r_los, theta_los, freqs))

    if k_db is None:
        return h_los

    k_linear = 10 ** (k_db / 10.0)
    r_fracs = reflector_config["r_fracs"]
    thetas_deg = reflector_config["thetas_deg"]
    phases = reflector_config["phases"]
    n_reflectors = len(r_fracs)

    rayleigh = array.rayleigh_distance()
    scattered = np.zeros_like(h_los)
    for i in range(n_reflectors):
        r_k = r_fracs[i] * rayleigh
        theta_k = np.deg2rad(thetas_deg[i])
        h_k = _normalize_channel(near_field_channel(array, r_k, theta_k, freqs))
        scattered = scattered + np.exp(1j * phases[i]) * h_k

    # Each h_k has power N*F; n_reflectors random-phase-weighted copies
    # summed has EXPECTED power n_reflectors*N*F (cross terms average to
    # zero over random phase), so dividing by sqrt(n_reflectors) brings
    # the expected power back to N*F, matching h_los's normalization.
    scattered = scattered / np.sqrt(n_reflectors)

    h_total = np.sqrt(k_linear / (k_linear + 1)) * h_los + np.sqrt(1 / (k_linear + 1)) * scattered

    # Final rescale to exactly sum|h|^2 = N*F -- a single random-phase
    # realization won't land exactly on the expected power (that's
    # genuine small-scale fading), so this final step matches every
    # prior stage's convention of removing overall power scale before
    # computing gain, isolating shape mismatch as the metric of interest.
    return _normalize_channel(h_total)