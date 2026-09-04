"""
Stage 5 -- Sweep Engine & Failure Map

Sweeps array size (N), bandwidth (B), and distance (r) jointly, computing
Gain_loss_dB for each of the three "non-ideal" beamformer modes
(conventional, nearfield-aware, squint-aware) against the combined
corrector's gain, evaluated on the TRUE near-field wideband channel --
per the project's Section 2.6 metric definition.

Design:
    - Distance is swept as a FRACTION of each array's own Rayleigh
      distance (not an absolute meter value), since Rayleigh distance
      itself depends on N (grows as N^2). This keeps the sweep grid
      physically comparable across array sizes, and is exactly the
      normalization that made Stage 2's divergence curves collapse onto
      each other.
    - All channel realizations are power-normalized before computing gain
      (same rule used since Stage 3/4), so the near-field 1/r amplitude
      taper never contaminates the gain-loss comparison.
    - Fully vectorized per (N, B, r) combination (each combination's inner
      beamform_gain computation is already vectorized across elements and
      frequency bins); the sweep loop itself iterates over combinations in
      Python since each combination requires a different-shaped channel/
      steering vector (different N or F) -- this matches the project's
      "vectorize within each computation, loop across independent grid
      points" scope.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from arraymodel import ArrayGeometry
from channel import near_field_channel
from beamformers import beamform, beamform_gain

THETA = np.deg2rad(20.0)


def _normalize_channel(channel: np.ndarray) -> np.ndarray:
    """Normalize total channel power to N*F (unit power per element per
    frequency bin), preventing the near-field 1/r taper from contaminating
    gain comparisons across different (N, B, r) grid points."""
    N, F = channel.shape
    power = np.sum(np.abs(channel) ** 2)
    return channel * np.sqrt(N * F / power)


def gain_loss_at_point(
    N: int,
    bandwidth: float,
    r_frac_rayleigh: float,
    center_freq: float,
    n_freq_bins: int = 33,
) -> dict:
    """Compute gain-loss (dB, relative to the combined corrector) for all
    three non-ideal beamformer modes at a single (N, bandwidth, r) point.

    Args:
        N: array size.
        bandwidth: signal bandwidth, Hz.
        r_frac_rayleigh: true user distance, expressed as a fraction/
            multiple of THIS array's own Rayleigh distance.
        center_freq: carrier frequency, Hz.
        n_freq_bins: number of frequency bins to evaluate the channel at
            across the bandwidth (33 is enough to resolve squint's
            linear-in-frequency phase error without being wasteful).

    Returns:
        dict with keys: N, bandwidth, r, r_frac_rayleigh, rayleigh_distance,
        gain_loss_conventional_db, gain_loss_nearfield_db, gain_loss_squint_db.
    """
    array = ArrayGeometry(N=N, center_freq=center_freq)
    rayleigh = array.rayleigh_distance()
    r = r_frac_rayleigh * rayleigh

    if bandwidth > 0:
        freqs = center_freq + np.linspace(-bandwidth / 2, bandwidth / 2, n_freq_bins)
    else:
        freqs = np.array([center_freq])

    true_channel = _normalize_channel(near_field_channel(array, r, THETA, freqs))

    gain_combined = beamform_gain(
        beamform("combined", array, freqs, THETA, r=r), true_channel
    )
    gain_conventional = beamform_gain(
        beamform("conventional", array, freqs, THETA, center_freq=center_freq), true_channel
    )
    gain_nearfield = beamform_gain(
        beamform("nearfield", array, freqs, THETA, r=r, center_freq=center_freq), true_channel
    )
    gain_squint = beamform_gain(
        beamform("squint", array, freqs, THETA), true_channel
    )

    return {
        "N": N,
        "bandwidth": bandwidth,
        "r": r,
        "r_frac_rayleigh": r_frac_rayleigh,
        "rayleigh_distance": rayleigh,
        "gain_loss_conventional_db": 10 * np.log10(gain_combined / gain_conventional),
        "gain_loss_nearfield_db": 10 * np.log10(gain_combined / gain_nearfield),
        "gain_loss_squint_db": 10 * np.log10(gain_combined / gain_squint),
    }


def run_sweep(
    Ns: list,
    bandwidths: list,
    r_fracs: list,
    center_freq: float,
    n_freq_bins: int = 33,
) -> pd.DataFrame:
    """Run the full (N x bandwidth x r_frac) grid sweep.

    Returns a pandas DataFrame, one row per grid point, with all gain-loss
    columns from gain_loss_at_point.
    """
    rows = []
    for N in Ns:
        for bw in bandwidths:
            for r_frac in r_fracs:
                rows.append(
                    gain_loss_at_point(N, bw, r_frac, center_freq, n_freq_bins=n_freq_bins)
                )
    return pd.DataFrame(rows)