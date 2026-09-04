"""
Beam pattern visualization -- reusable across Stage 4's convergence checks
and later stages (Stage 5b's interactive dashboard/failure-map explorer).

This is the "watch the beam split apart and refocus" visual: given a fixed
steering vector, sweep the TRUE angle of arrival and plot the resulting
beamformer response magnitude on a polar axis. Doing this PER FREQUENCY
(rather than averaged) is what makes beam squint visible: a conventional
steering vector's peak response angle shifts with frequency, so plotting
several frequencies on the same polar axes shows multiple lobes pointing
in different directions.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from arraymodel import ArrayGeometry
from channel import far_field_channel, near_field_channel


def compute_beam_pattern(
    steering: np.ndarray,
    array: ArrayGeometry,
    freqs: np.ndarray,
    r: float = None,
    theta_grid: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sweep true angle-of-arrival and compute per-frequency response magnitude.

    Args:
        steering: (N, F) fixed steering vector under test (from beamform()).
        array: ArrayGeometry.
        freqs: (F,) frequencies matching the steering vector's F axis.
        r: if given, sweep using the exact near-field channel at this fixed
            distance (shows near-field blur). If None, sweep using the
            far-field model (classic angle-only beam pattern, shows squint
            cleanly without near-field effects mixed in).
        theta_grid: (n_theta,) angles to evaluate, radians. Defaults to
            -90..90 degrees at 1-degree resolution.

    Returns:
        (theta_grid, response_db) where response_db has shape
        (n_theta, F) -- one curve per frequency, in dB relative to the
        per-frequency peak.
    """
    if theta_grid is None:
        theta_grid = np.deg2rad(np.linspace(-90, 90, 361))

    F = np.atleast_1d(freqs).shape[0]
    n_theta = theta_grid.shape[0]
    response = np.zeros((n_theta, F), dtype=complex)

    for i, th in enumerate(theta_grid):
        if r is None:
            ch = far_field_channel(array, th, freqs)          # (N, F)
        else:
            ch = near_field_channel(array, r, th, freqs)      # (N, F)
        # Per-frequency inner product, vectorized across F (no loop over freq).
        response[i, :] = np.sum(np.conj(steering) * ch, axis=0)

    mag = np.abs(response) ** 2                                # (n_theta, F)
    mag_db = 10 * np.log10(mag / (mag.max(axis=0, keepdims=True) + 1e-30) + 1e-12)
    return theta_grid, mag_db


def plot_beam_pattern(
    steering: np.ndarray,
    array: ArrayGeometry,
    freqs: np.ndarray,
    r: float = None,
    freq_labels: list = None,
    title: str = "Beam pattern",
    ax=None,
    floor_db: float = -40.0,
):
    """Polar plot of beam pattern, one curve per frequency in `freqs`.

    Passing multiple frequencies spanning the signal bandwidth is what
    makes squint visible as multiple distinct lobes; passing r makes
    near-field blur visible (widened/shifted main lobe vs. the far-field
    case). This function does the plotting only -- compute_beam_pattern
    does the math, kept separate so Stage 5b's dashboard can reuse the
    math without matplotlib in the loop.

    Args:
        steering: (N, F) fixed steering vector under test.
        array: ArrayGeometry.
        freqs: (F,) frequencies matching steering's F axis.
        r: optional fixed distance for near-field sweep (see compute_beam_pattern).
        freq_labels: optional list of labels per frequency, for the legend.
        title: plot title.
        ax: optional existing polar matplotlib axis to draw on.
        floor_db: radial axis floor, dB.
    """
    theta_grid, mag_db = compute_beam_pattern(steering, array, freqs, r=r)
    mag_db_clipped = np.clip(mag_db, floor_db, 0)

    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))

    F = mag_db.shape[1]
    if freq_labels is None:
        freq_labels = [f"{f/1e9:.2f} GHz" for f in np.atleast_1d(freqs)]

    for fi in range(F):
        ax.plot(theta_grid, mag_db_clipped[:, fi], label=freq_labels[fi])

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.set_ylim(floor_db, 0)
    ax.set_title(title)
    if F > 1:
        ax.legend(loc="lower left", bbox_to_anchor=(-0.1, -0.1), fontsize=8)
    return ax