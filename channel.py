"""
Stage 2 -- Channel Modeling (Near-Field vs. Far-Field)

Provides:
    - far_field_channel: planar-wavefront approximation (phase-only)
    - near_field_channel: exact spherical model (phase AND amplitude)

Both are per-frequency capable (needed later for beam-squint / combined
corrector work in Stage 4), fully vectorized with NumPy broadcasting across
elements and frequencies -- no per-element Python loops anywhere.

Convention: user position given as (r, theta), where r is distance from the
array center (meters) and theta is angle-of-arrival (radians, 0 = broadside).
"""

from __future__ import annotations

import numpy as np

from arraymodel import C, ArrayGeometry


def far_field_channel(
    array: ArrayGeometry,
    theta: float,
    freqs: np.ndarray,
) -> np.ndarray:
    """Far-field planar-wavefront channel model.

    Single angle-of-arrival, linear phase progression across elements.
    Amplitude is constant (no 1/r falloff modeled) -- this is the
    "wrong at scale" model referenced in the project doc, Section 2.3,
    extended here to be evaluated per-frequency (needed for beam squint
    in Stage 4/5).

    a_farfield[n, f] = exp(j * 2*pi*f/c * d_n * sin(theta))

    Args:
        array: ArrayGeometry.
        theta: angle of arrival, radians.
        freqs: (F,) array of frequencies (Hz) to evaluate at.

    Returns:
        (N, F) complex channel matrix.
    """
    d_n = array.positions[:, None]          # (N, 1)
    f = np.atleast_1d(freqs)[None, :]       # (1, F)
    phase = 2 * np.pi * f / C * d_n * np.sin(theta)
    return np.exp(1j * phase)


def near_field_channel(
    array: ArrayGeometry,
    r: float,
    theta: float,
    freqs: np.ndarray,
) -> np.ndarray:
    """Exact near-field spherical channel model.

    Per-element propagation distance computed individually (not
    approximated from a single angle). Both phase AND amplitude vary
    across elements -- this is the key mathematical difference from the
    far-field model. See project doc Section 2.4.

    r_n = sqrt(r^2 + d_n^2 - 2*r*d_n*sin(theta))
    a_nearfield[n, f] = (1/r_n) * exp(-j * 2*pi*f/c * r_n)

    Args:
        array: ArrayGeometry.
        r: distance from array center to user, meters.
        theta: angle of arrival, radians.
        freqs: (F,) array of frequencies (Hz) to evaluate at.

    Returns:
        (N, F) complex channel matrix.
    """
    d_n = array.positions[:, None]          # (N, 1)
    f = np.atleast_1d(freqs)[None, :]       # (1, F)

    r_n = np.sqrt(r**2 + d_n**2 - 2 * r * d_n * np.sin(theta))  # (N, 1)
    amplitude = 1.0 / r_n                                        # (N, 1)
    phase = 2 * np.pi * f / C * r_n                              # (N, F) via broadcast
    return amplitude * np.exp(-1j * phase)