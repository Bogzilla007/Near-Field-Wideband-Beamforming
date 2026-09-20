"""
Stage 7 -- Sensing Illustration (optional stretch tier)

A single matched-filter range estimator, applied at exactly the 3
representative distance points already used as Stage 5's coarse sweep
distance axis (0.1x, 1x, 10x Rayleigh distance) -- deliberately reusing
those points rather than choosing new ones, so this stage stays visibly
tethered to Stage 5's results rather than becoming a fresh mini-study.

Not a full estimator (no MUSIC, no angle estimation) -- just: given the
known steering angle, scan candidate distances and find the one whose
combined-corrector steering vector best matches the received near-field
channel. This is the natural range-estimation analog of the same
gain-maximization machinery already built for beamforming.
"""

from __future__ import annotations

import numpy as np

from arraymodel import ArrayGeometry
from channel import near_field_channel
from beamformers import beamform, beamform_gain
from constants import THETA


def estimate_range(
    array: ArrayGeometry,
    r_true: float,
    freqs: np.ndarray,
    r_candidates: np.ndarray,
) -> dict:
    """Matched-filter range estimate: scan r_candidates, pick the one whose
    combined-corrector steering vector gives maximum gain against the true
    received near-field channel.

    Args:
        array: ArrayGeometry.
        r_true: true user distance, meters (used only to generate the
            received signal -- the estimator does not see this value).
        freqs: (F,) frequencies the signal occupies.
        r_candidates: (n_candidates,) grid of candidate distances to test.

    Returns:
        dict with 'r_true', 'r_estimated', 'relative_error'.
    """
    true_channel = near_field_channel(array, r_true, THETA, freqs)
    power = np.sum(np.abs(true_channel) ** 2)
    true_channel = true_channel * np.sqrt(array.N * len(np.atleast_1d(freqs)) / power)

    gains = np.array([
        beamform_gain(beamform("combined", array, freqs, THETA, r=rc), true_channel)
        for rc in r_candidates
    ])
    r_estimated = r_candidates[np.argmax(gains)]
    relative_error = abs(r_estimated - r_true) / r_true

    return {
        "r_true": r_true,
        "r_estimated": r_estimated,
        "relative_error": relative_error,
    }