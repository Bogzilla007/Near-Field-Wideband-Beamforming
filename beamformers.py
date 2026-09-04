"""
Stage 3 -- Baseline Conventional Beamforming

Provides:
    - conventional_steering_vector: single frequency-independent phase
      steering vector (the "broken at scale" baseline), applied uniformly
      across the whole signal bandwidth -- deliberately ignores frequency
      dependence, which is exactly what makes it vulnerable to beam squint
      later (Stage 4/5).
    - beamform_gain: computes Gain = |steering_vector^H . channel|^2 per
      the project's metric definition (Section 2.6), evaluated per
      frequency and summed/averaged across the signal band.

Gain convention (pinned down explicitly, per project doc Section 2.6):
    Gain = |a^H . h|^2, using the RAW (unnormalized, unit-per-element-
    amplitude) steering vector dotted with the channel vector. With a
    matched, unit-amplitude-per-element steering vector against a
    far-field channel of the same convention, coherent combining across
    N elements gives |a^H . h|^2 = |N|^2 = N^2 -- this is the classical
    "power pattern peak" scaling, not the N-scaling of an SNR-normalized
    array gain (which would come from a 1/N-normalized steering vector
    instead). This project uses the UNNORMALIZED convention, validated
    against N^2 in validate_stage3.py.
"""

from __future__ import annotations

import numpy as np

from arraymodel import C, ArrayGeometry


def conventional_steering_vector(
    array: ArrayGeometry,
    theta: float,
    freqs: np.ndarray,
    center_freq: float,
) -> np.ndarray:
    """Conventional phase-shifter steering vector.

    A single phase shift per element, computed at the CENTER frequency
    only, then applied unchanged across every frequency in `freqs`. This
    is what makes it "conventional" -- it has no frequency dependence,
    which is correct at the carrier but increasingly wrong for
    off-carrier frequencies as bandwidth grows (beam squint, Stage 5).

    a_conventional[n, f] = exp(j * 2*pi*center_freq/c * d_n * sin(theta))
                            for every f (no f-dependence)

    Args:
        array: ArrayGeometry.
        theta: steering angle, radians.
        freqs: (F,) frequencies the signal occupies (Hz). Only used to
            determine output shape -- the phase itself does not depend on
            these, by construction.
        center_freq: carrier frequency (Hz) the phase shift is computed at.

    Returns:
        (N, F) complex steering vector, constant across the F axis.
    """
    d_n = array.positions[:, None]  # (N, 1)
    phase = 2 * np.pi * center_freq / C * d_n * np.sin(theta)  # (N, 1), no f dependence
    a_single_freq = np.exp(1j * phase)  # (N, 1)
    F = np.atleast_1d(freqs).shape[0]
    return np.repeat(a_single_freq, F, axis=1)  # (N, F), broadcast across frequency


def beamform_gain(steering: np.ndarray, channel: np.ndarray) -> float:
    """Compute beamforming gain per the project's Section 2.6 metric.

    Gain = |steering^H . channel|^2, evaluated per-frequency and averaged
    across the frequency axis (so multi-frequency wideband channels/
    steering vectors reduce to a single scalar gain figure, consistent
    with how this will be used across narrowband and wideband cases alike
    in later stages).

    Args:
        steering: (N, F) steering vector used by the beamformer under test.
        channel: (N, F) true channel response (always the exact near-field,
            per-frequency model -- see project doc Section 2.6).

    Returns:
        Scalar gain (average over frequency bins of |a^H h|^2 per bin).
    """
    # Per-frequency inner product: sum over elements (axis 0), for each freq.
    inner = np.sum(np.conj(steering) * channel, axis=0)  # (F,)
    gain_per_freq = np.abs(inner) ** 2
    return float(np.mean(gain_per_freq))