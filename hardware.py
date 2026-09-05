"""
Stage 9 -- Quantized Phase Shifters

Real phase-shifter hardware can only realize a finite set of discrete
phase states (n_bits -> 2^n_bits levels), not a continuous phase value.
This module adds that constraint on top of the existing (ideal,
continuous-phase) steering vectors from beamformers.py, and applies it
to BOTH the broken baseline (conventional/near-field-aware) AND the
combined corrector -- per the Extension Plan's v2 scope, this is
deliberately NOT limited to just the baseline, so we can ask "how much
of Stage 6's dB-recovery survives once BOTH sides face the same
realistic hardware constraint", not just "how much worse does the
already-broken baseline get".

Amplitude is left untouched (only phase is quantized) -- this matches
the plan's stated scope ("amplitude untouched, or optionally also
quantized as a follow-up").
"""

from __future__ import annotations

import numpy as np

from arraymodel import ArrayGeometry
from beamformers import beamform


def quantize_phase(steering: np.ndarray, n_bits: int) -> np.ndarray:
    """Round each element's phase to the nearest of 2^n_bits discrete
    levels spanning [-pi, pi), leaving amplitude untouched.

    Args:
        steering: (N, F) complex steering vector (from beamform()).
        n_bits: phase-shifter resolution. Common real-hardware values
            are 3, 4, 6 bits; 1-2 bits are included in Stage 9's sweep
            specifically to show a visible, explainable gain penalty
            at unrealistically coarse resolution (sanity check that
            quantization is doing something, not a realistic target).

    Returns:
        (N, F) complex steering vector with quantized phase, same
        amplitude as the input.
    """
    n_levels = 2 ** n_bits
    amplitude = np.abs(steering)
    phase = np.angle(steering)

    # Map phase to the nearest of n_levels discrete points uniformly
    # spaced over [-pi, pi). Standard "round to nearest quantization
    # step" construction: divide by step size, round, multiply back.
    step = 2 * np.pi / n_levels
    quantized_phase = np.round(phase / step) * step

    return amplitude * np.exp(1j * quantized_phase)


def quantized_beamform(
    mode: str,
    array: ArrayGeometry,
    freqs: np.ndarray,
    theta: float,
    n_bits: int,
    r: float = None,
    center_freq: float = None,
) -> np.ndarray:
    """Build a steering vector via the existing beamform() dispatcher,
    then apply phase quantization on top. Same signature as beamform()
    plus n_bits, so Stage 9's checkpoint can call any of the four modes
    interchangeably at any bit depth, including n_bits=None meaning
    "no quantization" (passthrough) for convenient side-by-side
    comparison against the ideal case.

    Args:
        mode: one of "conventional", "nearfield", "squint", "combined"
            (see beamformers.beamform for details).
        n_bits: phase-shifter resolution. If None, returns the ideal
            (unquantized) steering vector unchanged -- lets callers
            treat "ideal" as just another point on the same code path.
        (all other args match beamform())

    Returns:
        (N, F) complex steering vector, phase-quantized if n_bits is
        not None.
    """
    steering = beamform(mode, array, freqs, theta, r=r, center_freq=center_freq)
    if n_bits is None:
        return steering
    return quantize_phase(steering, n_bits)