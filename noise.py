"""
Stage 10 -- SNR / Noise Floor

Adds a receiver noise floor on top of the existing (noiseless) gain
metric, and reframes results in terms a link-budget conversation
actually cares about: effective output SNR, BER, and EVM.

IMPORTANT DESIGN NOTE (found while building this stage, before writing
any checkpoint assertions -- see validate_stage10.py for the full
derivation and a numeric confirmation):

Every beamformer mode in this project (beamformers.py) is normalized so
that sum_n |a_n|^2 = N, identically, regardless of mode. Since receiver
noise is independent across elements, the NOISE power after combining
with any steering vector `a` is sigma^2 * sum_n|a_n|^2 = sigma^2 * N --
exactly the same for every mode. Only the SIGNAL power (the numerator,
|a^H h|^2) differs between modes.

Consequence: the RATIO of effective output SNR between any two modes is
IDENTICAL to the existing (noiseless) gain ratio, at every noise level.
The gain-loss-in-dB metric used throughout Stages 3-9 is therefore
provably INDEPENDENT of the receiver noise floor under this project's
fixed-total-array-energy convention -- it cannot "compress" or "blur"
with SNR the way one might naively expect. This is confirmed numerically
in validate_stage10.py before anything else is built on top of it.

What DOES depend on absolute SNR is downstream, NONLINEAR quantities --
BER and EVM. A fixed dB gap between two beamformers has a hugely
different PRACTICAL consequence depending on where you sit on the BER
curve: near the "waterfall" region a few dB matters enormously; deep in
the error-floor (very high SNR) or garbage (very low SNR) regions, a few
dB is nearly irrelevant. That's the actual link-budget story this stage
tells, and it's the reason this module computes BER/EVM rather than
re-deriving a "noise-dependent gain-loss" number that provably cannot
exist under this project's conventions.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erfc


def effective_output_snr(gain: float, n_elements: int, snr_in_db: float) -> float:
    """Effective output SNR (linear, not dB) after beamforming.

    Args:
        gain: beamforming gain for the mode/point under test, i.e.
            beamform_gain(steering, channel) from beamformers.py
            (already averaged over frequency, per this project's
            existing convention).
        n_elements: N, the array size (equivalently sum_n|a_n|^2 for any
            of this project's steering-vector modes, since all are
            normalized to that value).
        snr_in_db: the PER-ELEMENT input SNR (signal power per element
            per frequency bin / noise power per element per frequency
            bin), in dB. Consistent with this project's channel
            normalization (sum|h|^2 = N*F, i.e. unit average signal
            power per element per frequency bin), an input SNR of
            snr_in_db directly sets the per-element noise variance as
            noise_variance = 10^(-snr_in_db/10).

    Returns:
        Effective output SNR (linear).
    """
    noise_variance = 10 ** (-snr_in_db / 10.0)
    noise_power_out = noise_variance * n_elements
    return gain / noise_power_out


def ber_qpsk(snr_db: float) -> float:
    """Closed-form QPSK bit error rate as a function of (effective,
    post-beamforming) SNR per bit, in dB.

    QPSK carries 2 bits/symbol with orthogonal I/Q, so per-bit SNR
    equals per-symbol SNR (each bit sees the full symbol energy per
    dimension) -- standard result: BER = 0.5 * erfc(sqrt(SNR_linear)).
    Reuses the QPSK signal model already built in Stage 1's
    generate_ofdm_signal (arraymodel.py) -- no new modulation scheme
    introduced here, just the matching closed-form error-rate formula.

    Args:
        snr_db: effective SNR, dB.

    Returns:
        Bit error rate, in [0, 0.5].
    """
    snr_linear = 10 ** (snr_db / 10.0)
    return 0.5 * erfc(np.sqrt(snr_linear))


def evm_from_snr(snr_db: float) -> float:
    """Error Vector Magnitude (%) from effective SNR, standard closed-form
    relationship: EVM = 1/sqrt(SNR_linear).

    Args:
        snr_db: effective SNR, dB.

    Returns:
        EVM as a percentage (0-100 scale; values are typically well
        under 100 for any usable link).
    """
    snr_linear = 10 ** (snr_db / 10.0)
    return 100.0 / np.sqrt(snr_linear)