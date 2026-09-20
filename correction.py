"""
Stage 6 -- Correction & Recovery (Strong tier)

Applies the combined corrector (already built and validated in Stage 4) to
the worst-performing region identified by Stage 5's sweep, and quantifies
how much gain is recovered relative to the broken conventional baseline.

Worst-performing region, defined operationally per the project doc:
largest array, widest bandwidth, closest distance -- taken directly from
Stage 5's fine sweep grid (N=512, bandwidth=4GHz, smallest r_fracs
available: 0.02, 0.05, 0.1x Rayleigh distance).

No new beamformer or metric is introduced here -- this stage is purely an
application + measurement exercise, reusing gain_loss_at_point from
sweep.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sweep import gain_loss_at_point


def recovery_table(N: int, bandwidth: float, r_fracs: list, center_freq: float) -> pd.DataFrame:
    """Compute before/after gain and recovery statistics at each point.

    'Before' = conventional beamformer's gain, in dB relative to combined
    ('gain_loss_conventional_db', already 0-referenced against the ideal
    combined corrector). 'After' = combined corrector's gain, which by
    construction has 0 dB loss against itself -- the interesting number is
    how many dB were closed, i.e. exactly the gain_loss_conventional_db
    value itself, reframed as "dB recovered by applying the correction".
    """
    rows = []
    for r_frac in r_fracs:
        point = gain_loss_at_point(N, bandwidth, r_frac, center_freq)
        db_recovered = point["gain_loss_conventional_db"]  # gap closed by switching to combined
        rows.append({
            "N": N,
            "bandwidth": bandwidth,
            "r_frac_rayleigh": r_frac,
            "gain_loss_before_db": db_recovered,   # conventional's loss vs ideal
            "gain_loss_after_db": 0.0,             # combined vs itself, by construction
            "db_recovered": db_recovered,
        })
    return pd.DataFrame(rows)


def cost_estimate(N: int, n_freq_bins: int) -> dict:
    """Rough per-beam computational/hardware cost comparison.

    Conventional (phase-shifter) beamforming: one complex phase multiply
    per element, independent of frequency -- O(N) multiplies per beam
    update.

    Combined corrector (near-field + TTD): a distinct complex weight per
    element PER FREQUENCY BIN (true-time-delay hardware, or an
    frequency-domain equivalent) -- O(N * F) multiplies per beam update,
    where F is the number of frequency bins/taps needed to resolve the
    signal bandwidth.

    This is a rough multiply-count comparison, not a hardware cost model
    (real TTD hardware costs differ in other ways -- delay line
    precision, calibration, etc.) -- intentionally kept at this level of
    detail per the project's scope-control principle.
    """
    conventional_multiplies = N
    combined_multiplies = N * n_freq_bins
    return {
        "N": N,
        "n_freq_bins": n_freq_bins,
        "conventional_multiplies_per_beam": conventional_multiplies,
        "combined_multiplies_per_beam": combined_multiplies,
        "cost_ratio": combined_multiplies / conventional_multiplies,
    }