"""
Stage 7 checkpoint script.

Deliverable: a single table -- distance point x range estimation error --
using EXACTLY the 3 distance points from Stage 5's coarse sweep distance
axis (0.1x, 1x, 10x Rayleigh distance), at the same (N, bandwidth) as one
of that coarse grid's cells, so each row can be tied directly back to that
point's already-computed Stage 5 gain-loss number.

Run: python3 validate_stage7.py
"""

import numpy as np
import pandas as pd

from arraymodel import ArrayGeometry
from sensing import estimate_range

CENTER_FREQ = 28e9
N_TEST = 128           # matches a Stage 5 coarse sweep array size
BANDWIDTH = 400e6       # matches a Stage 5 coarse sweep bandwidth
R_FRACS = [0.1, 1.0, 10.0]  # exactly Stage 5's coarse distance axis


def main():
    print("=== Stage 7 Checkpoint (optional stretch) ===\n")

    array = ArrayGeometry(N=N_TEST, center_freq=CENTER_FREQ)
    rayleigh = array.rayleigh_distance()
    freqs = CENTER_FREQ + np.linspace(-BANDWIDTH / 2, BANDWIDTH / 2, 33)

    # Candidate grid for the matched-filter search -- log-spaced, spanning
    # well beyond the plausible range so the estimator isn't hand-fed the
    # answer's neighborhood.
    r_candidates = np.logspace(np.log10(0.01 * rayleigh), np.log10(200 * rayleigh), 400)

    # Pull the matching Stage 5 coarse-sweep gain-loss numbers for context.
    coarse = pd.read_csv("results/coarse_sweep.csv")

    print(f"Array: N={N_TEST}, bandwidth={BANDWIDTH/1e6:.0f} MHz, "
          f"Rayleigh distance={rayleigh:.2f} m\n")

    rows = []
    for r_frac in R_FRACS:
        r_true = r_frac * rayleigh
        result = estimate_range(array, r_true, freqs, r_candidates)

        match = coarse[(coarse.N == N_TEST) & (coarse.bandwidth == BANDWIDTH) &
                        (np.isclose(coarse.r_frac_rayleigh, r_frac))]
        gain_loss_conv = match.iloc[0].gain_loss_conventional_db if len(match) else float("nan")

        rows.append({
            "r_frac_rayleigh": r_frac,
            "r_true_m": r_true,
            "r_estimated_m": result["r_estimated"],
            "relative_error_pct": result["relative_error"] * 100,
            "stage5_gain_loss_conventional_db": gain_loss_conv,
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv("results/stage7_sensing.csv", index=False)

    print("\n-- One-line interpretation per point --")
    for _, row in df.iterrows():
        print(f"  At r={row.r_frac_rayleigh}x Rayleigh: range estimation error = "
              f"{row.relative_error_pct:.2f}%, coinciding with a "
              f"{row.stage5_gain_loss_conventional_db:.2f} dB conventional "
              f"beamforming gain loss at that same point.")

    assert (df.relative_error_pct < 20).all(), "range estimates should be broadly reasonable"
    print("\n=== Stage 7 checkpoint PASSED (stretch tier complete). ===")


if __name__ == "__main__":
    main()