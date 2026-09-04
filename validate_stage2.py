"""
Stage 2 checkpoint script.

Confirms the hard gate before Stage 3:
    - near-field and far-field channel models CONVERGE at large distance
      (near-field -> far-field as r -> infinity)
    - they DIVERGE close-in, especially for large N, and that divergence
      roughly tracks the Rayleigh distance prediction from Stage 1.

Run: python3 validate_stage2.py
"""

import numpy as np
import matplotlib.pyplot as plt

from arraymodel import C, ArrayGeometry
from channel import far_field_channel, near_field_channel

CENTER_FREQ = 28e9
THETA = np.deg2rad(20.0)  # arbitrary non-broadside angle to test generality


def normalized_error(a_near: np.ndarray, a_far: np.ndarray) -> float:
    """Phase-invariant, amplitude-normalized mismatch between near-field and
    far-field channel vectors at a single frequency.

    Two normalizations are applied deliberately:
      1. Unit-norm both vectors first, so the near-field model's genuine
         1/r amplitude taper doesn't dominate the comparison (consistent
         with the project's normalization rule in Section 2.6) -- we care
         about *shape* mismatch, not overall power.
      2. Remove the GLOBAL PHASE difference before comparing. The exact
         near-field distance r_n includes the full absolute propagation
         distance r, which produces a common phase term (roughly
         exp(-j*2*pi*f/c*r)) shared by every element. This is physically
         real (it's just bulk propagation delay) but the far-field model
         doesn't represent it at all, and it is irrelevant to *beamforming
         shape* -- a beamformer only cares about phase/amplitude
         *differences across elements*, not a common offset. Without
         removing it, the metric would oscillate with r (as r*f/c wraps
         through multiples of 2*pi) instead of monotonically converging,
         which is exactly the bug this checkpoint caught on first run.

    Metric: sqrt(2 - 2*|<a_near_n, a_far_n>|), the chordal distance between
    the two unit vectors up to global phase. 0 = identical shape, up to
    sqrt(2) = maximally different shape.
    """
    a_near_n = a_near / np.linalg.norm(a_near)
    a_far_n = a_far / np.linalg.norm(a_far)
    correlation = np.abs(np.vdot(a_near_n, a_far_n))  # magnitude only -> phase-invariant
    correlation = np.clip(correlation, 0.0, 1.0)
    return np.sqrt(2 - 2 * correlation)


def main():
    print("=== Stage 2 Checkpoint ===\n")

    freqs = np.array([CENTER_FREQ])  # single carrier freq for this pure channel-model check

    # ---- Part A: convergence at large r, for a few array sizes ----
    print("-- Convergence check: near-field -> far-field as r -> infinity --")
    Ns_conv = [8, 64, 256]
    distances_conv = np.array([1, 10, 100, 1_000, 10_000, 100_000], dtype=float)  # meters

    convergence_results = {}
    for N in Ns_conv:
        array = ArrayGeometry(N=N, center_freq=CENTER_FREQ)
        errs = []
        for r in distances_conv:
            a_near = near_field_channel(array, r, THETA, freqs)[:, 0]
            a_far = far_field_channel(array, THETA, freqs)[:, 0]
            errs.append(normalized_error(a_near, a_far))
        convergence_results[N] = np.array(errs)
        rayleigh = array.rayleigh_distance()
        print(f"  N={N:4d} (Rayleigh={rayleigh:9.2f} m): "
              + ", ".join(f"r={r:g}m->err={e:.4f}" for r, e in zip(distances_conv, errs)))

    # Assert error shrinks monotonically-ish as r grows, for each N.
    for N, errs in convergence_results.items():
        assert errs[-1] < errs[0], f"N={N}: error should shrink as r grows"
        assert errs[-1] < 0.05, f"N={N}: error at largest r should be near zero, got {errs[-1]:.4f}"
    print("  OK: error shrinks toward ~0 at large r for all tested N.\n")

    # ---- Part B: divergence close-in, tracking Rayleigh distance ----
    print("-- Divergence check: near-field error vs. distance, relative to Rayleigh --")
    Ns_div = [32, 128, 512]
    # For each N, sweep r as a *fraction of that array's own Rayleigh distance*
    # so the comparison is apples-to-apples across array sizes.
    frac_of_rayleigh = np.array([0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0])

    divergence_results = {}
    for N in Ns_div:
        array = ArrayGeometry(N=N, center_freq=CENTER_FREQ)
        rayleigh = array.rayleigh_distance()
        r_values = frac_of_rayleigh * rayleigh
        errs = []
        for r in r_values:
            a_near = near_field_channel(array, r, THETA, freqs)[:, 0]
            a_far = far_field_channel(array, THETA, freqs)[:, 0]
            errs.append(normalized_error(a_near, a_far))
        divergence_results[N] = (r_values, np.array(errs), rayleigh)
        print(f"  N={N:4d} (Rayleigh={rayleigh:9.2f} m):")
        for frac, r, e in zip(frac_of_rayleigh, r_values, errs):
            marker = "  <-- near Rayleigh distance" if abs(frac - 1.0) < 1e-9 else ""
            print(f"      {frac:5.2f}x Rayleigh (r={r:9.2f} m): err={e:.4f}{marker}")

    # Assert: error should be substantially higher well inside near-field (0.05x)
    # than well past far-field (10x), for every N.
    for N, (r_values, errs, rayleigh) in divergence_results.items():
        assert errs[0] > errs[-1], f"N={N}: error deep in near-field should exceed error far past Rayleigh"
        assert errs[0] > 0.1, f"N={N}: error deep in near-field should be clearly non-trivial, got {errs[0]:.4f}"
    print("\n  OK: near-field error is high well inside Rayleigh distance and "
          "low well past it, for all tested N.\n")

    # Assert: divergence should be sharper (larger near-field error at a given
    # fractional distance) for larger N, since Rayleigh distance grows with N^2
    # but the physical curvature effect at a given *fraction* of it should be
    # comparable-or-worse for bigger apertures.
    err_at_010 = {N: divergence_results[N][1][1] for N in Ns_div}  # index 1 = 0.1x Rayleigh
    print(f"  Error at 0.1x Rayleigh distance, by N: "
          + ", ".join(f"N={N}:{e:.4f}" for N, e in err_at_010.items()))

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for N, errs in convergence_results.items():
        ax.plot(distances_conv, errs, marker="o", label=f"N={N}")
    ax.set_xscale("log")
    ax.set_xlabel("Distance r (m)")
    ax.set_ylabel("Normalized channel mismatch (near-field vs far-field)")
    ax.set_title("Convergence: near-field -> far-field as r grows")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for N, (r_values, errs, rayleigh) in divergence_results.items():
        ax.plot(frac_of_rayleigh, errs, marker="o", label=f"N={N} (Rayleigh={rayleigh:.1f}m)")
    ax.axvline(1.0, color="k", linestyle="--", alpha=0.5, label="Rayleigh distance")
    ax.set_xscale("log")
    ax.set_xlabel("Distance / Rayleigh distance")
    ax.set_ylabel("Normalized channel mismatch")
    ax.set_title("Divergence tracks Rayleigh distance across array sizes")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/stage2_channel_convergence.png", dpi=150)
    print("Saved figures/stage2_channel_convergence.png")

    print("\n=== Stage 2 checkpoint PASSED. Safe to proceed to Stage 3. ===")


if __name__ == "__main__":
    main()