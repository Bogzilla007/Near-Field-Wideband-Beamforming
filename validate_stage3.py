"""
Stage 3 checkpoint script.

Hard gate before Stage 4 (non-negotiable per project doc): validate the
baseline conventional beamformer under "easy" conditions -- small N,
narrowband signal, far user (so squint and near-field error are both
negligible) -- and confirm gain matches the theoretical N^2 power-pattern
scaling (see beamformers.py docstring for why N^2, not N, is the pinned
convention here).

True channel is always generated from the exact near-field model (project
doc Section 2.6), evaluated at a distance far past Rayleigh distance so it
should closely approximate the far-field case for this "easy" validation.

Run: python3 validate_stage3.py
"""

import numpy as np
import matplotlib.pyplot as plt

from arraymodel import C, ArrayGeometry
from channel import near_field_channel
from beamformers import conventional_steering_vector, beamform_gain

CENTER_FREQ = 28e9
THETA = np.deg2rad(20.0)
NARROWBAND_HZ = 1e6  # 1 MHz -- effectively a single tone for this "easy" check


def main():
    print("=== Stage 3 Checkpoint ===\n")
    print("Validating baseline conventional beamformer: small N, narrowband, "
          "far-field distance (>> Rayleigh distance)\n")

    Ns = [8, 16, 32, 64]
    gains = []
    freqs = np.array([CENTER_FREQ])  # single frequency -> genuinely narrowband

    for N in Ns:
        array = ArrayGeometry(N=N, center_freq=CENTER_FREQ)
        rayleigh = array.rayleigh_distance()
        r_far = rayleigh * 50  # well past Rayleigh distance -> near-field ~ far-field

        # True channel: always the exact near-field model, per Section 2.6.
        true_channel = near_field_channel(array, r_far, THETA, freqs)  # (N, 1)

        # Normalize the channel to the SAME convention as the far-field
        # model (unit modulus per element -> total power = N), per the
        # project's normalization rule (Section 2.6): the near-field
        # model's genuine 1/r amplitude falloff must not be allowed to
        # contaminate the gain comparison across N. Without this, r_far
        # (= 50x Rayleigh distance, which itself grows as N^2) makes the
        # 1/r term collapse for large N and swamps the N^2 array-gain
        # signal we're actually trying to measure here -- this is exactly
        # the bug this checkpoint caught on first run.
        channel_power = np.sum(np.abs(true_channel) ** 2)
        true_channel = true_channel * np.sqrt(N / channel_power)

        # Baseline beamformer steers using the SAME angle, conventional
        # (frequency-independent) steering vector.
        steering = conventional_steering_vector(array, THETA, freqs, CENTER_FREQ)  # (N, 1)

        gain = beamform_gain(steering, true_channel)
        gains.append(gain)

        theoretical_N2 = float(N**2)
        print(f"  N={N:3d}: raw_gain={gain:14.4f}  theoretical_N^2={theoretical_N2:10.1f}  "
              f"ratio(raw/N^2)={gain/theoretical_N2:.4f}")

    gains = np.array(gains)
    Ns_arr = np.array(Ns, dtype=float)

    # ---- Fit scaling exponent: gain ~ N^p, check p ~ 2 ----
    log_N = np.log(Ns_arr)
    log_gain = np.log(gains)
    p, log_c = np.polyfit(log_N, log_gain, 1)
    print(f"\n  Fitted scaling exponent p (gain ~ N^p): {p:.3f}  (expect ~2.0)")

    assert 1.85 < p < 2.15, f"Expected N^2 scaling, got exponent {p:.3f}"
    print("  OK: gain scales as N^2, matching the pinned power-pattern convention.\n")

    # ---- Sanity: squint/near-field error should be negligible here ----
    # Compare against the exact same channel used, at the SAME angle, to
    # confirm no systematic mismatch is being introduced by the "easy"
    # far-distance / narrowband setup itself.
    print("-- Sanity: gain-loss-style check should be ~0 dB in this easy regime --")
    for N in [8, 64]:
        array = ArrayGeometry(N=N, center_freq=CENTER_FREQ)
        rayleigh = array.rayleigh_distance()
        r_far = rayleigh * 50
        true_channel = near_field_channel(array, r_far, THETA, freqs)
        channel_power = np.sum(np.abs(true_channel) ** 2)
        true_channel = true_channel * np.sqrt(N / channel_power)  # same normalization as above
        steering = conventional_steering_vector(array, THETA, freqs, CENTER_FREQ)

        # "Ideal" reference here = perfectly phase-matched to the true channel
        # (i.e. steering = normalized true_channel itself), since Stage 4's
        # combined corrector doesn't exist yet -- this is a local sanity
        # check, not the project's final metric.
        ideal_steering = true_channel / np.abs(true_channel)  # unit modulus, matched phase
        gain_conventional = beamform_gain(steering, true_channel)
        gain_ideal = beamform_gain(ideal_steering, true_channel)
        gain_loss_db = 10 * np.log10(gain_ideal / gain_conventional)
        print(f"  N={N:3d}: gain_loss_conventional_vs_ideal = {gain_loss_db:.4f} dB "
              f"(expect ~0 dB in this easy regime)")
        assert gain_loss_db < 0.5, f"N={N}: unexpected gain loss in easy regime ({gain_loss_db:.4f} dB)"

    print("\n  OK: negligible gain loss confirms conventional beamformer behaves "
          "correctly under easy conditions.\n")

        # ---- Negative control: deliberately mis-steer the angle ----
    # If the formula were broken in a way that ignores angle (e.g. always
    # returning N^2 regardless of input), this check would fail to show any
    # drop. A correct implementation must show gain collapsing well below
    # N^2 when the steering vector points somewhere the signal isn't.
    print("-- Negative control: mis-steered angle should NOT give N^2 --")
    for N in [8, 64]:
        array = ArrayGeometry(N=N, center_freq=CENTER_FREQ)
        rayleigh = array.rayleigh_distance()
        r_far = rayleigh * 50
        true_channel = near_field_channel(array, r_far, THETA, freqs)
        channel_power = np.sum(np.abs(true_channel) ** 2)
        true_channel = true_channel * np.sqrt(N / channel_power)

        theta_wrong = THETA + np.deg2rad(30.0)  # 30 degrees off target
        steering_wrong = conventional_steering_vector(array, theta_wrong, freqs, CENTER_FREQ)
        gain_wrong = beamform_gain(steering_wrong, true_channel)
        theoretical_N2 = float(N**2)
        print(f"  N={N:3d}: mis-steered gain={gain_wrong:10.4f}  "
              f"vs theoretical_N^2={theoretical_N2:10.1f}  "
              f"ratio={gain_wrong/theoretical_N2:.4f}  (expect << 1.0)")
        assert gain_wrong < 0.5 * theoretical_N2, (
            f"N={N}: mis-steered gain should collapse well below N^2, "
            f"got ratio {gain_wrong/theoretical_N2:.4f} -- formula may be "
            f"ignoring angle entirely"
        )
    print("\n  OK: gain collapses when mis-steered, confirming the formula is "
          "genuinely angle-sensitive, not trivially returning N^2 regardless "
          "of input.\n")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(Ns_arr, gains, marker="o", label="Measured gain")
    ax.loglog(Ns_arr, Ns_arr**2 * (gains[0] / Ns_arr[0]**2), linestyle="--",
              label="Theoretical $N^2$ (anchored at N=8)")
    ax.set_xlabel("Array size N")
    ax.set_ylabel("Beamforming gain")
    ax.set_title("Stage 3: conventional beamformer gain vs. N (easy regime)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/stage3_gain_vs_N.png", dpi=150)
    print("Saved figures/stage3_gain_vs_N.png")

    print("\n=== Stage 3 checkpoint PASSED. Baseline beamformer validated. "
          "Safe to proceed to Stage 4. ===")


if __name__ == "__main__":
    main()